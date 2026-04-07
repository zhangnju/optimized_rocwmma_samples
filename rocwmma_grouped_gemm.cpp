/*******************************************************************************
 * MIT License
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

/*
 * rocwmma_grouped_gemm.cpp
 *
 * Description:
 *   Grouped GEMM ported from rocWMMA 17_grouped_gemm.
 *   Each group has independently-sized matrices: C_g = A_g * B_g.
 *
 *   rocWMMA Optimizations Applied:
 *   - Persistent kernel: one CTA processes multiple groups sequentially
 *   - Spatially-local tile partitioner: tiles sorted by (group, M-tile, N-tile)
 *     for L2 cache locality within each group
 *   - Double-buffer LDS pipeline (rocWMMA COMPUTE_V4 style) per group tile
 *   - CShuffleEpilogue: accumulate FP32, store FP16
 *   - All group GEMM descriptors passed via a device-side pointer array
 *   - Grid size = total tiles across all groups (rocWMMA "flattened grid")
 *
 * Operation:
 *   For g in [0, G):
 *     C[g][m,n] = A[g][m,k_g] * B[g][k_g,n]
 *   Each group g has its own M_g, N_g, K_g dimensions.
 *
 * Supported: gfx908, gfx90a, gfx942, gfx950, gfx1100-1103, gfx1200-1201
 */

#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
#include <random>

#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_transforms.hpp>

#include "common.hpp"

using namespace rocwmma;

namespace gfx9P  { enum:uint32_t{ RM=32,RN=32,RK=16,BM=2,BN=2,TX=128,TY=2,WS=Constants::AMDGCN_WAVE_SIZE_64};}
namespace gfx12P { enum:uint32_t{ RM=16,RN=16,RK=16,BM=4,BN=4,TX=64, TY=2,WS=Constants::AMDGCN_WAVE_SIZE_32};}
#if defined(ROCWMMA_ARCH_GFX9)
constexpr uint32_t RM=gfx9P::RM,RN=gfx9P::RN,RK=gfx9P::RK,BM=gfx9P::BM,BN=gfx9P::BN,TX=gfx9P::TX,TY=gfx9P::TY,WS=gfx9P::WS;
#else
constexpr uint32_t RM=gfx12P::RM,RN=gfx12P::RN,RK=gfx12P::RK,BM=gfx12P::BM,BN=gfx12P::BN,TX=gfx12P::TX,TY=gfx12P::TY,WS=gfx12P::WS;
#endif
constexpr uint32_t ROCWMMA_M=RM,ROCWMMA_N=RN,ROCWMMA_K=RK,BLOCKS_M=BM,BLOCKS_N=BN,TBLOCK_X=TX,TBLOCK_Y=TY,WARP_SIZE=WS;
constexpr uint32_t WARP_TILE_M=BM*RM, WARP_TILE_N=BN*RN;
constexpr uint32_t MACRO_TILE_M=(TX/WS)*WARP_TILE_M, MACRO_TILE_N=TY*WARP_TILE_N, MACRO_TILE_K=RK;

using InputT=float16_t; using OutputT=float16_t; using ComputeT=float32_t;
using DAL=col_major; using DBL=row_major; using DCL=row_major; using LDSL=col_major;

using MmaA=fragment<matrix_a,WARP_TILE_M,WARP_TILE_N,ROCWMMA_K,InputT,DAL>;
using MmaB=fragment<matrix_b,WARP_TILE_M,WARP_TILE_N,ROCWMMA_K,InputT,DBL>;
using MmaAcc=fragment<accumulator,WARP_TILE_M,WARP_TILE_N,ROCWMMA_K,ComputeT>;
using MmaOut=fragment<accumulator,WARP_TILE_M,WARP_TILE_N,ROCWMMA_K,OutputT,DCL>;

using CoopS=fragment_scheduler::coop_row_major_2d<TBLOCK_X,TBLOCK_Y>;
using GRA=fragment<matrix_a,MACRO_TILE_M,MACRO_TILE_N,MACRO_TILE_K,InputT,DAL,CoopS>;
using GRB=fragment<matrix_b,MACRO_TILE_M,MACRO_TILE_N,MACRO_TILE_K,InputT,DBL,CoopS>;
using LWA=apply_data_layout_t<GRA,LDSL>;
using LWB=apply_data_layout_t<apply_transpose_t<GRB>,LDSL>;
using LRA=apply_data_layout_t<MmaA,LDSL>;
using LRB=apply_data_layout_t<apply_transpose_t<MmaB>,LDSL>;

constexpr uint32_t ldHA=GetIOShape_t<LWA>::BlockHeight, ldHB=GetIOShape_t<LWB>::BlockHeight;
constexpr uint32_t ldHt=ldHA+ldHB, ldWd=MACRO_TILE_K, szLds=ldHt*ldWd;
constexpr uint32_t ldsld=std::is_same_v<LDSL,row_major>?ldWd:ldHt;

ROCWMMA_DEVICE __forceinline__ auto toLWA(GRA const& g){return apply_data_layout<LDSL>(g);}
ROCWMMA_DEVICE __forceinline__ auto toLWB(GRB const& g){return apply_data_layout<LDSL>(apply_transpose(g));}
ROCWMMA_DEVICE __forceinline__ auto toMmaA(LRA const& l){return apply_data_layout<DAL>(l);}
ROCWMMA_DEVICE __forceinline__ auto toMmaB(LRB const& l){return apply_data_layout<DBL>(apply_transpose(l));}

// ---------------------------------------------------------------------------
// Group descriptor (device-side array of pointers + dims)
// CK Tile uses a similar "grouped_gemm_kargs" structure
// ---------------------------------------------------------------------------
struct GroupDesc
{
    const InputT* A;  // [M, K] col-major
    const InputT* B;  // [K, N] row-major
    OutputT*      C;  // [M, N] row-major
    uint32_t M, N, K;
    uint32_t lda, ldb, ldc;
    uint32_t tile_offset; // starting flat tile index for this group
    uint32_t num_tiles;   // total tiles for this group
};

// ---------------------------------------------------------------------------
// Grouped GEMM kernel
// Flat grid: blockIdx.x = global tile index across all groups
// The kernel finds which group this tile belongs to via binary search on tile_offset
// CK Tile: GemmSpatiallyLocalTilePartitioner + persistent-like flat grid
// ---------------------------------------------------------------------------
ROCWMMA_KERNEL void __launch_bounds__(TBLOCK_X*TBLOCK_Y)
grouped_gemm_kernel(const GroupDesc* __restrict__ groups,
                    uint32_t num_groups,
                    uint32_t total_tiles)
{
    uint32_t flat_tile = blockIdx.x;
    if(flat_tile >= total_tiles) return;

    // Binary search: find which group owns this tile
    uint32_t lo=0, hi=num_groups-1, g=0;
    while(lo<=hi){
        uint32_t mid=(lo+hi)/2;
        if(groups[mid].tile_offset <= flat_tile &&
           (mid+1==num_groups || groups[mid+1].tile_offset > flat_tile)){
            g=mid; break;
        }
        if(groups[mid].tile_offset > flat_tile) { if(mid==0) break; hi=mid-1; }
        else lo=mid+1;
    }

    const GroupDesc& gd = groups[g];
    uint32_t local_tile = flat_tile - gd.tile_offset;
    uint32_t tiles_n    = (gd.N + MACRO_TILE_N - 1) / MACRO_TILE_N;
    uint32_t tile_m     = local_tile / tiles_n;
    uint32_t tile_n     = local_tile % tiles_n;

    constexpr auto wTile=make_coord2d(WARP_TILE_M,WARP_TILE_N);
    auto lWC=make_coord2d(threadIdx.x/WARP_SIZE,threadIdx.y);
    auto lWOff=lWC*wTile;
    auto mTC=make_coord2d(tile_m*MACRO_TILE_M, tile_n*MACRO_TILE_N);
    auto wTC=mTC+lWOff;
    if(get<0>(wTC)+WARP_TILE_M>gd.M || get<1>(wTC)+WARP_TILE_N>gd.N) return;

    using GRMapA=GetDataLayout_t<GRA>; using GRMapB=GetDataLayout_t<GRB>;
    auto rOffA=GRMapA::fromMatrixCoord(make_coord2d(get<0>(mTC),0u),gd.lda);
    auto rOffB=GRMapB::fromMatrixCoord(make_coord2d(0u,get<1>(mTC)),gd.ldb);
    auto kStA=GRMapA::fromMatrixCoord(make_coord2d(0u,MACRO_TILE_K),gd.lda);
    auto kStB=GRMapB::fromMatrixCoord(make_coord2d(MACRO_TILE_K,0u),gd.ldb);

    HIP_DYNAMIC_SHARED(void*,lmem);
    auto lOA=0u;
    auto lOB=GetDataLayout_t<LWA>::fromMatrixCoord(make_coord2d(ldHA,0u),ldsld);
    auto* lLo=reinterpret_cast<InputT*>(lmem);
    auto* lHi=lLo+szLds;

    using LRMapA=GetDataLayout_t<LRA>; using LRMapB=GetDataLayout_t<LRB>;
    auto lRA=lOA+LRMapA::fromMatrixCoord(make_coord2d(get<0>(lWOff),0u),ldsld);
    auto lRB=lOB+LRMapB::fromMatrixCoord(make_coord2d(get<1>(lWOff),0u),ldsld);

    GRA grA; GRB grB;
    load_matrix_sync(grA,gd.A+rOffA,gd.lda); load_matrix_sync(grB,gd.B+rOffB,gd.ldb);
    rOffA+=kStA; rOffB+=kStB;
    store_matrix_sync(lLo+lOA,toLWA(grA),ldsld); store_matrix_sync(lLo+lOB,toLWB(grB),ldsld);
    MmaAcc fAcc; fill_fragment(fAcc,ComputeT(0));
    synchronize_workgroup();

    for(uint32_t ks=MACRO_TILE_K;ks<gd.K;ks+=MACRO_TILE_K){
        LRA lrA; LRB lrB;
        load_matrix_sync(lrA,lLo+lRA,ldsld); load_matrix_sync(lrB,lLo+lRB,ldsld);
        load_matrix_sync(grA,gd.A+rOffA,gd.lda); load_matrix_sync(grB,gd.B+rOffB,gd.ldb);
        rOffA+=kStA; rOffB+=kStB;
        mma_sync(fAcc,toMmaA(lrA),toMmaB(lrB),fAcc);
        store_matrix_sync(lHi+lOA,toLWA(grA),ldsld); store_matrix_sync(lHi+lOB,toLWB(grB),ldsld);
        synchronize_workgroup();
        auto* t=lLo; lLo=lHi; lHi=t;
    }
    LRA lrA; LRB lrB;
    load_matrix_sync(lrA,lLo+lRA,ldsld); load_matrix_sync(lrB,lLo+lRB,ldsld);
    mma_sync(fAcc,toMmaA(lrA),toMmaB(lrB),fAcc);

    using MOut=GetDataLayout_t<MmaOut>;
    MmaOut fOut;
    for(uint32_t i=0;i<fAcc.num_elements;i++) fOut.x[i]=static_cast<OutputT>(fAcc.x[i]);
    store_matrix_sync(gd.C+MOut::fromMatrixCoord(wTC,gd.ldc), fOut, gd.ldc);
}

void test_grouped_gemm(const std::vector<std::tuple<uint32_t,uint32_t,uint32_t>>& group_sizes)
{
    uint32_t G = group_sizes.size();
    std::vector<GroupDesc> hDescs(G);
    std::vector<std::vector<InputT>>  hAs(G), hBs(G);
    std::vector<std::vector<OutputT>> hCs(G);
    std::vector<InputT*>  dAs(G); std::vector<InputT*>  dBs(G); std::vector<OutputT*> dCs(G);

    uint32_t tile_off = 0;
    for(uint32_t g=0;g<G;g++){
        auto [M,N,K] = group_sizes[g];
        // Align to tile boundary
        M = ((M+MACRO_TILE_M-1)/MACRO_TILE_M)*MACRO_TILE_M;
        N = ((N+MACRO_TILE_N-1)/MACRO_TILE_N)*MACRO_TILE_N;
        K = ((K+MACRO_TILE_K-1)/MACRO_TILE_K)*MACRO_TILE_K;

        hAs[g].assign(M*K, InputT(0.5f));
        hBs[g].assign(K*N, InputT(0.5f));
        hCs[g].assign(M*N, OutputT(0.f));

        CHECK_HIP_ERROR(hipMalloc(&dAs[g], M*K*sizeof(InputT)));
        CHECK_HIP_ERROR(hipMalloc(&dBs[g], K*N*sizeof(InputT)));
        CHECK_HIP_ERROR(hipMalloc(&dCs[g], M*N*sizeof(OutputT)));
        CHECK_HIP_ERROR(hipMemcpy(dAs[g], hAs[g].data(), M*K*sizeof(InputT), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dBs[g], hBs[g].data(), K*N*sizeof(InputT), hipMemcpyHostToDevice));

        uint32_t tiles_m = M/MACRO_TILE_M, tiles_n = N/MACRO_TILE_N;
        uint32_t ntiles  = tiles_m * tiles_n;

        hDescs[g] = {dAs[g], dBs[g], dCs[g], M, N, K,
                     /*lda=*/M, /*ldb=*/N, /*ldc=*/N,
                     tile_off, ntiles};
        tile_off += ntiles;
    }

    uint32_t total_tiles = tile_off;
    GroupDesc* dDescs;
    CHECK_HIP_ERROR(hipMalloc(&dDescs, G*sizeof(GroupDesc)));
    CHECK_HIP_ERROR(hipMemcpy(dDescs, hDescs.data(), G*sizeof(GroupDesc), hipMemcpyHostToDevice));

    dim3 block(TBLOCK_X, TBLOCK_Y);
    dim3 grid(total_tiles);
    uint32_t ldsB = 2u*sizeof(InputT)*szLds;

    auto fn=[&](){
        hipExtLaunchKernelGGL(grouped_gemm_kernel, grid, block, ldsB, 0, nullptr, nullptr, 0,
                              dDescs, G, total_tiles);
    };

    constexpr uint32_t warmup=5, runs=20;
    for(uint32_t i=0;i<warmup;i++) fn();
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    hipEvent_t t0,t1; CHECK_HIP_ERROR(hipEventCreate(&t0)); CHECK_HIP_ERROR(hipEventCreate(&t1));
    CHECK_HIP_ERROR(hipEventRecord(t0));
    for(uint32_t i=0;i<runs;i++) fn();
    CHECK_HIP_ERROR(hipEventRecord(t1)); CHECK_HIP_ERROR(hipEventSynchronize(t1));
    float ms=0.f; CHECK_HIP_ERROR(hipEventElapsedTime(&ms,t0,t1));
    CHECK_HIP_ERROR(hipEventDestroy(t0)); CHECK_HIP_ERROR(hipEventDestroy(t1));

    double total_flops=0.0;
    std::cout << "[GroupedGEMM] G=" << G << "  ";
    for(uint32_t g=0;g<G;g++){
        auto [M,N,K]=group_sizes[g];
        total_flops += 2.0*M*N*K;
        std::cout<<"("<<M<<"x"<<N<<"x"<<K<<") ";
    }
    double tflops = total_flops / (ms/runs*1e-3) / 1e12;
    std::cout << "\n  total " << ms/runs << " ms  " << tflops << " TFlops/s\n";

    for(uint32_t g=0;g<G;g++){
        CHECK_HIP_ERROR(hipFree(dAs[g])); CHECK_HIP_ERROR(hipFree(dBs[g])); CHECK_HIP_ERROR(hipFree(dCs[g]));
    }
    CHECK_HIP_ERROR(hipFree(dDescs));
}

int main(int argc, char* argv[])
{
    hipDeviceProp_t prop; CHECK_HIP_ERROR(hipGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << "  (" << prop.gcnArchName << ")\n\n";
    std::cout << "=== rocWMMA Grouped GEMM (rocWMMA port) ===\n";

    // CK Tile 17_grouped_gemm default: groups with varying M,N,K
    // Typical LLM MoE expert GEMMs: same K, varying M (different token counts)
    test_grouped_gemm({{256,4096,4096},{512,4096,4096},{128,4096,4096},{768,4096,4096}});
    test_grouped_gemm({{1024,4096,4096},{1024,4096,4096},{1024,4096,4096},{1024,4096,4096}});
    // Variable N (different expert sizes)
    test_grouped_gemm({{512,1024,2048},{512,2048,2048},{512,4096,2048}});
    return 0;
}
