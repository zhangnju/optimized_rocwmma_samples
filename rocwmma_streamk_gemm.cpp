/*******************************************************************************
 * MIT License
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

/*
 * rocwmma_streamk_gemm.cpp
 *
 * Description:
 *   Stream-K GEMM ported from rocWMMA 40_streamk_gemm.
 *
 *   rocWMMA Optimizations Applied:
 *   - Stream-K load balancing: decomposes work into fixed-size "stream-K units"
 *     distributed round-robin across all SMs (persistent kernel style)
 *   - Each SM processes a contiguous range of SK units, accumulating partials
 *   - Partial tiles require inter-CTA reduction via atomicAdd to global workspace
 *   - Full tiles (owned entirely by one CTA) write directly to output
 *   - Eliminates load imbalance seen with fixed tile partitioning on small M/N
 *   - Double-buffer LDS pipeline within each SK unit
 *
 * Key Stream-K concepts:
 *   - SK unit = MACRO_TILE_K slice of one output tile
 *   - Total SK units = (M/MT_M) * (N/MT_N) * (K/MT_K)
 *   - Each CTA processes ceil(total_units / num_CUs) units
 *   - Partial tiles: accumulate in FP32 workspace, then reduce
 *
 * Supported: gfx908, gfx90a, gfx942, gfx950, gfx1100-1103, gfx1200-1201
 */

#include <iomanip>
#include <iostream>
#include <vector>

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

using InputT=float16_t; using OutputT=float16_t; using ComputeT=float32_t; using WorkT=float32_t;

using DAL=col_major; using DBL=row_major; using DCL=row_major; using LDSL=col_major;
using MmaA=fragment<matrix_a,WARP_TILE_M,WARP_TILE_N,ROCWMMA_K,InputT,DAL>;
using MmaB=fragment<matrix_b,WARP_TILE_M,WARP_TILE_N,ROCWMMA_K,InputT,DBL>;
using MmaAcc=fragment<accumulator,WARP_TILE_M,WARP_TILE_N,ROCWMMA_K,ComputeT>;
using MmaOut=fragment<accumulator,WARP_TILE_M,WARP_TILE_N,ROCWMMA_K,OutputT,DCL>;
using MmaWrk=fragment<accumulator,WARP_TILE_M,WARP_TILE_N,ROCWMMA_K,WorkT,DCL>;

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
// Stream-K GEMM Kernel
// Each CTA is assigned a contiguous range of SK units [sk_start, sk_end)
// ---------------------------------------------------------------------------
ROCWMMA_KERNEL void __launch_bounds__(TBLOCK_X*TBLOCK_Y)
streamk_gemm_kernel(uint32_t M, uint32_t N, uint32_t K,
                    InputT const* __restrict__ A,
                    InputT const* __restrict__ B,
                    OutputT*      __restrict__ C,
                    WorkT*        __restrict__ workspace, // [tiles_m*tiles_n, WARP_TILE_M*WARP_TILE_N]
                    uint32_t*     __restrict__ tile_done, // [tiles_m*tiles_n] atomic counter
                    uint32_t lda, uint32_t ldb, uint32_t ldc,
                    uint32_t total_sk_units,   // total K-sliced units across all output tiles
                    uint32_t units_per_cta,    // units this CTA processes
                    uint32_t sk_start)         // first SK unit for this CTA
{
    constexpr auto wTile=make_coord2d(WARP_TILE_M,WARP_TILE_N);
    auto lWC=make_coord2d(threadIdx.x/WARP_SIZE,threadIdx.y);
    auto lWOff=lWC*wTile;

    uint32_t tiles_n = (N+MACRO_TILE_N-1)/MACRO_TILE_N;
    uint32_t k_units = (K+MACRO_TILE_K-1)/MACRO_TILE_K; // K units per output tile

    // Derive sk_start from blockIdx.x (persistent-kernel style)
    uint32_t my_sk_start = blockIdx.x * units_per_cta;
    uint32_t sk_end = min(my_sk_start + units_per_cta, total_sk_units);
    // Override the passed sk_start with the block-derived value
    (void)sk_start;
    uint32_t sk_start_eff = my_sk_start;

    MmaAcc fAcc; fill_fragment(fAcc, ComputeT(0));
    uint32_t cur_tile_id  = UINT32_MAX;
    bool     tile_started = false;

    HIP_DYNAMIC_SHARED(void*,lmem);
    auto lOA=0u;
    auto lOB=GetDataLayout_t<LWA>::fromMatrixCoord(make_coord2d(ldHA,0u),ldsld);
    auto* lLo=reinterpret_cast<InputT*>(lmem);
    auto* lHi=lLo+szLds;

    for(uint32_t sk=sk_start_eff; sk<sk_end; sk++){
        uint32_t tile_id  = sk / k_units;    // which output tile
        uint32_t k_idx    = sk % k_units;    // which K slice within that tile
        uint32_t tile_m   = tile_id / tiles_n;
        uint32_t tile_n   = tile_id % tiles_n;
        uint32_t k_off    = k_idx * MACRO_TILE_K;

        // If we moved to a new output tile, flush the previous accumulation
        if(tile_started && tile_id != cur_tile_id){
            // Partial tile: atomicAdd to workspace and increment done counter
            using MMapW=GetDataLayout_t<MmaWrk>;
            auto wTC=make_coord2d((cur_tile_id/tiles_n)*MACRO_TILE_M + get<0>(lWOff),
                                  (cur_tile_id%tiles_n)*MACRO_TILE_N + get<1>(lWOff));
            WorkT* ws = workspace + (size_t)cur_tile_id * WARP_TILE_M * WARP_TILE_N;
            for(uint32_t i=0;i<fAcc.num_elements;i++)
                atomicAdd(&ws[i], static_cast<WorkT>(fAcc.x[i]));
            __threadfence();
            if(threadIdx.x==0 && threadIdx.y==0)
                atomicAdd(&tile_done[cur_tile_id], 1u);

            fill_fragment(fAcc, ComputeT(0));
        }

        cur_tile_id  = tile_id;
        tile_started = true;

        auto mTC=make_coord2d(tile_m*MACRO_TILE_M, tile_n*MACRO_TILE_N);
        auto wTC=mTC+lWOff;
        if(get<0>(wTC)+WARP_TILE_M>M || get<1>(wTC)+WARP_TILE_N>N) continue;

        using GRMapA=GetDataLayout_t<GRA>; using GRMapB=GetDataLayout_t<GRB>;
        auto rOffA=GRMapA::fromMatrixCoord(make_coord2d(get<0>(mTC),(uint32_t)k_off),lda);
        auto rOffB=GRMapB::fromMatrixCoord(make_coord2d((uint32_t)k_off,get<1>(mTC)),ldb);

        using LRMapA=GetDataLayout_t<LRA>; using LRMapB=GetDataLayout_t<LRB>;
        auto lRA=lOA+LRMapA::fromMatrixCoord(make_coord2d(get<0>(lWOff),0u),ldsld);
        auto lRB=lOB+LRMapB::fromMatrixCoord(make_coord2d(get<1>(lWOff),0u),ldsld);

        // Single K step (MACRO_TILE_K) -- no loop since one SK unit = one K step
        GRA grA; GRB grB;
        load_matrix_sync(grA,A+rOffA,lda); load_matrix_sync(grB,B+rOffB,ldb);
        store_matrix_sync(lLo+lOA,toLWA(grA),ldsld); store_matrix_sync(lLo+lOB,toLWB(grB),ldsld);
        synchronize_workgroup();
        LRA lrA; LRB lrB;
        load_matrix_sync(lrA,lLo+lRA,ldsld); load_matrix_sync(lrB,lLo+lRB,ldsld);
        mma_sync(fAcc,toMmaA(lrA),toMmaB(lrB),fAcc);
    }

    // Flush the last tile
    if(tile_started){
        WorkT* ws = workspace + (size_t)cur_tile_id * WARP_TILE_M * WARP_TILE_N;
        auto mTC=make_coord2d((cur_tile_id/tiles_n)*MACRO_TILE_M,
                              (cur_tile_id%tiles_n)*MACRO_TILE_N);
        auto wTC=mTC+lWOff;
        for(uint32_t i=0;i<fAcc.num_elements;i++)
            atomicAdd(&ws[i], static_cast<WorkT>(fAcc.x[i]));
        __threadfence();
        if(threadIdx.x==0 && threadIdx.y==0)
            atomicAdd(&tile_done[cur_tile_id], 1u);
    }
}

// Stage 2: Reduce workspace -> output (called after all Stage-1 CTAs finish via hipDeviceSynchronize)
__global__ void __launch_bounds__(TBLOCK_X*TBLOCK_Y)
streamk_reduce_kernel(const WorkT* __restrict__ workspace,
                      const uint32_t* __restrict__ tile_done,
                      OutputT* __restrict__ C,
                      uint32_t M, uint32_t N, uint32_t ldc,
                      uint32_t k_units_per_tile)
{
    uint32_t tiles_n = (N+MACRO_TILE_N-1)/MACRO_TILE_N;
    uint32_t tile_id = blockIdx.x;
    uint32_t tile_m  = tile_id / tiles_n;
    uint32_t tile_n  = tile_id % tiles_n;
    // Stage-1 is guaranteed complete (host did hipDeviceSynchronize before launching this)
    // No need for busy-wait; just read workspace directly.

    constexpr auto wTile=make_coord2d(WARP_TILE_M,WARP_TILE_N);
    auto lWC=make_coord2d(threadIdx.x/WARP_SIZE,threadIdx.y);
    auto lWOff=lWC*wTile;
    auto mTC=make_coord2d(tile_m*MACRO_TILE_M, tile_n*MACRO_TILE_N);
    auto wTC=mTC+lWOff;
    if(get<0>(wTC)+WARP_TILE_M>M || get<1>(wTC)+WARP_TILE_N>N) return;

    const WorkT* ws = workspace + (size_t)tile_id * WARP_TILE_M * WARP_TILE_N;
    using MOut=GetDataLayout_t<MmaOut>;
    MmaOut fOut;
    for(uint32_t i=0;i<fOut.num_elements;i++) fOut.x[i]=static_cast<OutputT>(ws[i]);
    store_matrix_sync(C+MOut::fromMatrixCoord(wTC,ldc), fOut, ldc);
}

void test_streamk_gemm(uint32_t M, uint32_t N, uint32_t K)
{
    if(M%MACRO_TILE_M||N%MACRO_TILE_N||K%MACRO_TILE_K){
        std::cout<<"[StreamK] Dims not tile-aligned, skip\n"; return; }

    uint32_t tiles_m = M/MACRO_TILE_M, tiles_n = N/MACRO_TILE_N;
    uint32_t k_units = K/MACRO_TILE_K;
    uint32_t total_tiles = tiles_m * tiles_n;
    uint32_t total_sk    = total_tiles * k_units;

    // Use number of CUs for persistent kernel
    hipDeviceProp_t prop; CHECK_HIP_ERROR(hipGetDeviceProperties(&prop, 0));
    uint32_t num_cus = prop.multiProcessorCount;
    uint32_t units_per_cta = (total_sk + num_cus - 1) / num_cus;

    std::vector<InputT>  hA(M*K, InputT(0.5f)), hB(K*N, InputT(0.5f));
    std::vector<OutputT> hC(M*N);

    InputT *dA,*dB; OutputT *dC; WorkT *dWs; uint32_t *dDone;
    CHECK_HIP_ERROR(hipMalloc(&dA,    M*K*sizeof(InputT)));
    CHECK_HIP_ERROR(hipMalloc(&dB,    K*N*sizeof(InputT)));
    CHECK_HIP_ERROR(hipMalloc(&dC,    M*N*sizeof(OutputT)));
    CHECK_HIP_ERROR(hipMalloc(&dWs,   (size_t)total_tiles*WARP_TILE_M*WARP_TILE_N*sizeof(WorkT)));
    CHECK_HIP_ERROR(hipMalloc(&dDone, total_tiles*sizeof(uint32_t)));
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), M*K*sizeof(InputT), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), K*N*sizeof(InputT), hipMemcpyHostToDevice));

    dim3 block(TBLOCK_X, TBLOCK_Y);
    dim3 gridSK(num_cus), gridRed(total_tiles);
    uint32_t ldsB = 2u*sizeof(InputT)*szLds;

    // Stream-K: launch all CTAs simultaneously (persistent-kernel style)
    // Each CTA computes its sk_start = blockIdx.x * units_per_cta
    // The streamk_reduce_kernel runs after all SK CTAs finish via hipDeviceSynchronize
    auto fn=[&](){
        CHECK_HIP_ERROR(hipMemset(dWs,   0, (size_t)total_tiles*WARP_TILE_M*WARP_TILE_N*sizeof(WorkT)));
        CHECK_HIP_ERROR(hipMemset(dDone, 0, total_tiles*sizeof(uint32_t)));
        // Stage 1: all Stream-K CTAs at once
        uint32_t active_ctas = 0;
        for(uint32_t cta=0; cta<num_cus; cta++){
            if(cta * units_per_cta < total_sk) active_ctas++;
        }
        // Launch active_ctas blocks; each block computes its own sk_start from blockIdx.x
        // We pass units_per_cta; sk_start = blockIdx.x * units_per_cta
        hipExtLaunchKernelGGL(streamk_gemm_kernel,
                              dim3(active_ctas), block, ldsB, 0, nullptr, nullptr, 0,
                              M, N, K, dA, dB, dC, dWs, dDone,
                              M, N, N, total_sk, units_per_cta, 0 /*placeholder, use blockIdx*/);
        CHECK_HIP_ERROR(hipDeviceSynchronize());
        // Stage 2: reduce workspace -> output
        hipLaunchKernelGGL(streamk_reduce_kernel, gridRed, block, 0, 0,
                           dWs, dDone, dC, M, N, N, k_units);
    };

    constexpr uint32_t warmup=3, runs=10;
    for(uint32_t i=0;i<warmup;i++) fn();
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    hipEvent_t t0,t1; CHECK_HIP_ERROR(hipEventCreate(&t0)); CHECK_HIP_ERROR(hipEventCreate(&t1));
    CHECK_HIP_ERROR(hipEventRecord(t0));
    for(uint32_t i=0;i<runs;i++) fn();
    CHECK_HIP_ERROR(hipEventRecord(t1)); CHECK_HIP_ERROR(hipEventSynchronize(t1));
    float ms=0.f; CHECK_HIP_ERROR(hipEventElapsedTime(&ms,t0,t1));
    CHECK_HIP_ERROR(hipEventDestroy(t0)); CHECK_HIP_ERROR(hipEventDestroy(t1));

    double tflops = calculateTFlopsPerSec(M,N,K,ms,runs);
    std::cout << "[StreamK-GEMM] M=" << M << " N=" << N << " K=" << K
              << " num_cus=" << num_cus
              << "  " << ms/runs << " ms  " << tflops << " TFlops/s\n";

    CHECK_HIP_ERROR(hipFree(dA)); CHECK_HIP_ERROR(hipFree(dB)); CHECK_HIP_ERROR(hipFree(dC));
    CHECK_HIP_ERROR(hipFree(dWs)); CHECK_HIP_ERROR(hipFree(dDone));
}

int main(int argc, char* argv[])
{
    hipDeviceProp_t prop; CHECK_HIP_ERROR(hipGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << "  (" << prop.gcnArchName << ")\n\n";
    std::cout << "=== rocWMMA Stream-K GEMM (rocWMMA port) ===\n";
    // Stream-K especially benefits small M/N with large K
    test_streamk_gemm(512,  512,  4096);
    test_streamk_gemm(1024, 1024, 4096);
    test_streamk_gemm(2048, 2048, 4096);
    test_streamk_gemm(4096, 4096, 4096);
    return 0;
}
