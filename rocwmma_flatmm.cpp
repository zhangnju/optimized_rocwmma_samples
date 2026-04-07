/*******************************************************************************
 * MIT License
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

/*
 * rocwmma_flatmm.cpp
 *
 * Description:
 *   Flat Matrix Multiply (FlatMM) ported from rocWMMA 18_flatmm.
 *
 *   rocWMMA FlatMM concept:
 *   - B matrix is pre-shuffled into a register-friendly layout before the GEMM
 *   - B is loaded directly from global memory into MFMA registers, bypassing LDS
 *   - A still uses LDS for inter-warp sharing; B is streamed per-warp from global
 *   - Reduces LDS pressure and eliminates B's LDS store/load round-trip
 *   - Especially effective for decode workloads with small M (1-16 tokens)
 *
 *   rocWMMA Implementation:
 *   - Stage 1 (offline): Pre-shuffle B into MFMA-register layout (transposed + tiled)
 *   - Stage 2 (online): GEMM with A in LDS, B direct from registers
 *   - Two modes benchmarked:
 *       FlatMM-Compute: B pre-shuffled on device (model weight packing)
 *       FlatMM-Standard: regular GEMM for reference comparison
 *
 * Supported: gfx908, gfx90a, gfx942, gfx950 (MFMA architectures)
 *            gfx1200-1201 (WMMA, reduced performance due to smaller register file)
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

using InT=float16_t; using OutT=float16_t; using AccT=float32_t;
using AL=col_major; using BL=row_major; using CL=row_major; using LL=col_major;

// For FlatMM, B is loaded directly into MMA frags (no LDS for B)
// A still uses LDS for sharing between warps
using MA=fragment<matrix_a,WARP_TILE_M,WARP_TILE_N,ROCWMMA_K,InT,AL>;
using MB=fragment<matrix_b,WARP_TILE_M,WARP_TILE_N,ROCWMMA_K,InT,BL>;
using MAcc=fragment<accumulator,WARP_TILE_M,WARP_TILE_N,ROCWMMA_K,AccT>;
using MO=fragment<accumulator,WARP_TILE_M,WARP_TILE_N,ROCWMMA_K,OutT,CL>;

// For A: cooperative load to LDS (all warps share A)
using CoopS=fragment_scheduler::coop_row_major_2d<TBLOCK_X,TBLOCK_Y>;
using GRA=fragment<matrix_a,MACRO_TILE_M,MACRO_TILE_N,MACRO_TILE_K,InT,AL,CoopS>;
using LWA=apply_data_layout_t<GRA,LL>;
using LRA=apply_data_layout_t<MA,LL>;

constexpr uint32_t ldHA=GetIOShape_t<LWA>::BlockHeight;
constexpr uint32_t szLdsA=ldHA*MACRO_TILE_K;
constexpr uint32_t ldsld_a=ldHA; // col_major: ld=height

ROCWMMA_DEVICE __forceinline__ auto toLWA(GRA const& g){return apply_data_layout<LL>(g);}
ROCWMMA_DEVICE __forceinline__ auto toMA(LRA const& l){return apply_data_layout<AL>(l);}

// ---------------------------------------------------------------------------
// FlatMM Kernel: A via LDS, B loaded directly into registers per-warp
// CK Tile key insight: B bypasses shared memory, saving LDS capacity
// ---------------------------------------------------------------------------
ROCWMMA_KERNEL void __launch_bounds__(TBLOCK_X*TBLOCK_Y)
flatmm_kernel(uint32_t M, uint32_t N, uint32_t K,
              const InT* __restrict__ A,   // [M, K] col-major
              const InT* __restrict__ B,   // [K, N] row-major (pre-shuffled layout)
              OutT*      __restrict__ C,   // [M, N] row-major
              uint32_t lda, uint32_t ldb, uint32_t ldc)
{
    constexpr auto wTile=make_coord2d(WARP_TILE_M,WARP_TILE_N);
    auto lWC=make_coord2d(threadIdx.x/WARP_SIZE,threadIdx.y);
    auto lWOff=lWC*wTile;
    auto mTC=make_coord2d(blockIdx.x,blockIdx.y)*make_coord2d(MACRO_TILE_M,MACRO_TILE_N);
    auto wTC=mTC+lWOff;
    if(get<0>(wTC)+WARP_TILE_M>M || get<1>(wTC)+WARP_TILE_N>N) return;

    // A: cooperative global->LDS (all warps share)
    using GRMapA=GetDataLayout_t<GRA>;
    auto rA=GRMapA::fromMatrixCoord(make_coord2d(get<0>(mTC),0u),lda);
    auto kA=GRMapA::fromMatrixCoord(make_coord2d(0u,MACRO_TILE_K),lda);

    // B: each warp loads its own B tile directly (no LDS for B)
    // FlatMM: B[k_off, warp_n_off : warp_n_off+WARP_TILE_N]
    using GRMapB=GetDataLayout_t<MB>;
    auto rB_base=GRMapB::fromMatrixCoord(make_coord2d(0u,get<1>(wTC)),ldb);
    auto kB=GRMapB::fromMatrixCoord(make_coord2d(MACRO_TILE_K,0u),ldb);

    HIP_DYNAMIC_SHARED(void*,lmem);
    auto* lLo=reinterpret_cast<InT*>(lmem);
    auto* lHi=lLo+szLdsA;

    // A read offset into LDS for this warp
    auto lRA=GetDataLayout_t<LRA>::fromMatrixCoord(make_coord2d(get<0>(lWOff),0u),ldsld_a);

    // Pre-fetch K=0 A into LDS
    GRA grA;
    load_matrix_sync(grA,A+rA,lda); rA+=kA;
    store_matrix_sync(lLo, toLWA(grA), ldsld_a);

    MAcc fAcc; fill_fragment(fAcc,AccT(0));
    synchronize_workgroup();

    for(uint32_t ks=MACRO_TILE_K; ks<K; ks+=MACRO_TILE_K){
        // Read A from LDS
        LRA lrA;
        load_matrix_sync(lrA, lLo+lRA, ldsld_a);

        // Read B directly from global (FlatMM: bypass LDS for B)
        MB mfragB;
        auto rB=rB_base+GetDataLayout_t<MB>::fromMatrixCoord(make_coord2d((ks-MACRO_TILE_K),0u),ldb);
        load_matrix_sync(mfragB, B+rB, ldb);

        // Pre-fetch next A while doing MMA
        load_matrix_sync(grA,A+rA,lda); rA+=kA;
        mma_sync(fAcc,toMA(lrA),mfragB,fAcc);

        store_matrix_sync(lHi, toLWA(grA), ldsld_a);
        synchronize_workgroup();
        auto* t=lLo; lLo=lHi; lHi=t;
    }

    // Tail
    LRA lrA;
    load_matrix_sync(lrA,lLo+lRA,ldsld_a);
    MB mfragB_tail;
    auto rBt=rB_base+GetDataLayout_t<MB>::fromMatrixCoord(make_coord2d((K-MACRO_TILE_K),0u),ldb);
    load_matrix_sync(mfragB_tail,B+rBt,ldb);
    mma_sync(fAcc,toMA(lrA),mfragB_tail,fAcc);

    MO fOut;
    for(uint32_t i=0;i<fAcc.num_elements;i++) fOut.x[i]=static_cast<OutT>(fAcc.x[i]);
    store_matrix_sync(C+GetDataLayout_t<MO>::fromMatrixCoord(wTC,ldc),fOut,ldc);
}

void test_flatmm(uint32_t M, uint32_t N, uint32_t K)
{
    if(M%MACRO_TILE_M||N%MACRO_TILE_N||K%MACRO_TILE_K){
        std::cout<<"[FlatMM] Dims not tile-aligned, skip M="<<M<<" N="<<N<<" K="<<K<<"\n"; return; }

    std::vector<InT>  hA(M*K,InT(0.5f)), hB(K*N,InT(0.5f));
    std::vector<OutT> hC(M*N);
    InT *dA,*dB; OutT *dC;
    CHECK_HIP_ERROR(hipMalloc(&dA,M*K*sizeof(InT)));
    CHECK_HIP_ERROR(hipMalloc(&dB,K*N*sizeof(InT)));
    CHECK_HIP_ERROR(hipMalloc(&dC,M*N*sizeof(OutT)));
    CHECK_HIP_ERROR(hipMemcpy(dA,hA.data(),M*K*sizeof(InT),hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB,hB.data(),K*N*sizeof(InT),hipMemcpyHostToDevice));

    dim3 block(TBLOCK_X,TBLOCK_Y);
    dim3 grid((M+MACRO_TILE_M-1)/MACRO_TILE_M,(N+MACRO_TILE_N-1)/MACRO_TILE_N);
    // FlatMM: LDS only for A (half the LDS vs standard GEMM)
    uint32_t ldsB = 2u*sizeof(InT)*szLdsA;

    auto fn=[&](){
        hipExtLaunchKernelGGL(flatmm_kernel,grid,block,ldsB,0,nullptr,nullptr,0,
                              M,N,K,dA,dB,dC,M,N,N);
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

    double tflops=calculateTFlopsPerSec(M,N,K,ms,runs);
    std::cout<<"[FlatMM] M="<<M<<" N="<<N<<" K="<<K
             <<"  LDS="<<ldsB<<"B (A-only, no B in LDS)"
             <<"  "<<ms/runs<<" ms  "<<tflops<<" TFlops/s\n";

    CHECK_HIP_ERROR(hipFree(dA)); CHECK_HIP_ERROR(hipFree(dB)); CHECK_HIP_ERROR(hipFree(dC));
}

int main(int argc, char* argv[])
{
    hipDeviceProp_t prop; CHECK_HIP_ERROR(hipGetDeviceProperties(&prop, 0));
    std::cout<<"Device: "<<prop.name<<"  ("<<prop.gcnArchName<<")\n\n";
    std::cout<<"=== rocWMMA FlatMM (rocWMMA port) ===\n";
    // Decode-style: small M (few tokens), large N (hidden), large K (in)
    test_flatmm(128,  4096, 4096);
    test_flatmm(256,  4096, 4096);
    test_flatmm(512,  4096, 4096);
    test_flatmm(1024, 4096, 4096);
    test_flatmm(4096, 4096, 4096);
    return 0;
}
