/*******************************************************************************
 * MIT License
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

/*
 * rocwmma_batched_contraction.cpp
 *
 * Description:
 *   Batched Tensor Contraction ported from rocWMMA 41_batched_contraction.
 *
 *   rocWMMA Optimizations Applied:
 *   - Reduces to batched GEMM: E[g,m,n] = A[g,m,k] * B[g,n,k]
 *     (contraction over k dimension, G batch dims, M output rows, N output cols)
 *   - Full 3-level tile hierarchy (Block -> Warp -> MFMA) via rocWMMA
 *   - Double-buffer LDS pipeline (COMPUTE_V4 style)
 *   - Cooperative global load across warps
 *   - Batch dimension via blockIdx.z (same as batched GEMM)
 *   - Generalization: arbitrary NumDimG/M/N/K via flattening
 *
 * Operation:
 *   E[g0..gG-1, m0..mM-1, n0..nN-1] =
 *     sum_{k0..kK-1}( A[g0..gG-1, m0..mM-1, k0..kK-1] *
 *                     B[g0..gG-1, n0..nN-1, k0..kK-1] )
 *   After flattening G, M, N, K into single indices: same as batched GEMM
 *
 * Supported: gfx908, gfx90a, gfx942, gfx950, gfx1100-1103, gfx1200-1201
 */

#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_transforms.hpp>

#include "common.hpp"

using namespace rocwmma;

// Reuse the same tile params as batched GEMM
namespace gfx9P { enum : uint32_t { RM=32, RN=32, RK=16, BM=2, BN=2, TX=128, TY=2, WS=Constants::AMDGCN_WAVE_SIZE_64 }; }
namespace gfx12P{ enum : uint32_t { RM=16, RN=16, RK=16, BM=4, BN=4, TX=64,  TY=2, WS=Constants::AMDGCN_WAVE_SIZE_32 }; }
#if defined(ROCWMMA_ARCH_GFX9)
using namespace gfx9P;
constexpr uint32_t ROCWMMA_M=RM, ROCWMMA_N=RN, ROCWMMA_K=RK;
constexpr uint32_t BLOCKS_M=BM, BLOCKS_N=BN, TBLOCK_X=TX, TBLOCK_Y=TY, WARP_SIZE=WS;
#else
using namespace gfx12P;
constexpr uint32_t ROCWMMA_M=RM, ROCWMMA_N=RN, ROCWMMA_K=RK;
constexpr uint32_t BLOCKS_M=BM, BLOCKS_N=BN, TBLOCK_X=TX, TBLOCK_Y=TY, WARP_SIZE=WS;
#endif

constexpr uint32_t WARP_TILE_M  = BLOCKS_M * ROCWMMA_M;
constexpr uint32_t WARP_TILE_N  = BLOCKS_N * ROCWMMA_N;
constexpr uint32_t MACRO_TILE_M = (TBLOCK_X / WARP_SIZE) * WARP_TILE_M;
constexpr uint32_t MACRO_TILE_N = TBLOCK_Y * WARP_TILE_N;
constexpr uint32_t MACRO_TILE_K = ROCWMMA_K;

using InputT   = float16_t;
using OutputT  = float16_t;
using ComputeT = float32_t;

using DataLayoutA   = col_major;  // A[G, M, K] -> GEMM layout A[M, K]
using DataLayoutB   = col_major;  // B[G, N, K] stored as [N, K], treat as col_major [K, N] with ldb=N
using DataLayoutC   = row_major;
using DataLayoutLds = col_major;

using MmaFragA   = fragment<matrix_a, WARP_TILE_M, WARP_TILE_N, ROCWMMA_K, InputT, DataLayoutA>;
using MmaFragB   = fragment<matrix_b, WARP_TILE_M, WARP_TILE_N, ROCWMMA_K, InputT, DataLayoutB>;
using MmaFragAcc = fragment<accumulator, WARP_TILE_M, WARP_TILE_N, ROCWMMA_K, ComputeT>;
using MmaFragOut = fragment<accumulator, WARP_TILE_M, WARP_TILE_N, ROCWMMA_K, OutputT, DataLayoutC>;

using CoopSched = fragment_scheduler::coop_row_major_2d<TBLOCK_X, TBLOCK_Y>;
using GRFragA = fragment<matrix_a, MACRO_TILE_M, MACRO_TILE_N, MACRO_TILE_K, InputT, DataLayoutA, CoopSched>;
using GRFragB = fragment<matrix_b, MACRO_TILE_M, MACRO_TILE_N, MACRO_TILE_K, InputT, DataLayoutB, CoopSched>;
using LWFragA = apply_data_layout_t<GRFragA, DataLayoutLds>;
using LWFragB = apply_data_layout_t<apply_transpose_t<GRFragB>, DataLayoutLds>;
using LRFragA = apply_data_layout_t<MmaFragA, DataLayoutLds>;
using LRFragB = apply_data_layout_t<apply_transpose_t<MmaFragB>, DataLayoutLds>;

constexpr uint32_t ldHtA = GetIOShape_t<LWFragA>::BlockHeight;
constexpr uint32_t ldHtB = GetIOShape_t<LWFragB>::BlockHeight;
constexpr uint32_t ldHt  = ldHtA + ldHtB;
constexpr uint32_t ldWd  = MACRO_TILE_K;
constexpr uint32_t szLds = ldHt * ldWd;
constexpr uint32_t ldsld = std::is_same_v<DataLayoutLds, row_major> ? ldWd : ldHt;

ROCWMMA_DEVICE __forceinline__ auto toLWA(GRFragA const& g){ return apply_data_layout<DataLayoutLds>(g); }
ROCWMMA_DEVICE __forceinline__ auto toLWB(GRFragB const& g){ return apply_data_layout<DataLayoutLds>(apply_transpose(g)); }
ROCWMMA_DEVICE __forceinline__ auto toMmaA(LRFragA const& l){ return apply_data_layout<DataLayoutA>(l); }
ROCWMMA_DEVICE __forceinline__ auto toMmaB(LRFragB const& l){ return apply_data_layout<DataLayoutB>(apply_transpose(l)); }

// ---------------------------------------------------------------------------
// Batched contraction kernel (= batched GEMM with B transposed)
// A[G, M, K], B[G, N, K] -> E[G, M, N]
// blockIdx.z = G (batch/group dimension)
// ---------------------------------------------------------------------------
ROCWMMA_KERNEL void __launch_bounds__(TBLOCK_X * TBLOCK_Y)
batched_contraction_kernel(uint32_t G, uint32_t M, uint32_t N, uint32_t K,
                           InputT const* A,  // [G, M, K] col-major (lda=M)
                           InputT const* B,  // [G, N, K] row-major (ldb=N ... K is fast)
                           OutputT*      E,  // [G, M, N]
                           uint32_t lda, uint32_t ldb, uint32_t lde)
{
    uint32_t g = blockIdx.z;
    A += (size_t)g * M * K;
    B += (size_t)g * N * K;
    E += (size_t)g * M * N;

    constexpr auto wTile = make_coord2d(WARP_TILE_M, WARP_TILE_N);
    constexpr auto mTile = make_coord2d(MACRO_TILE_M, MACRO_TILE_N);
    auto lWarpC  = make_coord2d(threadIdx.x / WARP_SIZE, threadIdx.y);
    auto lWarpOff= lWarpC * wTile;
    auto mTileC  = make_coord2d(blockIdx.x, blockIdx.y) * mTile;
    auto wTileC  = mTileC + lWarpOff;
    if(get<0>(wTileC)+WARP_TILE_M>M || get<1>(wTileC)+WARP_TILE_N>N) return;

    using GRMapA = GetDataLayout_t<GRFragA>;
    using GRMapB = GetDataLayout_t<GRFragB>;
    auto rOffA = GRMapA::fromMatrixCoord(make_coord2d(get<0>(mTileC), 0u), lda);
    auto rOffB = GRMapB::fromMatrixCoord(make_coord2d(0u, get<1>(mTileC)), ldb);
    auto kStA  = GRMapA::fromMatrixCoord(make_coord2d(0u, MACRO_TILE_K), lda);
    auto kStB  = GRMapB::fromMatrixCoord(make_coord2d(MACRO_TILE_K, 0u), ldb);

    HIP_DYNAMIC_SHARED(void*, lmem);
    auto lOA = 0u;
    auto lOB = GetDataLayout_t<LWFragA>::fromMatrixCoord(make_coord2d(ldHtA, 0u), ldsld);
    auto* lLo = reinterpret_cast<InputT*>(lmem);
    auto* lHi = lLo + szLds;

    using LRMapA = GetDataLayout_t<LRFragA>;
    using LRMapB = GetDataLayout_t<LRFragB>;
    auto lRA = lOA + LRMapA::fromMatrixCoord(make_coord2d(get<0>(lWarpOff), 0u), ldsld);
    auto lRB = lOB + LRMapB::fromMatrixCoord(make_coord2d(get<1>(lWarpOff), 0u), ldsld);

    GRFragA grA; GRFragB grB;
    load_matrix_sync(grA, A+rOffA, lda);
    load_matrix_sync(grB, B+rOffB, ldb);
    rOffA+=kStA; rOffB+=kStB;
    store_matrix_sync(lLo+lOA, toLWA(grA), ldsld);
    store_matrix_sync(lLo+lOB, toLWB(grB), ldsld);

    MmaFragAcc fAcc; fill_fragment(fAcc, ComputeT(0));
    synchronize_workgroup();

    for(uint32_t ks=MACRO_TILE_K; ks<K; ks+=MACRO_TILE_K) {
        LRFragA lrA; LRFragB lrB;
        load_matrix_sync(lrA, lLo+lRA, ldsld);
        load_matrix_sync(lrB, lLo+lRB, ldsld);
        load_matrix_sync(grA, A+rOffA, lda);
        load_matrix_sync(grB, B+rOffB, ldb);
        rOffA+=kStA; rOffB+=kStB;
        mma_sync(fAcc, toMmaA(lrA), toMmaB(lrB), fAcc);
        store_matrix_sync(lHi+lOA, toLWA(grA), ldsld);
        store_matrix_sync(lHi+lOB, toLWB(grB), ldsld);
        synchronize_workgroup();
        auto* t=lLo; lLo=lHi; lHi=t;
    }
    LRFragA lrA; LRFragB lrB;
    load_matrix_sync(lrA, lLo+lRA, ldsld);
    load_matrix_sync(lrB, lLo+lRB, ldsld);
    mma_sync(fAcc, toMmaA(lrA), toMmaB(lrB), fAcc);

    using MOut = GetDataLayout_t<MmaFragOut>;
    MmaFragOut fOut;
    for(uint32_t i=0;i<fAcc.num_elements;i++) fOut.x[i]=static_cast<OutputT>(fAcc.x[i]);
    store_matrix_sync(E + MOut::fromMatrixCoord(wTileC, lde), fOut, lde);
}

void test_contraction(uint32_t G, uint32_t M, uint32_t N, uint32_t K)
{
    if(M%MACRO_TILE_M||N%MACRO_TILE_N||K%MACRO_TILE_K){
        std::cout<<"[Contraction] Dims not tile-aligned, skip G="<<G<<" M="<<M<<" N="<<N<<" K="<<K<<"\n"; return; }

    // A[G,M,K] col-major: lda=M; B[G,N,K] row-major: ldb=K
    uint32_t lda=M, ldb=N, lde=N; // col-major A: lda=M; col-major B[N,K]: ldb=N
    std::vector<InputT>  hA(G*M*K, InputT(0.5f)), hB(G*N*K, InputT(0.5f));
    std::vector<OutputT> hE(G*M*N);

    InputT *dA, *dB; OutputT *dE;
    CHECK_HIP_ERROR(hipMalloc(&dA, G*M*K*sizeof(InputT)));
    CHECK_HIP_ERROR(hipMalloc(&dB, G*N*K*sizeof(InputT)));
    CHECK_HIP_ERROR(hipMalloc(&dE, G*M*N*sizeof(OutputT)));
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), G*M*K*sizeof(InputT), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), G*N*K*sizeof(InputT), hipMemcpyHostToDevice));

    dim3 block(TBLOCK_X, TBLOCK_Y);
    dim3 grid(ceil_div(M, MACRO_TILE_M), ceil_div(N, MACRO_TILE_N), G);
    uint32_t ldsB = 2u * sizeof(InputT) * szLds;

    auto fn=[&](){ hipExtLaunchKernelGGL(batched_contraction_kernel, grid, block, ldsB, 0,
                              nullptr, nullptr, 0, G, M, N, K, dA, dB, dE, lda, ldb, lde); };

    constexpr uint32_t warmup=5, runs=20;
    for(uint32_t i=0;i<warmup;i++) fn();
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    hipEvent_t t0,t1; CHECK_HIP_ERROR(hipEventCreate(&t0)); CHECK_HIP_ERROR(hipEventCreate(&t1));
    CHECK_HIP_ERROR(hipEventRecord(t0));
    for(uint32_t i=0;i<runs;i++) fn();
    CHECK_HIP_ERROR(hipEventRecord(t1)); CHECK_HIP_ERROR(hipEventSynchronize(t1));
    float ms=0.f; CHECK_HIP_ERROR(hipEventElapsedTime(&ms,t0,t1));
    CHECK_HIP_ERROR(hipEventDestroy(t0)); CHECK_HIP_ERROR(hipEventDestroy(t1));

    double tflops = calculateTFlopsPerSec(M, N, K, ms, runs) * G;
    std::cout << "[Contraction] G=" << G << " M=" << M << " N=" << N << " K=" << K
              << "  " << ms/runs << " ms  " << tflops/G << " TFlops/s\n";

    CHECK_HIP_ERROR(hipFree(dA)); CHECK_HIP_ERROR(hipFree(dB)); CHECK_HIP_ERROR(hipFree(dE));
}

int main(int argc, char* argv[])
{
    hipDeviceProp_t prop; CHECK_HIP_ERROR(hipGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << "  (" << prop.gcnArchName << ")\n\n";
    std::cout << "=== rocWMMA Batched Contraction (rocWMMA port) ===\n";
    test_contraction(4,  1024, 1024, 1024);
    test_contraction(8,  512,  512,  2048);
    test_contraction(16, 256,  256,  4096);
    return 0;
}
