/*******************************************************************************
 * MIT License
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

/*
 * rocwmma_gemm_splitk.cpp
 *
 * Description:
 *   Split-K two-stage GEMM ported from rocWMMA 03_gemm/gemm_splitk_two_stage.
 *
 *   rocWMMA Optimizations Applied:
 *   - Split-K: K dimension partitioned across `split_k` thread-block groups
 *   - Stage 1: Each group computes partial C[m,n] = A[m, k/split_k] * B[k/split_k, n]
 *              accumulating in FP32 workspace (higher precision for large K)
 *   - Stage 2: A separate elementwise reduction kernel sums the `split_k` partials
 *              and converts back to FP16
 *   - Double-buffer LDS pipeline inside Stage 1 (rocWMMA COMPUTE_V4 style)
 *   - Workspace in FP32 for lossless partial accumulation (rocWMMA WorkspaceType=float)
 *
 * Operation:
 *   workspace[sk, m, n] = A[m, sk*Ks:(sk+1)*Ks] * B[sk*Ks:(sk+1)*Ks, n]  (Stage 1)
 *   C[m, n] = sum_sk(workspace[sk, m, n])                                   (Stage 2)
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

namespace gfx9Params { enum : uint32_t {
    ROCWMMA_M=32, ROCWMMA_N=32, ROCWMMA_K=16,
    BLOCKS_M=2, BLOCKS_N=2, TBLOCK_X=128, TBLOCK_Y=2,
    WARP_SIZE=Constants::AMDGCN_WAVE_SIZE_64 }; }
namespace gfx12Params { enum : uint32_t {
    ROCWMMA_M=16, ROCWMMA_N=16, ROCWMMA_K=16,
    BLOCKS_M=4, BLOCKS_N=4, TBLOCK_X=64, TBLOCK_Y=2,
    WARP_SIZE=Constants::AMDGCN_WAVE_SIZE_32 }; }
#if defined(ROCWMMA_ARCH_GFX9)
using namespace gfx9Params;
#else
using namespace gfx12Params;
#endif

constexpr uint32_t WARP_TILE_M  = BLOCKS_M * ROCWMMA_M;
constexpr uint32_t WARP_TILE_N  = BLOCKS_N * ROCWMMA_N;
constexpr uint32_t WARP_TILE_K  = ROCWMMA_K;
constexpr uint32_t WARPS_M      = TBLOCK_X / WARP_SIZE;
constexpr uint32_t WARPS_N      = TBLOCK_Y;
constexpr uint32_t MACRO_TILE_M = WARPS_M * WARP_TILE_M;
constexpr uint32_t MACRO_TILE_N = WARPS_N * WARP_TILE_N;
constexpr uint32_t MACRO_TILE_K = ROCWMMA_K;

using InputT    = float16_t;
using OutputT   = float16_t;
using ComputeT  = float32_t;
using WorkspaceT = float32_t; // CK Tile uses float for workspace (lossless)

using DataLayoutA   = col_major;
using DataLayoutB   = row_major;
using DataLayoutC   = row_major;
using DataLayoutLds = col_major;

using MmaFragA   = fragment<matrix_a, WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, InputT, DataLayoutA>;
using MmaFragB   = fragment<matrix_b, WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, InputT, DataLayoutB>;
using MmaFragAcc = fragment<accumulator, WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, ComputeT>;

using CoopScheduler = fragment_scheduler::coop_row_major_2d<TBLOCK_X, TBLOCK_Y>;
using GRFragA = fragment<matrix_a, MACRO_TILE_M, MACRO_TILE_N, MACRO_TILE_K, InputT, DataLayoutA, CoopScheduler>;
using GRFragB = fragment<matrix_b, MACRO_TILE_M, MACRO_TILE_N, MACRO_TILE_K, InputT, DataLayoutB, CoopScheduler>;
using LWFragA = apply_data_layout_t<GRFragA, DataLayoutLds>;
using LWFragB = apply_data_layout_t<apply_transpose_t<GRFragB>, DataLayoutLds>;
using LRFragA = apply_data_layout_t<MmaFragA, DataLayoutLds>;
using LRFragB = apply_data_layout_t<apply_transpose_t<MmaFragB>, DataLayoutLds>;

constexpr uint32_t ldsHeightA = GetIOShape_t<LWFragA>::BlockHeight;
constexpr uint32_t ldsHeightB = GetIOShape_t<LWFragB>::BlockHeight;
constexpr uint32_t ldsHeight  = ldsHeightA + ldsHeightB;
constexpr uint32_t ldsWidth   = MACRO_TILE_K;
constexpr uint32_t sizeLds    = ldsHeight * ldsWidth;
constexpr uint32_t ldsld = std::is_same_v<DataLayoutLds, row_major> ? ldsWidth : ldsHeight;

ROCWMMA_DEVICE __forceinline__ auto toLWA(GRFragA const& g){ return apply_data_layout<DataLayoutLds>(g); }
ROCWMMA_DEVICE __forceinline__ auto toLWB(GRFragB const& g){ return apply_data_layout<DataLayoutLds>(apply_transpose(g)); }
ROCWMMA_DEVICE __forceinline__ auto toMmaA(LRFragA const& l){ return apply_data_layout<DataLayoutA>(l); }
ROCWMMA_DEVICE __forceinline__ auto toMmaB(LRFragB const& l){ return apply_data_layout<DataLayoutB>(apply_transpose(l)); }

// ---------------------------------------------------------------------------
// Stage 1: Split-K partial GEMM -> FP32 workspace
// blockIdx.z = split_k index
// ---------------------------------------------------------------------------
ROCWMMA_KERNEL void __launch_bounds__(TBLOCK_X * TBLOCK_Y)
gemm_splitk_stage1(uint32_t m, uint32_t n, uint32_t k,
                   InputT const*   a, InputT const*  b,
                   WorkspaceT*     workspace,    // [split_k, m, n]
                   uint32_t lda, uint32_t ldb,
                   uint32_t split_k)
{
    uint32_t sk    = blockIdx.z;
    uint32_t k_per = (k + split_k - 1) / split_k;
    uint32_t k_off = sk * k_per;
    uint32_t k_len = (k_off + k_per <= k) ? k_per : (k - k_off);
    k_len = (k_len / MACRO_TILE_K) * MACRO_TILE_K; // truncate to tile boundary
    if(k_len == 0) return;

    constexpr auto warpTileSize  = make_coord2d(WARP_TILE_M, WARP_TILE_N);
    constexpr auto macroTileSize = make_coord2d(MACRO_TILE_M, MACRO_TILE_N);
    auto localWarpCoord  = make_coord2d(threadIdx.x / WARP_SIZE, threadIdx.y);
    auto localWarpOffset = localWarpCoord * warpTileSize;
    auto macroTileCoord  = make_coord2d(blockIdx.x, blockIdx.y) * macroTileSize;
    auto warpTileCoord   = macroTileCoord + localWarpOffset;
    if(get<0>(warpTileCoord) + WARP_TILE_M > m || get<1>(warpTileCoord) + WARP_TILE_N > n) return;

    using GRMapA = GetDataLayout_t<GRFragA>;
    using GRMapB = GetDataLayout_t<GRFragB>;
    // Offset into split-K slice
    auto gReadOffA = GRMapA::fromMatrixCoord(make_coord2d(get<0>(macroTileCoord), k_off), lda);
    auto gReadOffB = GRMapB::fromMatrixCoord(make_coord2d(k_off, get<1>(macroTileCoord)), ldb);
    auto kStepA    = GRMapA::fromMatrixCoord(make_coord2d(0u, MACRO_TILE_K), lda);
    auto kStepB    = GRMapB::fromMatrixCoord(make_coord2d(MACRO_TILE_K, 0u), ldb);

    HIP_DYNAMIC_SHARED(void*, localMemPtr);
    using LWMapA = GetDataLayout_t<LWFragA>;
    auto ldsOffA = 0u;
    auto ldsOffB = GetDataLayout_t<LWFragA>::fromMatrixCoord(make_coord2d(ldsHeightA, 0u), ldsld);
    auto* ldsPtrLo = reinterpret_cast<InputT*>(localMemPtr);
    auto* ldsPtrHi = ldsPtrLo + sizeLds;

    using LRMapA = GetDataLayout_t<LRFragA>;
    using LRMapB = GetDataLayout_t<LRFragB>;
    auto ldsRdA = ldsOffA + LRMapA::fromMatrixCoord(make_coord2d(get<0>(localWarpOffset), 0u), ldsld);
    auto ldsRdB = ldsOffB + LRMapB::fromMatrixCoord(make_coord2d(get<1>(localWarpOffset), 0u), ldsld);

    GRFragA grA; GRFragB grB;
    load_matrix_sync(grA, a + gReadOffA, lda);
    load_matrix_sync(grB, b + gReadOffB, ldb);
    gReadOffA += kStepA; gReadOffB += kStepB;
    store_matrix_sync(ldsPtrLo + ldsOffA, toLWA(grA), ldsld);
    store_matrix_sync(ldsPtrLo + ldsOffB, toLWB(grB), ldsld);

    MmaFragAcc fragAcc; fill_fragment(fragAcc, ComputeT(0));
    synchronize_workgroup();

    for(uint32_t ks = MACRO_TILE_K; ks < k_len; ks += MACRO_TILE_K) {
        LRFragA lrA; LRFragB lrB;
        load_matrix_sync(lrA, ldsPtrLo + ldsRdA, ldsld);
        load_matrix_sync(lrB, ldsPtrLo + ldsRdB, ldsld);
        load_matrix_sync(grA, a + gReadOffA, lda);
        load_matrix_sync(grB, b + gReadOffB, ldb);
        gReadOffA += kStepA; gReadOffB += kStepB;
        mma_sync(fragAcc, toMmaA(lrA), toMmaB(lrB), fragAcc);
        store_matrix_sync(ldsPtrHi + ldsOffA, toLWA(grA), ldsld);
        store_matrix_sync(ldsPtrHi + ldsOffB, toLWB(grB), ldsld);
        synchronize_workgroup();
        auto* t = ldsPtrLo; ldsPtrLo = ldsPtrHi; ldsPtrHi = t;
    }
    LRFragA lrA; LRFragB lrB;
    load_matrix_sync(lrA, ldsPtrLo + ldsRdA, ldsld);
    load_matrix_sync(lrB, ldsPtrLo + ldsRdB, ldsld);
    mma_sync(fragAcc, toMmaA(lrA), toMmaB(lrB), fragAcc);

    // Write FP32 partial to workspace[sk, m, n]
    // Use a row_major accumulator fragment with explicit DataLayout for store_matrix_sync
    using MmaFragAccRW = fragment<accumulator, WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, ComputeT, row_major>;
    using MmaMapAccRW  = GetDataLayout_t<MmaFragAccRW>;
    WorkspaceT* ws_base = workspace + (size_t)sk * m * n;
    auto warpOff = MmaMapAccRW::fromMatrixCoord(warpTileCoord, n);
    // Reinterpret fragAcc as MmaFragAccRW (same bit layout, just adds DataLayout tag)
    MmaFragAccRW fragAccRW;
    for(uint32_t i = 0; i < fragAcc.num_elements; i++) fragAccRW.x[i] = fragAcc.x[i];
    store_matrix_sync(ws_base + warpOff, fragAccRW, n);
}

// ---------------------------------------------------------------------------
// Stage 2: Reduce split_k partial results to FP16 output
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
gemm_splitk_reduce(const WorkspaceT* __restrict__ workspace, // [split_k, m, n]
                   float16_t*        __restrict__ c,
                   uint32_t m, uint32_t n, uint32_t split_k)
{
    uint32_t idx = blockIdx.x * 256 + threadIdx.x;
    if(idx >= m * n) return;

    float acc = 0.f;
    for(uint32_t sk = 0; sk < split_k; sk++)
        acc += workspace[sk * m * n + idx];
    c[idx] = __float2half(acc);
}

void test_splitk_gemm(uint32_t m, uint32_t n, uint32_t k, uint32_t split_k)
{
    uint32_t lda = m, ldb = n, ldc = n;
    std::vector<InputT>    hA(m*k, InputT(0.5f)), hB(k*n, InputT(0.5f));
    std::vector<OutputT>   hC(m*n, OutputT(0.f));
    std::vector<WorkspaceT> hWs(split_k * m * n, 0.f);

    InputT *dA, *dB; OutputT *dC; WorkspaceT *dWs;
    CHECK_HIP_ERROR(hipMalloc(&dA,  m*k*sizeof(InputT)));
    CHECK_HIP_ERROR(hipMalloc(&dB,  k*n*sizeof(InputT)));
    CHECK_HIP_ERROR(hipMalloc(&dC,  m*n*sizeof(OutputT)));
    CHECK_HIP_ERROR(hipMalloc(&dWs, split_k*m*n*sizeof(WorkspaceT)));
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), m*k*sizeof(InputT), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), k*n*sizeof(InputT), hipMemcpyHostToDevice));

    dim3 block1(TBLOCK_X, TBLOCK_Y);
    dim3 grid1(ceil_div(m, MACRO_TILE_M), ceil_div(n, MACRO_TILE_N), split_k);
    dim3 grid2((m*n+255)/256), block2(256);

    uint32_t ldsBytes = 2u * sizeof(InputT) * sizeLds;
    auto fn = [&]() {
        hipExtLaunchKernelGGL(gemm_splitk_stage1, grid1, block1, ldsBytes, 0,
                              nullptr, nullptr, 0,
                              m, n, k, dA, dB, dWs, lda, ldb, split_k);
        hipLaunchKernelGGL(gemm_splitk_reduce, grid2, block2, 0, 0,
                           dWs, dC, m, n, split_k);
    };

    constexpr uint32_t warmup=5, runs=20;
    for(uint32_t i=0;i<warmup;i++) fn();
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    hipEvent_t t0,t1;
    CHECK_HIP_ERROR(hipEventCreate(&t0)); CHECK_HIP_ERROR(hipEventCreate(&t1));
    CHECK_HIP_ERROR(hipEventRecord(t0));
    for(uint32_t i=0;i<runs;i++) fn();
    CHECK_HIP_ERROR(hipEventRecord(t1)); CHECK_HIP_ERROR(hipEventSynchronize(t1));
    float ms=0.f; CHECK_HIP_ERROR(hipEventElapsedTime(&ms, t0, t1));
    CHECK_HIP_ERROR(hipEventDestroy(t0)); CHECK_HIP_ERROR(hipEventDestroy(t1));

    double tflops = calculateTFlopsPerSec(m, n, k, ms, runs);
    std::cout << "[SplitK-GEMM] M=" << m << " N=" << n << " K=" << k
              << " split_k=" << split_k
              << "  " << ms/runs << " ms  " << tflops << " TFlops/s\n";

    CHECK_HIP_ERROR(hipFree(dA)); CHECK_HIP_ERROR(hipFree(dB));
    CHECK_HIP_ERROR(hipFree(dC)); CHECK_HIP_ERROR(hipFree(dWs));
}

int main(int argc, char* argv[])
{
    hipDeviceProp_t prop; CHECK_HIP_ERROR(hipGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << "  (" << prop.gcnArchName << ")\n\n";

    std::cout << "=== rocWMMA Split-K GEMM (rocWMMA port) ===\n";
    // Small M/N with large K -- classic split-K use case
    test_splitk_gemm(512,  512,  16384, 4);
    test_splitk_gemm(1024, 1024, 16384, 8);
    test_splitk_gemm(2048, 2048, 8192,  4);
    test_splitk_gemm(4096, 4096, 4096,  2);
    return 0;
}
