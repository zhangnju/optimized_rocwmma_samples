/*******************************************************************************
 * MIT License
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

/*
 * rocwmma_batched_gemm.cpp
 *
 * Description:
 *   Batched GEMM ported from rocWMMA 16_batched_gemm using rocWMMA.
 *
 *   rocWMMA Optimizations Applied:
 *   - 3-level tile hierarchy: Block(128x128) -> Warp(64x64) -> MFMA(32x32x16)
 *   - Double-buffer LDS pipeline (rocWMMA COMPUTE_V4 style)
 *   - Cooperative global read across all warps in a block
 *   - Transposed B in LDS for bank-conflict-free access
 *   - Batch dimension handled by blockIdx.z
 *   - CShuffleEpilogue: accumulate in FP32, write FP16
 *
 * Operation:
 *   C[b, m, n] = A[b, m, k] * B[b, k, n]   for b in [0, batch)
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

// ---------------------------------------------------------------------------
// Kernel parameters (same as perf_gemm_ck_style.cpp, reuse arch selection)
// ---------------------------------------------------------------------------
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

using InputT   = float16_t;
using OutputT  = float16_t;
using ComputeT = float32_t;

using DataLayoutA   = col_major;
using DataLayoutB   = row_major;
using DataLayoutC   = row_major;
using DataLayoutLds = col_major;

using MmaFragA   = fragment<matrix_a, WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, InputT, DataLayoutA>;
using MmaFragB   = fragment<matrix_b, WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, InputT, DataLayoutB>;
using MmaFragC   = fragment<accumulator, WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, OutputT, DataLayoutC>;
using MmaFragAcc = fragment<accumulator, WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, ComputeT>;

using CoopScheduler = fragment_scheduler::coop_row_major_2d<TBLOCK_X, TBLOCK_Y>;
using GRFragA = fragment<matrix_a, MACRO_TILE_M, MACRO_TILE_N, MACRO_TILE_K, InputT, DataLayoutA, CoopScheduler>;
using GRFragB = fragment<matrix_b, MACRO_TILE_M, MACRO_TILE_N, MACRO_TILE_K, InputT, DataLayoutB, CoopScheduler>;
using LWFragA = apply_data_layout_t<GRFragA, DataLayoutLds>;
using LWFragB = apply_data_layout_t<apply_transpose_t<GRFragB>, DataLayoutLds>;
using LRFragA = apply_data_layout_t<MmaFragA, DataLayoutLds>;
using LRFragB = apply_data_layout_t<apply_transpose_t<MmaFragB>, DataLayoutLds>;

using LWShapeA = GetIOShape_t<LWFragA>;
using LWShapeB = GetIOShape_t<LWFragB>;
constexpr uint32_t ldsHeightA = LWShapeA::BlockHeight;
constexpr uint32_t ldsHeightB = LWShapeB::BlockHeight;
constexpr uint32_t ldsHeight  = ldsHeightA + ldsHeightB;
constexpr uint32_t ldsWidth   = MACRO_TILE_K;
constexpr uint32_t sizeLds    = ldsHeight * ldsWidth;
constexpr uint32_t ldsld = std::is_same_v<DataLayoutLds, row_major> ? ldsWidth : ldsHeight;

ROCWMMA_DEVICE __forceinline__ auto toLWFragA(GRFragA const& gr) { return apply_data_layout<DataLayoutLds>(gr); }
ROCWMMA_DEVICE __forceinline__ auto toLWFragB(GRFragB const& gr) { return apply_data_layout<DataLayoutLds>(apply_transpose(gr)); }
ROCWMMA_DEVICE __forceinline__ auto toMmaFragA(LRFragA const& lr) { return apply_data_layout<DataLayoutA>(lr); }
ROCWMMA_DEVICE __forceinline__ auto toMmaFragB(LRFragB const& lr) { return apply_data_layout<DataLayoutB>(apply_transpose(lr)); }

// ---------------------------------------------------------------------------
// Batched GEMM kernel (CK Tile double-buffer pipeline, batch via blockIdx.z)
// ---------------------------------------------------------------------------
ROCWMMA_KERNEL void __launch_bounds__(TBLOCK_X * TBLOCK_Y)
batched_gemm_kernel(uint32_t m, uint32_t n, uint32_t k,
                    InputT const* a, InputT const* b, OutputT* c,
                    uint32_t lda, uint32_t ldb, uint32_t ldc,
                    uint32_t stride_a, uint32_t stride_b, uint32_t stride_c)
{
    // Batch offset
    uint32_t batch = blockIdx.z;
    a += batch * stride_a;
    b += batch * stride_b;
    c += batch * stride_c;

    constexpr auto warpTileSize  = make_coord2d(WARP_TILE_M, WARP_TILE_N);
    constexpr auto macroTileSize = make_coord2d(MACRO_TILE_M, MACRO_TILE_N);

    auto localWarpCoord  = make_coord2d(threadIdx.x / WARP_SIZE, threadIdx.y);
    auto localWarpOffset = localWarpCoord * warpTileSize;
    auto macroTileCoord  = make_coord2d(blockIdx.x, blockIdx.y) * macroTileSize;
    auto warpTileCoord   = macroTileCoord + localWarpOffset;

    if(get<0>(warpTileCoord) + WARP_TILE_M > m || get<1>(warpTileCoord) + WARP_TILE_N > n)
        return;

    using GRMapA = GetDataLayout_t<GRFragA>;
    using GRMapB = GetDataLayout_t<GRFragB>;
    auto gReadOffA = GRMapA::fromMatrixCoord(make_coord2d(get<0>(macroTileCoord), 0u), lda);
    auto gReadOffB = GRMapB::fromMatrixCoord(make_coord2d(0u, get<1>(macroTileCoord)), ldb);
    auto kStepA    = GRMapA::fromMatrixCoord(make_coord2d(0u, MACRO_TILE_K), lda);
    auto kStepB    = GRMapB::fromMatrixCoord(make_coord2d(MACRO_TILE_K, 0u), ldb);

    HIP_DYNAMIC_SHARED(void*, localMemPtr);
    using LWMapA = GetDataLayout_t<LWFragA>;
    using LWMapB = GetDataLayout_t<LWFragB>;
    auto ldsOffA = 0u;
    auto ldsOffB = LWMapA::fromMatrixCoord(make_coord2d(ldsHeightA, 0u), ldsld);

    auto* ldsPtrLo = reinterpret_cast<InputT*>(localMemPtr);
    auto* ldsPtrHi = ldsPtrLo + sizeLds;

    using LRMapA = GetDataLayout_t<LRFragA>;
    using LRMapB = GetDataLayout_t<LRFragB>;
    auto ldsRdA = ldsOffA + LRMapA::fromMatrixCoord(make_coord2d(get<0>(localWarpOffset), 0u), ldsld);
    auto ldsRdB = ldsOffB + LRMapB::fromMatrixCoord(make_coord2d(get<1>(localWarpOffset), 0u), ldsld);

    // Prefetch K=0
    GRFragA grA; GRFragB grB;
    load_matrix_sync(grA, a + gReadOffA, lda);
    load_matrix_sync(grB, b + gReadOffB, ldb);
    gReadOffA += kStepA; gReadOffB += kStepB;
    store_matrix_sync(ldsPtrLo + ldsOffA, toLWFragA(grA), ldsld);
    store_matrix_sync(ldsPtrLo + ldsOffB, toLWFragB(grB), ldsld);

    MmaFragAcc fragAcc; fill_fragment(fragAcc, ComputeT(0));
    synchronize_workgroup();

    for(uint32_t kStep = MACRO_TILE_K; kStep < k; kStep += MACRO_TILE_K) {
        LRFragA lrA; LRFragB lrB;
        load_matrix_sync(lrA, ldsPtrLo + ldsRdA, ldsld);
        load_matrix_sync(lrB, ldsPtrLo + ldsRdB, ldsld);
        load_matrix_sync(grA, a + gReadOffA, lda);
        load_matrix_sync(grB, b + gReadOffB, ldb);
        gReadOffA += kStepA; gReadOffB += kStepB;
        mma_sync(fragAcc, toMmaFragA(lrA), toMmaFragB(lrB), fragAcc);
        store_matrix_sync(ldsPtrHi + ldsOffA, toLWFragA(grA), ldsld);
        store_matrix_sync(ldsPtrHi + ldsOffB, toLWFragB(grB), ldsld);
        synchronize_workgroup();
        auto* tmp = ldsPtrLo; ldsPtrLo = ldsPtrHi; ldsPtrHi = tmp;
    }

    LRFragA lrA; LRFragB lrB;
    load_matrix_sync(lrA, ldsPtrLo + ldsRdA, ldsld);
    load_matrix_sync(lrB, ldsPtrLo + ldsRdB, ldsld);
    mma_sync(fragAcc, toMmaFragA(lrA), toMmaFragB(lrB), fragAcc);

    // Epilogue
    using MmaMapC = GetDataLayout_t<MmaFragC>;
    MmaFragC fragC; MmaFragC fragD;
    fill_fragment(fragC, OutputT(0));
    constexpr uint32_t cs=8, nc=MmaFragC::num_elements/cs, nr=MmaFragC::num_elements%cs;
    auto doFma = [&](uint32_t s, uint32_t sz){ for(uint32_t i=s;i<s+sz;i++) fragD.x[i]=static_cast<OutputT>(fragAcc.x[i]); };
    for(uint32_t ci=0;ci<nc;ci++) doFma(ci*cs,cs); doFma(nc*cs,nr);
    store_matrix_sync(c + MmaMapC::fromMatrixCoord(warpTileCoord, ldc), fragD, ldc);
}

void test_batched_gemm(uint32_t batch, uint32_t m, uint32_t n, uint32_t k)
{
    uint32_t lda = m, ldb = n, ldc = n; // col-major A, row-major B, row-major C
    if(std::is_same_v<DataLayoutA, row_major>) lda = k;
    if(std::is_same_v<DataLayoutB, col_major>) ldb = k;

    size_t szA = (size_t)batch * m * k, szB = (size_t)batch * k * n, szC = (size_t)batch * m * n;

    std::vector<InputT>  hA(szA, InputT(0.5f)), hB(szB, InputT(0.5f));
    std::vector<OutputT> hC(szC, OutputT(0.f));

    InputT *dA, *dB; OutputT *dC;
    CHECK_HIP_ERROR(hipMalloc(&dA, szA*sizeof(InputT)));
    CHECK_HIP_ERROR(hipMalloc(&dB, szB*sizeof(InputT)));
    CHECK_HIP_ERROR(hipMalloc(&dC, szC*sizeof(OutputT)));
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), szA*sizeof(InputT), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), szB*sizeof(InputT), hipMemcpyHostToDevice));

    dim3 blockDim(TBLOCK_X, TBLOCK_Y);
    dim3 gridDim(ceil_div(m, MACRO_TILE_M), ceil_div(n, MACRO_TILE_N), batch);

    uint32_t ldsBytes = 2u * sizeof(InputT) * sizeLds;
    auto fn = [&]() {
        hipExtLaunchKernelGGL(batched_gemm_kernel, gridDim, blockDim, ldsBytes, 0,
                              nullptr, nullptr, 0,
                              m, n, k, dA, dB, dC, lda, ldb, ldc,
                              (uint32_t)(m*k), (uint32_t)(k*n), (uint32_t)(m*n));
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

    double tflops = calculateTFlopsPerSec(m, n, k, ms, runs) * batch;
    std::cout << "[BatchedGEMM] batch=" << batch << " M=" << m << " N=" << n << " K=" << k
              << "  " << ms/runs << " ms  " << tflops/batch << " TFlops/s (per batch)"
              << "  " << tflops << " eff TFlops/s\n";

    CHECK_HIP_ERROR(hipFree(dA)); CHECK_HIP_ERROR(hipFree(dB)); CHECK_HIP_ERROR(hipFree(dC));
}

int main(int argc, char* argv[])
{
    hipDeviceProp_t prop; CHECK_HIP_ERROR(hipGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << "  (" << prop.gcnArchName << ")\n\n";

    uint32_t batch=4, m=1024, n=1024, k=1024;
    if(argc>=5){ batch=std::atoi(argv[1]); m=std::atoi(argv[2]); n=std::atoi(argv[3]); k=std::atoi(argv[4]); }

    std::cout << "=== rocWMMA Batched GEMM (rocWMMA port) ===\n";
    test_batched_gemm(batch, m, n, k);
    test_batched_gemm(8, 512, 512, 2048);
    test_batched_gemm(16, 256, 256, 4096);
    return 0;
}
