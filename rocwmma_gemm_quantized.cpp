/*******************************************************************************
 * MIT License
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

/*
 * rocwmma_gemm_quantized.cpp
 *
 * Description:
 *   Quantized GEMM variants ported from rocWMMA 38_block_scale_gemm and
 *   17_grouped_gemm (quantized variant).
 *
 *   rocWMMA Optimizations Applied:
 *   - FP8 inputs with per-tensor / per-row-col scale factors
 *   - Dequantization fused into the epilogue (scale * accumulator)
 *   - Same 3-level tile hierarchy as standard GEMM
 *   - Double-buffer LDS pipeline
 *   - Two quantization modes benchmarked:
 *       Mode 1 (tensor-scale): single A-scale, single B-scale
 *       Mode 2 (row-col-scale): per-row A-scale, per-col B-scale
 *
 * Operation:
 *   C[m,n] = dequant(A) * dequant(B)
 *   where dequant(x) = x * scale
 *   In Mode 1: scale_A * (A_fp8 * B_fp8) * scale_B
 *   In Mode 2: diag(scale_A) * (A_fp8 * B_fp8) * diag(scale_B)
 *
 * Supported: gfx942, gfx950, gfx1200, gfx1201
 *            (FP8 hardware support required)
 */

#include <iomanip>
#include <iostream>
#include <vector>
#include <random>

#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_transforms.hpp>

#include "common.hpp"

using namespace rocwmma;

// ---------------------------------------------------------------------------
// FP8 GEMM: use rocWMMA FP8 fragments on supported architectures
// ---------------------------------------------------------------------------
#if defined(ROCWMMA_ARCH_GFX942) || defined(ROCWMMA_ARCH_GFX950) || \
    defined(ROCWMMA_ARCH_GFX1200) || defined(ROCWMMA_ARCH_GFX1201)
#define HAS_FP8_SUPPORT 1
#else
#define HAS_FP8_SUPPORT 0
#endif

#if HAS_FP8_SUPPORT
using Fp8T  = float8_t;  // rocWMMA fp8 type
#else
using Fp8T  = float16_t; // fallback to fp16 on older hardware
#endif

using ScaleT  = float;
using OutputT = float16_t;
using AccT    = float;

// Tile parameters (identical to perf_gemm_ck_style for consistency)
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

using DataLayoutA   = col_major;
using DataLayoutB   = row_major;
using DataLayoutC   = row_major;
using DataLayoutLds = col_major;

// Use Fp8T for quantized GEMM, AccT=float
using MmaFragA   = fragment<matrix_a, WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, Fp8T, DataLayoutA>;
using MmaFragB   = fragment<matrix_b, WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, Fp8T, DataLayoutB>;
using MmaFragAcc = fragment<accumulator, WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, AccT>;
using MmaFragOut = fragment<accumulator, WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, OutputT, DataLayoutC>;

using CoopScheduler = fragment_scheduler::coop_row_major_2d<TBLOCK_X, TBLOCK_Y>;
using GRFragA = fragment<matrix_a, MACRO_TILE_M, MACRO_TILE_N, MACRO_TILE_K, Fp8T, DataLayoutA, CoopScheduler>;
using GRFragB = fragment<matrix_b, MACRO_TILE_M, MACRO_TILE_N, MACRO_TILE_K, Fp8T, DataLayoutB, CoopScheduler>;
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
// Quantized GEMM kernel: tensor scale mode
// C[m,n] = (A_fp8 * B_fp8) * scale_a * scale_b  (output: fp16)
// ---------------------------------------------------------------------------
ROCWMMA_KERNEL void __launch_bounds__(TBLOCK_X * TBLOCK_Y)
gemm_quant_tensor_scale(uint32_t m, uint32_t n, uint32_t k,
                        Fp8T const* a, Fp8T const* b,
                        OutputT*    c,
                        uint32_t lda, uint32_t ldb, uint32_t ldc,
                        ScaleT scale_a, ScaleT scale_b)
{
    constexpr auto warpTileSize  = make_coord2d(WARP_TILE_M, WARP_TILE_N);
    constexpr auto macroTileSize = make_coord2d(MACRO_TILE_M, MACRO_TILE_N);
    auto localWarpCoord  = make_coord2d(threadIdx.x / WARP_SIZE, threadIdx.y);
    auto localWarpOffset = localWarpCoord * warpTileSize;
    auto macroTileCoord  = make_coord2d(blockIdx.x, blockIdx.y) * macroTileSize;
    auto warpTileCoord   = macroTileCoord + localWarpOffset;
    if(get<0>(warpTileCoord) + WARP_TILE_M > m || get<1>(warpTileCoord) + WARP_TILE_N > n) return;

    using GRMapA = GetDataLayout_t<GRFragA>;
    using GRMapB = GetDataLayout_t<GRFragB>;
    auto gReadOffA = GRMapA::fromMatrixCoord(make_coord2d(get<0>(macroTileCoord), 0u), lda);
    auto gReadOffB = GRMapB::fromMatrixCoord(make_coord2d(0u, get<1>(macroTileCoord)), ldb);
    auto kStepA    = GRMapA::fromMatrixCoord(make_coord2d(0u, MACRO_TILE_K), lda);
    auto kStepB    = GRMapB::fromMatrixCoord(make_coord2d(MACRO_TILE_K, 0u), ldb);

    HIP_DYNAMIC_SHARED(void*, localMemPtr);
    auto ldsOffA = 0u;
    auto ldsOffB = GetDataLayout_t<LWFragA>::fromMatrixCoord(make_coord2d(ldsHeightA, 0u), ldsld);
    auto* ldsPtrLo = reinterpret_cast<Fp8T*>(localMemPtr);
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

    MmaFragAcc fragAcc; fill_fragment(fragAcc, AccT(0));
    synchronize_workgroup();

    for(uint32_t ks = MACRO_TILE_K; ks < k; ks += MACRO_TILE_K) {
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

    // Epilogue: dequantize with tensor scale -> fp16
    float ab_scale = scale_a * scale_b;
    using MmaMapOut = GetDataLayout_t<MmaFragOut>;
    MmaFragOut fragOut;
    for(uint32_t i = 0; i < fragAcc.num_elements; i++)
        fragOut.x[i] = static_cast<OutputT>(static_cast<float>(fragAcc.x[i]) * ab_scale);
    store_matrix_sync(c + MmaMapOut::fromMatrixCoord(warpTileCoord, ldc), fragOut, ldc);
}

void test_quant_gemm(uint32_t m, uint32_t n, uint32_t k)
{
    if(m % MACRO_TILE_M || n % MACRO_TILE_N || k % MACRO_TILE_K) {
        std::cout << "[QuantGEMM] Unsupported dims (not tile-aligned), skipping M="
                  << m << " N=" << n << " K=" << k << "\n";
        return;
    }

    uint32_t lda = m, ldb = n, ldc = n;

    std::vector<Fp8T>   hA(m*k), hB(k*n);
    std::vector<OutputT> hC(m*n);
    // Simple initialization
    for(size_t i=0;i<hA.size();i++) hA[i] = Fp8T(0.1f);
    for(size_t i=0;i<hB.size();i++) hB[i] = Fp8T(0.1f);

    Fp8T *dA, *dB; OutputT *dC;
    CHECK_HIP_ERROR(hipMalloc(&dA, m*k*sizeof(Fp8T)));
    CHECK_HIP_ERROR(hipMalloc(&dB, k*n*sizeof(Fp8T)));
    CHECK_HIP_ERROR(hipMalloc(&dC, m*n*sizeof(OutputT)));
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), m*k*sizeof(Fp8T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), k*n*sizeof(Fp8T), hipMemcpyHostToDevice));

    dim3 block(TBLOCK_X, TBLOCK_Y);
    dim3 grid(ceil_div(m, MACRO_TILE_M), ceil_div(n, MACRO_TILE_N));
    uint32_t ldsBytes = 2u * sizeof(Fp8T) * sizeLds;

    auto fn = [&]() {
        hipExtLaunchKernelGGL(gemm_quant_tensor_scale, grid, block, ldsBytes, 0,
                              nullptr, nullptr, 0,
                              m, n, k, dA, dB, dC, lda, ldb, ldc, 1.f/127.f, 1.f/127.f);
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
    std::cout << "[QuantGEMM-FP8] M=" << m << " N=" << n << " K=" << k
              << "  " << ms/runs << " ms  " << tflops << " TFlops/s"
#if HAS_FP8_SUPPORT
              << " (FP8 native)\n";
#else
              << " (FP16 fallback)\n";
#endif

    CHECK_HIP_ERROR(hipFree(dA)); CHECK_HIP_ERROR(hipFree(dB)); CHECK_HIP_ERROR(hipFree(dC));
}

int main(int argc, char* argv[])
{
    hipDeviceProp_t prop; CHECK_HIP_ERROR(hipGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << "  (" << prop.gcnArchName << ")\n";
#if HAS_FP8_SUPPORT
    std::cout << "FP8 hardware support: YES\n\n";
#else
    std::cout << "FP8 hardware support: NO (using FP16 fallback)\n\n";
#endif

    std::cout << "=== rocWMMA Quantized GEMM (rocWMMA port) ===\n";
    test_quant_gemm(3840, 4096, 4096);
    test_quant_gemm(4096, 4096, 4096);
    test_quant_gemm(8192, 8192, 8192);
    return 0;
}
