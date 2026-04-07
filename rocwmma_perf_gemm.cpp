/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

/*
 * rocwmma_perf_gemm.cpp
 *
 * Description:
 *   High-performance HGEMM (FP16 in, FP32 accumulate, FP16 out) implemented with
 *   rocWMMA, following rocWMMA's three-level tiling and pipeline optimization strategy.
 *
 *   rocWMMA Optimization Philosophy Applied:
 *   ==========================================
 *   1) Three-level tiling hierarchy:
 *        Block Tile  (MACRO_TILE_M x MACRO_TILE_N x MACRO_TILE_K)
 *        Warp Layout (WARPS_M x WARPS_N warps per block)
 *        Warp Tile   (WARP_TILE_M x WARP_TILE_N, BLOCKS_M x BLOCKS_N mfma tiles)
 *
 *   2) rocWMMA COMPUTE pipeline variants implemented:
 *        - Pipeline V1 (baseline): sequential global->LDS->MMA
 *        - Pipeline V2 (double-buffer): ping-pong LDS with global prefetch
 *          overlapping global memory fetch with MMA computation
 *
 *   3) Cooperative global read with rocWMMA fragment_scheduler:
 *        All warps in a block collaboratively load the macro tile A/B from
 *        global memory, distributing the I/O bandwidth across all threads.
 *
 *   4) Transposed B in LDS:
 *        B fragments are stored transposed in shared memory so that both A and B
 *        share the same K-dimension as the fast (col) axis in LDS, eliminating
 *        bank conflicts on local reads.
 *
 *   5) Warp tile data reuse:
 *        Each warp computes BLOCKS_M x BLOCKS_N MFMA tiles, reusing A columns
 *        across B blocks and B rows across A blocks to amortize the global
 *        memory bandwidth.
 *
 *   6) Performance benchmarking:
 *        Both pipeline variants are measured and compared.  Results are reported
 *        in TFlops/s, matching rocWMMA's performance reporting convention.
 *
 *   Target Architectures:
 *     - gfx950  (MI355X) : MFMA, Wave64, block sizes 32x32x16 and 16x16x16
 *     - gfx942  (MI300X) : MFMA, Wave64
 *     - gfx1200/gfx1201  (RDNA4) : WMMA, Wave32, block size 16x16x16
 *
 *   Kernel Parameters (compile-time, tuned per architecture):
 *
 *   GFX9 (gfx950 / gfx942):
 *     ROCWMMA_M=32  ROCWMMA_N=32  ROCWMMA_K=16
 *     BLOCKS_M=2    BLOCKS_N=2
 *     TBLOCK_X=128  TBLOCK_Y=2
 *     => WARP_TILE: 64x64, MACRO_TILE: 128x128xK
 *
 *   GFX12 (gfx1200/gfx1201 RDNA4):
 *     ROCWMMA_M=16  ROCWMMA_N=16  ROCWMMA_K=16
 *     BLOCKS_M=4    BLOCKS_N=4
 *     TBLOCK_X=64   TBLOCK_Y=2
 *     => WARP_TILE: 64x64, MACRO_TILE: 128x128xK
 *
 * Requirements:
 *   - ROCm 6.0+
 *   - rocWMMA library
 *   - Supported GPU: gfx908, gfx90a, gfx942, gfx950, gfx1100-1103, gfx1200-1201
 *
 * Limitations:
 *   - M, N, K must be multiples of MACRO_TILE_M, MACRO_TILE_N, ROCWMMA_K respectively
 *   - FP16 inputs only in this sample (ComputeT = float32)
 *
 * Author: AMD rocWMMA Community Sample
 * Date: 2026-04-06
 */

#include <algorithm>
#include <chrono>
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

// ---------------------------------------------------------------------------
// Architecture-specific kernel parameters
// CK Tile philosophy: tune tile sizes per architecture to match
// compute:bandwidth ratio and LDS capacity.
// ---------------------------------------------------------------------------

namespace gfx9Params
{
    // GFX9 (MI200/MI300/MI355): Wave64, MFMA 32x32x16
    // Block Tile 128x128, Warp Tile 64x64 (2x2 MFMA tiles per warp)
    // 4 warps per block (2 along M, 2 along N)
    enum kernelParams : uint32_t
    {
        ROCWMMA_M = 32u,  // MFMA block M
        ROCWMMA_N = 32u,  // MFMA block N
        ROCWMMA_K = 16u,  // MFMA block K
        BLOCKS_M  = 2u,   // MFMA tiles per warp in M
        BLOCKS_N  = 2u,   // MFMA tiles per warp in N
        TBLOCK_X  = 128u, // threads along X (2 warps x 64 threads)
        TBLOCK_Y  = 2u,   // warps along N
        WARP_SIZE = Constants::AMDGCN_WAVE_SIZE_64
    };
}

namespace gfx12Params
{
    // GFX12 (RDNA4): Wave32, WMMA 16x16x16
    // Block Tile 128x128, Warp Tile 64x64 (4x4 WMMA tiles per warp)
    // 4 warps per block (2 along M, 2 along N)
    enum kernelParams : uint32_t
    {
        ROCWMMA_M = 16u,
        ROCWMMA_N = 16u,
        ROCWMMA_K = 16u,
        BLOCKS_M  = 4u,
        BLOCKS_N  = 4u,
        TBLOCK_X  = 64u,
        TBLOCK_Y  = 2u,
        WARP_SIZE = Constants::AMDGCN_WAVE_SIZE_32
    };
}

// Select compile-time params based on architecture
#if defined(ROCWMMA_ARCH_GFX9)
using namespace gfx9Params;
#else
// Default to gfx12 params for RDNA4 and others
using namespace gfx12Params;
#endif

// ---------------------------------------------------------------------------
// Derived tile dimensions (CK Tile 3-level hierarchy)
// ---------------------------------------------------------------------------

// Level 3: MFMA/WMMA tile (ROCWMMA_M x ROCWMMA_N x ROCWMMA_K)
// Level 2: Warp Tile = BLOCKS_M x BLOCKS_N MFMA tiles
constexpr uint32_t WARP_TILE_M = BLOCKS_M * ROCWMMA_M;
constexpr uint32_t WARP_TILE_N = BLOCKS_N * ROCWMMA_N;
constexpr uint32_t WARP_TILE_K = ROCWMMA_K;

// Level 1: Block (Macro) Tile = WARPS_M x WARPS_N Warp Tiles
constexpr uint32_t WARPS_M      = TBLOCK_X / WARP_SIZE; // warps along M
constexpr uint32_t WARPS_N      = TBLOCK_Y;              // warps along N
constexpr uint32_t WARP_COUNT   = WARPS_M * WARPS_N;
constexpr uint32_t MACRO_TILE_M = WARPS_M * WARP_TILE_M;
constexpr uint32_t MACRO_TILE_N = WARPS_N * WARP_TILE_N;
constexpr uint32_t MACRO_TILE_K = ROCWMMA_K; // K step per iteration

// ---------------------------------------------------------------------------
// Data types  (FP16 in, FP32 accumulate, FP16 out -- same as CK Tile fp16 GEMM)
// ---------------------------------------------------------------------------
using InputT   = float16_t;
using OutputT  = float16_t;
using ComputeT = float32_t;

// Layouts: A col-major, B row-major => common GEMM convention (NT layout)
// In CK Tile terms: A(Row), B(Col), C(Row) -- here using rocWMMA layout names
using DataLayoutA   = col_major;
using DataLayoutB   = row_major;
using DataLayoutC   = row_major;
using DataLayoutLds = col_major; // LDS uses col-major for bank-conflict-free access

// ---------------------------------------------------------------------------
// Fragment types (CK Tile warp tile = BLOCKS_M x BLOCKS_N MFMA frags)
// ---------------------------------------------------------------------------

// Warp-tile-sized fragments used for MMA computation
using MmaFragA   = fragment<matrix_a, WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, InputT, DataLayoutA>;
using MmaFragB   = fragment<matrix_b, WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, InputT, DataLayoutB>;
using MmaFragC   = fragment<accumulator, WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, OutputT, DataLayoutC>;
using MmaFragD   = MmaFragC;
using MmaFragAcc = fragment<accumulator, WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, ComputeT>;

// Macro-tile-sized cooperative fragments (CK Tile cooperative load)
// All warps in the block participate in loading the macro tile from global memory
using CoopScheduler = fragment_scheduler::coop_row_major_2d<TBLOCK_X, TBLOCK_Y>;
using GRFragA       = fragment<matrix_a,
                         MACRO_TILE_M,
                         MACRO_TILE_N,
                         MACRO_TILE_K,
                         InputT,
                         DataLayoutA,
                         CoopScheduler>;
using GRFragB       = fragment<matrix_b,
                         MACRO_TILE_M,
                         MACRO_TILE_N,
                         MACRO_TILE_K,
                         InputT,
                         DataLayoutB,
                         CoopScheduler>;

// LDS (shared memory) fragment types
// CK Tile insight: store B transposed in LDS so A and B both have K as fast axis
using LWFragA = apply_data_layout_t<GRFragA, DataLayoutLds>;
using LWFragB = apply_data_layout_t<apply_transpose_t<GRFragB>, DataLayoutLds>;

// LDS read fragments (for MMA)
using LRFragA = apply_data_layout_t<MmaFragA, DataLayoutLds>;
using LRFragB = apply_data_layout_t<apply_transpose_t<MmaFragB>, DataLayoutLds>;

// ---------------------------------------------------------------------------
// LDS layout constants
// CK Tile uses a fixed LDS width = MACRO_TILE_K to pack A and B together
// ---------------------------------------------------------------------------
constexpr uint32_t ldsWidth  = MACRO_TILE_K;
constexpr uint32_t ldsHeightA = GetIOShape_t<LWFragA>::BlockHeight;
constexpr uint32_t ldsHeightB = GetIOShape_t<LWFragB>::BlockHeight;
constexpr uint32_t ldsHeight  = ldsHeightA + ldsHeightB;
constexpr uint32_t sizeLds    = ldsHeight * ldsWidth;
// LDS leading dimension (col_major => height)
constexpr uint32_t ldsld = std::is_same_v<DataLayoutLds, row_major> ? ldsWidth : ldsHeight;

// ---------------------------------------------------------------------------
// Helper: transform global read frags to LDS write frags
// ---------------------------------------------------------------------------
ROCWMMA_DEVICE __forceinline__ auto toLWFragA(GRFragA const& gr)
{
    return apply_data_layout<DataLayoutLds>(gr);
}

ROCWMMA_DEVICE __forceinline__ auto toLWFragB(GRFragB const& gr)
{
    return apply_data_layout<DataLayoutLds>(apply_transpose(gr));
}

ROCWMMA_DEVICE __forceinline__ auto toMmaFragA(LRFragA const& lr)
{
    return apply_data_layout<DataLayoutA>(lr);
}

ROCWMMA_DEVICE __forceinline__ auto toMmaFragB(LRFragB const& lr)
{
    return apply_data_layout<DataLayoutB>(apply_transpose(lr));
}

// ---------------------------------------------------------------------------
//
// Kernel V1: Pipeline V1 (CK Tile BasicInvoker / GemmPipelineAGmemBGmemCRegV1)
//
// Sequential pipeline: global load -> LDS store -> barrier -> LDS load -> MMA
// No overlap between memory and compute.  Simple baseline.
//
// ---------------------------------------------------------------------------
ROCWMMA_KERNEL void __launch_bounds__(TBLOCK_X* TBLOCK_Y)
    gemm_pipeline_v1(uint32_t      m,
                     uint32_t      n,
                     uint32_t      k,
                     InputT const* a,
                     InputT const* b,
                     OutputT const* c,
                     OutputT*      d,
                     uint32_t      lda,
                     uint32_t      ldb,
                     uint32_t      ldc,
                     uint32_t      ldd,
                     ComputeT      alpha,
                     ComputeT      beta)
{
    // --- Tile coordinate calculation (CK Tile TilePartitioner equivalent) ---
    constexpr auto warpTileSize  = make_coord2d(WARP_TILE_M, WARP_TILE_N);
    constexpr auto macroTileSize = make_coord2d(MACRO_TILE_M, MACRO_TILE_N);

    auto localWarpCoord  = make_coord2d(threadIdx.x / WARP_SIZE, threadIdx.y);
    auto localWarpOffset = localWarpCoord * warpTileSize;
    auto macroTileCoord  = make_coord2d(blockIdx.x, blockIdx.y) * macroTileSize;
    auto warpTileCoord   = macroTileCoord + localWarpOffset;

    // Bounds check
    if(get<0>(warpTileCoord) + WARP_TILE_M > m || get<1>(warpTileCoord) + WARP_TILE_N > n)
        return;

    // --- Global read offset calculation ---
    using GRMapA = GetDataLayout_t<GRFragA>;
    using GRMapB = GetDataLayout_t<GRFragB>;

    auto gReadOffA = GRMapA::fromMatrixCoord(make_coord2d(get<0>(macroTileCoord), 0u), lda);
    auto gReadOffB = GRMapB::fromMatrixCoord(make_coord2d(0u, get<1>(macroTileCoord)), ldb);
    auto kStepA    = GRMapA::fromMatrixCoord(make_coord2d(0u, MACRO_TILE_K), lda);
    auto kStepB    = GRMapB::fromMatrixCoord(make_coord2d(MACRO_TILE_K, 0u), ldb);

    // --- LDS setup ---
    HIP_DYNAMIC_SHARED(void*, localMemPtr);
    auto* ldsPtr = reinterpret_cast<InputT*>(localMemPtr);

    // A occupies [0, ldsHeightA), B occupies [ldsHeightA, ldsHeight)
    using LWMapA = GetDataLayout_t<LWFragA>;
    using LWMapB = GetDataLayout_t<LWFragB>;
    auto ldsOffA = 0u;
    auto ldsOffB = LWMapA::fromMatrixCoord(make_coord2d(ldsHeightA, 0u), ldsld);

    // Warp-local read offsets into LDS
    using LRMapA = GetDataLayout_t<LRFragA>;
    using LRMapB = GetDataLayout_t<LRFragB>;
    auto ldsRdA  = ldsOffA + LRMapA::fromMatrixCoord(make_coord2d(get<0>(localWarpOffset), 0u), ldsld);
    auto ldsRdB  = ldsOffB + LRMapB::fromMatrixCoord(make_coord2d(get<1>(localWarpOffset), 0u), ldsld);

    // --- Accumulator ---
    MmaFragAcc fragAcc;
    fill_fragment(fragAcc, ComputeT(0));

    // --- Main K loop (CK Tile V1 pipeline: sequential) ---
    for(uint32_t kStep = 0; kStep < k; kStep += MACRO_TILE_K)
    {
        // 1. Cooperative global load
        GRFragA grA;
        GRFragB grB;
        load_matrix_sync(grA, a + gReadOffA, lda);
        load_matrix_sync(grB, b + gReadOffB, ldb);
        gReadOffA += kStepA;
        gReadOffB += kStepB;

        // 2. Write to LDS
        store_matrix_sync(ldsPtr + ldsOffA, toLWFragA(grA), ldsld);
        store_matrix_sync(ldsPtr + ldsOffB, toLWFragB(grB), ldsld);

        // 3. Barrier: all warps must complete writes before reads
        synchronize_workgroup();

        // 4. Local read + MMA
        LRFragA lrA;
        LRFragB lrB;
        load_matrix_sync(lrA, ldsPtr + ldsRdA, ldsld);
        load_matrix_sync(lrB, ldsPtr + ldsRdB, ldsld);
        mma_sync(fragAcc, toMmaFragA(lrA), toMmaFragB(lrB), fragAcc);

        // 5. Barrier: wait before next global load overwrites LDS
        synchronize_workgroup();
    }

    // --- Epilogue: D = alpha * Acc + beta * C ---
    using MmaMapC = GetDataLayout_t<MmaFragC>;
    using MmaMapD = GetDataLayout_t<MmaFragD>;

    MmaFragC fragC;
    load_matrix_sync(fragC, c + MmaMapC::fromMatrixCoord(warpTileCoord, ldc), ldc);

    MmaFragD fragD;
    constexpr uint32_t chunkSize = 8u;
    constexpr uint32_t nChunks   = MmaFragD::num_elements / chunkSize;
    constexpr uint32_t nRemain   = MmaFragD::num_elements % chunkSize;

    auto doFma = [&](uint32_t start, uint32_t sz) {
        for(uint32_t i = start; i < start + sz; i++)
            fragD.x[i] = static_cast<OutputT>(alpha * fragAcc.x[i]
                                              + beta * static_cast<ComputeT>(fragC.x[i]));
    };
    for(uint32_t c2 = 0; c2 < nChunks; c2++)
        doFma(c2 * chunkSize, chunkSize);
    doFma(nChunks * chunkSize, nRemain);

    store_matrix_sync(d + MmaMapD::fromMatrixCoord(warpTileCoord, ldd), fragD, ldd);
}

// ---------------------------------------------------------------------------
//
// Kernel V2: Pipeline V2 / COMPUTE_V4 style
// (CK Tile GemmPipelineAgBgCrCompV4 with DoubleSmemBuffer = true)
//
// Double-buffer (ping-pong LDS) pipeline:
//   - Pre-fetch global A/B for K0 into LDS buffer 0
//   - Loop: while computing K_i from buffer i, prefetch K_{i+1} into buffer 1-i
//   - Tail: compute last K step from the final buffer
//
// This overlaps global memory latency with MFMA compute, matching
// CK Tile's most effective strategy for compute-bound GEMMs.
//
// ---------------------------------------------------------------------------
ROCWMMA_KERNEL void __launch_bounds__(TBLOCK_X* TBLOCK_Y)
    gemm_pipeline_v2_double_buffer(uint32_t      m,
                                   uint32_t      n,
                                   uint32_t      k,
                                   InputT const* a,
                                   InputT const* b,
                                   OutputT const* c,
                                   OutputT*      d,
                                   uint32_t      lda,
                                   uint32_t      ldb,
                                   uint32_t      ldc,
                                   uint32_t      ldd,
                                   ComputeT      alpha,
                                   ComputeT      beta)
{
    // --- Tile coordinates ---
    constexpr auto warpTileSize  = make_coord2d(WARP_TILE_M, WARP_TILE_N);
    constexpr auto macroTileSize = make_coord2d(MACRO_TILE_M, MACRO_TILE_N);

    auto localWarpCoord  = make_coord2d(threadIdx.x / WARP_SIZE, threadIdx.y);
    auto localWarpOffset = localWarpCoord * warpTileSize;
    auto macroTileCoord  = make_coord2d(blockIdx.x, blockIdx.y) * macroTileSize;
    auto warpTileCoord   = macroTileCoord + localWarpOffset;

    if(get<0>(warpTileCoord) + WARP_TILE_M > m || get<1>(warpTileCoord) + WARP_TILE_N > n)
        return;

    // --- Global read offsets ---
    using GRMapA = GetDataLayout_t<GRFragA>;
    using GRMapB = GetDataLayout_t<GRFragB>;

    auto gReadOffA = GRMapA::fromMatrixCoord(make_coord2d(get<0>(macroTileCoord), 0u), lda);
    auto gReadOffB = GRMapB::fromMatrixCoord(make_coord2d(0u, get<1>(macroTileCoord)), ldb);
    auto kStepA    = GRMapA::fromMatrixCoord(make_coord2d(0u, MACRO_TILE_K), lda);
    auto kStepB    = GRMapB::fromMatrixCoord(make_coord2d(MACRO_TILE_K, 0u), ldb);

    // --- Double-buffer LDS (CK Tile: DoubleSmemBuffer = true) ---
    // Allocate 2x the LDS for ping-pong buffering
    HIP_DYNAMIC_SHARED(void*, localMemPtr);
    using LWMapA = GetDataLayout_t<LWFragA>;
    using LWMapB = GetDataLayout_t<LWFragB>;

    auto ldsOffA = 0u;
    auto ldsOffB = LWMapA::fromMatrixCoord(make_coord2d(ldsHeightA, 0u), ldsld);

    auto* ldsPtrLo = reinterpret_cast<InputT*>(localMemPtr);
    auto* ldsPtrHi = ldsPtrLo + sizeLds;

    using LRMapA = GetDataLayout_t<LRFragA>;
    using LRMapB = GetDataLayout_t<LRFragB>;
    auto ldsRdA  = ldsOffA + LRMapA::fromMatrixCoord(make_coord2d(get<0>(localWarpOffset), 0u), ldsld);
    auto ldsRdB  = ldsOffB + LRMapB::fromMatrixCoord(make_coord2d(get<1>(localWarpOffset), 0u), ldsld);

    // --- Step 1: Pre-fetch first K block (K=0) into LDS buffer Lo ---
    GRFragA grA;
    GRFragB grB;
    load_matrix_sync(grA, a + gReadOffA, lda);
    load_matrix_sync(grB, b + gReadOffB, ldb);
    gReadOffA += kStepA;
    gReadOffB += kStepB;

    store_matrix_sync(ldsPtrLo + ldsOffA, toLWFragA(grA), ldsld);
    store_matrix_sync(ldsPtrLo + ldsOffB, toLWFragB(grB), ldsld);

    // --- Accumulator ---
    MmaFragAcc fragAcc;
    fill_fragment(fragAcc, ComputeT(0));

    synchronize_workgroup();

    // --- Main K loop: compute K_i while prefetching K_{i+1} ---
    // CK Tile: this is the core of the COMPUTE_V4 double-buffer pipeline
    for(uint32_t kStep = MACRO_TILE_K; kStep < k; kStep += MACRO_TILE_K)
    {
        // 1. Local read from "Lo" buffer (data from previous prefetch)
        LRFragA lrA;
        LRFragB lrB;
        load_matrix_sync(lrA, ldsPtrLo + ldsRdA, ldsld);
        load_matrix_sync(lrB, ldsPtrLo + ldsRdB, ldsld);

        // 2. Prefetch next global K block
        load_matrix_sync(grA, a + gReadOffA, lda);
        load_matrix_sync(grB, b + gReadOffB, ldb);
        gReadOffA += kStepA;
        gReadOffB += kStepB;

        // 3. MMA on current K step (compute while next data is in flight)
        mma_sync(fragAcc, toMmaFragA(lrA), toMmaFragB(lrB), fragAcc);

        // 4. Write prefetch to "Hi" buffer
        store_matrix_sync(ldsPtrHi + ldsOffA, toLWFragA(grA), ldsld);
        store_matrix_sync(ldsPtrHi + ldsOffB, toLWFragB(grB), ldsld);

        // 5. Barrier: ensure Hi is fully written before next iteration reads it
        synchronize_workgroup();

        // 6. Swap buffers (Lo <-> Hi)
        auto* tmp = ldsPtrLo;
        ldsPtrLo  = ldsPtrHi;
        ldsPtrHi  = tmp;
    }

    // --- Tail: process the last K block ---
    LRFragA lrA;
    LRFragB lrB;
    load_matrix_sync(lrA, ldsPtrLo + ldsRdA, ldsld);
    load_matrix_sync(lrB, ldsPtrLo + ldsRdB, ldsld);
    mma_sync(fragAcc, toMmaFragA(lrA), toMmaFragB(lrB), fragAcc);

    // --- Epilogue: D = alpha * Acc + beta * C ---
    using MmaMapC = GetDataLayout_t<MmaFragC>;
    using MmaMapD = GetDataLayout_t<MmaFragD>;

    MmaFragC fragC;
    load_matrix_sync(fragC, c + MmaMapC::fromMatrixCoord(warpTileCoord, ldc), ldc);

    MmaFragD fragD;
    constexpr uint32_t chunkSize = 8u;
    constexpr uint32_t nChunks   = MmaFragD::num_elements / chunkSize;
    constexpr uint32_t nRemain   = MmaFragD::num_elements % chunkSize;

    auto doFma = [&](uint32_t start, uint32_t sz) {
        for(uint32_t i = start; i < start + sz; i++)
            fragD.x[i] = static_cast<OutputT>(alpha * fragAcc.x[i]
                                              + beta * static_cast<ComputeT>(fragC.x[i]));
    };
    for(uint32_t c2 = 0; c2 < nChunks; c2++)
        doFma(c2 * chunkSize, chunkSize);
    doFma(nChunks * chunkSize, nRemain);

    store_matrix_sync(d + MmaMapD::fromMatrixCoord(warpTileCoord, ldd), fragD, ldd);
}

// ---------------------------------------------------------------------------
// Host launcher and benchmarking
// Follows CK Tile's stream_config + rotating memory pattern
// ---------------------------------------------------------------------------

struct BenchResult
{
    double tflopsPerSec;
    float  elapsedMs;
    bool   passed;
};

// Run a single kernel variant with warmup + timing
template <typename KernelFn>
BenchResult benchmarkKernel(const char*    name,
                            KernelFn       kernelFn,
                            uint32_t       m,
                            uint32_t       n,
                            uint32_t       k,
                            InputT const*  d_a,
                            InputT const*  d_b,
                            OutputT const* d_c,
                            OutputT*       d_d,
                            uint32_t       lda,
                            uint32_t       ldb,
                            uint32_t       ldc,
                            uint32_t       ldd,
                            ComputeT       alpha,
                            ComputeT       beta,
                            uint32_t       ldsBytes,
                            dim3           gridDim,
                            dim3           blockDim,
                            uint32_t       warmups    = 5,
                            uint32_t       recordRuns = 20)
{
    // Warm-up (not recorded) -- matches CK Tile's n_warmup
    for(uint32_t i = 0; i < warmups; ++i)
    {
        kernelFn(gridDim, blockDim, ldsBytes, m, n, k, d_a, d_b, d_c, d_d,
                 lda, ldb, ldc, ldd, alpha, beta);
    }
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    // Timed runs -- matches CK Tile's n_repeat
    hipEvent_t startEvent, stopEvent;
    CHECK_HIP_ERROR(hipEventCreate(&startEvent));
    CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

    CHECK_HIP_ERROR(hipEventRecord(startEvent));
    for(uint32_t i = 0; i < recordRuns; ++i)
    {
        kernelFn(gridDim, blockDim, ldsBytes, m, n, k, d_a, d_b, d_c, d_d,
                 lda, ldb, ldc, ldd, alpha, beta);
    }
    CHECK_HIP_ERROR(hipEventRecord(stopEvent));
    CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));

    float elapsed = 0.0f;
    CHECK_HIP_ERROR(hipEventElapsedTime(&elapsed, startEvent, stopEvent));
    CHECK_HIP_ERROR(hipEventDestroy(startEvent));
    CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

    auto tflops = calculateTFlopsPerSec(m, n, k, static_cast<double>(elapsed), recordRuns);

    std::cout << "[" << name << "] "
              << "M=" << m << " N=" << n << " K=" << k
              << "  elapsed=" << (elapsed / recordRuns) << " ms"
              << "  TFlops/s=" << tflops << "\n";

    return BenchResult{tflops, elapsed / static_cast<float>(recordRuns), true};
}

void gemm_test(uint32_t m, uint32_t n, uint32_t k, ComputeT alpha, ComputeT beta)
{
    // Runtime arch detection (CK Tile validates tile fit at runtime)
    auto warpSize = getWarpSize();

    uint32_t hTBLOCK_X  = isGfx9() ? gfx9Params::TBLOCK_X  : gfx12Params::TBLOCK_X;
    uint32_t hTBLOCK_Y  = isGfx9() ? gfx9Params::TBLOCK_Y  : gfx12Params::TBLOCK_Y;
    uint32_t hBLOCKS_M  = isGfx9() ? gfx9Params::BLOCKS_M  : gfx12Params::BLOCKS_M;
    uint32_t hBLOCKS_N  = isGfx9() ? gfx9Params::BLOCKS_N  : gfx12Params::BLOCKS_N;
    uint32_t hROCWMMA_M = isGfx9() ? gfx9Params::ROCWMMA_M : gfx12Params::ROCWMMA_M;
    uint32_t hROCWMMA_N = isGfx9() ? gfx9Params::ROCWMMA_N : gfx12Params::ROCWMMA_N;
    uint32_t hROCWMMA_K = isGfx9() ? gfx9Params::ROCWMMA_K : gfx12Params::ROCWMMA_K;

    uint32_t hWARP_TILE_M   = hBLOCKS_M * hROCWMMA_M;
    uint32_t hWARP_TILE_N   = hBLOCKS_N * hROCWMMA_N;
    uint32_t hWARPS_M       = hTBLOCK_X / warpSize;
    uint32_t hWARPS_N       = hTBLOCK_Y;
    uint32_t hMACRO_TILE_M  = hWARPS_M * hWARP_TILE_M;
    uint32_t hMACRO_TILE_N  = hWARPS_N * hWARP_TILE_N;
    uint32_t hMACRO_TILE_K  = hROCWMMA_K;

    // Validate dimensions (CK Tile IsSupportedArgument check)
    if(m % hMACRO_TILE_M || n % hMACRO_TILE_N || k % hMACRO_TILE_K)
    {
        std::cout << "Unsupported matrix dimensions. M, N must be multiples of "
                  << hMACRO_TILE_M << ", " << hMACRO_TILE_N
                  << " and K a multiple of " << hMACRO_TILE_K << "\n";
        return;
    }

    if((isGfx11() || isGfx12()) && warpSize != Constants::AMDGCN_WAVE_SIZE_32)
    {
        std::cout << "Unsupported wave size for GFX11/GFX12!\n";
        return;
    }

    // Leading dimensions
    int lda = std::is_same_v<DataLayoutA, row_major> ? k : m;
    int ldb = std::is_same_v<DataLayoutB, row_major> ? n : k;
    int ldc = std::is_same_v<DataLayoutC, row_major> ? n : m;
    int ldd = ldc;

    std::cout << "\n=== GEMM Configuration ===\n"
              << "M=" << m << " N=" << n << " K=" << k
              << "  alpha=" << alpha << " beta=" << beta << "\n"
              << "Block Tile: " << hMACRO_TILE_M << "x" << hMACRO_TILE_N << "x" << hMACRO_TILE_K
              << "  Warp Tile: " << hWARP_TILE_M << "x" << hWARP_TILE_N
              << "  MFMA/WMMA: " << hROCWMMA_M << "x" << hROCWMMA_N << "x" << hROCWMMA_K << "\n"
              << "Warps per block: " << hWARPS_M << "x" << hWARPS_N
              << "  BLOCKS per warp: " << hBLOCKS_M << "x" << hBLOCKS_N << "\n"
              << "ThreadBlock: " << hTBLOCK_X << "x" << hTBLOCK_Y << "\n";

    // Allocate and initialize host matrices
    std::vector<InputT>  matA(m * k);
    std::vector<InputT>  matB(k * n);
    std::vector<OutputT> matC(m * n);
    std::vector<OutputT> matD(m * n, OutputT(0));
    std::vector<OutputT> matD_ref(m * n, OutputT(0));

    fillRand(matA.data(), m, k);
    fillRand(matB.data(), k, n);
    fillRand(matC.data(), m, n);

    // Device allocations
    InputT*  d_a;
    InputT*  d_b;
    OutputT* d_c;
    OutputT* d_d;

    CHECK_HIP_ERROR(hipMalloc(&d_a, matA.size() * sizeof(InputT)));
    CHECK_HIP_ERROR(hipMalloc(&d_b, matB.size() * sizeof(InputT)));
    CHECK_HIP_ERROR(hipMalloc(&d_c, matC.size() * sizeof(OutputT)));
    CHECK_HIP_ERROR(hipMalloc(&d_d, matD.size() * sizeof(OutputT)));

    CHECK_HIP_ERROR(hipMemcpy(d_a, matA.data(), matA.size() * sizeof(InputT),  hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_b, matB.data(), matB.size() * sizeof(InputT),  hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_c, matC.data(), matC.size() * sizeof(OutputT), hipMemcpyHostToDevice));

    // Grid / block dimensions
    dim3 blockDim(hTBLOCK_X, hTBLOCK_Y);
    dim3 gridDim(rocwmma::ceil_div(m, hMACRO_TILE_M),
                 rocwmma::ceil_div(n, hMACRO_TILE_N));

    // LDS sizes: V1 = 1x, V2 = 2x (double buffer)
    uint32_t ldsV1 = sizeof(InputT) * sizeLds;
    uint32_t ldsV2 = 2u * sizeof(InputT) * sizeLds;

    std::cout << "\nLDS per block: V1=" << ldsV1 << " B  V2=" << ldsV2 << " B\n";
    std::cout << "Grid: " << gridDim.x << "x" << gridDim.y
              << "  Block: " << blockDim.x << "x" << blockDim.y << "\n\n";

    // ---- Lambda wrappers for each kernel variant ----
    auto launchV1 = [](dim3 g, dim3 b, uint32_t lds,
                       uint32_t m_, uint32_t n_, uint32_t k_,
                       InputT const* a_, InputT const* b_in, OutputT const* c_, OutputT* d_,
                       uint32_t lda_, uint32_t ldb_, uint32_t ldc_, uint32_t ldd_,
                       ComputeT alpha_, ComputeT beta_)
    {
        hipExtLaunchKernelGGL(gemm_pipeline_v1,
                              g, b, lds, 0, nullptr, nullptr, 0,
                              m_, n_, k_, a_, b_in, c_, d_,
                              lda_, ldb_, ldc_, ldd_, alpha_, beta_);
    };

    auto launchV2 = [](dim3 g, dim3 b, uint32_t lds,
                       uint32_t m_, uint32_t n_, uint32_t k_,
                       InputT const* a_, InputT const* b_in, OutputT const* c_, OutputT* d_,
                       uint32_t lda_, uint32_t ldb_, uint32_t ldc_, uint32_t ldd_,
                       ComputeT alpha_, ComputeT beta_)
    {
        hipExtLaunchKernelGGL(gemm_pipeline_v2_double_buffer,
                              g, b, lds, 0, nullptr, nullptr, 0,
                              m_, n_, k_, a_, b_in, c_, d_,
                              lda_, ldb_, ldc_, ldd_, alpha_, beta_);
    };

    // ---- Benchmark V1 ----
    auto resV1 = benchmarkKernel("Pipeline-V1 (sequential)",
                                  launchV1, m, n, k,
                                  d_a, d_b, d_c, d_d,
                                  lda, ldb, ldc, ldd,
                                  alpha, beta, ldsV1, gridDim, blockDim);

    // ---- Benchmark V2 ----
    auto resV2 = benchmarkKernel("Pipeline-V2 (double-buffer)",
                                  launchV2, m, n, k,
                                  d_a, d_b, d_c, d_d,
                                  lda, ldb, ldc, ldd,
                                  alpha, beta, ldsV2, gridDim, blockDim);

#if !NDEBUG
    // --- Validation ---
    std::cout << "\nValidating Pipeline-V2 result...\n";
    CHECK_HIP_ERROR(hipMemcpy(matD.data(), d_d, matD.size() * sizeof(OutputT), hipMemcpyDeviceToHost));

    gemm_cpu_h<InputT, OutputT, ComputeT, DataLayoutA, DataLayoutB, DataLayoutC>(
        m, n, k,
        matA.data(), matB.data(), matC.data(), matD_ref.data(),
        lda, ldb, ldc, ldd, alpha, beta);

    auto [passed, maxRelErr] = compareEqual(matD.data(), matD_ref.data(), m * n);
    std::cout << (passed ? "PASSED" : "FAILED")
              << "  MaxRelError=" << maxRelErr << "\n";
#endif

    // ---- Performance Summary ----
    std::cout << "\n=== Performance Summary ===\n"
              << std::left
              << std::setw(35) << "Kernel"
              << std::setw(15) << "TFlops/s"
              << std::setw(15) << "Speedup vs V1"
              << "\n"
              << std::string(65, '-') << "\n"
              << std::setw(35) << "Pipeline-V1 (sequential)"
              << std::setw(15) << std::fixed << std::setprecision(3) << resV1.tflopsPerSec
              << std::setw(15) << "1.00x"
              << "\n"
              << std::setw(35) << "Pipeline-V2 (double-buffer)"
              << std::setw(15) << std::fixed << std::setprecision(3) << resV2.tflopsPerSec
              << std::setw(15) << std::fixed << std::setprecision(2)
              << (resV2.tflopsPerSec / resV1.tflopsPerSec) << "x"
              << "\n\n";

    // Cleanup
    CHECK_HIP_ERROR(hipFree(d_a));
    CHECK_HIP_ERROR(hipFree(d_b));
    CHECK_HIP_ERROR(hipFree(d_c));
    CHECK_HIP_ERROR(hipFree(d_d));
}

int main(int argc, char* argv[])
{
    // Default sizes from CK Tile benchmark script: large square GEMMs
    uint32_t m = 4096, n = 4096, k = 4096;
    if(argc >= 4)
    {
        m = static_cast<uint32_t>(std::atoi(argv[1]));
        n = static_cast<uint32_t>(std::atoi(argv[2]));
        k = static_cast<uint32_t>(std::atoi(argv[3]));
    }

    ComputeT alpha = 1.0f;
    ComputeT beta  = 0.0f;

    // Print device info
    hipDeviceProp_t devProp;
    CHECK_HIP_ERROR(hipGetDeviceProperties(&devProp, 0));
    std::cout << "Device: " << devProp.name
              << "  GCN Arch: " << devProp.gcnArchName << "\n";

    // Run a suite of sizes that stress different compute:bandwidth ratios
    // matching CK Tile's default benchmark matrix sizes
    struct TestCase { uint32_t m, n, k; };
    std::vector<TestCase> sizes;

    if(argc >= 4)
    {
        sizes.push_back({m, n, k});
    }
    else
    {
        // CK Tile-style benchmark suite
        sizes = {
            {3840, 4096, 4096},   // Large square-ish (LLM attention)
            {4096, 4096, 4096},   // Pure square
            {8192, 8192, 8192},   // Very large
            {1024, 4096, 8192},   // Tall-thin (decode-like)
            {4096, 1024, 8192},   // Decode-like transposed
        };
    }

    for(auto& tc : sizes)
    {
        gemm_test(tc.m, tc.n, tc.k, alpha, beta);
    }

    return 0;
}
