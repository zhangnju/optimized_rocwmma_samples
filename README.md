# rocWMMA Community Samples: CK Tile Optimization Porting Guide

## Overview

This document covers the core optimization techniques, code analysis, and benchmark results (MI355X and Radeon AI PRO R9700)
for 26 GPU kernel samples ported from CK Tile (ComposableKernel Tile) using the rocWMMA API.

All samples are located at: `rocwmma/samples/community/rocwmma_*.cpp`

**Test environments (community samples in this repo):**

| Item | MI355X (gfx950) | Radeon AI PRO R9700 (gfx1201) |
|---|---|---|
| **GPU** | AMD Instinct MI355X (×8) | AMD Radeon AI PRO R9700 |
| **Architecture** | CDNA 3.5, 256 CUs, 2400 MHz | RDNA 4, 64 CUs, Wave32, WMMA 16×16×16 (boost up to 2.92 GHz) |
| **ROCm** | 7.2.0 | 7.2.0 |
| **Compiler** | AMD Clang 22.0.0 (HIP 7.2) | AMD Clang 22.0.0 (HIP 7.2) |
| **rocWMMA** | 2.2.0 | 2.2.0 |
| **FP16 peak (matrix)** | ~1300 TFlops/s (CDNA MFMA class) | **191** TFlops/s ([AMD specs](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html), FP16 matrix) |
| **FP8 peak (matrix, E4M3/E5M2)** | ~2600 TFlops/s (CDNA class) | **383** TFlops/s (same source, FP8 matrix) |
| **Sample benchmark** | See §11 (MI355X tables) | Measured **2026-04-07**; log `benchmark_results_r9700_gfx1201.txt` / `.csv` |

Measured kernel throughputs for both GPUs are in **[Section 11](#11-complete-performance-data)** (MI355X first, R9700 second in each subsection). Manufacturer peaks are theoretical upper bounds; achievable TFlops depend on kernel and memory behavior.

---

## Table of Contents

0. [Quick Start: Build and Run](#0-quick-start-build-and-run)
1. [Three-Level Tile Hierarchy](#1-three-level-tile-hierarchy)
2. [Double-Buffer LDS Pipeline (COMPUTE_V4)](#2-double-buffer-lds-pipeline-compute_v4)
3. [Cooperative Global Memory Load](#3-cooperative-global-memory-load)
4. [LDS Transpose of B (Eliminating Bank Conflicts)](#4-lds-transpose-of-b-eliminating-bank-conflicts)
5. [Warp Tile Data Reuse](#5-warp-tile-data-reuse)
6. [Vectorized Memory Access](#6-vectorized-memory-access)
7. [Warp-Level Butterfly Reduction](#7-warp-level-butterfly-reduction)
8. [Online Softmax (Flash Attention)](#8-online-softmax-flash-attention)
9. [FlatMM: Register-Direct Load of B](#9-flatmm-register-direct-load-of-b)
10. [Stream-K Load Balancing](#10-stream-k-load-balancing)
11. [Complete Performance Data (MI355X gfx950 & R9700 gfx1201)](#11-complete-performance-data)
12. [CK Tile vs rocWMMA Full Comparison (Including Elementwise/Pooling/Contraction)](#12-ck-tile-vs-rocwmma-full-comparison-mi355x--gfx950)

---

## 0. Quick Start: Build and Run

This section provides complete documentation and usage examples for `build.sh` and `benchmark.sh`,
located in `rocwmma/samples/community/`.

### 0.1 Prerequisites

```bash
# Verify ROCm version (requires >= 6.0)
rocminfo | grep -E "ROCm|Version"
hipcc --version

# Verify GPU visibility
rocminfo | grep "Marketing Name"
# Expected: AMD Instinct MI355X (or other supported GPU)
```

### 0.2 Build Script: `build.sh`

Located at `samples/community/build.sh` (or `/home/optimized_rocwmma_samples/build.sh` for standalone deployments).
Run `./build.sh -h` for full options.

**Common build commands:**

```bash
cd rocwmma/samples/community

# 1) Auto-detect GPU, build all community samples
./build.sh

# 2) Specify MI355X (gfx950), build all samples
./build.sh -g gfx950

# 2b) AMD Radeon AI PRO R9700 / RDNA 4 (gfx1201)
./build.sh -g gfx1201

# 3) Multi-GPU targets (MI355X + RDNA4), build simultaneously
./build.sh -g "gfx950;gfx1200;gfx1201"

# 4) Build a single sample (fast iteration)
./build.sh -g gfx950 -t rocwmma_perf_gemm

# 5) Clean rebuild (use when switching GPU targets)
./build.sh -g gfx950 -c

# 6) Parallel build with 32 cores
./build.sh -g gfx950 -j 32

# 7) Debug mode (enables CPU verification)
./build.sh -g gfx950 -d -t rocwmma_layernorm2d
```

**Sample build output:**

```
============================================================
 rocWMMA Community Samples Build
============================================================
 Repo root   : /home/user/rocwmma
 Build dir   : /home/user/rocwmma/build
 Build type  : Release
 GPU targets : gfx950
 CMake target: rocwmma_community_samples
 Jobs        : 128
============================================================

[INFO] Configuring with CMake...
-- GPU_TARGETS=gfx950

[INFO] Building target: rocwmma_community_samples (jobs=128)...
[  4%] Built target rocwmma_tile_distr_reg_map
[  8%] Built target rocwmma_layernorm2d
...
[100%] Built target rocwmma_community_samples

============================================================
 Build complete!
 Binaries: /home/user/rocwmma/build/samples/community/
============================================================
rocwmma_batched_contraction
rocwmma_batched_gemm
...
```

---

### 0.3 Benchmark Script: `benchmark.sh`

Located at `samples/community/benchmark.sh` (or `/home/optimized_rocwmma_samples/benchmark.sh` for standalone deployments).
Covers all 26 samples with timeout protection, CSV output, and GPU device selection.
Run `./benchmark.sh -h` for full options.

**Common benchmark commands:**

```bash
cd rocwmma/samples/community

# 1) Run all samples, write results to default file
./benchmark.sh -b ../../build

# 2) Specify GPU (multi-GPU systems)
HIP_VISIBLE_DEVICES=2 ./benchmark.sh -b ../../build

# 3) Run specific samples only
./benchmark.sh -b ../../build -s rocwmma_perf_gemm
./benchmark.sh -b ../../build -s "rocwmma_layernorm2d,rocwmma_rmsnorm2d,rocwmma_smoothquant"

# 4) Generate CSV results (for further analysis)
./benchmark.sh -b ../../build --csv -o results_mi355x.txt

# 5) Specify output file
./benchmark.sh -b ../../build -o /tmp/bench_$(date +%Y%m%d).txt --csv
```

---

### 0.4 Manual Execution (Without Scripts)

**Build a single sample:**

```bash
# Enter build directory
cd /path/to/rocwmma/build

# Reconfigure (only on first run or when parameters change)
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGPU_TARGETS="gfx950" \
    -DROCWMMA_BUILD_SAMPLES=ON \
    -DROCWMMA_BUILD_COMMUNITY_SAMPLES=ON \
    -DROCWMMA_BUILD_TESTS=OFF

# Build all community samples
make -j$(nproc) rocwmma_community_samples

# Or build a single sample
make -j$(nproc) rocwmma_perf_gemm
```

**Run individual samples:**

```bash
BIN=./samples/community  # shorthand

# ── GEMM ─────────────────────────────────────────────────────────────────────
# HGEMM Pipeline V1/V2 comparison (runs 5 default sizes)
${BIN}/rocwmma_perf_gemm

# Specify dimensions (M N K)
${BIN}/rocwmma_perf_gemm 4096 4096 4096
${BIN}/rocwmma_perf_gemm 8192 8192 8192
${BIN}/rocwmma_perf_gemm 1024 4096 8192   # decode-like (small M)

# Split-K GEMM
${BIN}/rocwmma_gemm_splitk

# FP8 quantized GEMM
${BIN}/rocwmma_gemm_quantized

# Batched GEMM (batch M N K)
${BIN}/rocwmma_batched_gemm 4  1024 1024 1024
${BIN}/rocwmma_batched_gemm 16 512  512  4096

# GEMM + fused epilogue (bias + ReLU)
${BIN}/rocwmma_gemm_multi_d

# Grouped GEMM
${BIN}/rocwmma_grouped_gemm

# Stream-K GEMM
${BIN}/rocwmma_streamk_gemm

# FlatMM (B bypasses LDS)
${BIN}/rocwmma_flatmm

# MX format GEMM (native gfx950)
${BIN}/rocwmma_mx_gemm

# Batched tensor contraction
${BIN}/rocwmma_batched_contraction

# ── Attention ────────────────────────────────────────────────────────────────
# Flash Attention forward (B H Sq Sk D causal)
${BIN}/rocwmma_fmha_fwd

# Sparse attention (Jenga LUT)
${BIN}/rocwmma_sparse_attn

# ── Normalization ─────────────────────────────────────────────────────────────
# LayerNorm2D (M N)
${BIN}/rocwmma_layernorm2d 3328 4096
${BIN}/rocwmma_layernorm2d 3328 8192

# RMSNorm2D + Add+RMSNorm fused (M N)
${BIN}/rocwmma_rmsnorm2d 3328 4096

# SmoothQuant FP16→INT8 (M N)
${BIN}/rocwmma_smoothquant 3328 4096

# ── MoE ──────────────────────────────────────────────────────────────────────
# TopK Softmax (tokens experts topk)
${BIN}/rocwmma_topk_softmax 3328 32 5

# MoE Token Sorting (tokens experts topk)
${BIN}/rocwmma_moe_sorting 3328 64 5

# MoE Smooth Quantization
${BIN}/rocwmma_moe_smoothquant

# Fully fused MoE (sort + gate_up GEMM + SiLU + down GEMM)
${BIN}/rocwmma_fused_moe

# ── Reduction & Elementwise ───────────────────────────────────────────────────
# Row-wise Reduce Sum/Max (M N)
${BIN}/rocwmma_reduce 3328 4096

# Elementwise (M N)
${BIN}/rocwmma_elementwise 3840 4096

# ── Convolution & Data Layout ─────────────────────────────────────────────────
# Im2Col + GEMM (CNN layers)
${BIN}/rocwmma_img2col_gemm

# Grouped convolution forward
${BIN}/rocwmma_grouped_conv_fwd

# 3D Pooling
${BIN}/rocwmma_pooling

# Tensor Permute / NCHW↔NHWC
${BIN}/rocwmma_permute

# ── Utilities ─────────────────────────────────────────────────────────────────
# Tile Distribution Register Map (diagnostic: prints fragment shapes)
${BIN}/rocwmma_tile_distr_reg_map
```

---

### 0.5 Complete End-to-End Workflow (From Scratch)

```bash
# 1. Clone repository
git clone https://github.com/ROCm/rocWMMA.git
cd rocWMMA

# 2. Build all community samples (MI355X / gfx950)
./samples/community/build.sh -g gfx950

# 3. Run full benchmark and generate report
./samples/community/benchmark.sh \
    -b build \
    -o samples/community/benchmark_results.txt \
    --csv

# 4. View results
cat samples/community/benchmark_results.txt | grep -E "TFlops|GB/s"

# 5. View CSV and sort by performance
sort -t',' -k4 -rn samples/community/benchmark_results.csv | head -20
```

---

### 0.6 RDNA4 (gfx1200/gfx1201) Build Notes

RDNA4 uses Wave32 + WMMA 16×16×16, which differs from gfx9's Wave64 + MFMA 32×32×16.
All samples automatically select parameters via compile-time macros `ROCWMMA_ARCH_GFX9` / `ROCWMMA_ARCH_GFX12`:

```bash
# Build for RDNA4
./build.sh -g "gfx1200;gfx1201"

# Build fat binary for both MI355X and RDNA4
./build.sh -g "gfx950;gfx1200;gfx1201"
```

Key parameter differences (auto-selected in code):

```cpp
// rocwmma_perf_gemm.cpp
#if defined(ROCWMMA_ARCH_GFX9)
// MI355X / gfx950: Wave64 + MFMA 32×32×16
constexpr uint32_t ROCWMMA_M=32, ROCWMMA_N=32, ROCWMMA_K=16;
constexpr uint32_t WARP_SIZE = Constants::AMDGCN_WAVE_SIZE_64;  // = 64
#else
// RDNA4 / gfx1200: Wave32 + WMMA 16×16×16
constexpr uint32_t ROCWMMA_M=16, ROCWMMA_N=16, ROCWMMA_K=16;
constexpr uint32_t WARP_SIZE = Constants::AMDGCN_WAVE_SIZE_32;  // = 32
#endif
```

---

## 1. Three-Level Tile Hierarchy

**CK Tile equivalent:** `TileGemmShape<sequence<M,N,K>, sequence<Mw,Nw,Kw>, sequence<Mwt,Nwt,Kwt>>`

CK Tile decomposes GEMM computation into three levels: Block Tile → Warp Layout → MFMA/WMMA Tile.
rocWMMA implements the same decomposition via `fragment` dimensions and thread block configuration.

```cpp
// =========================================================
// File: rocwmma_perf_gemm.cpp  ~line 100-130
// =========================================================

// Level 3: Matrix block handled by a single MFMA instruction (gfx9: 32×32×16)
//          Corresponds to CK Tile M_Warp_Tile × N_Warp_Tile × K_Warp_Tile
namespace gfx9Params { enum : uint32_t {
    ROCWMMA_M = 32u,   // MFMA block M
    ROCWMMA_N = 32u,   // MFMA block N
    ROCWMMA_K = 16u,   // MFMA block K
    BLOCKS_M  = 2u,    // Number of MFMA tiles per warp along M
    BLOCKS_N  = 2u,    // Number of MFMA tiles per warp along N
    TBLOCK_X  = 128u,  // Thread block X threads (= WARPS_M × 64)
    TBLOCK_Y  = 2u,    // Thread block Y threads (= WARPS_N)
    WARP_SIZE = Constants::AMDGCN_WAVE_SIZE_64
}; }

// Level 2: Warp Tile = BLOCKS_M × ROCWMMA_M
//          Corresponds to CK Tile M_Warp × M_Warp_Tile
constexpr uint32_t WARP_TILE_M  = BLOCKS_M * ROCWMMA_M;  // 2×32 = 64
constexpr uint32_t WARP_TILE_N  = BLOCKS_N * ROCWMMA_N;  // 2×32 = 64

// Level 1: Block (Macro) Tile = WARPS_M × WARP_TILE_M
//          Corresponds to CK Tile M_Tile (Block Tile)
constexpr uint32_t WARPS_M      = TBLOCK_X / WARP_SIZE;  // 128/64 = 2
constexpr uint32_t WARPS_N      = TBLOCK_Y;               // 2
constexpr uint32_t MACRO_TILE_M = WARPS_M * WARP_TILE_M; // 2×64 = 128
constexpr uint32_t MACRO_TILE_N = WARPS_N * WARP_TILE_N; // 2×64 = 128
constexpr uint32_t MACRO_TILE_K = ROCWMMA_K;             // 16
```

**Hierarchy Visualization:**

```
Block Tile (128×128)
├── Warp(0,0): Warp Tile (64×64) ← 2×2 MFMA 32×32×16 tiles
├── Warp(1,0): Warp Tile (64×64)
├── Warp(0,1): Warp Tile (64×64)
└── Warp(1,1): Warp Tile (64×64)

Inside each Warp Tile (BLOCKS_M=2, BLOCKS_N=2):
[MFMA(0,0)][MFMA(0,1)]    Each MFMA = 32×32×16
[MFMA(1,0)][MFMA(1,1)]    4 mma_sync calls total
```

**Design rationale:**
- Block Tile determines L2 cache reuse: 128×128×FP16 = 32KB, fits in first-level cache
- Warp Tile determines register occupancy: 64×64×FP32 acc = 16KB/warp (within MI355X's 256KB register file)
- MFMA Tile is fixed by hardware: 32×32×16 is the native size of gfx9 MFMA instructions

---

## 2. Double-Buffer LDS Pipeline (COMPUTE_V4)

**CK Tile equivalent:** `GemmPipelineAgBgCrCompV4` with `DoubleSmemBuffer = true`

This is CK Tile's most critical optimization: while computing the current K step, prefetch the next K step's data into a second LDS buffer, hiding global memory latency.

```cpp
// =========================================================
// File: rocwmma_perf_gemm.cpp  ~line 270-340 (Pipeline V2)
// =========================================================

// ─── Allocate two LDS buffers (ping-pong double buffer) ───
// CK Tile: allocates 2×sizeLds when DoubleSmemBuffer = true
HIP_DYNAMIC_SHARED(void*, localMemPtr);
auto* ldsPtrLo = reinterpret_cast<InputT*>(localMemPtr);
auto* ldsPtrHi = ldsPtrLo + sizeLds;   // Second buffer offset by sizeLds elements

// ─── Prefetch K=0 into buffer Lo ───
GRFragA grA; GRFragB grB;
load_matrix_sync(grA, a + gReadOffA, lda);   // Global memory → registers
load_matrix_sync(grB, b + gReadOffB, ldb);
gReadOffA += kStepA;  // Advance to K=1 address
gReadOffB += kStepB;
store_matrix_sync(ldsPtrLo + ldsOffA, toLWFragA(grA), ldsld);  // Registers → LDS Lo
store_matrix_sync(ldsPtrLo + ldsOffB, toLWFragB(grB), ldsld);

MmaFragAcc fragAcc;
fill_fragment(fragAcc, ComputeT(0));
synchronize_workgroup();   // Ensure all warps' LDS writes complete

// ─── Main K loop: compute K_i while prefetching K_{i+1} ───
for(uint32_t kStep = MACRO_TILE_K; kStep < k; kStep += MACRO_TILE_K) {
    // Step A: Read current K step data from Lo (low-latency LDS)
    LRFragA lrA; LRFragB lrB;
    load_matrix_sync(lrA, ldsPtrLo + ldsRdA, ldsld);
    load_matrix_sync(lrB, ldsPtrLo + ldsRdB, ldsld);

    // Step B: Simultaneously prefetch next K step (high-latency global memory masked by MMA)
    load_matrix_sync(grA, a + gReadOffA, lda);   // ← runs in parallel with MMA
    load_matrix_sync(grB, b + gReadOffB, ldb);
    gReadOffA += kStepA;
    gReadOffB += kStepB;

    // Step C: Compute MMA for current K step
    mma_sync(fragAcc, toMmaFragA(lrA), toMmaFragB(lrB), fragAcc);

    // Step D: Write prefetched results into Hi buffer
    store_matrix_sync(ldsPtrHi + ldsOffA, toLWFragA(grA), ldsld);
    store_matrix_sync(ldsPtrHi + ldsOffB, toLWFragB(grB), ldsld);

    synchronize_workgroup();  // Wait for Hi writes to complete

    // Step E: Swap Lo/Hi pointers (O(1) operation, no data copy)
    auto* tmp = ldsPtrLo;
    ldsPtrLo  = ldsPtrHi;
    ldsPtrHi  = tmp;
}
```

**Execution Timeline:**

```
Time →
K=0: [Global Load A0,B0] → [LDS Store Lo]
K=1: [LDS Read Lo] | [Global Load A1,B1]   ← parallel!
     [MMA K0]      | [LDS Store Hi]
                     [Swap Lo↔Hi]
K=2: [LDS Read Lo] | [Global Load A2,B2]   ← parallel!
     [MMA K1]      | [LDS Store Hi]
...
```

**LDS Size Calculation:**

```cpp
// LDS layout: A block (ldHA rows) followed by B block (ldHB rows), both width MACRO_TILE_K
constexpr uint32_t ldsHeightA = GetIOShape_t<LWFragA>::BlockHeight;  // = 128
constexpr uint32_t ldsHeightB = GetIOShape_t<LWFragB>::BlockHeight;  // = 128
constexpr uint32_t ldsHeight  = ldsHeightA + ldsHeightB;             // = 256
constexpr uint32_t ldsWidth   = MACRO_TILE_K;                        // = 16
constexpr uint32_t sizeLds    = ldsHeight * ldsWidth;                // = 4096 elements

// Single-buffer LDS: 4096 × 2 bytes = 8 KB
// Double-buffer LDS: 8192 × 2 bytes = 16 KB (CK Tile DoubleSmemBuffer = true)
```

---

## 3. Cooperative Global Memory Load

**CK Tile equivalent:** `CoopScheduler` + `GemmPipelineProblem::CoopA/B`

All warps in a block cooperatively share the global memory load of the Macro Tile, improving bandwidth utilization.

```cpp
// =========================================================
// File: rocwmma_perf_gemm.cpp  ~line 175-200
// =========================================================

// Use coop_row_major_2d scheduler: distributes Macro Tile I/O work
// evenly among warps within the TBLOCK_X × TBLOCK_Y thread block
// CK Tile equivalent: fragment<..., CoopScheduler> + coop load
using CoopScheduler = fragment_scheduler::coop_row_major_2d<TBLOCK_X, TBLOCK_Y>;

// Macro Tile-level cooperative load fragment
// Each warp only loads 1/(WARPS_M×WARPS_N) of GRFragA
using GRFragA = fragment<matrix_a,
                         MACRO_TILE_M,   // = 128 (entire block M range)
                         MACRO_TILE_N,   // = 128 (entire block N range)
                         MACRO_TILE_K,   // = 16
                         InputT,
                         DataLayoutA,
                         CoopScheduler>; // ← key: enables cooperative load

// Each warp computes its own row offset for the portion it loads
// In practice, load_matrix_sync internally assigns work based on CoopScheduler
load_matrix_sync(grA, a + gReadOffA, lda);  // Each warp loads 1/4 of data

// Then all warps write to their respective positions in LDS
store_matrix_sync(ldsPtrLo + ldsOffA, toLWFragA(grA), ldsld);
```

**Benefits of Cooperative Loading:**
```
Without cooperation: each warp independently loads the full Macro Tile
  → 4 warps × 128×16×2 bytes = 4× redundant traffic

With cooperation: 4 warps each load 1/4
  → Total traffic = 128×16×2 bytes (no redundancy)
  → Improved L2 cache hit rate and global bandwidth utilization
```

---

## 4. LDS Transpose of B (Eliminating Bank Conflicts)

**CK Tile equivalent:** `CLayout::LdsB = Transposed`

Matrix B is transposed when stored to LDS, making the K dimension the fast axis (column) for both A and B,
ensuring bank-conflict-free LDS access patterns.

```cpp
// =========================================================
// File: rocwmma_perf_gemm.cpp  ~line 155-170
// =========================================================

// Global B: row_major layout, shape [K, N], fast axis is N
using DataLayoutB   = row_major;

// LDS layout: col_major, making K the fast axis
using DataLayoutLds = col_major;

// Transpose B when writing to LDS:
//   Original B[k, n] → stored in LDS as B^T[n, k]
//   Now the LDS column direction = K direction = shared contraction dimension
using LWFragB = apply_data_layout_t<
    apply_transpose_t<GRFragB>,  // ← transpose first (swap rows and columns)
    DataLayoutLds>;               // ← then map to LDS in col_major

// Transpose B again when reading from LDS, restoring MFMA-expected format
using LRFragB = apply_data_layout_t<
    apply_transpose_t<MmaFragB>,  // ← transpose
    DataLayoutLds>;

// Actual write:
ROCWMMA_DEVICE auto toLWFragB(GRFragB const& gr)
{
    return apply_data_layout<DataLayoutLds>(
        apply_transpose(gr)  // Transpose B, then store to LDS in col_major
    );
}

// Actual read:
ROCWMMA_DEVICE auto toMmaFragB(LRFragB const& lr)
{
    return apply_data_layout<DataLayoutB>(
        apply_transpose(lr)  // Read from LDS and transpose again, restoring row_major
    );
}
```

**LDS Bank Conflict Analysis:**

```
gfx9 LDS has 32 banks, each 4 bytes wide.
Wave 64 has 64 lanes accessing LDS simultaneously.

Without transpose (B in row_major, K=16, N=128):
  lanes 0-15 read B[0][0..15], stride=1 → banks 0,1,2,...,15 → no conflict
  lanes 16-31 read B[0][16..31] → banks 16..31 → no conflict
  But next row: lanes read B[1][0..15] → same banks 0..15 → 16-way conflict!

With transpose (B^T in col_major, K=16 as rows):
  All lanes read different rows of the same column → each lane accesses a different bank → 0 conflicts
```

---

## 5. Warp Tile Data Reuse

**CK Tile equivalent:** `BLOCKS_M × BLOCKS_N` loop in `BlockGemmASmemBSmemCRegV1`

Each warp computes `BLOCKS_M × BLOCKS_N` MFMA tiles, reusing a column of A and a row of B in the inner loop.

```cpp
// =========================================================
// File: rocwmma_batched_gemm.cpp  ~line 160-200 (conceptually equivalent to)
// rocwmma_perf_gemm.cpp implements via WARP_TILE_M/N = BLOCKS*ROCWMMA
// =========================================================

// MmaFragA size = WARP_TILE_M × WARP_TILE_N = 64×64
// Internally aggregates BLOCKS_M × BLOCKS_N = 2×2 MFMA 32×32 tiles
using MmaFragA = fragment<matrix_a,
    WARP_TILE_M,  // = BLOCKS_M * ROCWMMA_M = 2*32 = 64
    WARP_TILE_N,  // = BLOCKS_N * ROCWMMA_N = 2*32 = 64
    WARP_TILE_K,  // = ROCWMMA_K = 16
    InputT, DataLayoutA>;

// One mma_sync is equivalent to CK Tile's:
// for i in range(BLOCKS_M):
//   for j in range(BLOCKS_N):
//     mfma(acc[i,j], A[i], B[j], acc[i,j])
// rocWMMA handles the 2×2 sub-matrix automatically at the fragment level
mma_sync(fragAcc, toMmaFragA(lrA), toMmaFragB(lrB), fragAcc);
// ↑ equivalent to 2×2=4 MFMA 32×32×16 instructions
```

**Data Reuse Analysis (BLOCKS=2×2):**

```
A[row0]: used once each with B's col0 and col1  ← N-direction reuse ×BLOCKS_N
A[row1]: used once each with B's col0 and col1

B[col0]: used once each with A's row0 and row1  ← M-direction reuse ×BLOCKS_M
B[col1]: used once each with A's row0 and row1

Total reuse = BLOCKS_M × BLOCKS_N = 4×
Each LDS read is used by MMA 4 times (instead of once)
```

---

## 6. Vectorized Memory Access

**CK Tile equivalent:** `VectorSize = 8` (`fp16 × 8 = 128-bit`)

In elementwise, reduce, and normalization kernels, each thread loads 8 fp16 values at once (one float4).

```cpp
// =========================================================
// File: rocwmma_reduce.cpp  ~line 60-95
//       rocwmma_layernorm2d.cpp  ~line 80-110
// =========================================================

constexpr uint32_t VEC = 8;  // 8 × fp16 = 128-bit, aligned to 128-bit memory access

// Vectorized load: reads 128 bits at once (4 × float32 = 8 × float16)
// CK Tile equivalent: CK_TILE_UNROLL, ThreadwiseTensorSliceTransfer with VectorSize=8
for(uint32_t col = threadIdx.x * VEC; col + VEC <= N; col += TBLOCK * VEC)
{
    // Cast float16* to float4* for 128-bit aligned read
    const float4* ptr = reinterpret_cast<const float4*>(row_ptr + col);
    float4 v = *ptr;   // Single 128-bit load, equivalent to 4×ds_read_b128

    // Unpack: float4 contains 4 __half2 values (each __half2 = 2×float16)
    const __half2* h = reinterpret_cast<const __half2*>(&v);

    for(int i = 0; i < 4; i++) {
        float2 f = __half22float2(h[i]);  // __half2 → float2 (SIMD conversion)
        acc += f.x + f.y;                 // accumulate 2 elements
    }
}
```

**Theoretical Performance Comparison:**

```
Without vectorization: 16-bit read per thread per access
  Throughput = 1× (limited by number of memory transactions)

With vectorization (float4): 128-bit read per thread per access
  Throughput ≈ 8× (8 fp16 per transaction)
  Actual improvement: ~2-4× (when bandwidth-bound, not latency-bound)
```

---

## 7. Warp-Level Butterfly Reduction

**CK Tile equivalent:** `WarpReduce<ReduceOp::Add>` / `warp_reduce`

LayerNorm, RMSNorm, Reduce, SmoothQuant and other kernels use the butterfly pattern
with `__shfl_down` for efficient LDS-free intra-warp reduction.

```cpp
// =========================================================
// File: rocwmma_reduce.cpp  ~line 40-55
//       rocwmma_layernorm2d.cpp  ~line 65-80
// =========================================================

// Wave64 butterfly reduction: log2(64) = 6 steps, each step halves the reduction range
// CK Tile: WarpReduce<AccType, ReduceOp::Add>
template <typename T, typename ReduceOp>
__device__ __forceinline__ T warpReduce(T val, ReduceOp op)
{
    // Step 1: offset=32 → merge lane[0..31] and lane[32..63]
    // Step 2: offset=16 → merge lane[0..15] and lane[16..31] within each group
    // ...
    // Step 6: offset=1  → merge adjacent lanes
    for(int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = op(val, __shfl_down(val, offset, WARP_SIZE));
    //          ↑ no LDS needed, direct lane-to-lane communication (1 cycle/step)
    return val;
    // Final: lane 0 holds the complete reduction result
}

// Welford online variance warp reduction (for LayerNorm)
// CK Tile: WelfordWarpReduce
struct WelfordVar { float mean, m2; uint32_t count; };

__device__ WelfordVar welfordWarpReduce(WelfordVar v)
{
    for(int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float  om  = __shfl_down(v.mean,  offset, WARP_SIZE);
        float  om2 = __shfl_down(v.m2,    offset, WARP_SIZE);
        uint32_t oc = __shfl_down(v.count, offset, WARP_SIZE);

        // Parallel Welford merge (mathematically equivalent to sequential Welford)
        uint32_t nc = v.count + oc;
        if(nc == 0) continue;
        float delta = om - v.mean;
        v.mean   = v.mean + delta * oc / nc;
        v.m2    += om2 + delta * delta * (float)v.count * oc / nc;
        v.count  = nc;
    }
    return v;
}

// Block-level reduction: warp results written to LDS, merged in first warp
__shared__ float smem[WARPS];  // one slot per warp
if(lid == 0) smem[wid] = acc;   // only lane 0 writes (carries warp reduction result)
__syncthreads();
if(wid == 0) {                   // first warp does final reduction
    acc = (lid < WARPS) ? smem[lid] : 0.f;
    acc = warpReduce(acc, [](float a, float b){ return a + b; });
}
```

**Butterfly Reduction Steps (WARP_SIZE=8 simplified):**

```
Initial: [a0, a1, a2, a3, a4, a5, a6, a7]

offset=4:
lane 0 ← a0+a4,  lane 1 ← a1+a5,  lane 2 ← a2+a6,  lane 3 ← a3+a7

offset=2:
lane 0 ← (a0+a4)+(a2+a6),  lane 1 ← (a1+a5)+(a3+a7)

offset=1:
lane 0 ← sum(all)   ← final result in lane 0

log2(8) = 3 steps total, no LDS operations required
```

---

## 8. Online Softmax (Flash Attention)

**CK Tile equivalent:** `FmhaPipelineQRKSVS` (Flash Attention 2 algorithm)

The core of Flash Attention is online softmax: maintaining running max and sum while iterating over KV blocks,
avoiding materialization of the full S=QK^T matrix.

```cpp
// =========================================================
// File: rocwmma_fmha_fwd.cpp  ~line 140-200
//       rocwmma_sparse_attn.cpp  ~line 170-220
// =========================================================

// Each warp maintains independent running max/sum for MFMA_M=32 rows of Q
float row_max[MFMA_M], row_sum[MFMA_M];
for(uint32_t i = 0; i < MFMA_M; i++) { row_max[i] = -1e30f; row_sum[i] = 0.f; }

// Initialize O accumulator
for(uint32_t t = 0; t < N_TILES; t++) fill_fragment(fragO[t], AccT(0));

// Iterate over all KV blocks (sparse attention skips invalid blocks)
for(int32_t kv_ptr = kv_start; kv_ptr < kv_end; kv_ptr++) {
    // Compute S = Q * K^T / sqrt(d)
    FragS fragS; fill_fragment(fragS, AccT(0));
    for(uint32_t d = 0; d < D; d += MFMA_K) {
        FragQ fragQ; FragK fragK;
        load_matrix_sync(fragQ, Qh + q_local*D + d, D);
        load_matrix_sync(fragK, Kh + kv_off*D + d, D);
        mma_sync(fragS, fragQ, fragK, fragS);
    }

    // Online softmax update (key Flash Attention step)
    // CK Tile: online_softmax_update<AccDataType>
    for(uint32_t elem = 0; elem < fragS.num_elements; elem++) {
        uint32_t r = elem % MFMA_M;
        float s = static_cast<float>(fragS.x[elem]) * scale;

        float new_max = fmaxf(row_max[r], s);
        float exp_s   = expf(s - new_max);          // numerically stable exp
        float exp_old = expf(row_max[r] - new_max);  // decay factor for old max

        // Update running sum (compensate for difference between old and new max)
        row_sum[r] = row_sum[r] * exp_old + exp_s;
        row_max[r] = new_max;

        fragS.x[elem] = static_cast<AccT>(exp_s);  // used for P×V
    }

    // O += P × V (P is the normalized weight)
    for(uint32_t nt = 0; nt < N_TILES; nt++) {
        FragV fragV;
        load_matrix_sync(fragV, Vh + kv_off*D + nt*MFMA_N, D);
        mma_sync(fragO[nt], fragP_cast, fragV, fragO[nt]);
    }
}

// Final normalization: O /= row_sum (done in registers, no extra memory access)
for(uint32_t nt = 0; nt < N_TILES; nt++) {
    for(uint32_t elem = 0; elem < fragO[nt].num_elements; elem++) {
        uint32_t r = elem % MFMA_M;
        if(row_sum[r] > 0.f) fragO[nt].x[elem] /= row_sum[r];
    }
}
```

**Flash Attention vs Standard Attention Memory Comparison:**

```
Sequence length S=4096, head dim D=128, FP16:

Standard Attention:
  S matrix: S×S = 4096×4096×2 = 32 MB  ← must be fully written to HBM
  Memory traffic: O(S²×D)

Flash Attention:
  S matrix: never written to HBM, computed block-by-block in registers/LDS
  Memory traffic: O(S×D) (linear!)
```

---

## 9. FlatMM: Register-Direct Load of B

**CK Tile equivalent:** `WeightPreshufflePipelineAGmemBGmemCRegV2` / FlatMM

FlatMM is CK Tile's specialized optimization for decode scenarios (small M): matrix B bypasses LDS
and is loaded directly from global memory into each warp's registers, halving LDS usage and increasing occupancy.

```cpp
// =========================================================
// File: rocwmma_flatmm.cpp  ~line 150-210
// =========================================================

// Standard GEMM: both A and B go through LDS
// LDS requirement = (ldHA + ldHB) × MACRO_TILE_K × sizeof(fp16)
//                 = (128 + 128) × 16 × 2 = 8192 bytes (single buffer)

// FlatMM: only A goes through LDS, B each warp loads independently
// LDS requirement = ldHA × MACRO_TILE_K × sizeof(fp16)
//                 = 128 × 16 × 2 = 4096 bytes (single buffer) = halved!

constexpr uint32_t szLdsA = ldHA * MACRO_TILE_K;  // Only A in LDS

// A: cooperatively written to LDS (same as standard GEMM)
GRA grA;
load_matrix_sync(grA, A + rA, lda);
store_matrix_sync(lLo, toLWA(grA), ldsld_a);
synchronize_workgroup();

for(uint32_t ks = MACRO_TILE_K; ks < K; ks += MACRO_TILE_K) {
    // A: read from LDS (shared by all warps)
    LRA lrA;
    load_matrix_sync(lrA, lLo + lRA, ldsld_a);

    // B: each warp independently loads its own WARP_TILE_N columns from global memory
    // CK Tile: B is pre-shuffled into MFMA register-friendly layout (preshuffle)
    MB mfragB;
    auto rB = rB_base + GetDataLayout_t<MB>::fromMatrixCoord(
                  make_coord2d(ks - MACRO_TILE_K, 0u), ldb);
    load_matrix_sync(mfragB, B + rB, ldb);  // Direct global → registers (bypasses LDS)

    // Prefetch next A step (parallel with current MMA)
    load_matrix_sync(grA, A + rA, lda);
    rA += kA;

    mma_sync(fAcc, toMA(lrA), mfragB, fAcc);  // MMA

    // Only update A in LDS (B not needed)
    store_matrix_sync(lHi, toLWA(grA), ldsld_a);
    synchronize_workgroup();
    // Swap Lo/Hi
    auto* t = lLo; lLo = lHi; lHi = t;
}
```

**FlatMM Use Cases:**

```
Decode (small M):  M=128, N=4096, K=4096
  B matrix size: 4096×4096×2 = 32 MB (weights, in HBM)
  B loaded per warp: 64×4096×2 = 512 KB

B Preshuffle (offline rearrangement):
  Rearrange B offline into MFMA register-friendly layout
  → Zero shuffle overhead at runtime
  → Bypasses LDS store/load (saves ~16 KB LDS)

LDS Usage Comparison:
  Standard GEMM: 16 KB (double buffer) → N blocks can run concurrently per CU
  FlatMM:         8 KB (double buffer) → ~2× occupancy improvement
```

---

## 10. Stream-K Load Balancing

**CK Tile equivalent:** `GemmStreamKPartitioner`

Stream-K divides the K dimension of GEMM into "SK units" (each unit = MACRO_TILE_K steps),
distributed evenly across all SMs, eliminating tail-wave load imbalance.

```cpp
// =========================================================
// File: rocwmma_streamk_gemm.cpp  ~line 95-175
// =========================================================

// Total SK units = number of output tiles × K steps
uint32_t total_tiles = tiles_m * tiles_n;         // = (M/MT_M) × (N/MT_N)
uint32_t k_units     = K / MACRO_TILE_K;          // = K steps per output tile
uint32_t total_sk    = total_tiles * k_units;     // total K slices

// SK units per CTA (SM), assigned via round-robin scheduling
uint32_t units_per_cta = (total_sk + num_cus - 1) / num_cus;

// In kernel: each block knows its range via blockIdx.x
uint32_t my_sk_start = blockIdx.x * units_per_cta;
uint32_t sk_end      = min(my_sk_start + units_per_cta, total_sk);

// Iterate over SK units assigned to this block
for(uint32_t sk = my_sk_start; sk < sk_end; sk++) {
    uint32_t tile_id = sk / k_units;  // which output tile this belongs to
    uint32_t k_idx   = sk % k_units;  // which K slice within that tile

    // When crossing output tile boundaries, flush accumulated results for previous tile
    if(tile_started && tile_id != cur_tile_id) {
        // Use atomicAdd to write into FP32 workspace (multiple blocks contribute to one tile)
        for(uint32_t i = 0; i < fragAcc.num_elements; i++)
            atomicAdd(&workspace[cur_tile_id * WARP_TILE_M * WARP_TILE_N + i],
                      static_cast<WorkT>(fragAcc.x[i]));
        // Atomically increment completion counter
        atomicAdd(&tile_done[cur_tile_id], 1u);
        fill_fragment(fAcc, ComputeT(0));  // reset accumulator
    }
    // Compute single SK unit's MMA (K slice of width MACRO_TILE_K)
    // ...（same as single-step computation in standard GEMM）
}

// Stage 2: after all SK blocks complete, reduce workspace → final output
// hipDeviceSynchronize() ensures Stage 1 is done
hipLaunchKernelGGL(streamk_reduce_kernel, ...);
```

**Stream-K's Solution to "Tail Wave Bubbles":**

```
Standard 2D tile assignment (M=512, N=512, MT=128):
  Total tiles = 4×4 = 16
  MI355X has 256 CUs
  Wave 0: CUs 0-15 each process 1 tile (100% utilization)
  → No tail-wave waste, but large K means each tile processes serially

Stream-K (K=4096, MT_K=16):
  Total SK units = 16 × 256 = 4096
  Each CU processes 4096/256 = 16 units
  → All 256 CUs work continuously, no idle time
  → Especially effective for large K (decode scenarios)
```

---

## 11. Complete Performance Data

Measured on **AMD Instinct MI355X (gfx950, CDNA 3.5)** unless noted otherwise, and on **AMD Radeon AI PRO R9700 (gfx1201, RDNA 4)** (benchmark **2026-04-07**, logs `benchmark_results_r9700_gfx1201.txt` / `.csv`). In subsections **11.1–11.6**, **MI355X** tables appear first, then **R9700**. Section **11.7** compares CK Tile vs rocWMMA on MI355X only. **11.8** lists build/benchmark commands for both GPUs.

**HGEMM on RDNA 4:** On gfx1201, **Pipeline V1 is often faster than V2 double-buffer** for square sizes in `rocwmma_perf_gemm` (unlike MI355X); tune per shape.

### 11.1 GEMM

**MI355X (gfx950)**

#### Standard HGEMM (FP16 input, FP32 accumulation, FP16 output)

`rocwmma_perf_gemm.cpp`

| M | N | K | V1 Sequential | V2 Double-Buffer | Speedup |
|---|---|---|---|---|---|
| 3840 | 4096 | 4096 | 365 TF/s | **492 TF/s** | 1.35× |
| 4096 | 4096 | 4096 | 393 TF/s | **536 TF/s** | 1.36× |
| 8192 | 8192 | 8192 | 464 TF/s | **591 TF/s** | 1.27× |
| 1024 | 4096 | 8192 | 197 TF/s | **274 TF/s** | 1.39× |
| 4096 | 1024 | 8192 | 197 TF/s | **281 TF/s** | 1.43× |

#### Split-K GEMM (FP16)

`rocwmma_gemm_splitk.cpp`

| M | N | K | split_k | TFlops/s |
|---|---|---|---|---|
| 512  | 512  | 16384 | 4 | 18.8 |
| 1024 | 1024 | 16384 | 8 | 26.9 |
| 2048 | 2048 | 8192  | 4 | 27.1 |
| 4096 | 4096 | 4096  | 2 | 26.7 |

#### FP8 Quantized GEMM

`rocwmma_gemm_quantized.cpp`

| M | N | K | TFlops/s | vs FP16 Peak |
|---|---|---|---|---|
| 3840 | 4096 | 4096 | **109 TF/s** | — |
| 4096 | 4096 | 4096 | **112 TF/s** | — |
| 8192 | 8192 | 8192 | **147 TF/s** | — |

#### MX Format GEMM (E4M3 FP8, E8M0 block scale)

`rocwmma_mx_gemm.cpp`

| M | N | K | MX_BLOCK | TFlops/s |
|---|---|---|---|---|
| 3840 | 4096 | 4096 | 32 | 56.1 |
| 4096 | 4096 | 4096 | 32 | 58.3 |
| 8192 | 8192 | 8192 | 32 | 69.7 |

#### GEMM + Fused Epilogue (bias + scale + ReLU)

`rocwmma_gemm_multi_d.cpp`

| M | N | K | ReLU | TFlops/s |
|---|---|---|---|---|
| 3840 | 4096 | 4096 | No  | 155 |
| 4096 | 4096 | 4096 | Yes | 164 |
| 8192 | 8192 | 8192 | Yes | 189 |

#### Batched GEMM

`rocwmma_batched_gemm.cpp`

| batch | M | N | K | TFlops/s (per batch) | Total Effective TFlops/s |
|---|---|---|---|---|---|
| 4 | 1024 | 1024 | 1024 | 61.4 | 245.6 |
| 8 | 512  | 512  | 2048 | 16.8 | 134.8 |
| 16 | 256 | 256  | 4096 | 4.4  | 70.4  |

#### Grouped GEMM

`rocwmma_grouped_gemm.cpp`

| G | Group Sizes | Total TFlops/s |
|---|---|---|
| 4 | 256/512/128/768 × 4096 × 4096 (mixed) | **109** |
| 4 | 1024 × 4096 × 4096 (uniform) | **129** |
| 3 | 512×1024/2048/4096 × 2048 (varying N) | 66 |

#### Stream-K GEMM

`rocwmma_streamk_gemm.cpp` (256 CUs running continuously)

| M | N | K | TFlops/s |
|---|---|---|---|
| 512  | 512  | 4096 | 8.1 |
| 1024 | 1024 | 4096 | 27.8 |
| 2048 | 2048 | 4096 | 66.0 |
| 4096 | 4096 | 4096 | 64.4 |

#### FlatMM (A via LDS, B directly to registers bypassing LDS)

`rocwmma_flatmm.cpp`

| M | N | K | LDS Usage | TFlops/s |
|---|---|---|---|---|
| 128  | 4096 | 4096 | 8 KB (A only) | 16.2 |
| 256  | 4096 | 4096 | 8 KB | 32.4 |
| 512  | 4096 | 4096 | 8 KB | 64.5 |
| 1024 | 4096 | 4096 | 8 KB | **91.7** |
| 4096 | 4096 | 4096 | 8 KB | **127** |

#### Batched Tensor Contraction

`rocwmma_batched_contraction.cpp`

| G | M | N | K | TFlops/s (per group) |
|---|---|---|---|---|
| 4  | 1024 | 1024 | 1024 | 8.1 |
| 8  | 512  | 512  | 2048 | 3.8 |
| 16 | 256  | 256  | 4096 | 1.2 |

**Radeon AI PRO R9700 (gfx1201)**

#### Standard HGEMM (FP16 in, FP32 acc, FP16 out) — `rocwmma_perf_gemm.cpp`

| M | N | K | V1 Sequential | V2 Double-Buffer | Comment |
|---|---|---|---|---|---|
| 3840 | 4096 | 4096 | **130** TF/s | 118 TF/s | V1 faster |
| 4096 | 4096 | 4096 | **132** TF/s | 120 TF/s | V1 faster |
| 8192 | 8192 | 8192 | **133** TF/s | 123 TF/s | V1 faster |
| 1024 | 4096 | 8192 | **115** TF/s | 114 TF/s | ~tie |
| 4096 | 1024 | 8192 | 105 TF/s | **118** TF/s | V2 faster |

#### Split-K GEMM — `rocwmma_gemm_splitk.cpp`

| M | N | K | split_k | TFlops/s |
|---|---|---|---|---|
| 512 | 512 | 16384 | 4 | 1.56 |
| 1024 | 1024 | 16384 | 8 | 0.80 |
| 2048 | 2048 | 8192 | 4 | 0.80 |
| 4096 | 4096 | 4096 | 2 | 0.80 |

#### FP8 quantized GEMM — `rocwmma_gemm_quantized.cpp`

| M | N | K | TFlops/s |
|---|---|---|---|
| 3840 | 4096 | 4096 | 2.39 |
| 4096 | 4096 | 4096 | 2.39 |
| 8192 | 8192 | 8192 | 5.13 |

#### MX-format GEMM — `rocwmma_mx_gemm.cpp` (runs on gfx1201; binary may print “gfx950 native”)

| M | N | K | MX_BLOCK | TFlops/s |
|---|---|---|---|---|
| 3840 | 4096 | 4096 | 32 | 3.09 |
| 4096 | 4096 | 4096 | 32 | 3.09 |
| 8192 | 8192 | 8192 | 32 | 5.73 |

#### GEMM + fused epilogue — `rocwmma_gemm_multi_d.cpp`

| M | N | K | ReLU | TFlops/s |
|---|---|---|---|---|
| 3840 | 4096 | 4096 | No | 14.6 |
| 4096 | 4096 | 4096 | Yes | 13.8 |
| 8192 | 8192 | 8192 | Yes | 16.8 |

#### Batched GEMM — `rocwmma_batched_gemm.cpp`

| batch | M | N | K | TFlops/s (per batch) | Total effective TFlops/s |
|---|---|---|---|---|---|
| 4 | 1024 | 1024 | 1024 | 15.4 | 61.8 |
| 8 | 512 | 512 | 2048 | 9.15 | 73.2 |
| 16 | 256 | 256 | 4096 | 3.85 | 61.6 |

#### Grouped GEMM — `rocwmma_grouped_gemm.cpp`

| G | Group sizes | Total TFlops/s |
|---|---|---|
| 4 | 256/512/128/768 × 4096 × 4096 (mixed) | 2.65 |
| 4 | 1024 × 4096 × 4096 (uniform) | 2.66 |
| 3 | 512×1024/2048/4096 × 2048 | 1.38 |

#### Stream-K GEMM — `rocwmma_streamk_gemm.cpp` (num_cus=32 in sample)

| M | N | K | TFlops/s |
|---|---|---|---|
| 512 | 512 | 4096 | 2.51 |
| 1024 | 1024 | 4096 | 4.90 |
| 2048 | 2048 | 4096 | 5.63 |
| 4096 | 4096 | 4096 | 5.52 |

#### FlatMM — `rocwmma_flatmm.cpp`

| M | N | K | LDS | TFlops/s |
|---|---|---|---|---|
| 128 | 4096 | 4096 | 8 KB (A only) | 3.10 |
| 256 | 4096 | 4096 | 8 KB | 2.40 |
| 512 | 4096 | 4096 | 8 KB | 2.46 |
| 1024 | 4096 | 4096 | 8 KB | 2.58 |
| 4096 | 4096 | 4096 | 8 KB | 2.65 |

#### Batched tensor contraction — `rocwmma_batched_contraction.cpp`

| G | M | N | K | TFlops/s (per group) |
|---|---|---|---|---|
| 4 | 1024 | 1024 | 1024 | 0.22 |
| 8 | 512 | 512 | 2048 | 0.21 |
| 16 | 256 | 256 | 4096 | 0.20 |

---

### 11.2 Attention

**MI355X (gfx950)**

#### Flash Attention Forward

`rocwmma_fmha_fwd.cpp`

| B | H | Sq | Sk | D | Causal | TFlops/s |
|---|---|---|---|---|---|---|
| 4 | 32 | 2048 | 2048 | 128 | No  | **75.1** |
| 4 | 32 | 2048 | 2048 | 128 | Yes | 74.8 |
| 1 | 32 | 4096 | 4096 | 128 | Yes | **88.4** |

#### Sparse Attention (Jenga LUT + Flash Attention)

`rocwmma_sparse_attn.cpp`

| B | H | Sq/Sk | D | Density | Active Tile Pairs | TFlops/s |
|---|---|---|---|---|---|---|
| 4 | 32 | 2048 | 128 | 79.4% | 813/1024   | 59.8 |
| 4 | 32 | 4096 | 128 | 27.5% | 1125/4096  | 68.9 |
| 4 | 32 | 8192 | 128 | 14.3% | 2341/16384 | **71.6** |

**Radeon AI PRO R9700 (gfx1201)**

#### Flash Attention forward — `rocwmma_fmha_fwd.cpp`

| B | H | Sq | Sk | D | Causal | TFlops/s |
|---|---|---|---|---|---|---|
| 4 | 32 | 2048 | 2048 | 128 | No | 19.3 |
| 4 | 32 | 2048 | 2048 | 128 | Yes | **36.4** |
| 1 | 32 | 4096 | 4096 | 128 | Yes | **36.8** |

#### Sparse attention — `rocwmma_sparse_attn.cpp`

| B | H | Sq/Sk | D | Density | Active tile pairs | TFlops/s |
|---|---|---|---|---|---|---|
| 4 | 32 | 2048 | 128 | 79.4% | 813/1024 | 6.28 |
| 4 | 32 | 4096 | 128 | 27.5% | 1125/4096 | 5.42 |
| 4 | 32 | 8192 | 128 | 14.3% | 2341/16384 | 5.62 |

---

### 11.3 Normalization

**MI355X (gfx950)**

#### LayerNorm 2D (Welford online variance + vectorized IO)

`rocwmma_layernorm2d.cpp`

| M | N | Time | Bandwidth |
|---|---|---|---|
| 3328 | 1024 | 0.017 ms | 801 GB/s |
| 3328 | 2048 | 0.015 ms | 1772 GB/s |
| 3328 | 4096 | 0.021 ms | **2613 GB/s** |
| 3328 | 8192 | 0.031 ms | **3538 GB/s** |

#### RMSNorm 2D

`rocwmma_rmsnorm2d.cpp`

| M | N | Operation | Bandwidth |
|---|---|---|---|
| 3328 | 4096 | RMSNorm     | **4787 GB/s** |
| 3328 | 4096 | Add+RMSNorm | **5303 GB/s** |
| 3328 | 8192 | RMSNorm     | 4737 GB/s |

#### Smooth Quantization (FP16 → INT8)

`rocwmma_smoothquant.cpp`

| M | N | Bandwidth |
|---|---|---|
| 3328 | 1024 | 1213 GB/s |
| 3328 | 2048 | 2999 GB/s |
| 3328 | 4096 | **3929 GB/s** |
| 3328 | 8192 | 4069 GB/s |

**Radeon AI PRO R9700 (gfx1201)**

#### LayerNorm 2D — `rocwmma_layernorm2d.cpp`

| M | N | Bandwidth |
|---|---|---|
| 3328 | 1024 | 238 GB/s |
| 3328 | 2048 | 480 GB/s |
| 3328 | 4096 | **548** GB/s |
| 3328 | 8192 | 426 GB/s |

#### RMSNorm 2D — `rocwmma_rmsnorm2d.cpp`

| M | N | Operation | Bandwidth |
|---|---|---|---|
| 3328 | 4096 | RMSNorm | **1368** GB/s |
| 3328 | 4096 | Add+RMSNorm | 487 GB/s |
| 3328 | 8192 | RMSNorm | 477 GB/s |

#### SmoothQuant — `rocwmma_smoothquant.cpp`

| M | N | Bandwidth |
|---|---|---|
| 3328 | 1024 | 328 GB/s |
| 3328 | 2048 | 1010 GB/s |
| 3328 | 4096 | **1249** GB/s |
| 3328 | 8192 | 509–551 GB/s |

---

### 11.4 MoE

**MI355X (gfx950)**

#### TopK Softmax (warp-per-token)

`rocwmma_topk_softmax.cpp`

| tokens | experts | topk | Bandwidth |
|---|---|---|---|
| 3328 | 8  | 2 | 25.4 GB/s |
| 3328 | 16 | 2 | 40.2 GB/s |
| 3328 | 32 | 5 | 56.1 GB/s |
| 3328 | 64 | 5 | 89.7 GB/s |

#### MoE Token Sorting (counting sort)

`rocwmma_moe_sorting.cpp`

| tokens | experts | topk | Bandwidth |
|---|---|---|---|
| 3328 | 8   | 4 | 3.26 GB/s |
| 3328 | 32  | 4 | 4.94 GB/s |
| 3328 | 128 | 8 | 7.65 GB/s |

#### MoE Smooth Quantization (per-expert scale)

`rocwmma_moe_smoothquant.cpp`

| tokens | hidden | experts | topk | Bandwidth |
|---|---|---|---|---|
| 3328 | 4096 | 8  | 2 | **2800 GB/s** |
| 3328 | 4096 | 32 | 5 | 2259 GB/s |
| 3328 | 8192 | 64 | 5 | 1575 GB/s |

#### Fully Fused MoE (Sort + Gate/Up GEMM + SiLU + Down GEMM)

`rocwmma_fused_moe.cpp`

| tokens | hidden | inter | experts | topk | TFlops/s |
|---|---|---|---|---|---|
| 3328 | 4096 | 14336 | 8 | 2 | **110** |

**Radeon AI PRO R9700 (gfx1201)**

#### TopK Softmax — `rocwmma_topk_softmax.cpp`

| tokens | experts | topk | Bandwidth |
|---|---|---|---|
| 3328 | 8 | 2 | 11.6 GB/s |
| 3328 | 16 | 2 | 17.3 GB/s |
| 3328 | 32 | 5 | 24.2 GB/s |
| 3328 | 64 | 5 | 40.1 GB/s |

#### MoE token sorting — `rocwmma_moe_sorting.cpp`

| tokens | experts | topk | Bandwidth |
|---|---|---|---|
| 3328 | 8 | 4 | 7.67 GB/s |
| 3328 | 32 | 4 | 7.50 GB/s |
| 3328 | 64 | 5 | 8.64 GB/s |
| 3328 | 128 | 8 | 12.4 GB/s |

#### MoE SmoothQuant — `rocwmma_moe_smoothquant.cpp`

| tokens | hidden | experts | topk | Bandwidth |
|---|---|---|---|---|
| 3328 | 4096 | 8 | 2 | **940** GB/s |
| 3328 | 4096 | 32 | 5 | 430 GB/s |
| 3328 | 8192 | 64 | 5 | 483 GB/s |

#### Fused MoE — `rocwmma_fused_moe.cpp`

| tokens | hidden | inter | experts | topk | TFlops/s |
|---|---|---|---|---|---|
| 3328 | 4096 | 14336 | 8 | 2 | 10.9 |
| 3328 | 4096 | 14336 | 32 | 5 | 9.56 |
| 3328 | 7168 | 2048 | 64 | 6 | 17.3 |

---

### 11.5 Reduction & Elementwise

**MI355X (gfx950)**

#### Row Reduction (Sum/Max)

`rocwmma_reduce.cpp`

| M | N | ReduceSum | ReduceMax |
|---|---|---|---|
| 3328 | 1024 | 2046 GB/s | 1938 GB/s |
| 3328 | 2048 | 3903 GB/s | 3710 GB/s |
| 3328 | 4096 | **6802 GB/s** | **6412 GB/s** |
| 3328 | 8192 | 5741 GB/s | 5598 GB/s |

#### Elementwise Operations

`rocwmma_elementwise.cpp`

| M | N | Operation | Bandwidth |
|---|---|---|---|
| 3840 | 4096 | Add2D       | 2080 GB/s |
| 3840 | 4096 | Square2D    | 1622 GB/s |
| 3840 | 4096 | Transpose2D | **2588 GB/s** |

**Radeon AI PRO R9700 (gfx1201)**

#### Row reduction — `rocwmma_reduce.cpp`

| M | N | ReduceSum | ReduceMax |
|---|---|---|---|
| 3328 | 1024 | 660 GB/s | 646 GB/s |
| 3328 | 2048 | 1333 GB/s | 1292 GB/s |
| 3328 | 4096 | **1747** GB/s | 1715 GB/s |
| 3328 | 8192 | 1987 GB/s | 1987 GB/s |

#### Elementwise — `rocwmma_elementwise.cpp` (M=3840, N=4096)

| Operation | Bandwidth |
|---|---|
| Add2D | 375 GB/s |
| Square2D | 1367 GB/s |
| Transpose2D | 260 GB/s |

---

### 11.6 Convolution & Data Layout

**MI355X (gfx950)**

#### Grouped Convolution Forward (Im2col + rocWMMA GEMM)

`rocwmma_grouped_conv_fwd.cpp`

| N | H×W | G | Cin | Cout | K | TFlops/s |
|---|---|---|---|---|---|---|
| 8 | 56×56 | 1 | 64  | 64  | 3×3 | 8.60 |
| 8 | 28×28 | 1 | 128 | 128 | 3×3 | 8.82 |
| 8 | 14×14 | 1 | 256 | 256 | 3×3 | 6.89 |
| 8 | 7×7   | 1 | 512 | 512 | 3×3 | 3.98 |

#### Tensor Permute (NCHW ↔ NHWC)

`rocwmma_permute.cpp`

| Operation | Shape | Bandwidth |
|---|---|---|
| Transpose2D | 8192×8192 | **2732 GB/s** |
| NCHW→NHWC | N=8, C=256, H=W=56 | 2047 GB/s |
| NCHW→NHWC | N=32, C=128, H=W=64 | **2531 GB/s** |

#### 3D Pooling

`rocwmma_pooling.cpp`

| N | D×H×W | C | K | MaxPool | AvgPool |
|---|---|---|---|---|---|
| 2 | 16×28×28 | 256 | 2×2×2 | **2123 GB/s** | 1999 GB/s |
| 2 | 8×56×56  | 128 | 3×3×3 | 196 GB/s | 194 GB/s |

**Radeon AI PRO R9700 (gfx1201)**

#### Grouped convolution forward — `rocwmma_grouped_conv_fwd.cpp`

| N | H×W | G | Cin | Cout | K | TFlops/s |
|---|---|---|---|---|---|---|
| 8 | 56×56 | 1 | 64 | 64 | 3×3 | 0.18 |
| 8 | 28×28 | 1 | 128 | 128 | 3×3 | 0.83 |
| 8 | 14×14 | 1 | 256 | 256 | 3×3 | 2.11 |
| 8 | 7×7 | 1 | 512 | 512 | 3×3 | 1.47 |

#### Permute — `rocwmma_permute.cpp`

| Operation | Shape | Bandwidth |
|---|---|---|
| Transpose2D | 3840×4096 | 515 GB/s |
| Transpose2D | 4096×4096 | 518 GB/s |
| Transpose2D | 8192×8192 | 171 GB/s |
| NCHW→NHWC | N=32, C=128, H=W=64 | **534** GB/s |

#### 3D pooling — `rocwmma_pooling.cpp`

| N | D×H×W | C | K | MaxPool | AvgPool |
|---|---|---|---|---|---|
| 2 | 16×28×28 | 256 | 2×2×2 | **552** GB/s | 528 GB/s |
| 2 | 8×56×56 | 128 | 3×3×3 | 48.4 GB/s | 47.1 GB/s |

---

### 11.7 CK Tile vs rocWMMA Comparison (HGEMM, M=N=K=4096)

| Implementation | TFlops/s | Notes |
|---|---|---|
| CK Tile Basic (V1 pipeline) | **640** | 256×256 block tile, gfx9-specific compiler optimizations |
| rocWMMA V1 (sequential)     | 393 | 128×128 block tile, standard pipeline |
| rocWMMA V2 (double-buffer)  | **536** | 128×128 block tile, double-buffer LDS |
| Speedup (V2 vs V1)          | 1.36× | double-buffer effect |
| rocWMMA V2 vs CK Tile       | 83.7% | gap mainly from block tile size and compiler optimizations |

> **Note:** CK Tile uses a 256×256 larger block tile (requiring 64KB LDS) and applies
> low-level compiler flags like `-mllvm -enable-noalias-to-md-conversion=0`.
> rocWMMA uses the standard HIP compilation path (128×128 block tile, 16KB LDS).

### 11.8 Reproducing benchmarks (MI355X vs R9700)

```bash
cd /home/optimized_rocwmma_samples   # or your clone path

# MI355X / CDNA (example target)
./build.sh -g gfx950
./benchmark.sh -b build --csv -o benchmark_results_mi355x.txt --timeout 300

# Radeon AI PRO R9700 / RDNA 4 (gfx1201)
./build.sh -g gfx1201
./benchmark.sh -b build --csv -o benchmark_results_r9700_gfx1201.txt --timeout 300
```

Requires CMake ≥3.16 and a full ROCm install (`HIPConfig.cmake` under `$ROCM_PATH/lib/cmake/hip`). This project prepends `/opt/rocm` (or `$ROCM_PATH`) to `CMAKE_PREFIX_PATH` so `find_package(HIP)` works on typical setups.

---

## Appendix: File Index

| File | CK Tile Equivalent | Core Optimization |
|---|---|---|
| `rocwmma_perf_gemm.cpp` | 03_gemm | Double-buffer LDS, cooperative load, Warp Tile |
| `rocwmma_gemm_splitk.cpp` | 03_gemm splitk | FP32 workspace two-stage split-K |
| `rocwmma_gemm_quantized.cpp` | 38_block_scale_gemm | FP8 tensor-scale epilogue |
| `rocwmma_batched_gemm.cpp` | 16_batched_gemm | blockIdx.z batch dimension |
| `rocwmma_gemm_multi_d.cpp` | 19_gemm_multi_d | Fused bias/scale/ReLU epilogue |
| `rocwmma_batched_contraction.cpp` | 41_batched_contraction | Contraction → batched GEMM |
| `rocwmma_grouped_gemm.cpp` | 17_grouped_gemm | Flat grid + binary search dispatch |
| `rocwmma_streamk_gemm.cpp` | 40_streamk_gemm | Stream-K + atomic workspace reduction |
| `rocwmma_flatmm.cpp` | 18_flatmm | B bypasses LDS, direct register load |
| `rocwmma_mx_gemm.cpp` | 42_mx_gemm | MX E8M0 block-scale FP8 |
| `rocwmma_fmha_fwd.cpp` | 01_fmha | Flash Attention online softmax |
| `rocwmma_sparse_attn.cpp` | 50_sparse_attn | Jenga LUT sparse block skipping |
| `rocwmma_layernorm2d.cpp` | 02_layernorm2d | Welford online variance + vectorized IO |
| `rocwmma_rmsnorm2d.cpp` | 10/11_rmsnorm2d | Single-pass RMS + fused Add+RMS |
| `rocwmma_smoothquant.cpp` | 12_smoothquant | Two-pass quantization: abs-max + quantize |
| `rocwmma_moe_smoothquant.cpp` | 14_moe_smoothquant | Per-expert smooth scale |
| `rocwmma_fused_moe.cpp` | 15_fused_moe | Sort+GateUp+SiLU+Down fully fused |
| `rocwmma_reduce.cpp` | 05_reduce | Butterfly warp reduce + LDS block reduce |
| `rocwmma_elementwise.cpp` | 21_elementwise | float4 vectorization, LDS padding transpose |
| `rocwmma_topk_softmax.cpp` | 09_topk_softmax | Warp-per-row softmax + top-K |
| `rocwmma_moe_sorting.cpp` | 13_moe_sorting | Three-phase counting sort |
| `rocwmma_img2col_gemm.cpp` | 04_img2col | Virtual im2col + rocWMMA GEMM |
| `rocwmma_grouped_conv_fwd.cpp` | 20_grouped_convolution | Grouped im2col + per-group GEMM |
| `rocwmma_pooling.cpp` | 36_pooling | NDHWC vectorized window reduction |
| `rocwmma_permute.cpp` | 06/35_permute | LDS padding 2D tile transpose |
| `rocwmma_tile_distr_reg_map.cpp` | 51_tile_distr | Host diagnostic: fragment shape printer |

---

## 12. CK Tile vs rocWMMA Full Comparison (MI355X / gfx950)

This section provides measured data from running CK Tile original and rocWMMA ports on the same MI355X with identical warmup/repeat parameters.

**Test Conditions:**

| Item | CK Tile | rocWMMA |
|---|---|---|
| Compiler | AMD Clang 22.0 | AMD Clang 22.0 |
| ROCm | 7.2.0 | 7.2.0 |
| GPU | MI355X (gfx950) | MI355X (gfx950) |
| Warmup | 5 | 5 |
| Repeat | 20 | 20 |
| CK Tile Special Flags | `-mllvm -enable-noalias-to-md-conversion=0` | None |
| Block Tile (GEMM) | 256×256 (CK Tile default) | 128×128 (rocWMMA limitation) |

> **Note:** CK Tile GEMM uses a 256×256 Block Tile requiring 64 KB LDS,
> and relies on low-level LLVM passes and `noalias` conversion disabling for optimal ILP.
> rocWMMA uses the standard HIP compilation path, Block Tile limited to 128×128 (16 KB LDS).
> This is the primary source of the GEMM performance gap between the two.

---

### 12.1 GEMM Comparison

#### HGEMM (FP16, Col-major A, Row-major B)

`tile_example_gemm_basic` (V1 pipeline) vs `rocwmma_perf_gemm` (V1/V2)

| M | N | K | CK Tile V1 | rocWMMA V1 | rocWMMA V2 | rocWMMA/CK Ratio (V2) |
|---|---|---|---|---|---|---|
| 3840 | 4096 | 4096 | **634 TF/s** | 365 TF/s | 492 TF/s | 77.6% |
| 4096 | 4096 | 4096 | **647 TF/s** | 391 TF/s | 536 TF/s | 82.8% |
| 8192 | 8192 | 8192 | **773 TF/s** | 464 TF/s | 592 TF/s | 76.6% |
| 1024 | 4096 | 8192 | 219 TF/s | 197 TF/s | **275 TF/s** | **125.6%** |
| 4096 | 1024 | 8192 | 218 TF/s | 197 TF/s | **281 TF/s** | **128.9%** |

> rocWMMA V2 outperforms CK Tile V1 in thin-tile scenarios (small M/N), because double-buffering
> more effectively hides memory latency (CK Tile's advantage with large block tiles shows when K is large).

#### HGEMM Universal (Async Pipeline)

`tile_example_gemm_universal` (async) vs `rocwmma_perf_gemm` (V2 double-buffer)

| M | N | K | CK Tile Async | rocWMMA V2 | rocWMMA/CK Ratio |
|---|---|---|---|---|---|
| 4096 | 4096 | 4096 | **757 TF/s** | 536 TF/s | 70.8% |
| 8192 | 8192 | 8192 | **931 TF/s** | 592 TF/s | 63.6% |

> CK Tile Universal uses asynchronous memory copy instructions (CDNA equivalent of `cp.async`),
> further decoupling memory loads from computation — hardware capability not exposed by rocWMMA.

#### Split-K GEMM (FP16)

`tile_example_gemm_splitk_two_stage` vs `rocwmma_gemm_splitk`

| M | N | K | split_k | CK Tile | rocWMMA | Ratio |
|---|---|---|---|---|---|---|
| 512  | 512  | 16384 | 4 | **173 TF/s** | 18.8 TF/s | 10.9% |
| 1024 | 1024 | 16384 | 8 | **174 TF/s** | 26.9 TF/s | 15.5% |
| 4096 | 4096 | 4096  | 2 | **155 TF/s** | 26.7 TF/s | 17.2% |

> CK Tile split-K uses persistent kernels and CShuffleEpilogue.
> rocWMMA uses per-CTA dispatch + workspace reduce, with higher launch overhead.

#### Batched GEMM (FP16, default sizes)

`tile_example_batched_gemm` vs `rocwmma_batched_gemm`

| Configuration | CK Tile | rocWMMA | Ratio |
|---|---|---|---|
| Default (M≈1024,N≈1024) | **341 TF/s** | 61.4 TF/s (per batch) | 18% per-batch |

> Note: CK Tile batched GEMM default sizes don't exactly match rocWMMA.
> Effective comparison requires aligned sizes.

#### GEMM + MultiD Epilogue (FP16)

`tile_example_gemm_multi_d_fp16` vs `rocwmma_gemm_multi_d`

| Configuration | CK Tile | rocWMMA |
|---|---|---|
| Default (M=3840,N=4096) | **924 TF/s** | 155 TF/s |

> CK Tile uses `CShuffleEpilogue` + 256×256 block tile for extreme efficiency.
> rocWMMA simplifies to register-level per-element operations, lower launch overhead but worse compute utilization.

---

### 12.2 Normalization Comparison

#### LayerNorm 2D (FP16 in/out)

`tile_example_layernorm2d_fwd` vs `rocwmma_layernorm2d`

| M | N | CK Tile | rocWMMA | Ratio |
|---|---|---|---|---|
| 3328 | 1024 | **2036 GB/s** | 801 GB/s | 39.3% |
| 3328 | 2048 | **2340 GB/s** | 1772 GB/s | 75.7% |
| 3328 | 4096 | **3557 GB/s** | 2613 GB/s | 73.5% |
| 3328 | 8192 | **3644 GB/s** | 3538 GB/s | 97.1% |

> At N=8192, rocWMMA approaches CK Tile (97.1%), showing vectorization width is the key bottleneck.
> At N=1024, larger gap (39%) because CK Tile uses more aggressive thread organization.

#### RMSNorm 2D (FP16 in/out)

`tile_rmsnorm2d_fwd` vs `rocwmma_rmsnorm2d`

| M | N | CK Tile | rocWMMA | Ratio |
|---|---|---|---|---|
| 3328 | 1024 | **3469 GB/s** | 1830 GB/s | 52.8% |
| 3328 | 2048 | **4463 GB/s** | 4110 GB/s | 92.1% |
| 3328 | 4096 | **5576 GB/s** | 4787 GB/s | 85.9% |
| 3328 | 8192 | **6443 GB/s** | 4737 GB/s | 73.5% |

> CK Tile RMSNorm uses codegen to produce optimal thread organization for gfx950.
> rocWMMA uses generic Wave64 butterfly reduction.

#### Smooth Quantization (FP16 → INT8)

`tile_smoothquant` vs `rocwmma_smoothquant`

| M | N | CK Tile | rocWMMA | Ratio |
|---|---|---|---|---|
| 3328 | 1024 | **2066 GB/s** | 1213 GB/s | 58.7% |
| 3328 | 2048 | **2715 GB/s** | 2999 GB/s | **110.4%** |
| 3328 | 4096 | **3288 GB/s** | 3929 GB/s | **119.5%** |
| 3328 | 8192 | 3209 GB/s | **4069 GB/s** | **126.8%** |

> rocWMMA outperforms CK Tile SmoothQuant at N≥2048!
> Reason: rocWMMA uses `int64_t`-aligned writes (8×INT8 per write),
> while CK Tile's INT8 output vectorization has limitations.

---

### 12.3 MoE Comparison

#### TopK Softmax

`tile_example_topk_softmax` vs `rocwmma_topk_softmax`

| tokens | experts | topk | CK Tile (ms) | rocWMMA (ms) | Winner |
|---|---|---|---|---|---|
| 3328 | 8  | 2 | **0.00265 ms** | 0.00419 ms | CK 1.58× faster |
| 3328 | 16 | 2 | **0.00294 ms** | 0.00397 ms | CK 1.35× faster |
| 3328 | 32 | 5 | **0.00399 ms** | 0.00617 ms | CK 1.55× faster |
| 3328 | 64 | 5 | 0.00544 ms | **0.00623 ms** | Close (CK 1.15× faster) |

#### MoE Token Sorting

`tile_example_moe_sorting` vs `rocwmma_moe_sorting`

| tokens | experts | topk | CK Tile (ms) | rocWMMA (ms) | Ratio |
|---|---|---|---|---|---|
| 3328 | 8   | 4 | **0.0132 ms** | 0.0493 ms | CK 3.7× faster |
| 3328 | 32  | 4 | **0.0125 ms** | 0.0332 ms | CK 2.7× faster |
| 3328 | 64  | 4 | **0.0122 ms** | 0.0397 ms | CK 3.3× faster |
| 3328 | 128 | 4 | **0.0121 ms** | 0.0439 ms | CK 3.6× faster |

> CK Tile MoE Sorting uses a specialized `GpuSort` algorithm (radix sort based).
> rocWMMA uses three-phase counting sort (histogram+prefix+scatter), with extra overhead for small expert counts.

---

### 12.4 Reduction & Data Movement Comparison

#### Row-wise Reduce (FP16 → FP32)

`tile_example_reduce` (reduce over N, keep C) vs `rocwmma_reduce`

> **Note:** CK Tile reduce operates over `(N,H,W)` keeping `C`,
> semantically equivalent to rocWMMA's row reduction (keeping columns), but with different dimension layout.

| M (reduced dim) | N (kept dim) | CK Tile | rocWMMA | Ratio |
|---|---|---|---|---|
| 3328 | 1024 | **1113 GB/s** | 2046 GB/s | rocWMMA **1.84×** faster |
| 3328 | 2048 | **1230 GB/s** | 3903 GB/s | rocWMMA **3.17×** faster |
| 3328 | 4096 | **1114 GB/s** | 6802 GB/s | rocWMMA **6.10×** faster |
| 3328 | 8192 | **1166 GB/s** | 5741 GB/s | rocWMMA **4.92×** faster |

> rocWMMA reduce significantly outperforms CK Tile!
> Reason: CK Tile reduce's default configuration uses 4D tensors (N,H,W,C),
> reducing over N×H×W dimensions — not exactly equivalent to rocWMMA's row reduction.
> These numbers are for reference; real workloads require specific validation.

#### 2D Transpose / Permute

`tile_example_permute` vs `rocwmma_permute`

| M | N | CK Tile | rocWMMA | Ratio |
|---|---|---|---|---|
| 3840 | 4096 | 341 GB/s | **2463 GB/s** | rocWMMA **7.2×** faster |
| 4096 | 4096 | 416 GB/s | **2470 GB/s** | rocWMMA **5.9×** faster |
| 8192 | 8192 | 437 GB/s | **2732 GB/s** | rocWMMA **6.3×** faster |

> rocWMMA uses LDS-padded tiled transpose (32×32 tiles).
> CK Tile's default configuration uses small shapes (`2,3,4`), not optimized for large matrices.
> Using CK Tile's `PERMUTE_USE_ALTERNATIVE_IMPL` (matrix-core swizzle) would yield higher performance,
> but requires additional compile flags.

---

### 12.5 Elementwise, Pooling, Batched Contraction Comparison

#### Elementwise 2D Add (FP16, three-tensor C = A + B)

`tile_example_elementwise` vs `rocwmma_elementwise`

| M | N | CK Tile | rocWMMA | Ratio |
|---|---|---|---|---|
| 3840 | 4096 | **3358 GB/s** | 2080 GB/s | CK Tile +61.4% |
| 4096 | 4096 | **3407 GB/s** | — | — |
| 8192 | 8192 | **2670 GB/s** | — | — |

> CK Tile elementwise uses the `GenericPermute` framework for efficient tile dispatch.
> rocWMMA uses `float4` vectorization but loads each row independently, with lower L2 reuse.

#### 3D Pooling (FP16, NDHWC layout)

`tile_example_pool3d` vs `rocwmma_pooling`

| N | D×H×W | C | Window | CK Tile | rocWMMA | Ratio |
|---|---|---|---|---|---|---|
| 2 | 16×28×28 | 256 | 2×2×2 | **497 GB/s** | 2123 GB/s | rocWMMA **4.3×** faster |
| 2 | 8×56×56 | 128 | 3×3×3 | **79 GB/s** | 196 GB/s | rocWMMA **2.5×** faster |

> rocWMMA pooling massively outperforms CK Tile on large window sizes!
> Reason: rocWMMA uses `float4` vectorized C-dimension reads for NDHWC layout,
> while CK Tile default Pool3D uses row-major traversal, unfriendly to large C dimensions.

#### Batched Tensor Contraction (FP16)

`tile_example_batched_contraction` vs `rocwmma_batched_contraction`

| G | M | N | K | CK Tile | rocWMMA | Ratio |
|---|---|---|---|---|---|---|
| 4 | 1024 | 1024 | 1024 | **181 TF/s** | 8.1 TF/s | CK Tile **22×** |
| 8 | 512  | 512  | 2048 | **186 TF/s** | 3.8 TF/s | CK Tile **49×** |

> CK Tile batched_contraction uses persistent kernels + CShuffleEpilogue.
> rocWMMA uses simple blockIdx.z batch scheduling, where dispatch overhead dominates for small M/N.

---

### 12.6 Comprehensive Comparison Summary

| Operation | Typical Size | CK Tile | rocWMMA V2 | Winner | Gap Explanation |
|---|---|---|---|---|---|
| **HGEMM square** | 8192³ | **773 TF/s** | 592 TF/s | CK Tile | 256×256 block tile + async pipeline |
| **HGEMM thin** | 1024×4096×8192 | 219 TF/s | **275 TF/s** | rocWMMA | double-buffer latency hiding more effective |
| **HGEMM async** | 4096³ | **757 TF/s** | 536 TF/s | CK Tile | cp.async hardware instruction |
| **LayerNorm** | 3328×4096 | **3557 GB/s** | 2613 GB/s | CK Tile | codegen thread optimization |
| **RMSNorm** | 3328×4096 | **5576 GB/s** | 4787 GB/s | CK Tile | codegen |
| **SmoothQuant** | 3328×4096 | 3288 GB/s | **3929 GB/s** | rocWMMA | int64_t aligned writes |
| **SmoothQuant** | 3328×8192 | 3209 GB/s | **4069 GB/s** | rocWMMA | +26.8% |
| **Row Reduce** | 3328×4096 | 1114 GB/s | **6802 GB/s** | rocWMMA | warp butterfly more efficient |
| **Transpose** | 8192² | 437 GB/s | **2732 GB/s** | rocWMMA | LDS tiled transpose |
| **TopK Softmax** | 3328,32 experts,5k | **0.40 ms** | 0.62 ms | CK Tile | warp-level optimization |
| **MoE Sorting** | 3328,64 experts,4k | **0.012 ms** | 0.040 ms | CK Tile | radix sort vs counting sort |
| **Elementwise Add** | 3840×4096 | **3358 GB/s** | 2080 GB/s | CK Tile | tile dispatch framework +61% |
| **Pool3D MaxPool** | N=2,D×H×W=16×28×28,K=2³ | 497 GB/s | **2123 GB/s** | rocWMMA | NDHWC float4 vectorization +327% |
| **Batched Contraction** | G=4,M=N=K=1024 | **181 TF/s** | 8.1 TF/s | CK Tile | persistent+CShuflle vs simple batching |

**Root Cause Analysis of Performance Gaps:**

| Cause | CK Tile Advantage Scenarios | rocWMMA Advantage Scenarios |
|---|---|---|
| **Block Tile Size** | 256×256 (GEMM), higher L2 reuse | 128×128, more flexible when LDS is insufficient |
| **Async Memory Access** | cp.async instructions hide global memory latency | Standard load + barrier double-buffering |
| **Codegen Instantiation** | Generates optimal parameters per (M,N,K,dtype) | Runtime generic parameters |
| **Compiler Passes** | `-enable-noalias-to-md-conversion=0` and other optimizations | Standard HIP compilation |
| **Vectorized Output** | Well-optimized for large element count scenarios | int64_t aligned writes (INT8), more efficient for small elements |
| **Thread Organization** | Normalization ops rigorously tuned via codegen | Generic Wave64 butterfly reduction |

**Conclusions:**
- CK Tile is the clear leader in **compute-intensive** (large square GEMM) and **codegen-tuned normalization** (LayerNorm, RMSNorm) scenarios
- rocWMMA performs better in **memory-intensive** (Transpose, Row Reduce) and certain **quantization** (SmoothQuant N≥2048) scenarios
- For production environments, CK Tile is recommended; rocWMMA samples are ideal for rapid prototyping, education, and RDNA4 porting

