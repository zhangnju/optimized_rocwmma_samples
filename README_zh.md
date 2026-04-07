# rocWMMA Community Samples: CK Tile 优化思路移植指南

## 概述

本文档整理了基于 CK Tile (ComposableKernel Tile) 优化思路，使用 rocWMMA API 移植实现的
26 个 GPU kernel 样例的核心优化技术、代码解析与 **MI355X、Radeon AI PRO R9700** 等平台上的实测性能。

所有样例位于：`rocwmma/samples/community/rocwmma_*.cpp`

**测试环境（本文档涉及的 community sample）：**

| 项目 | MI355X（gfx950） | Radeon AI PRO R9700（gfx1201） |
|---|---|---|
| **GPU** | AMD Instinct MI355X（×8） | AMD Radeon AI PRO R9700 |
| **架构** | CDNA 3.5，256 CUs，2400 MHz | RDNA 4，64 CUs，Wave32，WMMA 16×16×16（加速频率最高约 2.92 GHz） |
| **ROCm** | 7.2.0 | 7.2.0 |
| **编译器** | AMD Clang 22.0.0（HIP 7.2） | AMD Clang 22.0.0（HIP 7.2） |
| **rocWMMA** | 2.2.0 | 2.2.0 |
| **FP16 峰值（矩阵）** | ~1300 TFlops/s（CDNA MFMA 量级） | **191** TFlops/s（[AMD 规格页](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html)，FP16 matrix） |
| **FP8 峰值（矩阵，E4M3/E5M2）** | ~2600 TFlops/s（CDNA 量级） | **383** TFlops/s（同上，FP8 matrix） |
| **样例 benchmark** | 见 §11（MI355X 表） | 测量日期 **2026-04-07**；日志 `benchmark_results_r9700_gfx1201.txt` / `.csv` |

两套 GPU 的实测吞吐均在 **[第 11 节](#11-完整性能数据)**（各小节先 MI355X、后 R9700）。表中峰值为厂商给出的理论上限，实际可达 TFlops 取决于算子与访存。

---

## 目录

0. [快速开始：编译与运行](#0-快速开始编译与运行)
1. [三级 Tile 层次结构](#1-三级-tile-层次结构)
2. [双缓冲 LDS Pipeline（COMPUTE_V4）](#2-双缓冲-lds-pipelinecompute_v4)
3. [协作式全局内存加载](#3-协作式全局内存加载)
4. [LDS 转置 B（消除 bank conflict）](#4-lds-转置-b消除-bank-conflict)
5. [Warp Tile 数据复用](#5-warp-tile-数据复用)
6. [向量化内存访问](#6-向量化内存访问)
7. [Warp 级蝶形规约](#7-warp-级蝶形规约)
8. [Online Softmax（Flash Attention）](#8-online-softmaxflash-attention)
9. [FlatMM：寄存器直接加载 B](#9-flatmm寄存器直接加载-b)
10. [Stream-K 负载均衡](#10-stream-k-负载均衡)
11. [完整性能数据（MI355X gfx950 与 R9700 gfx1201）](#11-完整性能数据)
12. [CK Tile vs rocWMMA 完整性能对比（含 Elementwise/Pooling/Contraction）](#12-ck-tile-vs-rocwmma-完整性能对比mi355x--gfx950)

---

## 0. 快速开始：编译与运行

本节提供 `build.sh` 和 `benchmark.sh` 两个脚本的完整说明与使用示例，
脚本位于 `rocwmma/samples/community/`。

### 0.1 依赖环境

```bash
# 验证 ROCm 版本（需 >= 6.0）
rocminfo | grep -E "ROCm|Version"
hipcc --version

# 验证 GPU 可见
rocminfo | grep "Marketing Name"
# 期望输出：AMD Instinct MI355X（或其他支持的 GPU）
```

### 0.2 编译脚本：`build.sh`

脚本位于 `samples/community/build.sh`（独立部署时在 `/home/optimized_rocwmma_samples/build.sh`）。
运行 `./build.sh -h` 查看完整选项说明。

**常用编译命令：**

```bash
cd rocwmma/samples/community

# 1) 自动检测 GPU，编译所有社区 sample
./build.sh

# 2) 指定 MI355X（gfx950），编译所有 sample
./build.sh -g gfx950

# 2b) AMD Radeon AI PRO R9700 / RDNA 4（gfx1201）
./build.sh -g gfx1201

# 3) 多 GPU 目标（MI355X + RDNA4），同时编译
./build.sh -g "gfx950;gfx1200;gfx1201"

# 4) 只编译单个 sample（快速迭代）
./build.sh -g gfx950 -t rocwmma_perf_gemm

# 5) 清除重建（切换 GPU 目标时使用）
./build.sh -g gfx950 -c

# 6) 并行 32 核编译
./build.sh -g gfx950 -j 32

# 7) Debug 模式（启用 CPU 验证）
./build.sh -g gfx950 -d -t rocwmma_layernorm2d
```

**编译输出示例：**

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

### 0.3 Benchmark 脚本：`benchmark.sh`

脚本位于 `samples/community/benchmark.sh`（独立部署时在 `/home/optimized_rocwmma_samples/benchmark.sh`）。
覆盖全部 26 个 sample，支持超时保护、CSV 输出、GPU 设备选择。
运行 `./benchmark.sh -h` 查看完整选项说明。

**常用 benchmark 命令：**

```bash
cd rocwmma/samples/community

# 1) 运行所有 sample，结果写到默认文件
./benchmark.sh -b ../../build

# 2) 指定 GPU（多 GPU 系统）
HIP_VISIBLE_DEVICES=2 ./benchmark.sh -b ../../build

# 3) 只跑特定 sample
./benchmark.sh -b ../../build -s rocwmma_perf_gemm
./benchmark.sh -b ../../build -s "rocwmma_layernorm2d,rocwmma_rmsnorm2d,rocwmma_smoothquant"

# 4) 生成 CSV 结果（便于后续分析）
./benchmark.sh -b ../../build --csv -o results_mi355x.txt

# 5) 指定输出文件
./benchmark.sh -b ../../build -o /tmp/bench_$(date +%Y%m%d).txt --csv
```

---

### 0.4 逐个手动运行（不使用脚本）

**编译单个 sample：**

```bash
# 进入 build 目录
cd /path/to/rocwmma/build

# 重新配置（仅首次或参数变化时）
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGPU_TARGETS="gfx950" \
    -DROCWMMA_BUILD_SAMPLES=ON \
    -DROCWMMA_BUILD_COMMUNITY_SAMPLES=ON \
    -DROCWMMA_BUILD_TESTS=OFF

# 编译所有社区 sample
make -j$(nproc) rocwmma_community_samples

# 或编译单个
make -j$(nproc) rocwmma_perf_gemm
```

**运行各 sample：**

```bash
BIN=./samples/community  # 简写

# ── GEMM 类 ──────────────────────────────────────────────────────────────────
# HGEMM Pipeline V1/V2 对比（默认跑 5 组尺寸）
${BIN}/rocwmma_perf_gemm

# 指定尺寸（M N K）
${BIN}/rocwmma_perf_gemm 4096 4096 4096
${BIN}/rocwmma_perf_gemm 8192 8192 8192
${BIN}/rocwmma_perf_gemm 1024 4096 8192   # decode-like (small M)

# Split-K GEMM
${BIN}/rocwmma_gemm_splitk

# FP8 量化 GEMM
${BIN}/rocwmma_gemm_quantized

# 批量 GEMM（batch M N K）
${BIN}/rocwmma_batched_gemm 4  1024 1024 1024
${BIN}/rocwmma_batched_gemm 16 512  512  4096

# GEMM + fused epilogue（bias + ReLU）
${BIN}/rocwmma_gemm_multi_d

# 分组 GEMM
${BIN}/rocwmma_grouped_gemm

# Stream-K GEMM
${BIN}/rocwmma_streamk_gemm

# FlatMM（B 绕过 LDS）
${BIN}/rocwmma_flatmm

# MX 格式 GEMM（gfx950 原生）
${BIN}/rocwmma_mx_gemm

# 批量张量收缩
${BIN}/rocwmma_batched_contraction

# ── 注意力类 ─────────────────────────────────────────────────────────────────
# Flash Attention 前向（B H Sq Sk D causal）
${BIN}/rocwmma_fmha_fwd

# 稀疏注意力（Jenga LUT）
${BIN}/rocwmma_sparse_attn

# ── 规范化类 ─────────────────────────────────────────────────────────────────
# LayerNorm2D（M N）
${BIN}/rocwmma_layernorm2d 3328 4096
${BIN}/rocwmma_layernorm2d 3328 8192

# RMSNorm2D + Add+RMSNorm fused（M N）
${BIN}/rocwmma_rmsnorm2d 3328 4096

# SmoothQuant FP16→INT8（M N）
${BIN}/rocwmma_smoothquant 3328 4096

# ── MoE 类 ───────────────────────────────────────────────────────────────────
# TopK Softmax（tokens experts topk）
${BIN}/rocwmma_topk_softmax 3328 32 5

# MoE Token Sorting（tokens experts topk）
${BIN}/rocwmma_moe_sorting 3328 64 5

# MoE Smooth Quantization
${BIN}/rocwmma_moe_smoothquant

# 全融合 MoE（sort + gate_up GEMM + SiLU + down GEMM）
${BIN}/rocwmma_fused_moe

# ── Reduction & Elementwise ───────────────────────────────────────────────────
# Row-wise Reduce Sum/Max（M N）
${BIN}/rocwmma_reduce 3328 4096

# Elementwise（M N）
${BIN}/rocwmma_elementwise 3840 4096

# ── 卷积 & 数据布局 ────────────────────────────────────────────────────────────
# Im2Col + GEMM（CNN layers）
${BIN}/rocwmma_img2col_gemm

# 分组卷积前向
${BIN}/rocwmma_grouped_conv_fwd

# 3D Pooling
${BIN}/rocwmma_pooling

# Tensor Permute / NCHW↔NHWC
${BIN}/rocwmma_permute

# ── 工具 ─────────────────────────────────────────────────────────────────────
# Tile Distribution Register Map（诊断工具，打印 fragment 形状）
${BIN}/rocwmma_tile_distr_reg_map
```

---

### 0.5 完整端到端流程（从零开始）

```bash
# 1. 克隆仓库
git clone https://github.com/ROCm/rocWMMA.git
cd rocWMMA

# 2. 编译所有社区 sample（MI355X / gfx950）
./samples/community/build.sh -g gfx950

# 3. 运行全量 benchmark，生成报告
./samples/community/benchmark.sh \
    -b build \
    -o samples/community/benchmark_results.txt \
    --csv

# 4. 查看结果
cat samples/community/benchmark_results.txt | grep -E "TFlops|GB/s"

# 5. 查看 CSV 并排序
sort -t',' -k4 -rn samples/community/benchmark_results.csv | head -20
```

---

### 0.6 RDNA4（gfx1200/gfx1201）编译注意事项

RDNA4 使用 Wave32 + WMMA 16×16×16，与 gfx9 的 Wave64 + MFMA 32×32×16 不同。
所有 sample 通过编译期宏 `ROCWMMA_ARCH_GFX9` / `ROCWMMA_ARCH_GFX12` 自动选择参数：

```bash
# 编译 RDNA4 目标
./build.sh -g "gfx1200;gfx1201"

# 同时编译 MI355X 和 RDNA4（fat binary）
./build.sh -g "gfx950;gfx1200;gfx1201"
```

关键参数差异（代码中自动切换）：

```cpp
// rocwmma_perf_gemm.cpp
#if defined(ROCWMMA_ARCH_GFX9)
// MI355X / gfx950：Wave64 + MFMA 32×32×16
constexpr uint32_t ROCWMMA_M=32, ROCWMMA_N=32, ROCWMMA_K=16;
constexpr uint32_t WARP_SIZE = Constants::AMDGCN_WAVE_SIZE_64;  // = 64
#else
// RDNA4 / gfx1200：Wave32 + WMMA 16×16×16
constexpr uint32_t ROCWMMA_M=16, ROCWMMA_N=16, ROCWMMA_K=16;
constexpr uint32_t WARP_SIZE = Constants::AMDGCN_WAVE_SIZE_32;  // = 32
#endif
```

---

## 1. 三级 Tile 层次结构

**对应 CK Tile：** `TileGemmShape<sequence<M,N,K>, sequence<Mw,Nw,Kw>, sequence<Mwt,Nwt,Kwt>>`

CK Tile 将 GEMM 计算分解为三个层次：Block Tile → Warp Layout → MFMA/WMMA Tile。
rocWMMA 通过 `fragment` 尺寸和线程块配置实现同样的分解。

```cpp
// =========================================================
// 文件：rocwmma_perf_gemm.cpp  行 ~100-130
// =========================================================

// Level 3: 单个 MFMA 指令处理的矩阵块 (gfx9: 32×32×16)
//          对应 CK Tile M_Warp_Tile × N_Warp_Tile × K_Warp_Tile
namespace gfx9Params { enum : uint32_t {
    ROCWMMA_M = 32u,   // MFMA block M
    ROCWMMA_N = 32u,   // MFMA block N
    ROCWMMA_K = 16u,   // MFMA block K
    BLOCKS_M  = 2u,    // 每个 warp 沿 M 方向的 MFMA tile 数
    BLOCKS_N  = 2u,    // 每个 warp 沿 N 方向的 MFMA tile 数
    TBLOCK_X  = 128u,  // 线程块 X 方向线程数 (= WARPS_M × 64)
    TBLOCK_Y  = 2u,    // 线程块 Y 方向线程数 (= WARPS_N)
    WARP_SIZE = Constants::AMDGCN_WAVE_SIZE_64
}; }

// Level 2: Warp Tile = BLOCKS_M × ROCWMMA_M
//          对应 CK Tile M_Warp × M_Warp_Tile
constexpr uint32_t WARP_TILE_M  = BLOCKS_M * ROCWMMA_M;  // 2×32 = 64
constexpr uint32_t WARP_TILE_N  = BLOCKS_N * ROCWMMA_N;  // 2×32 = 64

// Level 1: Block (Macro) Tile = WARPS_M × WARP_TILE_M
//          对应 CK Tile M_Tile (Block Tile)
constexpr uint32_t WARPS_M      = TBLOCK_X / WARP_SIZE;  // 128/64 = 2
constexpr uint32_t WARPS_N      = TBLOCK_Y;               // 2
constexpr uint32_t MACRO_TILE_M = WARPS_M * WARP_TILE_M; // 2×64 = 128
constexpr uint32_t MACRO_TILE_N = WARPS_N * WARP_TILE_N; // 2×64 = 128
constexpr uint32_t MACRO_TILE_K = ROCWMMA_K;             // 16
```

**层次结构可视化：**

```
Block Tile (128×128)
├── Warp(0,0): Warp Tile (64×64) ← 2×2 个 MFMA 32×32×16
├── Warp(1,0): Warp Tile (64×64)
├── Warp(0,1): Warp Tile (64×64)
└── Warp(1,1): Warp Tile (64×64)

每个 Warp Tile 内 (BLOCKS_M=2, BLOCKS_N=2):
[MFMA(0,0)][MFMA(0,1)]    每个 MFMA = 32×32×16
[MFMA(1,0)][MFMA(1,1)]    共 4 次 mma_sync 调用
```

**为什么这样设计：**
- Block Tile 决定 L2 cache 复用效率：128×128×FP16 = 32KB，覆盖一级 cache
- Warp Tile 决定寄存器占用：64×64×FP32 acc = 16 KB/warp（在 MI355X 256KB 寄存器文件内）
- MFMA Tile 由硬件固定：32×32×16 是 gfx9 MFMA 指令的原生尺寸

---

## 2. 双缓冲 LDS Pipeline（COMPUTE_V4）

**对应 CK Tile：** `GemmPipelineAgBgCrCompV4` 中的 `DoubleSmemBuffer = true`

这是 CK Tile 最核心的优化：在计算当前 K 步的同时，预取下一个 K 步的数据到第二个 LDS 缓冲区，从而隐藏全局内存延迟。

```cpp
// =========================================================
// 文件：rocwmma_perf_gemm.cpp  行 ~270-340  (Pipeline V2)
// =========================================================

// ─── 分配两块 LDS 缓冲区（ping-pong 双缓冲）───
// CK Tile: DoubleSmemBuffer = true 时分配 2×sizeLds
HIP_DYNAMIC_SHARED(void*, localMemPtr);
auto* ldsPtrLo = reinterpret_cast<InputT*>(localMemPtr);
auto* ldsPtrHi = ldsPtrLo + sizeLds;   // 第二个缓冲区偏移 sizeLds 个元素

// ─── 预取 K=0 到缓冲区 Lo ───
GRFragA grA; GRFragB grB;
load_matrix_sync(grA, a + gReadOffA, lda);   // 全局内存 → 寄存器
load_matrix_sync(grB, b + gReadOffB, ldb);
gReadOffA += kStepA;  // 预先推进到 K=1 的地址
gReadOffB += kStepB;
store_matrix_sync(ldsPtrLo + ldsOffA, toLWFragA(grA), ldsld);  // 寄存器 → LDS Lo
store_matrix_sync(ldsPtrLo + ldsOffB, toLWFragB(grB), ldsld);

MmaFragAcc fragAcc;
fill_fragment(fragAcc, ComputeT(0));
synchronize_workgroup();   // 确保所有 warp 的 LDS 写入完成

// ─── 主 K 循环：计算 K_i 同时预取 K_{i+1} ───
for(uint32_t kStep = MACRO_TILE_K; kStep < k; kStep += MACRO_TILE_K) {
    // Step A: 从 Lo 读取当前 K 步的数据（LDS 低延迟）
    LRFragA lrA; LRFragB lrB;
    load_matrix_sync(lrA, ldsPtrLo + ldsRdA, ldsld);
    load_matrix_sync(lrB, ldsPtrLo + ldsRdB, ldsld);

    // Step B: 同时预取下一个 K 步（全局内存高延迟被 MMA 掩盖）
    load_matrix_sync(grA, a + gReadOffA, lda);   // ← 与 MMA 并行执行
    load_matrix_sync(grB, b + gReadOffB, ldb);
    gReadOffA += kStepA;
    gReadOffB += kStepB;

    // Step C: 计算当前 K 步的 MMA
    mma_sync(fragAcc, toMmaFragA(lrA), toMmaFragB(lrB), fragAcc);

    // Step D: 把预取结果写入 Hi 缓冲区
    store_matrix_sync(ldsPtrHi + ldsOffA, toLWFragA(grA), ldsld);
    store_matrix_sync(ldsPtrHi + ldsOffB, toLWFragB(grB), ldsld);

    synchronize_workgroup();  // 等待 Hi 写入完成

    // Step E: 交换 Lo/Hi 指针（O(1) 操作，无数据拷贝）
    auto* tmp = ldsPtrLo;
    ldsPtrLo  = ldsPtrHi;
    ldsPtrHi  = tmp;
}
```

**执行时间线：**

```
时间 →
K=0: [Global Load A0,B0] → [LDS Store Lo]
K=1: [LDS Read Lo] | [Global Load A1,B1]   ← 并行！
     [MMA K0]      | [LDS Store Hi]
                     [Swap Lo↔Hi]
K=2: [LDS Read Lo] | [Global Load A2,B2]   ← 并行！
     [MMA K1]      | [LDS Store Hi]
...
```

**LDS 尺寸计算：**

```cpp
// LDS 布局：A block (ldHA 行) 后接 B block (ldHB 行)，宽度均为 MACRO_TILE_K
constexpr uint32_t ldsHeightA = GetIOShape_t<LWFragA>::BlockHeight;  // = 128
constexpr uint32_t ldsHeightB = GetIOShape_t<LWFragB>::BlockHeight;  // = 128
constexpr uint32_t ldsHeight  = ldsHeightA + ldsHeightB;             // = 256
constexpr uint32_t ldsWidth   = MACRO_TILE_K;                        // = 16
constexpr uint32_t sizeLds    = ldsHeight * ldsWidth;                // = 4096 elements

// 单缓冲 LDS: 4096 × 2 bytes = 8 KB
// 双缓冲 LDS: 8192 × 2 bytes = 16 KB（CK Tile DoubleSmemBuffer = true）
```

---

## 3. 协作式全局内存加载

**对应 CK Tile：** `CoopScheduler` + `GemmPipelineProblem::CoopA/B`

block 内所有 warp 协作分担 Macro Tile 的全局内存加载，提升带宽利用率。

```cpp
// =========================================================
// 文件：rocwmma_perf_gemm.cpp  行 ~175-200
// =========================================================

// 使用 coop_row_major_2d 调度器：将 macro tile 的 I/O 工作
// 在 TBLOCK_X × TBLOCK_Y 的线程块内均匀分配给每个 warp
// CK Tile 等价：fragment<..., CoopScheduler> + coop load
using CoopScheduler = fragment_scheduler::coop_row_major_2d<TBLOCK_X, TBLOCK_Y>;

// Macro Tile 级别的协作加载 fragment
// 每个 warp 只负责加载 GRFragA 的 1/(WARPS_M×WARPS_N) 部分
using GRFragA = fragment<matrix_a,
                         MACRO_TILE_M,   // = 128（整个 block 的 M 范围）
                         MACRO_TILE_N,   // = 128（整个 block 的 N 范围）
                         MACRO_TILE_K,   // = 16
                         InputT,
                         DataLayoutA,
                         CoopScheduler>; // ← 关键：启用协作加载

// 每个 warp 计算自己负责加载的行偏移
// 实际上 load_matrix_sync 内部会根据 CoopScheduler 自动分配工作
load_matrix_sync(grA, a + gReadOffA, lda);  // 每个 warp 只加载 1/4 的数据

// 然后所有 warp 分别写入 LDS 的对应位置
store_matrix_sync(ldsPtrLo + ldsOffA, toLWFragA(grA), ldsld);
```

**协作加载的收益：**
```
无协作：每个 warp 独立加载完整 Macro Tile
  → 4 个 warp × 128×16×2 bytes = 4× 冗余流量

有协作：4 个 warp 各加载 1/4
  → 总流量 = 128×16×2 bytes（无冗余）
  → L2 cache 命中率提升，全局带宽利用率提升
```

---

## 4. LDS 转置 B（消除 bank conflict）

**对应 CK Tile：** `CLayout::LdsB = Transposed`

B 矩阵在存入 LDS 时进行转置，使得 A 和 B 的 K 维都成为 LDS 的快速轴（列），
从而保证 LDS 读写的 bank-conflict-free 访问。

```cpp
// =========================================================
// 文件：rocwmma_perf_gemm.cpp  行 ~155-170
// =========================================================

// 全局 B: row_major，形状 [K, N]，快速轴是 N
using DataLayoutB   = row_major;

// LDS 布局：col_major，使 K 成为快速轴
using DataLayoutLds = col_major;

// LDS 写入时对 B 做转置：
//   原始 B[k, n] → LDS 中存为 B^T[n, k]
//   这样 LDS 的列方向 = K 方向 = 两个矩阵共同的收缩维度
using LWFragB = apply_data_layout_t<
    apply_transpose_t<GRFragB>,  // ← 先转置（行列互换）
    DataLayoutLds>;               // ← 再以 col_major 方式映射到 LDS

// LDS 读取时对 B 再次转置，恢复为 MFMA 期望的格式
using LRFragB = apply_data_layout_t<
    apply_transpose_t<MmaFragB>,  // ← 转置
    DataLayoutLds>;

// 实际写入：
ROCWMMA_DEVICE auto toLWFragB(GRFragB const& gr)
{
    return apply_data_layout<DataLayoutLds>(
        apply_transpose(gr)  // 转置 B，然后按 col_major 存 LDS
    );
}

// 实际读取：
ROCWMMA_DEVICE auto toMmaFragB(LRFragB const& lr)
{
    return apply_data_layout<DataLayoutB>(
        apply_transpose(lr)  // 从 LDS 读出再转置，恢复原始 row_major 格式
    );
}
```

**LDS Bank Conflict 分析：**

```
gfx9 LDS 有 32 个 bank，每个 bank 4 字节宽。
Wave 64 中 64 个 lane 同时访问 LDS。

不转置时（B 为 row_major，K=16, N=128）：
  lane 0-15 读 B[0][0..15]，stride=1 → bank 0,1,2,...,15 → 无冲突
  lane 16-31 读 B[0][16..31] → bank 16..31 → 无冲突
  但下一行 lane 读 B[1][0..15] → 同样 bank 0..15 → 16路冲突！

转置后（B^T 为 col_major，K=16 作为行）：
  所有 lane 读同一列的不同行 → 每个 lane 访问不同 bank → 0 冲突
```

---

## 5. Warp Tile 数据复用

**对应 CK Tile：** `BlockGemmASmemBSmemCRegV1` 中的 `BLOCKS_M × BLOCKS_N` 循环

每个 warp 计算 `BLOCKS_M × BLOCKS_N` 个 MFMA tile，在内层循环中复用 A 的一列和 B 的一行。

```cpp
// =========================================================
// 文件：rocwmma_batched_gemm.cpp  行 ~160-200（概念等价于）
// rocwmma_perf_gemm.cpp 中通过 WARP_TILE_M/N = BLOCKS*ROCWMMA 实现
// =========================================================

// MmaFragA 的尺寸 = WARP_TILE_M × WARP_TILE_N = 64×64
// 内部是 BLOCKS_M × BLOCKS_N = 2×2 个 32×32 MFMA tile 的聚合
using MmaFragA = fragment<matrix_a,
    WARP_TILE_M,  // = BLOCKS_M * ROCWMMA_M = 2*32 = 64
    WARP_TILE_N,  // = BLOCKS_N * ROCWMMA_N = 2*32 = 64
    WARP_TILE_K,  // = ROCWMMA_K = 16
    InputT, DataLayoutA>;

// 一次 mma_sync 等价于 CK Tile 中
// for i in range(BLOCKS_M):
//   for j in range(BLOCKS_N):
//     mfma(acc[i,j], A[i], B[j], acc[i,j])
// rocWMMA 在 fragment 层面自动处理 2×2 的子矩阵
mma_sync(fragAcc, toMmaFragA(lrA), toMmaFragB(lrB), fragAcc);
// ↑ 等价于 2×2=4 次 32×32×16 MFMA 指令
```

**数据复用分析（BLOCKS=2×2）：**

```
A[row0]: 被 B 的 col0 和 col1 各用一次  ← N 方向复用 ×BLOCKS_N
A[row1]: 被 B 的 col0 和 col1 各用一次

B[col0]: 被 A 的 row0 和 row1 各用一次  ← M 方向复用 ×BLOCKS_M
B[col1]: 被 A 的 row0 和 row1 各用一次

总复用率 = BLOCKS_M × BLOCKS_N = 4×
每次 LDS 读出的数据被 MMA 使用 4 次（而非 1 次）
```

---

## 6. 向量化内存访问

**对应 CK Tile：** `VectorSize = 8`（`fp16 × 8 = 128-bit`）

在 elementwise、reduce、normalization 类 kernel 中，每个线程一次加载 8 个 fp16（一个 float4）。

```cpp
// =========================================================
// 文件：rocwmma_reduce.cpp  行 ~60-95
//       rocwmma_layernorm2d.cpp  行 ~80-110
// =========================================================

constexpr uint32_t VEC = 8;  // 8 × fp16 = 128-bit，对齐到 128-bit 内存访问

// 向量化加载：一次读取 128 bits（4 × float32 = 8 × float16）
// CK Tile 等价：CK_TILE_UNROLL, VectorSize=8 的 ThreadwiseTensorSliceTransfer
for(uint32_t col = threadIdx.x * VEC; col + VEC <= N; col += TBLOCK * VEC)
{
    // 将 float16* 转换为 float4* 实现 128-bit 对齐读取
    const float4* ptr = reinterpret_cast<const float4*>(row_ptr + col);
    float4 v = *ptr;   // 一次 128-bit 加载，等价于 4×ds_read_b128

    // 解包：float4 内含 4 个 __half2（每个 __half2 = 2×float16）
    const __half2* h = reinterpret_cast<const __half2*>(&v);

    for(int i = 0; i < 4; i++) {
        float2 f = __half22float2(h[i]);  // __half2 → float2（SIMD转换）
        acc += f.x + f.y;                 // 累加 2 个元素
    }
}
```

**性能对比（理论）：**

```
无向量化：每线程每次 16-bit 读取
  吞吐 = 1× （受 memory transaction 数量限制）

向量化（float4）：每线程每次 128-bit 读取
  吞吐 ≈ 8× （每次 transaction 传输 8 个 fp16）
  实际提升：约 2-4×（受带宽而非延迟限制时）
```

---

## 7. Warp 级蝶形规约

**对应 CK Tile：** `WarpReduce<ReduceOp::Add>` / `warp_reduce`

LayerNorm、RMSNorm、Reduce、SmoothQuant 等 kernel 使用 butterfly（蝶形）模式
通过 `__shfl_down` 在 warp 内进行无 LDS 的高效规约。

```cpp
// =========================================================
// 文件：rocwmma_reduce.cpp  行 ~40-55
//       rocwmma_layernorm2d.cpp  行 ~65-80
// =========================================================

// Wave64 蝶形归约：log2(64) = 6 步，每步使 规约范围缩小一半
// CK Tile: WarpReduce<AccType, ReduceOp::Add>
template <typename T, typename ReduceOp>
__device__ __forceinline__ T warpReduce(T val, ReduceOp op)
{
    // 步骤 1: offset=32 → 合并 lane[0..31] 和 lane[32..63]
    // 步骤 2: offset=16 → 合并每组内 lane[0..15] 和 lane[16..31]
    // ...
    // 步骤 6: offset=1  → 合并相邻 lane
    for(int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = op(val, __shfl_down(val, offset, WARP_SIZE));
    //          ↑ 无需 LDS，直接 lane → lane 通信（1 cycle/step）
    return val;
    // 最终 lane 0 持有完整规约结果
}

// Welford 在线方差的 warp 规约（LayerNorm 专用）
// CK Tile: WelfordWarpReduce
struct WelfordVar { float mean, m2; uint32_t count; };

__device__ WelfordVar welfordWarpReduce(WelfordVar v)
{
    for(int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float  om  = __shfl_down(v.mean,  offset, WARP_SIZE);
        float  om2 = __shfl_down(v.m2,    offset, WARP_SIZE);
        uint32_t oc = __shfl_down(v.count, offset, WARP_SIZE);

        // Parallel Welford merge（数学上等价于顺序 Welford）
        uint32_t nc = v.count + oc;
        if(nc == 0) continue;
        float delta = om - v.mean;
        v.mean   = v.mean + delta * oc / nc;
        v.m2    += om2 + delta * delta * (float)v.count * oc / nc;
        v.count  = nc;
    }
    return v;
}

// Block 级规约：warp 内规约结果写入 LDS，再在第一个 warp 内合并
__shared__ float smem[WARPS];  // 每个 warp 一个槽位
if(lid == 0) smem[wid] = acc;   // 只有 lane 0 写（携带 warp 规约结果）
__syncthreads();
if(wid == 0) {                   // 第一个 warp 做最终归约
    acc = (lid < WARPS) ? smem[lid] : 0.f;
    acc = warpReduce(acc, [](float a, float b){ return a + b; });
}
```

**蝶形规约步骤示意（WARP_SIZE=8 简化）：**

```
初始:  [a0, a1, a2, a3, a4, a5, a6, a7]

offset=4:
lane 0 ← a0+a4,  lane 1 ← a1+a5,  lane 2 ← a2+a6,  lane 3 ← a3+a7

offset=2:
lane 0 ← (a0+a4)+(a2+a6),  lane 1 ← (a1+a5)+(a3+a7)

offset=1:
lane 0 ← sum(all)   ← 最终结果在 lane 0

共 log2(8) = 3 步，无需任何 LDS 操作
```

---

## 8. Online Softmax（Flash Attention）

**对应 CK Tile：** `FmhaPipelineQRKSVS`（Flash Attention 2 算法）

Flash Attention 的核心是 online softmax：在遍历 KV block 时维护运行中的 max 和 sum，
避免材料化完整的 S=QK^T 矩阵。

```cpp
// =========================================================
// 文件：rocwmma_fmha_fwd.cpp  行 ~140-200
//       rocwmma_sparse_attn.cpp  行 ~170-220
// =========================================================

// 每个 warp 为 MFMA_M=32 行 Q 维护独立的 running max/sum
float row_max[MFMA_M], row_sum[MFMA_M];
for(uint32_t i = 0; i < MFMA_M; i++) { row_max[i] = -1e30f; row_sum[i] = 0.f; }

// 初始化 O 累加器
for(uint32_t t = 0; t < N_TILES; t++) fill_fragment(fragO[t], AccT(0));

// 遍历所有 KV block（稀疏注意力只遍历有效 block）
for(int32_t kv_ptr = kv_start; kv_ptr < kv_end; kv_ptr++) {
    // 计算 S = Q * K^T / sqrt(d)
    FragS fragS; fill_fragment(fragS, AccT(0));
    for(uint32_t d = 0; d < D; d += MFMA_K) {
        FragQ fragQ; FragK fragK;
        load_matrix_sync(fragQ, Qh + q_local*D + d, D);
        load_matrix_sync(fragK, Kh + kv_off*D + d, D);
        mma_sync(fragS, fragQ, fragK, fragS);
    }

    // Online softmax 更新（Flash Attention 关键步骤）
    // CK Tile: online_softmax_update<AccDataType>
    for(uint32_t elem = 0; elem < fragS.num_elements; elem++) {
        uint32_t r = elem % MFMA_M;
        float s = static_cast<float>(fragS.x[elem]) * scale;

        float new_max = fmaxf(row_max[r], s);
        float exp_s   = expf(s - new_max);          // 数值稳定的 exp
        float exp_old = expf(row_max[r] - new_max);  // 旧 max 的衰减因子

        // 更新 running sum（补偿旧 max 与新 max 的差异）
        row_sum[r] = row_sum[r] * exp_old + exp_s;
        row_max[r] = new_max;

        fragS.x[elem] = static_cast<AccT>(exp_s);  // 用于 P×V
    }

    // O += P × V（P 是已归一化的权重）
    for(uint32_t nt = 0; nt < N_TILES; nt++) {
        FragV fragV;
        load_matrix_sync(fragV, Vh + kv_off*D + nt*MFMA_N, D);
        mma_sync(fragO[nt], fragP_cast, fragV, fragO[nt]);
    }
}

// 最终归一化：O /= row_sum（在寄存器中完成，无额外内存访问）
for(uint32_t nt = 0; nt < N_TILES; nt++) {
    for(uint32_t elem = 0; elem < fragO[nt].num_elements; elem++) {
        uint32_t r = elem % MFMA_M;
        if(row_sum[r] > 0.f) fragO[nt].x[elem] /= row_sum[r];
    }
}
```

**Flash Attention vs 标准 Attention 内存对比：**

```
序列长度 S=4096，头维度 D=128，精度 FP16：

标准 Attention:
  S矩阵: S×S = 4096×4096×2 = 32 MB  ← 必须全部写到 HBM
  内存流量: O(S²×D)

Flash Attention:
  S矩阵: 不写到 HBM，在寄存器/LDS 中分块计算
  内存流量: O(S×D)（线性！）
```

---

## 9. FlatMM：寄存器直接加载 B

**对应 CK Tile：** `WeightPreshufflePipelineAGmemBGmemCRegV2` / FlatMM

FlatMM 是 CK Tile 针对解码场景（小 M）的专用优化：B 矩阵不经过 LDS，
直接从全局内存加载到每个 warp 的寄存器，减半 LDS 占用，增加 occupancy。

```cpp
// =========================================================
// 文件：rocwmma_flatmm.cpp  行 ~150-210
// =========================================================

// 标准 GEMM：A 和 B 都通过 LDS 共享
// LDS 需求 = (ldHA + ldHB) × MACRO_TILE_K × sizeof(fp16)
//          = (128 + 128) × 16 × 2 = 8192 bytes（单缓冲）

// FlatMM：只有 A 经过 LDS，B 每个 warp 独立加载
// LDS 需求 = ldHA × MACRO_TILE_K × sizeof(fp16)
//          = 128 × 16 × 2 = 4096 bytes（单缓冲）= 减半！

constexpr uint32_t szLdsA = ldHA * MACRO_TILE_K;  // 只有 A 的 LDS

// A: 协作写入 LDS（同标准 GEMM）
GRA grA;
load_matrix_sync(grA, A + rA, lda);
store_matrix_sync(lLo, toLWA(grA), ldsld_a);
synchronize_workgroup();

for(uint32_t ks = MACRO_TILE_K; ks < K; ks += MACRO_TILE_K) {
    // A: 从 LDS 读取（所有 warp 共享）
    LRA lrA;
    load_matrix_sync(lrA, lLo + lRA, ldsld_a);

    // B: 每个 warp 独自从全局内存加载自己的 WARP_TILE_N 列
    // CK Tile: B 已按 MFMA register layout 预排列（preshuffle）
    MB mfragB;
    auto rB = rB_base + GetDataLayout_t<MB>::fromMatrixCoord(
                  make_coord2d(ks - MACRO_TILE_K, 0u), ldb);
    load_matrix_sync(mfragB, B + rB, ldb);  // 直接全局 → 寄存器（绕过 LDS）

    // 预取下一步 A（与当前 MMA 并行）
    load_matrix_sync(grA, A + rA, lda);
    rA += kA;

    mma_sync(fAcc, toMA(lrA), mfragB, fAcc);  // MMA

    // 只更新 LDS 中的 A（B 不需要）
    store_matrix_sync(lHi, toLWA(grA), ldsld_a);
    synchronize_workgroup();
    // Swap Lo/Hi
    auto* t = lLo; lLo = lHi; lHi = t;
}
```

**FlatMM 适用场景：**

```
Decode（小 M）:  M=128, N=4096, K=4096
  B 矩阵大小: 4096×4096×2 = 32 MB（权重，在 HBM 中）
  B 每个 warp 读取: 64×4096×2 = 512 KB

B 预排列（Preshuffle）：
  离线将 B 重排为 MFMA 寄存器友好的布局
  → 在线读取时零 shuffle overhead
  → 绕过 LDS store/load 两步，节省约 16 KB LDS

LDS 占用对比:
  标准 GEMM: 16 KB（双缓冲）→ 每个 CU 可同时运行 N 个 block
  FlatMM:     8 KB（双缓冲）→ occupancy 提升约 2×
```

---

## 10. Stream-K 负载均衡

**对应 CK Tile：** `GemmStreamKPartitioner`

Stream-K 将 GEMM 的 K 维切分为 "SK unit"（每单元 = MACRO_TILE_K 步），
均匀分配给所有 SM，消除波量末尾的负载不均衡。

```cpp
// =========================================================
// 文件：rocwmma_streamk_gemm.cpp  行 ~95-175
// =========================================================

// SK unit 总数 = 输出 tile 数 × K steps
uint32_t total_tiles = tiles_m * tiles_n;         // = (M/MT_M) × (N/MT_N)
uint32_t k_units     = K / MACRO_TILE_K;          // = K steps per output tile
uint32_t total_sk    = total_tiles * k_units;     // 所有 K 切片的总数

// 每个 CTA（SM）分配的 SK unit 数（轮询调度）
uint32_t units_per_cta = (total_sk + num_cus - 1) / num_cus;

// Kernel 内：每个 block 通过 blockIdx.x 知道自己的范围
uint32_t my_sk_start = blockIdx.x * units_per_cta;
uint32_t sk_end      = min(my_sk_start + units_per_cta, total_sk);

// 遍历分配给本 block 的 SK units
for(uint32_t sk = my_sk_start; sk < sk_end; sk++) {
    uint32_t tile_id = sk / k_units;  // 属于哪个输出 tile
    uint32_t k_idx   = sk % k_units;  // 该 tile 的第几个 K slice

    // 跨 output tile 边界时，flush 上一个 tile 的累积结果
    if(tile_started && tile_id != cur_tile_id) {
        // 用 atomicAdd 写入 FP32 workspace（多 block 共同贡献一个 tile）
        for(uint32_t i = 0; i < fragAcc.num_elements; i++)
            atomicAdd(&workspace[cur_tile_id * WARP_TILE_M * WARP_TILE_N + i],
                      static_cast<WorkT>(fragAcc.x[i]));
        // 原子递增完成计数器
        atomicAdd(&tile_done[cur_tile_id], 1u);
        fill_fragment(fAcc, ComputeT(0));  // 重置累加器
    }
    // 计算本 SK unit 的单步 MMA（MACRO_TILE_K 宽度的 K slice）
    // ...（同标准 GEMM 的单步计算）
}

// Stage 2：所有 SK block 完成后，reduce workspace → 最终输出
// hipDeviceSynchronize() 保证 Stage 1 完成
hipLaunchKernelGGL(streamk_reduce_kernel, ...);
```

**Stream-K 消除的"尾部气泡"问题：**

```
标准 2D tile 分配（M=512, N=512, MT=128）：
  总 tile 数 = 4×4 = 16
  MI355X 有 256 CUs
  Wave 0: CU 0-15 各处理 1 tile（利用率 100%）
  → 没有波尾浪费，但 K 大时每个 tile 串行

Stream-K（K=4096, MT_K=16）：
  总 SK units = 16 × 256 = 4096
  每 CU 处理 4096/256 = 16 units
  → 所有 256 CU 持续工作，无空闲
  → 对 K 大（decode 场景）尤其有效
```

---

## 11. 完整性能数据

数据分别来自 **AMD Instinct MI355X（gfx950，CDNA 3.5）** 与 **AMD Radeon AI PRO R9700（gfx1201，RDNA 4）**（后者测量 **2026-04-07**，日志 `benchmark_results_r9700_gfx1201.txt` / `.csv`）。**11.1–11.6** 各小节中先列 **MI355X**，再列 **R9700**。**11.7** 仅为 MI355X 上 CK Tile 与 rocWMMA 对比。**11.8** 给出两套 GPU 的编译与 benchmark 复现命令。

**HGEMM（RDNA 4）：** 在 gfx1201 上，多数方阵尺寸下 **`rocwmma_perf_gemm` 的 V1 顺序流水线常快于 V2 双缓冲**（与 MI355X 不同），请按形状调参。

### 11.1 GEMM 类

**MI355X（gfx950）**

#### 标准 HGEMM（FP16 输入，FP32 累加，FP16 输出）

`rocwmma_perf_gemm.cpp`

| M | N | K | V1 Sequential | V2 Double-Buffer | 提升 |
|---|---|---|---|---|---|
| 3840 | 4096 | 4096 | 365 TF/s | **492 TF/s** | 1.35× |
| 4096 | 4096 | 4096 | 393 TF/s | **536 TF/s** | 1.36× |
| 8192 | 8192 | 8192 | 464 TF/s | **591 TF/s** | 1.27× |
| 1024 | 4096 | 8192 | 197 TF/s | **274 TF/s** | 1.39× |
| 4096 | 1024 | 8192 | 197 TF/s | **281 TF/s** | 1.43× |

#### Split-K GEMM（FP16）

`rocwmma_gemm_splitk.cpp`

| M | N | K | split_k | TFlops/s |
|---|---|---|---|---|
| 512  | 512  | 16384 | 4 | 18.8 |
| 1024 | 1024 | 16384 | 8 | 26.9 |
| 2048 | 2048 | 8192  | 4 | 27.1 |
| 4096 | 4096 | 4096  | 2 | 26.7 |

#### FP8 量化 GEMM

`rocwmma_gemm_quantized.cpp`

| M | N | K | TFlops/s | 相对 FP16 峰值 |
|---|---|---|---|---|
| 3840 | 4096 | 4096 | **109 TF/s** | — |
| 4096 | 4096 | 4096 | **112 TF/s** | — |
| 8192 | 8192 | 8192 | **147 TF/s** | — |

#### MX 格式 GEMM（E4M3 FP8，E8M0 block scale）

`rocwmma_mx_gemm.cpp`

| M | N | K | MX_BLOCK | TFlops/s |
|---|---|---|---|---|
| 3840 | 4096 | 4096 | 32 | 56.1 |
| 4096 | 4096 | 4096 | 32 | 58.3 |
| 8192 | 8192 | 8192 | 32 | 69.7 |

#### GEMM + 融合 Epilogue（bias + scale + ReLU）

`rocwmma_gemm_multi_d.cpp`

| M | N | K | ReLU | TFlops/s |
|---|---|---|---|---|
| 3840 | 4096 | 4096 | No  | 155 |
| 4096 | 4096 | 4096 | Yes | 164 |
| 8192 | 8192 | 8192 | Yes | 189 |

#### 批量 GEMM

`rocwmma_batched_gemm.cpp`

| batch | M | N | K | TFlops/s(单batch) | 总有效 TFlops/s |
|---|---|---|---|---|---|
| 4 | 1024 | 1024 | 1024 | 61.4 | 245.6 |
| 8 | 512  | 512  | 2048 | 16.8 | 134.8 |
| 16 | 256 | 256  | 4096 | 4.4  | 70.4  |

#### 分组 GEMM

`rocwmma_grouped_gemm.cpp`

| G | 各组尺寸 | 总 TFlops/s |
|---|---|---|
| 4 | 256/512/128/768 × 4096 × 4096 (混合) | **109** |
| 4 | 1024 × 4096 × 4096 (均匀) | **129** |
| 3 | 512×1024/2048/4096 × 2048 (变 N) | 66 |

#### Stream-K GEMM

`rocwmma_streamk_gemm.cpp`（256 CUs 持续运行）

| M | N | K | TFlops/s |
|---|---|---|---|
| 512  | 512  | 4096 | 8.1 |
| 1024 | 1024 | 4096 | 27.8 |
| 2048 | 2048 | 4096 | 66.0 |
| 4096 | 4096 | 4096 | 64.4 |

#### FlatMM（A经LDS，B绕过LDS直读寄存器）

`rocwmma_flatmm.cpp`

| M | N | K | LDS占用 | TFlops/s |
|---|---|---|---|---|
| 128  | 4096 | 4096 | 8 KB（仅A） | 16.2 |
| 256  | 4096 | 4096 | 8 KB | 32.4 |
| 512  | 4096 | 4096 | 8 KB | 64.5 |
| 1024 | 4096 | 4096 | 8 KB | **91.7** |
| 4096 | 4096 | 4096 | 8 KB | **127** |

#### 批量张量收缩

`rocwmma_batched_contraction.cpp`

| G | M | N | K | TFlops/s(per group) |
|---|---|---|---|---|
| 4  | 1024 | 1024 | 1024 | 8.1 |
| 8  | 512  | 512  | 2048 | 3.8 |
| 16 | 256  | 256  | 4096 | 1.2 |

**Radeon AI PRO R9700（gfx1201）**

#### 标准 HGEMM（FP16 入、FP32 累加、FP16 出）— `rocwmma_perf_gemm.cpp`

| M | N | K | V1 顺序 | V2 双缓冲 | 说明 |
|---|---|---|---|---|---|
| 3840 | 4096 | 4096 | **130** TF/s | 118 TF/s | V1 更快 |
| 4096 | 4096 | 4096 | **132** TF/s | 120 TF/s | V1 更快 |
| 8192 | 8192 | 8192 | **133** TF/s | 123 TF/s | V1 更快 |
| 1024 | 4096 | 8192 | **115** TF/s | 114 TF/s | 接近 |
| 4096 | 1024 | 8192 | 105 TF/s | **118** TF/s | V2 更快 |

#### Split-K GEMM — `rocwmma_gemm_splitk.cpp`

| M | N | K | split_k | TFlops/s |
|---|---|---|---|---|
| 512 | 512 | 16384 | 4 | 1.56 |
| 1024 | 1024 | 16384 | 8 | 0.80 |
| 2048 | 2048 | 8192 | 4 | 0.80 |
| 4096 | 4096 | 4096 | 2 | 0.80 |

#### FP8 量化 GEMM — `rocwmma_gemm_quantized.cpp`

| M | N | K | TFlops/s |
|---|---|---|---|
| 3840 | 4096 | 4096 | 2.39 |
| 4096 | 4096 | 4096 | 2.39 |
| 8192 | 8192 | 8192 | 5.13 |

#### MX 格式 GEMM — `rocwmma_mx_gemm.cpp`（gfx1201 可运行；程序可能打印 “gfx950 native”）

| M | N | K | MX_BLOCK | TFlops/s |
|---|---|---|---|---|
| 3840 | 4096 | 4096 | 32 | 3.09 |
| 4096 | 4096 | 4096 | 32 | 3.09 |
| 8192 | 8192 | 8192 | 32 | 5.73 |

#### GEMM + 融合 Epilogue — `rocwmma_gemm_multi_d.cpp`

| M | N | K | ReLU | TFlops/s |
|---|---|---|---|---|
| 3840 | 4096 | 4096 | No | 14.6 |
| 4096 | 4096 | 4096 | Yes | 13.8 |
| 8192 | 8192 | 8192 | Yes | 16.8 |

#### 批量 GEMM — `rocwmma_batched_gemm.cpp`

| batch | M | N | K | TFlops/s(单 batch) | 总有效 TFlops/s |
|---|---|---|---|---|---|
| 4 | 1024 | 1024 | 1024 | 15.4 | 61.8 |
| 8 | 512 | 512 | 2048 | 9.15 | 73.2 |
| 16 | 256 | 256 | 4096 | 3.85 | 61.6 |

#### 分组 GEMM — `rocwmma_grouped_gemm.cpp`

| G | 各组尺寸 | 总 TFlops/s |
|---|---|---|
| 4 | 256/512/128/768 × 4096 × 4096 (混合) | 2.65 |
| 4 | 1024 × 4096 × 4096 (均匀) | 2.66 |
| 3 | 512×1024/2048/4096 × 2048 | 1.38 |

#### Stream-K GEMM — `rocwmma_streamk_gemm.cpp`（样例内 num_cus=32）

| M | N | K | TFlops/s |
|---|---|---|---|
| 512 | 512 | 4096 | 2.51 |
| 1024 | 1024 | 4096 | 4.90 |
| 2048 | 2048 | 4096 | 5.63 |
| 4096 | 4096 | 4096 | 5.52 |

#### FlatMM — `rocwmma_flatmm.cpp`

| M | N | K | LDS | TFlops/s |
|---|---|---|---|---|
| 128 | 4096 | 4096 | 8 KB（仅 A） | 3.10 |
| 256 | 4096 | 4096 | 8 KB | 2.40 |
| 512 | 4096 | 4096 | 8 KB | 2.46 |
| 1024 | 4096 | 4096 | 8 KB | 2.58 |
| 4096 | 4096 | 4096 | 8 KB | 2.65 |

#### 批量张量收缩 — `rocwmma_batched_contraction.cpp`

| G | M | N | K | TFlops/s(per group) |
|---|---|---|---|---|
| 4 | 1024 | 1024 | 1024 | 0.22 |
| 8 | 512 | 512 | 2048 | 0.21 |
| 16 | 256 | 256 | 4096 | 0.20 |

---

### 11.2 注意力类

**MI355X（gfx950）**

#### Flash Attention 前向

`rocwmma_fmha_fwd.cpp`

| B | H | Sq | Sk | D | Causal | TFlops/s |
|---|---|---|---|---|---|---|
| 4 | 32 | 2048 | 2048 | 128 | No  | **75.1** |
| 4 | 32 | 2048 | 2048 | 128 | Yes | 74.8 |
| 1 | 32 | 4096 | 4096 | 128 | Yes | **88.4** |

#### 稀疏注意力（Jenga LUT + Flash Attention）

`rocwmma_sparse_attn.cpp`

| B | H | Sq/Sk | D | 密度 | 有效块对 | TFlops/s |
|---|---|---|---|---|---|---|
| 4 | 32 | 2048 | 128 | 79.4% | 813/1024   | 59.8 |
| 4 | 32 | 4096 | 128 | 27.5% | 1125/4096  | 68.9 |
| 4 | 32 | 8192 | 128 | 14.3% | 2341/16384 | **71.6** |

**Radeon AI PRO R9700（gfx1201）**

#### Flash Attention 前向 — `rocwmma_fmha_fwd.cpp`

| B | H | Sq | Sk | D | Causal | TFlops/s |
|---|---|---|---|---|---|---|
| 4 | 32 | 2048 | 2048 | 128 | No | 19.3 |
| 4 | 32 | 2048 | 2048 | 128 | Yes | **36.4** |
| 1 | 32 | 4096 | 4096 | 128 | Yes | **36.8** |

#### 稀疏注意力 — `rocwmma_sparse_attn.cpp`

| B | H | Sq/Sk | D | 密度 | 有效块对 | TFlops/s |
|---|---|---|---|---|---|---|
| 4 | 32 | 2048 | 128 | 79.4% | 813/1024 | 6.28 |
| 4 | 32 | 4096 | 128 | 27.5% | 1125/4096 | 5.42 |
| 4 | 32 | 8192 | 128 | 14.3% | 2341/16384 | 5.62 |

---

### 11.3 规范化类

**MI355X（gfx950）**

#### LayerNorm 2D（Welford在线方差 + 向量化IO）

`rocwmma_layernorm2d.cpp`

| M | N | 时间 | 带宽 |
|---|---|---|---|
| 3328 | 1024 | 0.017 ms | 801 GB/s |
| 3328 | 2048 | 0.015 ms | 1772 GB/s |
| 3328 | 4096 | 0.021 ms | **2613 GB/s** |
| 3328 | 8192 | 0.031 ms | **3538 GB/s** |

#### RMSNorm 2D

`rocwmma_rmsnorm2d.cpp`

| M | N | 操作 | 带宽 |
|---|---|---|---|
| 3328 | 4096 | RMSNorm     | **4787 GB/s** |
| 3328 | 4096 | Add+RMSNorm | **5303 GB/s** |
| 3328 | 8192 | RMSNorm     | 4737 GB/s |

#### Smooth Quantization（FP16 → INT8）

`rocwmma_smoothquant.cpp`

| M | N | 带宽 |
|---|---|---|
| 3328 | 1024 | 1213 GB/s |
| 3328 | 2048 | 2999 GB/s |
| 3328 | 4096 | **3929 GB/s** |
| 3328 | 8192 | 4069 GB/s |

**Radeon AI PRO R9700（gfx1201）**

#### LayerNorm 2D — `rocwmma_layernorm2d.cpp`

| M | N | 带宽 |
|---|---|---|
| 3328 | 1024 | 238 GB/s |
| 3328 | 2048 | 480 GB/s |
| 3328 | 4096 | **548** GB/s |
| 3328 | 8192 | 426 GB/s |

#### RMSNorm 2D — `rocwmma_rmsnorm2d.cpp`

| M | N | 操作 | 带宽 |
|---|---|---|---|
| 3328 | 4096 | RMSNorm | **1368** GB/s |
| 3328 | 4096 | Add+RMSNorm | 487 GB/s |
| 3328 | 8192 | RMSNorm | 477 GB/s |

#### SmoothQuant — `rocwmma_smoothquant.cpp`

| M | N | 带宽 |
|---|---|---|
| 3328 | 1024 | 328 GB/s |
| 3328 | 2048 | 1010 GB/s |
| 3328 | 4096 | **1249** GB/s |
| 3328 | 8192 | 509–551 GB/s |

---

### 11.4 MoE 类

**MI355X（gfx950）**

#### TopK Softmax（warp-per-token）

`rocwmma_topk_softmax.cpp`

| tokens | experts | topk | 带宽 |
|---|---|---|---|
| 3328 | 8  | 2 | 25.4 GB/s |
| 3328 | 16 | 2 | 40.2 GB/s |
| 3328 | 32 | 5 | 56.1 GB/s |
| 3328 | 64 | 5 | 89.7 GB/s |

#### MoE Token 排序（计数排序）

`rocwmma_moe_sorting.cpp`

| tokens | experts | topk | 带宽 |
|---|---|---|---|
| 3328 | 8   | 4 | 3.26 GB/s |
| 3328 | 32  | 4 | 4.94 GB/s |
| 3328 | 128 | 8 | 7.65 GB/s |

#### MoE Smooth Quantization（Per-expert scale）

`rocwmma_moe_smoothquant.cpp`

| tokens | hidden | experts | topk | 带宽 |
|---|---|---|---|---|
| 3328 | 4096 | 8  | 2 | **2800 GB/s** |
| 3328 | 4096 | 32 | 5 | 2259 GB/s |
| 3328 | 8192 | 64 | 5 | 1575 GB/s |

#### 全融合 MoE（Sort + Gate/Up GEMM + SiLU + Down GEMM）

`rocwmma_fused_moe.cpp`

| tokens | hidden | inter | experts | topk | TFlops/s |
|---|---|---|---|---|---|
| 3328 | 4096 | 14336 | 8 | 2 | **110** |

**Radeon AI PRO R9700（gfx1201）**

#### TopK Softmax — `rocwmma_topk_softmax.cpp`

| tokens | experts | topk | 带宽 |
|---|---|---|---|
| 3328 | 8 | 2 | 11.6 GB/s |
| 3328 | 16 | 2 | 17.3 GB/s |
| 3328 | 32 | 5 | 24.2 GB/s |
| 3328 | 64 | 5 | 40.1 GB/s |

#### MoE Token 排序 — `rocwmma_moe_sorting.cpp`

| tokens | experts | topk | 带宽 |
|---|---|---|---|
| 3328 | 8 | 4 | 7.67 GB/s |
| 3328 | 32 | 4 | 7.50 GB/s |
| 3328 | 64 | 5 | 8.64 GB/s |
| 3328 | 128 | 8 | 12.4 GB/s |

#### MoE SmoothQuant — `rocwmma_moe_smoothquant.cpp`

| tokens | hidden | experts | topk | 带宽 |
|---|---|---|---|---|
| 3328 | 4096 | 8 | 2 | **940** GB/s |
| 3328 | 4096 | 32 | 5 | 430 GB/s |
| 3328 | 8192 | 64 | 5 | 483 GB/s |

#### 全融合 MoE — `rocwmma_fused_moe.cpp`

| tokens | hidden | inter | experts | topk | TFlops/s |
|---|---|---|---|---|---|
| 3328 | 4096 | 14336 | 8 | 2 | 10.9 |
| 3328 | 4096 | 14336 | 32 | 5 | 9.56 |
| 3328 | 7168 | 2048 | 64 | 6 | 17.3 |

---

### 11.5 Reduction & Elementwise 类

**MI355X（gfx950）**

#### Row Reduction（Sum/Max）

`rocwmma_reduce.cpp`

| M | N | ReduceSum | ReduceMax |
|---|---|---|---|
| 3328 | 1024 | 2046 GB/s | 1938 GB/s |
| 3328 | 2048 | 3903 GB/s | 3710 GB/s |
| 3328 | 4096 | **6802 GB/s** | **6412 GB/s** |
| 3328 | 8192 | 5741 GB/s | 5598 GB/s |

#### Elementwise Operations

`rocwmma_elementwise.cpp`

| M | N | 操作 | 带宽 |
|---|---|---|---|
| 3840 | 4096 | Add2D       | 2080 GB/s |
| 3840 | 4096 | Square2D    | 1622 GB/s |
| 3840 | 4096 | Transpose2D | **2588 GB/s** |

**Radeon AI PRO R9700（gfx1201）**

#### Row Reduction — `rocwmma_reduce.cpp`

| M | N | ReduceSum | ReduceMax |
|---|---|---|---|
| 3328 | 1024 | 660 GB/s | 646 GB/s |
| 3328 | 2048 | 1333 GB/s | 1292 GB/s |
| 3328 | 4096 | **1747** GB/s | 1715 GB/s |
| 3328 | 8192 | 1987 GB/s | 1987 GB/s |

#### Elementwise — `rocwmma_elementwise.cpp`（M=3840, N=4096）

| 操作 | 带宽 |
|---|---|
| Add2D | 375 GB/s |
| Square2D | 1367 GB/s |
| Transpose2D | 260 GB/s |

---

### 11.6 卷积 & 数据布局类

**MI355X（gfx950）**

#### 分组卷积前向（Im2col + rocWMMA GEMM）

`rocwmma_grouped_conv_fwd.cpp`

| N | H×W | G | Cin | Cout | K | TFlops/s |
|---|---|---|---|---|---|---|
| 8 | 56×56 | 1 | 64  | 64  | 3×3 | 8.60 |
| 8 | 28×28 | 1 | 128 | 128 | 3×3 | 8.82 |
| 8 | 14×14 | 1 | 256 | 256 | 3×3 | 6.89 |
| 8 | 7×7   | 1 | 512 | 512 | 3×3 | 3.98 |

#### 张量置换（NCHW ↔ NHWC）

`rocwmma_permute.cpp`

| 操作 | 形状 | 带宽 |
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

**Radeon AI PRO R9700（gfx1201）**

#### 分组卷积前向 — `rocwmma_grouped_conv_fwd.cpp`

| N | H×W | G | Cin | Cout | K | TFlops/s |
|---|---|---|---|---|---|---|
| 8 | 56×56 | 1 | 64 | 64 | 3×3 | 0.18 |
| 8 | 28×28 | 1 | 128 | 128 | 3×3 | 0.83 |
| 8 | 14×14 | 1 | 256 | 256 | 3×3 | 2.11 |
| 8 | 7×7 | 1 | 512 | 512 | 3×3 | 1.47 |

#### Permute — `rocwmma_permute.cpp`

| 操作 | 形状 | 带宽 |
|---|---|---|
| Transpose2D | 3840×4096 | 515 GB/s |
| Transpose2D | 4096×4096 | 518 GB/s |
| Transpose2D | 8192×8192 | 171 GB/s |
| NCHW→NHWC | N=32, C=128, H=W=64 | **534** GB/s |

#### 3D Pooling — `rocwmma_pooling.cpp`

| N | D×H×W | C | K | MaxPool | AvgPool |
|---|---|---|---|---|---|
| 2 | 16×28×28 | 256 | 2×2×2 | **552** GB/s | 528 GB/s |
| 2 | 8×56×56 | 128 | 3×3×3 | 48.4 GB/s | 47.1 GB/s |

---

### 11.7 CK Tile vs rocWMMA 对比（HGEMM，M=N=K=4096）

| 实现 | TFlops/s | 说明 |
|---|---|---|
| CK Tile Basic (V1 pipeline) | **640** | 256×256 block tile，gfx9专用编译器优化 |
| rocWMMA V1 (sequential)     | 393 | 128×128 block tile，标准双缓冲 |
| rocWMMA V2 (double-buffer)  | **536** | 128×128 block tile，双缓冲LDS |
| 提升倍数 (V2 vs V1)          | 1.36× | 双缓冲效果 |
| rocWMMA V2 vs CK Tile       | 83.7% | 差距主要来自 block tile 尺寸和编译器优化 |

> **注：** CK Tile 使用 256×256 更大的 block tile（需 64KB LDS）并应用
> `-mllvm -enable-noalias-to-md-conversion=0` 等底层编译器 flag，
> rocWMMA 使用标准 HIP 编译路径（128×128 block tile，16KB LDS）。

### 11.8 复现 benchmark（MI355X 与 R9700）

```bash
cd /home/optimized_rocwmma_samples   # 或你的克隆路径

# MI355X / CDNA（示例目标）
./build.sh -g gfx950
./benchmark.sh -b build --csv -o benchmark_results_mi355x.txt --timeout 300

# Radeon AI PRO R9700 / RDNA 4（gfx1201）
./build.sh -g gfx1201
./benchmark.sh -b build --csv -o benchmark_results_r9700_gfx1201.txt --timeout 300
```

需要 CMake ≥3.16 与完整 ROCm（`HIPConfig.cmake` 位于 `$ROCM_PATH/lib/cmake/hip`）。工程会将 `/opt/rocm`（或 `$ROCM_PATH`）置于 `CMAKE_PREFIX_PATH` 前部，以便 `find_package(HIP)` 在常见环境下成功。

---

## 附录：文件索引

| 文件 | 对应 CK Tile | 核心优化 |
|---|---|---|
| `rocwmma_perf_gemm.cpp` | 03_gemm | 双缓冲 LDS、协作加载、Warp Tile |
| `rocwmma_gemm_splitk.cpp` | 03_gemm splitk | FP32 workspace 两阶段 split-K |
| `rocwmma_gemm_quantized.cpp` | 38_block_scale_gemm | FP8 tensor-scale epilogue |
| `rocwmma_batched_gemm.cpp` | 16_batched_gemm | blockIdx.z batch 维 |
| `rocwmma_gemm_multi_d.cpp` | 19_gemm_multi_d | 融合 bias/scale/ReLU epilogue |
| `rocwmma_batched_contraction.cpp` | 41_batched_contraction | 收缩 → 批量 GEMM |
| `rocwmma_grouped_gemm.cpp` | 17_grouped_gemm | 扁平网格 + 二分查找 dispatch |
| `rocwmma_streamk_gemm.cpp` | 40_streamk_gemm | Stream-K + 原子 workspace 归约 |
| `rocwmma_flatmm.cpp` | 18_flatmm | B 绕过 LDS，寄存器直接加载 |
| `rocwmma_mx_gemm.cpp` | 42_mx_gemm | MX E8M0 block-scale FP8 |
| `rocwmma_fmha_fwd.cpp` | 01_fmha | Flash Attention online softmax |
| `rocwmma_sparse_attn.cpp` | 50_sparse_attn | Jenga LUT 稀疏块跳过 |
| `rocwmma_layernorm2d.cpp` | 02_layernorm2d | Welford 在线方差 + 向量化 |
| `rocwmma_rmsnorm2d.cpp` | 10/11_rmsnorm2d | RMS 单遍 + Add+RMS 融合 |
| `rocwmma_smoothquant.cpp` | 12_smoothquant | 两遍量化：abs-max + 量化 |
| `rocwmma_moe_smoothquant.cpp` | 14_moe_smoothquant | Per-expert smooth scale |
| `rocwmma_fused_moe.cpp` | 15_fused_moe | Sort+GateUp+SiLU+Down 全融合 |
| `rocwmma_reduce.cpp` | 05_reduce | 蝶形 warp reduce + LDS block reduce |
| `rocwmma_elementwise.cpp` | 21_elementwise | float4 向量化，LDS padding transpose |
| `rocwmma_topk_softmax.cpp` | 09_topk_softmax | warp-per-row softmax + top-K |
| `rocwmma_moe_sorting.cpp` | 13_moe_sorting | 计数排序三阶段 |
| `rocwmma_img2col_gemm.cpp` | 04_img2col | 虚拟 im2col + rocWMMA GEMM |
| `rocwmma_grouped_conv_fwd.cpp` | 20_grouped_convolution | 分组 im2col + 逐组 GEMM |
| `rocwmma_pooling.cpp` | 36_pooling | NDHWC 向量化窗口规约 |
| `rocwmma_permute.cpp` | 06/35_permute | LDS padding 2D tile transpose |
| `rocwmma_tile_distr_reg_map.cpp` | 51_tile_distr | Host 诊断：fragment 形状打印 |

---

## 12. CK Tile vs rocWMMA 完整性能对比（MI355X / gfx950）

本节提供在同一台 MI355X 上使用相同 warmup/repeat 参数运行 CK Tile 原版与 rocWMMA 移植版的实测数据对比。

**测试条件：**

| 项目 | CK Tile | rocWMMA |
|---|---|---|
| 编译器 | AMD Clang 22.0 | AMD Clang 22.0 |
| ROCm | 7.2.0 | 7.2.0 |
| GPU | MI355X (gfx950) | MI355X (gfx950) |
| Warmup | 5 | 5 |
| Repeat | 20 | 20 |
| CK Tile 特殊 flag | `-mllvm -enable-noalias-to-md-conversion=0` | 无 |
| Block Tile (GEMM) | 256×256 (CK Tile 默认) | 128×128 (rocWMMA 限制) |

> **注意：** CK Tile GEMM 使用 256×256 的 Block Tile，需要 64 KB LDS，
> 并依赖底层 LLVM pass 和 `noalias` 转换禁用来实现最优 ILP。
> rocWMMA 使用标准 HIP 编译路径，Block Tile 限制在 128×128（16 KB LDS）。
> 这是两者 GEMM 性能差距的主要来源。

---

### 12.1 GEMM 类对比

#### HGEMM（FP16，Col-major A，Row-major B）

`tile_example_gemm_basic` (V1 pipeline) vs `rocwmma_perf_gemm` (V1/V2)

| M | N | K | CK Tile V1 | rocWMMA V1 | rocWMMA V2 | rocWMMA/CK比(V2) |
|---|---|---|---|---|---|---|
| 3840 | 4096 | 4096 | **634 TF/s** | 365 TF/s | 492 TF/s | 77.6% |
| 4096 | 4096 | 4096 | **647 TF/s** | 391 TF/s | 536 TF/s | 82.8% |
| 8192 | 8192 | 8192 | **773 TF/s** | 464 TF/s | 592 TF/s | 76.6% |
| 1024 | 4096 | 8192 | 219 TF/s | 197 TF/s | **275 TF/s** | **125.6%** |
| 4096 | 1024 | 8192 | 218 TF/s | 197 TF/s | **281 TF/s** | **128.9%** |

> rocWMMA V2 在小 M/N 的 thin-tile 场景下超越 CK Tile V1，因为双缓冲
> 更有效地隐藏了内存延迟（K 大时 CK Tile 优势体现在更大的 block tile）。

#### HGEMM Universal（异步 pipeline）

`tile_example_gemm_universal` (async) vs `rocwmma_perf_gemm` (V2 double-buffer)

| M | N | K | CK Tile Async | rocWMMA V2 | rocWMMA/CK比 |
|---|---|---|---|---|---|
| 4096 | 4096 | 4096 | **757 TF/s** | 536 TF/s | 70.8% |
| 8192 | 8192 | 8192 | **931 TF/s** | 592 TF/s | 63.6% |

> CK Tile Universal 使用异步内存拷贝指令（`cp.async` 等价的 CDNA 指令），
> 进一步解耦内存加载与计算，是 rocWMMA 未暴露的硬件能力。

#### Split-K GEMM（FP16）

`tile_example_gemm_splitk_two_stage` vs `rocwmma_gemm_splitk`

| M | N | K | split_k | CK Tile | rocWMMA | 比值 |
|---|---|---|---|---|---|---|
| 512  | 512  | 16384 | 4 | **173 TF/s** | 18.8 TF/s | 10.9% |
| 1024 | 1024 | 16384 | 8 | **174 TF/s** | 26.9 TF/s | 15.5% |
| 4096 | 4096 | 4096  | 2 | **155 TF/s** | 26.7 TF/s | 17.2% |

> CK Tile split-K 使用持久化 kernel（persistent）和 CShuffleEpilogue，
> rocWMMA 版本使用逐 CTA 调用 + workspace reduce，调用开销较大。

#### Batched GEMM（FP16，默认尺寸）

`tile_example_batched_gemm` vs `rocwmma_batched_gemm`

| 配置 | CK Tile | rocWMMA | 比值 |
|---|---|---|---|
| 默认 (M≈1024,N≈1024) | **341 TF/s** | 61.4 TF/s (per batch) | 18% per-batch |

> 注：CK Tile batched GEMM 默认尺寸与 rocWMMA 不完全一致，
> 有效比较需对齐尺寸。

#### GEMM + MultiD Epilogue（FP16）

`tile_example_gemm_multi_d_fp16` vs `rocwmma_gemm_multi_d`

| 配置 | CK Tile | rocWMMA |
|---|---|---|
| 默认 (M=3840,N=4096) | **924 TF/s** | 155 TF/s |

> CK Tile 使用 `CShuffleEpilogue` + 256×256 block tile 实现极高效率，
> rocWMMA 版本简化为寄存器内逐元素操作，调用开销更低但算力利用率差。

---

### 12.2 规范化类对比

#### LayerNorm 2D（FP16 in/out）

`tile_example_layernorm2d_fwd` vs `rocwmma_layernorm2d`

| M | N | CK Tile | rocWMMA | 比值 |
|---|---|---|---|---|
| 3328 | 1024 | **2036 GB/s** | 801 GB/s | 39.3% |
| 3328 | 2048 | **2340 GB/s** | 1772 GB/s | 75.7% |
| 3328 | 4096 | **3557 GB/s** | 2613 GB/s | 73.5% |
| 3328 | 8192 | **3644 GB/s** | 3538 GB/s | 97.1% |

> N=8192 时 rocWMMA 接近 CK Tile（97.1%），说明向量化宽度是关键瓶颈。
> N=1024 时差距大（39%），因为 CK Tile 内核使用更激进的线程组织。

#### RMSNorm 2D（FP16 in/out）

`tile_rmsnorm2d_fwd` vs `rocwmma_rmsnorm2d`

| M | N | CK Tile | rocWMMA | 比值 |
|---|---|---|---|---|
| 3328 | 1024 | **3469 GB/s** | 1830 GB/s | 52.8% |
| 3328 | 2048 | **4463 GB/s** | 4110 GB/s | 92.1% |
| 3328 | 4096 | **5576 GB/s** | 4787 GB/s | 85.9% |
| 3328 | 8192 | **6443 GB/s** | 4737 GB/s | 73.5% |

> CK Tile RMSNorm 使用 codegen 生成针对 gfx950 的最优线程组织，
> rocWMMA 版本使用通用的 Wave64 蝶形规约。

#### Smooth Quantization（FP16 → INT8）

`tile_smoothquant` vs `rocwmma_smoothquant`

| M | N | CK Tile | rocWMMA | 比值 |
|---|---|---|---|---|
| 3328 | 1024 | **2066 GB/s** | 1213 GB/s | 58.7% |
| 3328 | 2048 | **2715 GB/s** | 2999 GB/s | **110.4%** |
| 3328 | 4096 | **3288 GB/s** | 3929 GB/s | **119.5%** |
| 3328 | 8192 | 3209 GB/s | **4069 GB/s** | **126.8%** |

> rocWMMA 在 N≥2048 时超越 CK Tile SmoothQuant！
> 原因：rocWMMA 使用 `int64_t` 对齐写入（8×INT8 一次写），
> 而 CK Tile 版本对 INT8 输出的向量化支持有限制。

---

### 12.3 MoE 类对比

#### TopK Softmax

`tile_example_topk_softmax` vs `rocwmma_topk_softmax`

| tokens | experts | topk | CK Tile (ms) | rocWMMA (ms) | 比值(时间越小越好) |
|---|---|---|---|---|---|
| 3328 | 8  | 2 | **0.00265 ms** | 0.00419 ms | CK 快 1.58× |
| 3328 | 16 | 2 | **0.00294 ms** | 0.00397 ms | CK 快 1.35× |
| 3328 | 32 | 5 | **0.00399 ms** | 0.00617 ms | CK 快 1.55× |
| 3328 | 64 | 5 | 0.00544 ms | **0.00623 ms** | 接近（CK快 1.15×） |

#### MoE Token Sorting

`tile_example_moe_sorting` vs `rocwmma_moe_sorting`

| tokens | experts | topk | CK Tile (ms) | rocWMMA (ms) | 比值 |
|---|---|---|---|---|---|
| 3328 | 8   | 4 | **0.0132 ms** | 0.0493 ms | CK 快 3.7× |
| 3328 | 32  | 4 | **0.0125 ms** | 0.0332 ms | CK 快 2.7× |
| 3328 | 64  | 4 | **0.0122 ms** | 0.0397 ms | CK 快 3.3× |
| 3328 | 128 | 4 | **0.0121 ms** | 0.0439 ms | CK 快 3.6× |

> CK Tile MoE Sorting 使用专门的 `GpuSort` 算法（基于 radix sort），
> rocWMMA 使用三阶段计数排序（histogram+prefix+scatter），后者对小 expert 数有额外开销。

---

### 12.4 归约与数据搬运类对比

#### Row-wise Reduce（FP16 → FP32）

`tile_example_reduce` (reduce over N, keep C) vs `rocwmma_reduce`

> **注意：** CK Tile reduce 是对 `(N,H,W)` 进行规约保留 `C`，
> 与 rocWMMA reduce 的行规约（保留列）语义相同，但维度组织不同。

| M (被规约维) | N (保留维) | CK Tile | rocWMMA | 比值 |
|---|---|---|---|---|
| 3328 | 1024 | **1113 GB/s** | 2046 GB/s | rocWMMA 快 **1.84×** |
| 3328 | 2048 | **1230 GB/s** | 3903 GB/s | rocWMMA 快 **3.17×** |
| 3328 | 4096 | **1114 GB/s** | 6802 GB/s | rocWMMA 快 **6.10×** |
| 3328 | 8192 | **1166 GB/s** | 5741 GB/s | rocWMMA 快 **4.92×** |

> rocWMMA reduce 性能远超 CK Tile 版本！
> 原因：CK Tile reduce 的默认配置使用 4D tensor (N,H,W,C)，
> 规约维度为 N×H×W，与 rocWMMA 的行规约不完全等价（计算量不同）。
> 此处数字为参考对比，实际工作负载需具体验证。

#### 2D Transpose / Permute

`tile_example_permute` vs `rocwmma_permute`

| M | N | CK Tile | rocWMMA | 比值 |
|---|---|---|---|---|
| 3840 | 4096 | 341 GB/s | **2463 GB/s** | rocWMMA 快 **7.2×** |
| 4096 | 4096 | 416 GB/s | **2470 GB/s** | rocWMMA 快 **5.9×** |
| 8192 | 8192 | 437 GB/s | **2732 GB/s** | rocWMMA 快 **6.3×** |

> rocWMMA 版本使用 LDS padding 的 tiled transpose（32×32 tile），
> CK Tile 默认配置使用小 shape（`2,3,4`），在大矩阵上未针对性优化。
> 使用 CK Tile 的 `PERMUTE_USE_ALTERNATIVE_IMPL`（matrix-core swizzle）
> 可获得更高性能，但需要额外编译 flag。

---

### 12.5 Elementwise、Pooling、Batched Contraction 对比

#### Elementwise 2D Add（FP16，三张量 C = A + B）

`tile_example_elementwise` vs `rocwmma_elementwise`

| M | N | CK Tile | rocWMMA | 比值 |
|---|---|---|---|---|
| 3840 | 4096 | **3358 GB/s** | 2080 GB/s | CK Tile +61.4% |
| 4096 | 4096 | **3407 GB/s** | — | — |
| 8192 | 8192 | **2670 GB/s** | — | — |

> CK Tile elementwise 使用 `GenericPermute` 框架实现高效的 tile 分发，
> rocWMMA 版本使用 `float4` 向量化但每行独立加载，L2 复用率较低。

#### 3D Pooling（FP16，NDHWC layout）

`tile_example_pool3d` vs `rocwmma_pooling`

| N | D×H×W | C | Window | CK Tile | rocWMMA | 比值 |
|---|---|---|---|---|---|---|
| 2 | 16×28×28 | 256 | 2×2×2 | **497 GB/s** | 2123 GB/s | rocWMMA **4.3×** |
| 2 | 8×56×56 | 128 | 3×3×3 | **79 GB/s** | 196 GB/s | rocWMMA **2.5×** |

> rocWMMA pooling 在大窗口场景大幅超越 CK Tile！
> 原因：rocWMMA 版本针对 NDHWC layout 使用 `float4` 向量化 C 维读取，
> 而 CK Tile 默认 Pool3D 使用行优先遍历，对大 C 维场景不友好。

#### 批量张量收缩（FP16）

`tile_example_batched_contraction` vs `rocwmma_batched_contraction`

| G | M | N | K | CK Tile | rocWMMA | 比值 |
|---|---|---|---|---|---|---|
| 4 | 1024 | 1024 | 1024 | **181 TF/s** | 8.1 TF/s | CK Tile **22×** |
| 8 | 512  | 512  | 2048 | **186 TF/s** | 3.8 TF/s | CK Tile **49×** |

> CK Tile batched_contraction 使用持久化 kernel + CShuffleEpilogue，
> rocWMMA 版本使用简单的 blockIdx.z 批量调度，在小 M/N 时调度开销占主导。

---

### 12.6 综合对比总结

| 操作类型 | 典型尺寸 | CK Tile | rocWMMA V2 | 胜者 | 差距说明 |
|---|---|---|---|---|---|
| **HGEMM 方阵** | 8192³ | **773 TF/s** | 592 TF/s | CK Tile | 256×256 block tile + async pipeline |
| **HGEMM thin** | 1024×4096×8192 | 219 TF/s | **275 TF/s** | rocWMMA | 双缓冲延迟隐藏更有效 |
| **HGEMM async** | 4096³ | **757 TF/s** | 536 TF/s | CK Tile | cp.async 硬件指令 |
| **LayerNorm** | 3328×4096 | **3557 GB/s** | 2613 GB/s | CK Tile | codegen 线程优化 |
| **RMSNorm** | 3328×4096 | **5576 GB/s** | 4787 GB/s | CK Tile | codegen |
| **SmoothQuant** | 3328×4096 | 3288 GB/s | **3929 GB/s** | rocWMMA | int64_t 对齐写入 |
| **SmoothQuant** | 3328×8192 | 3209 GB/s | **4069 GB/s** | rocWMMA | +26.8% |
| **Row Reduce** | 3328×4096 | 1114 GB/s | **6802 GB/s** | rocWMMA | warp butterfly 更高效 |
| **Transpose** | 8192² | 437 GB/s | **2732 GB/s** | rocWMMA | LDS tiled transpose |
| **TopK Softmax** | 3328,32exp,5k | **0.40 ms** | 0.62 ms | CK Tile | warp 级优化 |
| **MoE Sorting** | 3328,64exp,4k | **0.012 ms** | 0.040 ms | CK Tile | radix sort vs 计数排序 |
| **Elementwise Add** | 3840×4096 | **3358 GB/s** | 2080 GB/s | CK Tile | tile 分发框架 +61% |
| **Pool3D MaxPool** | N=2,D×H×W=16×28×28,K=2³ | 497 GB/s | **2123 GB/s** | rocWMMA | NDHWC float4 向量化 +327% |
| **Batched Contraction** | G=4,M=N=K=1024 | **181 TF/s** | 8.1 TF/s | CK Tile | persistent+CShuflle vs 简单批量 |

**性能差距根本原因分析：**

| 原因 | CK Tile 优势场景 | rocWMMA 优势场景 |
|---|---|---|
| **Block Tile 尺寸** | 256×256 (GEMM)，更高 L2 复用率 | 128×128，LDS 不足时更灵活 |
| **异步内存访问** | cp.async 指令隐藏全局内存延迟 | 标准 load + barrier 双缓冲 |
| **Codegen 实例化** | 针对每个 (M,N,K,dtype) 生成最优参数 | 运行时通用参数 |
| **编译器 Pass** | `-enable-noalias-to-md-conversion=0` 等优化 | 标准 HIP 编译 |
| **向量化写出** | 大量元素场景优化好 | int64_t 对齐写出 (INT8)，小元素更高效 |
| **线程组织** | 规范化算子经过严格 codegen 调优 | 通用 Wave64 蝶形规约 |

**结论：**
- CK Tile 在 **计算密集型**（大方阵 GEMM）和 **codegen 规范化**（LayerNorm、RMSNorm）场景绝对领先
- rocWMMA 在 **内存密集型**（Transpose、Row Reduce）和部分**量化**（SmoothQuant N≥2048）场景表现更优
- 对于生产环境建议使用 CK Tile；rocWMMA 样例适合快速原型、教学与 RDNA4 移植

---

## 13. AMD Radeon AI PRO R9700（gfx1201）实测结果

下列数据在 **AMD Radeon AI PRO R9700**（gfx1201）上测量，ROCm **7.2.0**，命令：`./build.sh -g gfx1201` 与 `./benchmark.sh -b build --csv -o benchmark_results_r9700_gfx1201.txt`（每 sample 超时 300s）。报告日期：**2026-04-07**。原始输出见本目录 `benchmark_results_r9700_gfx1201.txt` / `.csv`。

**HGEMM 流水线说明（RDNA 4）：** 与 MI355X（gfx950）不同，在 gfx1201 上多数默认方阵尺寸下 **`rocwmma_perf_gemm` 的 V1 顺序流水线快于 V2 双缓冲**；仅在部分扁长 tile（如 M=4096, N=1024, K=8192）V2 更优。请按实际算子调参。

### 13.1 GEMM 类

#### 标准 HGEMM（FP16 入、FP32 累加、FP16 出）— `rocwmma_perf_gemm.cpp`

| M | N | K | V1 顺序 | V2 双缓冲 | 说明 |
|---|---|---|---|---|---|
| 3840 | 4096 | 4096 | **130** TF/s | 118 TF/s | V1 更快 |
| 4096 | 4096 | 4096 | **132** TF/s | 120 TF/s | V1 更快 |
| 8192 | 8192 | 8192 | **133** TF/s | 123 TF/s | V1 更快 |
| 1024 | 4096 | 8192 | **115** TF/s | 114 TF/s | 接近 |
| 4096 | 1024 | 8192 | 105 TF/s | **118** TF/s | V2 更快 |

#### Split-K GEMM — `rocwmma_gemm_splitk.cpp`

| M | N | K | split_k | TFlops/s |
|---|---|---|---|---|
| 512 | 512 | 16384 | 4 | 1.56 |
| 1024 | 1024 | 16384 | 8 | 0.80 |
| 2048 | 2048 | 8192 | 4 | 0.80 |
| 4096 | 4096 | 4096 | 2 | 0.80 |

#### FP8 量化 GEMM — `rocwmma_gemm_quantized.cpp`

| M | N | K | TFlops/s |
|---|---|---|---|
| 3840 | 4096 | 4096 | 2.39 |
| 4096 | 4096 | 4096 | 2.39 |
| 8192 | 8192 | 8192 | 5.13 |

#### MX 格式 GEMM — `rocwmma_mx_gemm.cpp`（在 gfx1201 上可运行；程序输出仍标注 “gfx950 native”）

| M | N | K | MX_BLOCK | TFlops/s |
|---|---|---|---|---|
| 3840 | 4096 | 4096 | 32 | 3.09 |
| 4096 | 4096 | 4096 | 32 | 3.09 |
| 8192 | 8192 | 8192 | 32 | 5.73 |

#### GEMM + 融合 Epilogue — `rocwmma_gemm_multi_d.cpp`

| M | N | K | ReLU | TFlops/s |
|---|---|---|---|---|
| 3840 | 4096 | 4096 | No | 14.6 |
| 4096 | 4096 | 4096 | Yes | 13.8 |
| 8192 | 8192 | 8192 | Yes | 16.8 |

#### 批量 GEMM — `rocwmma_batched_gemm.cpp`

| batch | M | N | K | TFlops/s(单 batch) | 总有效 TFlops/s |
|---|---|---|---|---|---|
| 4 | 1024 | 1024 | 1024 | 15.4 | 61.8 |
| 8 | 512 | 512 | 2048 | 9.15 | 73.2 |
| 16 | 256 | 256 | 4096 | 3.85 | 61.6 |

#### 分组 GEMM — `rocwmma_grouped_gemm.cpp`

| G | 各组尺寸 | 总 TFlops/s |
|---|---|---|
| 4 | 256/512/128/768 × 4096 × 4096 (混合) | 2.65 |
| 4 | 1024 × 4096 × 4096 (均匀) | 2.66 |
| 3 | 512×1024/2048/4096 × 2048 | 1.38 |

#### Stream-K GEMM — `rocwmma_streamk_gemm.cpp`（样例内 num_cus=32）

| M | N | K | TFlops/s |
|---|---|---|---|
| 512 | 512 | 4096 | 2.51 |
| 1024 | 1024 | 4096 | 4.90 |
| 2048 | 2048 | 4096 | 5.63 |
| 4096 | 4096 | 4096 | 5.52 |

#### FlatMM — `rocwmma_flatmm.cpp`

| M | N | K | LDS | TFlops/s |
|---|---|---|---|---|
| 128 | 4096 | 4096 | 8 KB（仅 A） | 3.10 |
| 256 | 4096 | 4096 | 8 KB | 2.40 |
| 512 | 4096 | 4096 | 8 KB | 2.46 |
| 1024 | 4096 | 4096 | 8 KB | 2.58 |
| 4096 | 4096 | 4096 | 8 KB | 2.65 |

#### 批量张量收缩 — `rocwmma_batched_contraction.cpp`

| G | M | N | K | TFlops/s(per group) |
|---|---|---|---|---|
| 4 | 1024 | 1024 | 1024 | 0.22 |
| 8 | 512 | 512 | 2048 | 0.21 |
| 16 | 256 | 256 | 4096 | 0.20 |

### 13.2 注意力类

#### Flash Attention 前向 — `rocwmma_fmha_fwd.cpp`

| B | H | Sq | Sk | D | Causal | TFlops/s |
|---|---|---|---|---|---|---|
| 4 | 32 | 2048 | 2048 | 128 | No | 19.3 |
| 4 | 32 | 2048 | 2048 | 128 | Yes | **36.4** |
| 1 | 32 | 4096 | 4096 | 128 | Yes | **36.8** |

#### 稀疏注意力 — `rocwmma_sparse_attn.cpp`

| B | H | Sq/Sk | D | 密度 | 有效块对 | TFlops/s |
|---|---|---|---|---|---|---|
| 4 | 32 | 2048 | 128 | 79.4% | 813/1024 | 6.28 |
| 4 | 32 | 4096 | 128 | 27.5% | 1125/4096 | 5.42 |
| 4 | 32 | 8192 | 128 | 14.3% | 2341/16384 | 5.62 |

### 13.3 规范化类

#### LayerNorm 2D — `rocwmma_layernorm2d.cpp`

| M | N | 带宽 |
|---|---|---|
| 3328 | 1024 | 238 GB/s |
| 3328 | 2048 | 480 GB/s |
| 3328 | 4096 | **548** GB/s |
| 3328 | 8192 | 426 GB/s |

#### RMSNorm 2D — `rocwmma_rmsnorm2d.cpp`

| M | N | 操作 | 带宽 |
|---|---|---|---|
| 3328 | 4096 | RMSNorm | **1368** GB/s |
| 3328 | 4096 | Add+RMSNorm | 487 GB/s |
| 3328 | 8192 | RMSNorm | 477 GB/s |

#### SmoothQuant — `rocwmma_smoothquant.cpp`

| M | N | 带宽 |
|---|---|---|
| 3328 | 1024 | 328 GB/s |
| 3328 | 2048 | 1010 GB/s |
| 3328 | 4096 | **1249** GB/s |
| 3328 | 8192 | 509–551 GB/s |

### 13.4 MoE 类

#### TopK Softmax — `rocwmma_topk_softmax.cpp`

| tokens | experts | topk | 带宽 |
|---|---|---|---|
| 3328 | 8 | 2 | 11.6 GB/s |
| 3328 | 16 | 2 | 17.3 GB/s |
| 3328 | 32 | 5 | 24.2 GB/s |
| 3328 | 64 | 5 | 40.1 GB/s |

#### MoE Token 排序 — `rocwmma_moe_sorting.cpp`

| tokens | experts | topk | 带宽 |
|---|---|---|---|
| 3328 | 8 | 4 | 7.67 GB/s |
| 3328 | 32 | 4 | 7.50 GB/s |
| 3328 | 64 | 5 | 8.64 GB/s |
| 3328 | 128 | 8 | 12.4 GB/s |

#### MoE SmoothQuant — `rocwmma_moe_smoothquant.cpp`

| tokens | hidden | experts | topk | 带宽 |
|---|---|---|---|---|
| 3328 | 4096 | 8 | 2 | **940** GB/s |
| 3328 | 4096 | 32 | 5 | 430 GB/s |
| 3328 | 8192 | 64 | 5 | 483 GB/s |

#### 全融合 MoE — `rocwmma_fused_moe.cpp`

| tokens | hidden | inter | experts | topk | TFlops/s |
|---|---|---|---|---|---|
| 3328 | 4096 | 14336 | 8 | 2 | 10.9 |
| 3328 | 4096 | 14336 | 32 | 5 | 9.56 |
| 3328 | 7168 | 2048 | 64 | 6 | 17.3 |

### 13.5 Reduction、Elementwise、卷积、Permute、Pooling

#### Row Reduction — `rocwmma_reduce.cpp`

| M | N | ReduceSum | ReduceMax |
|---|---|---|---|
| 3328 | 1024 | 660 GB/s | 646 GB/s |
| 3328 | 2048 | 1333 GB/s | 1292 GB/s |
| 3328 | 4096 | **1747** GB/s | 1715 GB/s |
| 3328 | 8192 | 1987 GB/s | 1987 GB/s |

#### Elementwise — `rocwmma_elementwise.cpp`（M=3840, N=4096）

| 操作 | 带宽 |
|---|---|
| Add2D | 375 GB/s |
| Square2D | 1367 GB/s |
| Transpose2D | 260 GB/s |

#### 分组卷积前向 — `rocwmma_grouped_conv_fwd.cpp`

| N | H×W | G | Cin | Cout | K | TFlops/s |
|---|---|---|---|---|---|---|
| 8 | 56×56 | 1 | 64 | 64 | 3×3 | 0.18 |
| 8 | 28×28 | 1 | 128 | 128 | 3×3 | 0.83 |
| 8 | 14×14 | 1 | 256 | 256 | 3×3 | 2.11 |
| 8 | 7×7 | 1 | 512 | 512 | 3×3 | 1.47 |

#### Permute — `rocwmma_permute.cpp`

| 操作 | 形状 | 带宽 |
|---|---|---|
| Transpose2D | 3840×4096 | 515 GB/s |
| Transpose2D | 4096×4096 | 518 GB/s |
| Transpose2D | 8192×8192 | 171 GB/s |
| NCHW→NHWC | N=32, C=128, H=W=64 | **534** GB/s |

#### 3D Pooling — `rocwmma_pooling.cpp`

| N | D×H×W | C | K | MaxPool | AvgPool |
|---|---|---|---|---|---|
| 2 | 16×28×28 | 256 | 2×2×2 | **552** GB/s | 528 GB/s |
| 2 | 8×56×56 | 128 | 3×3×3 | 48.4 GB/s | 47.1 GB/s |

### 13.6 编译与复现（R9700 / gfx1201）

```bash
cd /home/optimized_rocwmma_samples   # 或你的克隆路径

# RDNA 4 — R9700 对应 gfx1201（可用: rocminfo | grep -E "Marketing Name|Name:.*gfx"）
./build.sh -g gfx1201

./benchmark.sh -b build --csv -o benchmark_results_r9700_gfx1201.txt --timeout 300
```

需要 CMake ≥3.16 与完整 ROCm（`HIPConfig.cmake` 位于 `$ROCM_PATH/lib/cmake/hip`）。工程会将 `/opt/rocm`（或环境变量 `ROCM_PATH`）置于 `CMAKE_PREFIX_PATH` 前部，以便 `find_package(HIP)` 在常见 Linux 安装下可直接成功。

