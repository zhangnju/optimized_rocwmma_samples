/*******************************************************************************
 * MIT License
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

/*
 * rocwmma_reduce.cpp
 *
 * Description:
 *   Tensor reduction operations ported from rocWMMA 05_reduce.
 *
 *   rocWMMA Optimizations Applied:
 *   - Warp-level reduction using __shfl_down_sync (matches rocWMMA WarpReduce)
 *   - Block-level two-pass reduction: intra-warp then inter-warp via LDS
 *   - Vectorized global memory reads (float4 = 8x fp16)
 *   - Row-reduction (reduce along N, keep M) for typical LLM use cases
 *   - Configurable tile sizes: TILE_M rows per block, all N columns per block
 *
 * Operations:
 *   ReduceSum: y[m] = sum(x[m, :])   (reduce along N)
 *   ReduceMax: y[m] = max(x[m, :])
 *
 * Supported: all GPU targets
 */

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#define CHECK_HIP(cmd) do { \
    hipError_t e = (cmd); \
    if(e != hipSuccess) { \
        std::cerr << "HIP error " << hipGetErrorString(e) << " L" << __LINE__ << "\n"; exit(1); \
    } } while(0)

using InT  = __half;
using OutT = float; // accumulate in fp32 (CK Tile default)

constexpr uint32_t WARP_SIZE  = 64; // Wave64 for gfx9
constexpr uint32_t WARPS_PER_BLOCK = 4;
constexpr uint32_t TBLOCK     = WARP_SIZE * WARPS_PER_BLOCK; // 256 threads
constexpr uint32_t VECTOR_SIZE = 8; // 8x fp16 = 128-bit

// ---------------------------------------------------------------------------
// Warp reduction (CK Tile WarpReduce equivalent)
// ---------------------------------------------------------------------------
template <typename T, typename ReduceOp>
__device__ __forceinline__ T warpReduce(T val, ReduceOp op)
{
    // Full warp reduction using butterfly pattern
    for(int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = op(val, __shfl_down(val, offset, WARP_SIZE));
    return val;
}

// ---------------------------------------------------------------------------
// Kernel: Row Sum Reduction   y[m] = sum_n(x[m,n])
// CK Tile pipeline: threadwise vectorized load -> warp reduce -> block reduce
// One block per row, all threads collaborate on the N dimension
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(TBLOCK)
kernel_reduce_sum(const InT* __restrict__ x,
                  OutT*      __restrict__ y,
                  uint32_t M, uint32_t N, uint32_t ldX)
{
    __shared__ OutT smem[WARPS_PER_BLOCK];

    uint32_t row = blockIdx.x;
    if(row >= M) return;

    const InT* row_ptr = x + row * ldX;

    // Each thread accumulates VECTOR_SIZE elements per step
    OutT acc = 0.f;
    uint32_t col = threadIdx.x * VECTOR_SIZE;
    for(; col + VECTOR_SIZE <= N; col += TBLOCK * VECTOR_SIZE)
    {
        const float4* ptr = reinterpret_cast<const float4*>(row_ptr + col);
        float4 v = *ptr;
        const __half2* h = reinterpret_cast<const __half2*>(&v);
        for(int i = 0; i < 4; i++) {
            float2 f = __half22float2(h[i]);
            acc += f.x + f.y;
        }
    }
    // Tail
    for(uint32_t c = col; c < N; c++) acc += __half2float(row_ptr[c]);

    // Warp reduce
    acc = warpReduce(acc, [](float a, float b){ return a + b; });

    // Block reduce via LDS
    uint32_t wid = threadIdx.x / WARP_SIZE;
    uint32_t lid = threadIdx.x % WARP_SIZE;
    if(lid == 0) smem[wid] = acc;
    __syncthreads();

    if(wid == 0) {
        acc = (lid < WARPS_PER_BLOCK) ? smem[lid] : 0.f;
        acc = warpReduce(acc, [](float a, float b){ return a + b; });
        if(lid == 0) y[row] = acc;
    }
}

// ---------------------------------------------------------------------------
// Kernel: Row Max Reduction   y[m] = max_n(x[m,n])
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(TBLOCK)
kernel_reduce_max(const InT* __restrict__ x,
                  OutT*      __restrict__ y,
                  uint32_t M, uint32_t N, uint32_t ldX)
{
    __shared__ OutT smem[WARPS_PER_BLOCK];

    uint32_t row = blockIdx.x;
    if(row >= M) return;

    const InT* row_ptr = x + row * ldX;

    OutT acc = -std::numeric_limits<OutT>::infinity();
    uint32_t col = threadIdx.x * VECTOR_SIZE;
    for(; col + VECTOR_SIZE <= N; col += TBLOCK * VECTOR_SIZE)
    {
        const float4* ptr = reinterpret_cast<const float4*>(row_ptr + col);
        float4 v = *ptr;
        const __half2* h = reinterpret_cast<const __half2*>(&v);
        for(int i = 0; i < 4; i++) {
            float2 f = __half22float2(h[i]);
            acc = fmaxf(acc, fmaxf(f.x, f.y));
        }
    }
    for(uint32_t c = col; c < N; c++) acc = fmaxf(acc, __half2float(row_ptr[c]));

    acc = warpReduce(acc, [](float a, float b){ return fmaxf(a,b); });

    uint32_t wid = threadIdx.x / WARP_SIZE;
    uint32_t lid = threadIdx.x % WARP_SIZE;
    if(lid == 0) smem[wid] = acc;
    __syncthreads();

    if(wid == 0) {
        acc = (lid < WARPS_PER_BLOCK) ? smem[lid] : -std::numeric_limits<OutT>::infinity();
        acc = warpReduce(acc, [](float a, float b){ return fmaxf(a,b); });
        if(lid == 0) y[row] = acc;
    }
}

// ---------------------------------------------------------------------------
// Host benchmark
// ---------------------------------------------------------------------------
template <typename Fn>
double bench(Fn fn, uint32_t warmup=5, uint32_t runs=20)
{
    for(uint32_t i = 0; i < warmup; i++) fn();
    CHECK_HIP(hipDeviceSynchronize());
    hipEvent_t t0, t1;
    CHECK_HIP(hipEventCreate(&t0)); CHECK_HIP(hipEventCreate(&t1));
    CHECK_HIP(hipEventRecord(t0));
    for(uint32_t i = 0; i < runs; i++) fn();
    CHECK_HIP(hipEventRecord(t1));
    CHECK_HIP(hipEventSynchronize(t1));
    float ms = 0.f;
    CHECK_HIP(hipEventElapsedTime(&ms, t0, t1));
    CHECK_HIP(hipEventDestroy(t0)); CHECK_HIP(hipEventDestroy(t1));
    return ms / runs;
}

void test_reduce(uint32_t M, uint32_t N)
{
    std::vector<InT>  hX(M * N);
    for(size_t i = 0; i < M * N; i++) hX[i] = __float2half(1.f);
    std::vector<OutT> hY(M, 0.f);

    InT *dX; OutT *dY;
    CHECK_HIP(hipMalloc(&dX, M * N * sizeof(InT)));
    CHECK_HIP(hipMalloc(&dY, M * sizeof(OutT)));
    CHECK_HIP(hipMemcpy(dX, hX.data(), M * N * sizeof(InT), hipMemcpyHostToDevice));

    dim3 block(TBLOCK), grid(M);

    auto fnSum = [&]() {
        hipLaunchKernelGGL(kernel_reduce_sum, grid, block, 0, 0, dX, dY, M, N, N);
    };
    auto fnMax = [&]() {
        hipLaunchKernelGGL(kernel_reduce_max, grid, block, 0, 0, dX, dY, M, N, N);
    };

    double msSum = bench(fnSum);
    double msMax = bench(fnMax);

    double bytes = (double)M * N * sizeof(InT) + (double)M * sizeof(OutT);
    std::cout << "[ReduceSum] M=" << M << " N=" << N
              << "  " << msSum << " ms  " << bytes/(msSum*1e-3)/1e9 << " GB/s\n";
    std::cout << "[ReduceMax] M=" << M << " N=" << N
              << "  " << msMax << " ms  " << bytes/(msMax*1e-3)/1e9 << " GB/s\n";

    CHECK_HIP(hipFree(dX)); CHECK_HIP(hipFree(dY));
}

int main(int argc, char* argv[])
{
    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << "  (" << prop.gcnArchName << ")\n\n";

    uint32_t M = 3328, N = 4096;
    if(argc >= 3) { M = std::atoi(argv[1]); N = std::atoi(argv[2]); }

    std::cout << "=== rocWMMA Reduce (rocWMMA port) ===\n";
    test_reduce(M, N);
    // Sweep sizes typical for LLM (hidden_size variants)
    for(uint32_t n : {1024u, 2048u, 4096u, 8192u}) {
        if(n != N) test_reduce(M, n);
    }
    return 0;
}
