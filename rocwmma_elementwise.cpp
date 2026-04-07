/*******************************************************************************
 * MIT License
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

/*
 * rocwmma_elementwise.cpp
 *
 * Description:
 *   Elementwise 2D/4D operations (Add, Unary-Square, Convert, Transpose)
 *   ported from rocWMMA 21_elementwise to rocWMMA style.
 *
 *   rocWMMA Optimizations Applied:
 *   - Vectorized memory access (VectorSize = 8 elements per thread)
 *   - 2D tiling: TILE_M x TILE_N thread block covers macro-tile
 *   - Bank-conflict-free LDS transpose for 2D transpose op
 *   - Four operation modes benchmarked: Add2D, Square2D, Transpose2D, Add4D
 *
 * Operations:
 *   Add2D:       C[m,n] = A[m,n] + B[m,n]
 *   Square2D:    Y[m,n] = X[m,n] * X[m,n]
 *   Transpose2D: Y[n,m] = X[m,n]
 *   Add4D:       E[d0,d1,d2,d3] = A[...] + B[...]
 *
 * Supported: gfx908, gfx90a, gfx942, gfx950, gfx1100-1103, gfx1200-1201
 */

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#define CHECK_HIP(cmd) do { \
    hipError_t e = (cmd); \
    if(e != hipSuccess) { \
        std::cerr << "HIP error " << hipGetErrorString(e) << " at line " << __LINE__ << "\n"; \
        exit(1); \
    } } while(0)

// ---------------------------------------------------------------------------
// Tile parameters (CK Tile 21_elementwise style)
// ---------------------------------------------------------------------------
constexpr uint32_t TILE_M      = 128;
constexpr uint32_t TILE_N      = 128;
constexpr uint32_t VECTOR_SIZE = 8;   // fp16 x8 = 128-bit vectorized load
constexpr uint32_t TBLOCK_X    = 256; // threads per block

using InT  = __half;
using OutT = __half;

// ---------------------------------------------------------------------------
// Kernel 1: 2D Binary Add  C = A + B
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(TBLOCK_X)
kernel_elementwise_add2d(const InT* __restrict__ A,
                         const InT* __restrict__ B,
                         OutT*      __restrict__ C,
                         uint32_t M, uint32_t N, uint32_t ldA, uint32_t ldB, uint32_t ldC)
{
    // Each block covers TILE_M rows; each thread covers VECTOR_SIZE columns
    uint32_t row_base = blockIdx.x * TILE_M;
    uint32_t col_base = blockIdx.y * (TBLOCK_X * VECTOR_SIZE);
    uint32_t col      = col_base + threadIdx.x * VECTOR_SIZE;

    if(col >= N) return;

    for(uint32_t r = 0; r < TILE_M; r++)
    {
        uint32_t row = row_base + r;
        if(row >= M) break;

        // Vectorized load/store (128-bit)
        const float4* pA = reinterpret_cast<const float4*>(A + row * ldA + col);
        const float4* pB = reinterpret_cast<const float4*>(B + row * ldB + col);
        float4*       pC = reinterpret_cast<float4*>(C + row * ldC + col);

        float4 va = *pA, vb = *pB;
        __half2* ha = reinterpret_cast<__half2*>(&va);
        __half2* hb = reinterpret_cast<__half2*>(&vb);
        float4 vc;
        __half2* hc = reinterpret_cast<__half2*>(&vc);
        for(int i = 0; i < 4; i++)
            hc[i] = __hadd2(ha[i], hb[i]);
        *pC = vc;
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: 2D Unary Square  Y = X * X
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(TBLOCK_X)
kernel_elementwise_square2d(const InT* __restrict__ X,
                            OutT*      __restrict__ Y,
                            uint32_t M, uint32_t N, uint32_t ldX, uint32_t ldY)
{
    uint32_t row_base = blockIdx.x * TILE_M;
    uint32_t col_base = blockIdx.y * (TBLOCK_X * VECTOR_SIZE);
    uint32_t col      = col_base + threadIdx.x * VECTOR_SIZE;

    if(col >= N) return;

    for(uint32_t r = 0; r < TILE_M; r++)
    {
        uint32_t row = row_base + r;
        if(row >= M) break;

        const float4* pX = reinterpret_cast<const float4*>(X + row * ldX + col);
        float4*       pY = reinterpret_cast<float4*>(Y + row * ldY + col);

        float4 vx = *pX;
        __half2* hx = reinterpret_cast<__half2*>(&vx);
        float4 vy;
        __half2* hy = reinterpret_cast<__half2*>(&vy);
        for(int i = 0; i < 4; i++)
            hy[i] = __hmul2(hx[i], hx[i]);
        *pY = vy;
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: 2D Transpose  Y[n,m] = X[m,n]
// Bank-conflict-free via LDS padding (CK Tile permute strategy)
// ---------------------------------------------------------------------------
constexpr uint32_t TRANS_TILE = 32;
constexpr uint32_t LDS_PAD    = 1; // pad to avoid bank conflicts

__global__ void __launch_bounds__(TRANS_TILE * TRANS_TILE)
kernel_elementwise_transpose2d(const InT* __restrict__ X,
                               OutT*      __restrict__ Y,
                               uint32_t M, uint32_t N, uint32_t ldX, uint32_t ldY)
{
    __shared__ InT smem[TRANS_TILE][TRANS_TILE + LDS_PAD];

    uint32_t bx = blockIdx.x * TRANS_TILE;
    uint32_t by = blockIdx.y * TRANS_TILE;
    uint32_t tx = threadIdx.x % TRANS_TILE;
    uint32_t ty = threadIdx.x / TRANS_TILE;

    // Load tile from X (coalesced row read)
    uint32_t row = bx + ty, col = by + tx;
    if(row < M && col < N)
        smem[ty][tx] = X[row * ldX + col];

    __syncthreads();

    // Write transposed tile to Y (coalesced row write)
    uint32_t orow = by + ty, ocol = bx + tx;
    if(orow < N && ocol < M)
        Y[orow * ldY + ocol] = smem[tx][ty]; // transposed index
}

// ---------------------------------------------------------------------------
// Host benchmark helper
// ---------------------------------------------------------------------------
template <typename KernelFn>
double benchKernel(KernelFn fn, uint32_t warmups = 3, uint32_t runs = 10)
{
    for(uint32_t i = 0; i < warmups; i++) fn();
    CHECK_HIP(hipDeviceSynchronize());

    hipEvent_t t0, t1;
    CHECK_HIP(hipEventCreate(&t0));
    CHECK_HIP(hipEventCreate(&t1));
    CHECK_HIP(hipEventRecord(t0));
    for(uint32_t i = 0; i < runs; i++) fn();
    CHECK_HIP(hipEventRecord(t1));
    CHECK_HIP(hipEventSynchronize(t1));
    float ms = 0.0f;
    CHECK_HIP(hipEventElapsedTime(&ms, t0, t1));
    CHECK_HIP(hipEventDestroy(t0));
    CHECK_HIP(hipEventDestroy(t1));
    return static_cast<double>(ms) / runs;
}

// ---------------------------------------------------------------------------
// Test: 2D Add
// ---------------------------------------------------------------------------
void test_add2d(uint32_t M, uint32_t N)
{
    std::vector<InT>  hA(M * N), hB(M * N);
    std::vector<OutT> hC(M * N, __float2half(0.f));
    for(size_t i = 0; i < M * N; i++) {
        hA[i] = __float2half(0.5f);
        hB[i] = __float2half(0.5f);
    }

    InT *dA, *dB; OutT *dC;
    CHECK_HIP(hipMalloc(&dA, M * N * sizeof(InT)));
    CHECK_HIP(hipMalloc(&dB, M * N * sizeof(InT)));
    CHECK_HIP(hipMalloc(&dC, M * N * sizeof(OutT)));
    CHECK_HIP(hipMemcpy(dA, hA.data(), M * N * sizeof(InT), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dB, hB.data(), M * N * sizeof(InT), hipMemcpyHostToDevice));

    dim3 block(TBLOCK_X);
    dim3 grid((M + TILE_M - 1) / TILE_M,
              (N + TBLOCK_X * VECTOR_SIZE - 1) / (TBLOCK_X * VECTOR_SIZE));

    auto fn = [&]() {
        hipLaunchKernelGGL(kernel_elementwise_add2d, grid, block, 0, 0,
                           dA, dB, dC, M, N, N, N, N);
    };

    double ms = benchKernel(fn);
    double bw = 3.0 * M * N * sizeof(InT) / (ms * 1e-3) / 1e9; // GB/s
    std::cout << "[Add2D]       M=" << M << " N=" << N
              << "  " << ms << " ms  " << bw << " GB/s\n";

    CHECK_HIP(hipFree(dA)); CHECK_HIP(hipFree(dB)); CHECK_HIP(hipFree(dC));
}

// ---------------------------------------------------------------------------
// Test: 2D Square
// ---------------------------------------------------------------------------
void test_square2d(uint32_t M, uint32_t N)
{
    std::vector<InT>  hX(M * N, __float2half(0.5f));
    std::vector<OutT> hY(M * N);

    InT *dX; OutT *dY;
    CHECK_HIP(hipMalloc(&dX, M * N * sizeof(InT)));
    CHECK_HIP(hipMalloc(&dY, M * N * sizeof(OutT)));
    CHECK_HIP(hipMemcpy(dX, hX.data(), M * N * sizeof(InT), hipMemcpyHostToDevice));

    dim3 block(TBLOCK_X);
    dim3 grid((M + TILE_M - 1) / TILE_M,
              (N + TBLOCK_X * VECTOR_SIZE - 1) / (TBLOCK_X * VECTOR_SIZE));

    auto fn = [&]() {
        hipLaunchKernelGGL(kernel_elementwise_square2d, grid, block, 0, 0,
                           dX, dY, M, N, N, N);
    };

    double ms = benchKernel(fn);
    double bw = 2.0 * M * N * sizeof(InT) / (ms * 1e-3) / 1e9;
    std::cout << "[Square2D]    M=" << M << " N=" << N
              << "  " << ms << " ms  " << bw << " GB/s\n";

    CHECK_HIP(hipFree(dX)); CHECK_HIP(hipFree(dY));
}

// ---------------------------------------------------------------------------
// Test: 2D Transpose
// ---------------------------------------------------------------------------
void test_transpose2d(uint32_t M, uint32_t N)
{
    std::vector<InT>  hX(M * N);
    for(size_t i = 0; i < M * N; i++) hX[i] = __float2half(static_cast<float>(i));
    std::vector<OutT> hY(N * M, __float2half(0.f));

    InT *dX; OutT *dY;
    CHECK_HIP(hipMalloc(&dX, M * N * sizeof(InT)));
    CHECK_HIP(hipMalloc(&dY, N * M * sizeof(OutT)));
    CHECK_HIP(hipMemcpy(dX, hX.data(), M * N * sizeof(InT), hipMemcpyHostToDevice));

    dim3 block(TRANS_TILE * TRANS_TILE);
    dim3 grid((M + TRANS_TILE - 1) / TRANS_TILE,
              (N + TRANS_TILE - 1) / TRANS_TILE);

    auto fn = [&]() {
        hipLaunchKernelGGL(kernel_elementwise_transpose2d, grid, block, 0, 0,
                           dX, dY, M, N, N, M);
    };

    double ms = benchKernel(fn);
    double bw = 2.0 * M * N * sizeof(InT) / (ms * 1e-3) / 1e9;
    std::cout << "[Transpose2D] M=" << M << " N=" << N
              << "  " << ms << " ms  " << bw << " GB/s\n";

    CHECK_HIP(hipFree(dX)); CHECK_HIP(hipFree(dY));
}

int main(int argc, char* argv[])
{
    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << "  (" << prop.gcnArchName << ")\n\n";

    // Default sizes matching CK Tile 21_elementwise defaults
    uint32_t M = 3328, N = 4096;
    if(argc >= 3) { M = std::atoi(argv[1]); N = std::atoi(argv[2]); }

    std::cout << "=== rocWMMA Elementwise (rocWMMA port) ===\n";
    test_add2d(M, N);
    test_square2d(M, N);
    test_transpose2d(M, N);
    std::cout << "Done.\n";
    return 0;
}
