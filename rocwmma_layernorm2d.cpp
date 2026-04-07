/*******************************************************************************
 * MIT License
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

/*
 * rocwmma_layernorm2d.cpp
 *
 * Description:
 *   Layer Normalization 2D forward pass ported from rocWMMA 02_layernorm2d.
 *
 *   rocWMMA Optimizations Applied:
 *   - One block per row (M blocks total)
 *   - Two-pass online algorithm: first pass compute mean+var, second pass normalize
 *   - Welford's online algorithm for numerically stable variance
 *   - Vectorized FP16 loads (float4 = 8xfp16 per thread)
 *   - Warp-level reduction using __shfl_down + LDS inter-warp reduce
 *   - Fused gamma/beta application in single pass
 *
 * Operation:
 *   y[m,n] = (x[m,n] - mean[m]) / sqrt(var[m] + eps) * gamma[n] + beta[n]
 *
 * Supported: all GPU targets
 */

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#define CHECK_HIP(cmd) do { \
    hipError_t e=(cmd); if(e!=hipSuccess){ \
    std::cerr<<"HIP error "<<hipGetErrorString(e)<<" L"<<__LINE__<<"\n";exit(1);} }while(0)

using InT  = __half;
using OutT = __half;
using AccT = float;

constexpr uint32_t WARP_SIZE  = 64;
constexpr uint32_t WARPS      = 4;
constexpr uint32_t TBLOCK     = WARP_SIZE * WARPS;
constexpr uint32_t VEC        = 8; // fp16 x8 = 128-bit

// ---------------------------------------------------------------------------
// Welford online mean/variance reduction across a warp
// ---------------------------------------------------------------------------
struct WelfordVar { float mean, m2; uint32_t count; };

__device__ __forceinline__
WelfordVar welfordWarpReduce(WelfordVar v)
{
    for(int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float om   = __shfl_down(v.mean,  offset, WARP_SIZE);
        float om2  = __shfl_down(v.m2,    offset, WARP_SIZE);
        uint32_t oc = __shfl_down(v.count, offset, WARP_SIZE);
        // Parallel Welford merge
        uint32_t nc = v.count + oc;
        if(nc == 0) continue;
        float delta = om - v.mean;
        v.mean  = v.mean + delta * oc / nc;
        v.m2   += om2 + delta * delta * (float)v.count * oc / nc;
        v.count = nc;
    }
    return v;
}

// ---------------------------------------------------------------------------
// LayerNorm2D Forward Kernel
// One block per row. Threads cover N dimension collectively.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(TBLOCK)
kernel_layernorm2d_fwd(const InT* __restrict__ x,
                       const AccT* __restrict__ gamma,
                       const AccT* __restrict__ beta,
                       OutT*       __restrict__ y,
                       uint32_t M, uint32_t N, uint32_t ldX, uint32_t ldY,
                       float eps)
{
    __shared__ float smem_mean[WARPS];
    __shared__ float smem_m2  [WARPS];
    __shared__ uint32_t smem_cnt[WARPS];

    uint32_t row = blockIdx.x;
    if(row >= M) return;

    const InT* xrow = x + row * ldX;
    OutT*       yrow = y + row * ldY;

    // --- Pass 1: Welford online mean/variance ---
    WelfordVar wv = {0.f, 0.f, 0u};
    for(uint32_t col = threadIdx.x * VEC; col + VEC <= N; col += TBLOCK * VEC) {
        const float4* ptr = reinterpret_cast<const float4*>(xrow + col);
        float4 v = *ptr;
        const __half2* h = reinterpret_cast<const __half2*>(&v);
        for(int i = 0; i < 4; i++) {
            float2 f = __half22float2(h[i]);
            // Update Welford for each element
            wv.count++; float d = f.x - wv.mean; wv.mean += d / wv.count; wv.m2 += d * (f.x - wv.mean);
            wv.count++; d = f.y - wv.mean; wv.mean += d / wv.count; wv.m2 += d * (f.y - wv.mean);
        }
    }
    // Tail
    for(uint32_t col = (N / (TBLOCK * VEC)) * TBLOCK * VEC + threadIdx.x;
        col < N; col += TBLOCK) {
        float v = __half2float(xrow[col]);
        wv.count++; float d = v - wv.mean; wv.mean += d / wv.count; wv.m2 += d * (v - wv.mean);
    }

    // Warp reduce
    wv = welfordWarpReduce(wv);

    uint32_t wid = threadIdx.x / WARP_SIZE;
    uint32_t lid = threadIdx.x % WARP_SIZE;
    if(lid == 0) { smem_mean[wid] = wv.mean; smem_m2[wid] = wv.m2; smem_cnt[wid] = wv.count; }
    __syncthreads();

    // Block-level merge (done by first warp)
    if(wid == 0) {
        WelfordVar bv = { (lid < WARPS) ? smem_mean[lid] : 0.f,
                          (lid < WARPS) ? smem_m2  [lid] : 0.f,
                          (lid < WARPS) ? smem_cnt [lid] : 0u };
        bv = welfordWarpReduce(bv);
        if(lid == 0) { smem_mean[0] = bv.mean; smem_m2[0] = bv.m2; }
    }
    __syncthreads();

    float mean = smem_mean[0];
    float rstd = rsqrtf(smem_m2[0] / N + eps);

    // --- Pass 2: Normalize + apply gamma/beta ---
    for(uint32_t col = threadIdx.x * VEC; col + VEC <= N; col += TBLOCK * VEC) {
        const float4* px = reinterpret_cast<const float4*>(xrow + col);
        float4*       py = reinterpret_cast<float4*>(yrow + col);
        float4 vx = *px;
        const __half2* hx = reinterpret_cast<const __half2*>(&vx);
        float4 vy;
        __half2* hy = reinterpret_cast<__half2*>(&vy);
        for(int i = 0; i < 4; i++) {
            float2 f = __half22float2(hx[i]);
            uint32_t c0 = col + i*2, c1 = c0 + 1;
            float y0 = (f.x - mean) * rstd * gamma[c0] + beta[c0];
            float y1 = (f.y - mean) * rstd * gamma[c1] + beta[c1];
            hy[i] = __floats2half2_rn(y0, y1);
        }
        *py = vy;
    }
    for(uint32_t col = (N / (TBLOCK * VEC)) * TBLOCK * VEC + threadIdx.x;
        col < N; col += TBLOCK) {
        float v = (__half2float(xrow[col]) - mean) * rstd * gamma[col] + beta[col];
        yrow[col] = __float2half(v);
    }
}

template <typename Fn>
double bench(Fn fn, uint32_t w=5, uint32_t r=20)
{
    for(uint32_t i=0;i<w;i++) fn();
    CHECK_HIP(hipDeviceSynchronize());
    hipEvent_t t0,t1;
    CHECK_HIP(hipEventCreate(&t0)); CHECK_HIP(hipEventCreate(&t1));
    CHECK_HIP(hipEventRecord(t0));
    for(uint32_t i=0;i<r;i++) fn();
    CHECK_HIP(hipEventRecord(t1));
    CHECK_HIP(hipEventSynchronize(t1));
    float ms=0.f; CHECK_HIP(hipEventElapsedTime(&ms,t0,t1));
    CHECK_HIP(hipEventDestroy(t0)); CHECK_HIP(hipEventDestroy(t1));
    return ms/r;
}

void test_layernorm(uint32_t M, uint32_t N, float eps=1e-5f)
{
    std::vector<InT>  hX(M*N);
    std::vector<AccT> hGamma(N, 1.f), hBeta(N, 0.f);
    std::vector<OutT> hY(M*N);
    for(size_t i=0;i<M*N;i++) hX[i]=__float2half(static_cast<float>(rand())/RAND_MAX);

    InT *dX; OutT *dY; AccT *dG, *dB;
    CHECK_HIP(hipMalloc(&dX, M*N*sizeof(InT)));
    CHECK_HIP(hipMalloc(&dY, M*N*sizeof(OutT)));
    CHECK_HIP(hipMalloc(&dG, N*sizeof(AccT)));
    CHECK_HIP(hipMalloc(&dB, N*sizeof(AccT)));
    CHECK_HIP(hipMemcpy(dX, hX.data(),     M*N*sizeof(InT),  hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dG, hGamma.data(), N*sizeof(AccT),   hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dB, hBeta.data(),  N*sizeof(AccT),   hipMemcpyHostToDevice));

    dim3 block(TBLOCK), grid(M);
    auto fn = [&]() {
        hipLaunchKernelGGL(kernel_layernorm2d_fwd, grid, block, 0, 0,
                           dX, dG, dB, dY, M, N, N, N, eps);
    };

    double ms = bench(fn);
    // Bandwidth: read X, write Y, read gamma/beta (once)
    double bytes = (M*N*2 + N*2) * sizeof(InT);
    std::cout << "[LayerNorm2D] M=" << M << " N=" << N
              << "  " << ms << " ms  " << bytes/(ms*1e-3)/1e9 << " GB/s\n";

    CHECK_HIP(hipFree(dX)); CHECK_HIP(hipFree(dY));
    CHECK_HIP(hipFree(dG)); CHECK_HIP(hipFree(dB));
}

int main(int argc, char* argv[])
{
    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << "\n\n";

    uint32_t M = 3328, N = 4096;
    if(argc>=3){ M=std::atoi(argv[1]); N=std::atoi(argv[2]); }

    std::cout << "=== rocWMMA LayerNorm2D (rocWMMA port) ===\n";
    test_layernorm(M, N);
    for(uint32_t n : {1024u, 2048u, 8192u}) if(n!=N) test_layernorm(M, n);
    return 0;
}
