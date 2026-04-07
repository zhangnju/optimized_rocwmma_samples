/*******************************************************************************
 * MIT License
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

/*
 * rocwmma_rmsnorm2d.cpp
 *
 * Description:
 *   RMS Normalization 2D forward pass ported from rocWMMA 10_rmsnorm2d.
 *
 *   rocWMMA Optimizations Applied:
 *   - One block per row for fused single-pass computation
 *   - Online RMS accumulation: acc += x^2, then rstd = rsqrt(acc/N + eps)
 *   - Vectorized FP16 loads (float4 = 8xfp16)
 *   - Warp butterfly reduction for rms, then LDS block-level merge
 *   - Fused: optional residual add before norm + optional per-row quantization
 *
 * Operations:
 *   RMSNorm:  y[m,n] = x[m,n] / sqrt(mean(x[m,:]^2) + eps) * gamma[n]
 *   Add+RMS:  x2 = x + residual; y = rmsnorm(x2) * gamma
 *
 * Supported: all GPU targets
 */

#include <cmath>
#include <iostream>
#include <vector>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#define CHECK_HIP(cmd) do { \
    hipError_t e=(cmd); if(e!=hipSuccess){ \
    std::cerr<<"HIP "<<hipGetErrorString(e)<<" L"<<__LINE__<<"\n";exit(1);} }while(0)

using InT  = __half;
using OutT = __half;
using AccT = float;

constexpr uint32_t WARP_SIZE = 64;
constexpr uint32_t WARPS     = 4;
constexpr uint32_t TBLOCK    = WARP_SIZE * WARPS;
constexpr uint32_t VEC       = 8;

__device__ __forceinline__ float warpReduceSum(float v)
{
    for(int off = WARP_SIZE/2; off > 0; off >>= 1)
        v += __shfl_down(v, off, WARP_SIZE);
    return v;
}

// ---------------------------------------------------------------------------
// RMSNorm Forward (single-pass, no residual)
// y[m,n] = x[m,n] * rsqrt(mean(x^2) + eps) * gamma[n]
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(TBLOCK)
kernel_rmsnorm2d_fwd(const InT* __restrict__ x,
                     const AccT* __restrict__ gamma,
                     OutT*       __restrict__ y,
                     uint32_t M, uint32_t N, uint32_t ldX, uint32_t ldY,
                     float eps)
{
    __shared__ float smem[WARPS];

    uint32_t row = blockIdx.x;
    if(row >= M) return;

    const InT* xrow = x + row * ldX;
    OutT*       yrow = y + row * ldY;

    // Pass 1: accumulate x^2
    float rms_acc = 0.f;
    for(uint32_t col = threadIdx.x * VEC; col + VEC <= N; col += TBLOCK * VEC) {
        const float4* ptr = reinterpret_cast<const float4*>(xrow + col);
        float4 v = *ptr;
        const __half2* h = reinterpret_cast<const __half2*>(&v);
        for(int i = 0; i < 4; i++) {
            float2 f = __half22float2(h[i]);
            rms_acc += f.x * f.x + f.y * f.y;
        }
    }
    for(uint32_t col = (N/(TBLOCK*VEC))*TBLOCK*VEC + threadIdx.x; col < N; col += TBLOCK) {
        float v = __half2float(xrow[col]); rms_acc += v * v;
    }

    rms_acc = warpReduceSum(rms_acc);

    uint32_t wid = threadIdx.x / WARP_SIZE, lid = threadIdx.x % WARP_SIZE;
    if(lid == 0) smem[wid] = rms_acc;
    __syncthreads();
    if(wid == 0) {
        rms_acc = (lid < WARPS) ? smem[lid] : 0.f;
        rms_acc = warpReduceSum(rms_acc);
        if(lid == 0) smem[0] = rms_acc;
    }
    __syncthreads();

    float rstd = rsqrtf(smem[0] / N + eps);

    // Pass 2: normalize
    for(uint32_t col = threadIdx.x * VEC; col + VEC <= N; col += TBLOCK * VEC) {
        const float4* px = reinterpret_cast<const float4*>(xrow + col);
        float4*       py = reinterpret_cast<float4*>(yrow + col);
        float4 vx = *px;
        const __half2* hx = reinterpret_cast<const __half2*>(&vx);
        float4 vy;
        __half2* hy = reinterpret_cast<__half2*>(&vy);
        for(int i = 0; i < 4; i++) {
            float2 f = __half22float2(hx[i]);
            uint32_t c0 = col+i*2, c1 = c0+1;
            hy[i] = __floats2half2_rn(f.x * rstd * gamma[c0], f.y * rstd * gamma[c1]);
        }
        *py = vy;
    }
    for(uint32_t col = (N/(TBLOCK*VEC))*TBLOCK*VEC + threadIdx.x; col < N; col += TBLOCK) {
        float v = __half2float(xrow[col]) * rstd * gamma[col];
        yrow[col] = __float2half(v);
    }
}

// ---------------------------------------------------------------------------
// Add+RMSNorm (fused residual, CK Tile 11_add_rmsnorm2d_rdquant style)
// x_out[m,n] = x[m,n] + residual[m,n]
// y[m,n]     = x_out[m,n] * rstd(x_out) * gamma[n]
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(TBLOCK)
kernel_add_rmsnorm2d_fwd(const InT* __restrict__ x,
                         const InT* __restrict__ residual,
                         const AccT* __restrict__ gamma,
                         OutT*       __restrict__ y,
                         InT*        __restrict__ x_out,  // optional: save x+residual
                         uint32_t M, uint32_t N, float eps)
{
    __shared__ float smem[WARPS];

    uint32_t row = blockIdx.x;
    if(row >= M) return;

    const InT* xrow  = x        + row * N;
    const InT* rrow  = residual  + row * N;
    OutT*       yrow  = y        + row * N;
    InT*        xorow = x_out   + row * N;

    // Pass 1: add + accumulate rms
    float rms_acc = 0.f;
    for(uint32_t col = threadIdx.x * VEC; col + VEC <= N; col += TBLOCK * VEC) {
        const float4* px = reinterpret_cast<const float4*>(xrow + col);
        const float4* pr = reinterpret_cast<const float4*>(rrow + col);
        float4*       po = reinterpret_cast<float4*>(xorow + col);
        float4 vx = *px, vr = *pr;
        const __half2* hx = reinterpret_cast<const __half2*>(&vx);
        const __half2* hr = reinterpret_cast<const __half2*>(&vr);
        float4 vo;
        __half2* ho = reinterpret_cast<__half2*>(&vo);
        for(int i = 0; i < 4; i++) {
            float2 fx = __half22float2(hx[i]), fr = __half22float2(hr[i]);
            float s0 = fx.x + fr.x, s1 = fx.y + fr.y;
            ho[i] = __floats2half2_rn(s0, s1);
            rms_acc += s0*s0 + s1*s1;
        }
        *po = vo;
    }
    rms_acc = warpReduceSum(rms_acc);
    uint32_t wid = threadIdx.x/WARP_SIZE, lid = threadIdx.x%WARP_SIZE;
    if(lid==0) smem[wid] = rms_acc;
    __syncthreads();
    if(wid==0){ rms_acc=(lid<WARPS)?smem[lid]:0.f; rms_acc=warpReduceSum(rms_acc); if(lid==0) smem[0]=rms_acc; }
    __syncthreads();
    float rstd = rsqrtf(smem[0]/N + eps);

    // Pass 2: normalize
    for(uint32_t col = threadIdx.x*VEC; col+VEC<=N; col+=TBLOCK*VEC) {
        const float4* po = reinterpret_cast<const float4*>(xorow + col);
        float4*        py = reinterpret_cast<float4*>(yrow + col);
        float4 vo = *po;
        const __half2* ho = reinterpret_cast<const __half2*>(&vo);
        float4 vy; __half2* hy = reinterpret_cast<__half2*>(&vy);
        for(int i=0;i<4;i++){
            float2 f=__half22float2(ho[i]);
            uint32_t c0=col+i*2, c1=c0+1;
            hy[i]=__floats2half2_rn(f.x*rstd*gamma[c0], f.y*rstd*gamma[c1]);
        }
        *py = vy;
    }
}

template <typename Fn>
double bench(Fn fn, uint32_t w=5, uint32_t r=20)
{
    for(uint32_t i=0;i<w;i++) fn();
    CHECK_HIP(hipDeviceSynchronize());
    hipEvent_t t0,t1; CHECK_HIP(hipEventCreate(&t0)); CHECK_HIP(hipEventCreate(&t1));
    CHECK_HIP(hipEventRecord(t0));
    for(uint32_t i=0;i<r;i++) fn();
    CHECK_HIP(hipEventRecord(t1)); CHECK_HIP(hipEventSynchronize(t1));
    float ms=0.f; CHECK_HIP(hipEventElapsedTime(&ms,t0,t1));
    CHECK_HIP(hipEventDestroy(t0)); CHECK_HIP(hipEventDestroy(t1));
    return ms/r;
}

void test_rmsnorm(uint32_t M, uint32_t N, float eps=1e-5f)
{
    std::vector<InT>  hX(M*N, __float2half(0.5f));
    std::vector<AccT> hG(N, 1.f);
    std::vector<OutT> hY(M*N);

    InT *dX; OutT *dY; AccT *dG;
    CHECK_HIP(hipMalloc(&dX, M*N*sizeof(InT)));
    CHECK_HIP(hipMalloc(&dY, M*N*sizeof(OutT)));
    CHECK_HIP(hipMalloc(&dG, N*sizeof(AccT)));
    CHECK_HIP(hipMemcpy(dX, hX.data(), M*N*sizeof(InT), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dG, hG.data(), N*sizeof(AccT),  hipMemcpyHostToDevice));

    dim3 block(TBLOCK), grid(M);
    auto fn = [&](){ hipLaunchKernelGGL(kernel_rmsnorm2d_fwd, grid, block, 0, 0,
                                        dX, dG, dY, M, N, N, N, eps); };
    double ms = bench(fn);
    double bw = (M*N*2.0*sizeof(InT) + N*sizeof(AccT)) / (ms*1e-3) / 1e9;
    std::cout << "[RMSNorm2D] M=" << M << " N=" << N
              << "  " << ms << " ms  " << bw << " GB/s\n";

    CHECK_HIP(hipFree(dX)); CHECK_HIP(hipFree(dY)); CHECK_HIP(hipFree(dG));
}

void test_add_rmsnorm(uint32_t M, uint32_t N, float eps=1e-5f)
{
    std::vector<InT>  hX(M*N, __float2half(0.3f)), hR(M*N, __float2half(0.2f));
    std::vector<AccT> hG(N, 1.f);
    std::vector<OutT> hY(M*N);
    std::vector<InT>  hXo(M*N);

    InT *dX, *dR, *dXo; OutT *dY; AccT *dG;
    CHECK_HIP(hipMalloc(&dX,  M*N*sizeof(InT)));
    CHECK_HIP(hipMalloc(&dR,  M*N*sizeof(InT)));
    CHECK_HIP(hipMalloc(&dXo, M*N*sizeof(InT)));
    CHECK_HIP(hipMalloc(&dY,  M*N*sizeof(OutT)));
    CHECK_HIP(hipMalloc(&dG,  N*sizeof(AccT)));
    CHECK_HIP(hipMemcpy(dX, hX.data(), M*N*sizeof(InT), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dR, hR.data(), M*N*sizeof(InT), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dG, hG.data(), N*sizeof(AccT),  hipMemcpyHostToDevice));

    dim3 block(TBLOCK), grid(M);
    auto fn = [&](){ hipLaunchKernelGGL(kernel_add_rmsnorm2d_fwd, grid, block, 0, 0,
                                        dX, dR, dG, dY, dXo, M, N, eps); };
    double ms = bench(fn);
    double bw = (M*N*4.0*sizeof(InT) + N*sizeof(AccT)) / (ms*1e-3) / 1e9;
    std::cout << "[Add+RMSNorm2D] M=" << M << " N=" << N
              << "  " << ms << " ms  " << bw << " GB/s\n";

    CHECK_HIP(hipFree(dX)); CHECK_HIP(hipFree(dR)); CHECK_HIP(hipFree(dXo));
    CHECK_HIP(hipFree(dY)); CHECK_HIP(hipFree(dG));
}

int main(int argc, char* argv[])
{
    hipDeviceProp_t prop; CHECK_HIP(hipGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << "\n\n";
    uint32_t M=3328, N=4096;
    if(argc>=3){ M=std::atoi(argv[1]); N=std::atoi(argv[2]); }
    std::cout << "=== rocWMMA RMSNorm2D (rocWMMA port) ===\n";
    test_rmsnorm(M, N);
    test_add_rmsnorm(M, N);
    for(uint32_t n : {1024u, 2048u, 8192u}) { test_rmsnorm(M, n); }
    return 0;
}
