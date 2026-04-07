/*******************************************************************************
 * MIT License
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

/*
 * rocwmma_smoothquant.cpp
 *
 * Description:
 *   Smooth Quantization ported from rocWMMA 12_smoothquant.
 *
 *   rocWMMA Optimizations Applied:
 *   - One block per row, threads cover N collectively
 *   - Vectorized FP16 loads and int8 stores
 *   - Two-pass: pass1 = find abs-max after smooth-scale, pass2 = quantize
 *   - Warp + block reduction for per-row abs-max
 *   - Fused smooth-scale (per-channel) * per-row dynamic quantization
 *
 * Operation:
 *   x_scaled[m,n] = x[m,n] * smooth_scale[n]
 *   row_scale[m]  = max(|x_scaled[m,:]|) / 127
 *   y[m,n]        = round(x_scaled[m,n] / row_scale[m])   (int8 output)
 *
 * Supported: all GPU targets
 */

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#define CHECK_HIP(cmd) do { \
    hipError_t e=(cmd); if(e!=hipSuccess){ \
    std::cerr<<"HIP "<<hipGetErrorString(e)<<" L"<<__LINE__<<"\n";exit(1);} }while(0)

using InT    = __half;   // fp16 input
using ScaleT = float;    // smooth scale (per-channel)
using OutT   = int8_t;   // int8 output

constexpr uint32_t WARP_SIZE = 64;
constexpr uint32_t WARPS     = 4;
constexpr uint32_t TBLOCK    = WARP_SIZE * WARPS;
constexpr uint32_t VEC       = 8; // fp16 x8

__device__ __forceinline__ float warpReduceMax(float v)
{
    for(int off = WARP_SIZE/2; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_down(v, off, WARP_SIZE));
    return v;
}

// ---------------------------------------------------------------------------
// SmoothQuant kernel
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(TBLOCK)
kernel_smoothquant(const InT*    __restrict__ x,           // [M, N]
                   const ScaleT* __restrict__ smooth_scale, // [N]
                   OutT*         __restrict__ y,            // [M, N]
                   ScaleT*       __restrict__ row_scale,    // [M]
                   uint32_t M, uint32_t N, float quant_max)
{
    __shared__ float smem[WARPS];
    __shared__ float s_row_scale;

    uint32_t row = blockIdx.x;
    if(row >= M) return;

    const InT* xrow = x + row * N;
    OutT*       yrow = y + row * N;

    // --- Pass 1: compute abs-max of x * smooth_scale ---
    float abs_max = 0.f;
    for(uint32_t col = threadIdx.x * VEC; col + VEC <= N; col += TBLOCK * VEC) {
        const float4* px = reinterpret_cast<const float4*>(xrow + col);
        float4 vx = *px;
        const __half2* hx = reinterpret_cast<const __half2*>(&vx);
        for(int i = 0; i < 4; i++) {
            float2 f = __half22float2(hx[i]);
            uint32_t c0 = col+i*2, c1 = c0+1;
            abs_max = fmaxf(abs_max, fabsf(f.x * smooth_scale[c0]));
            abs_max = fmaxf(abs_max, fabsf(f.y * smooth_scale[c1]));
        }
    }
    for(uint32_t col=(N/(TBLOCK*VEC))*TBLOCK*VEC+threadIdx.x; col<N; col+=TBLOCK) {
        abs_max = fmaxf(abs_max, fabsf(__half2float(xrow[col]) * smooth_scale[col]));
    }

    abs_max = warpReduceMax(abs_max);
    uint32_t wid = threadIdx.x/WARP_SIZE, lid = threadIdx.x%WARP_SIZE;
    if(lid==0) smem[wid] = abs_max;
    __syncthreads();
    if(wid==0){ abs_max=(lid<WARPS)?smem[lid]:0.f; abs_max=warpReduceMax(abs_max); if(lid==0) { s_row_scale = abs_max / quant_max; row_scale[row] = s_row_scale; } }
    __syncthreads();

    float inv_scale = (s_row_scale > 0.f) ? (1.f / s_row_scale) : 0.f;

    // --- Pass 2: quantize ---
    for(uint32_t col = threadIdx.x * VEC; col + VEC <= N; col += TBLOCK * VEC) {
        const float4* px = reinterpret_cast<const float4*>(xrow + col);
        float4 vx = *px;
        const __half2* hx = reinterpret_cast<const __half2*>(&vx);
        // Write VEC int8s
        int8_t tmp[VEC];
        for(int i = 0; i < 4; i++) {
            float2 f = __half22float2(hx[i]);
            uint32_t c0=col+i*2, c1=c0+1;
            float s0 = f.x * smooth_scale[c0] * inv_scale;
            float s1 = f.y * smooth_scale[c1] * inv_scale;
            tmp[i*2]   = static_cast<int8_t>(fminf(fmaxf(rintf(s0), -quant_max), quant_max));
            tmp[i*2+1] = static_cast<int8_t>(fminf(fmaxf(rintf(s1), -quant_max), quant_max));
        }
        // Write 8 int8s at once (64-bit)
        *reinterpret_cast<int64_t*>(yrow + col) = *reinterpret_cast<int64_t*>(tmp);
    }
    for(uint32_t col=(N/(TBLOCK*VEC))*TBLOCK*VEC+threadIdx.x; col<N; col+=TBLOCK) {
        float v = __half2float(xrow[col]) * smooth_scale[col] * inv_scale;
        yrow[col] = static_cast<int8_t>(fminf(fmaxf(rintf(v), -quant_max), quant_max));
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

void test_smoothquant(uint32_t M, uint32_t N)
{
    std::vector<InT>    hX(M*N, __float2half(0.5f));
    std::vector<ScaleT> hS(N, 1.f);
    std::vector<OutT>   hY(M*N, 0);
    std::vector<ScaleT> hRS(M, 0.f);

    InT *dX; OutT *dY; ScaleT *dS, *dRS;
    CHECK_HIP(hipMalloc(&dX,  M*N*sizeof(InT)));
    CHECK_HIP(hipMalloc(&dY,  M*N*sizeof(OutT)));
    CHECK_HIP(hipMalloc(&dS,  N*sizeof(ScaleT)));
    CHECK_HIP(hipMalloc(&dRS, M*sizeof(ScaleT)));
    CHECK_HIP(hipMemcpy(dX, hX.data(), M*N*sizeof(InT),    hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dS, hS.data(), N*sizeof(ScaleT),   hipMemcpyHostToDevice));

    dim3 block(TBLOCK), grid(M);
    auto fn = [&](){ hipLaunchKernelGGL(kernel_smoothquant, grid, block, 0, 0,
                                        dX, dS, dY, dRS, M, N, 127.f); };
    double ms = bench(fn);
    double bw = ((double)M*N*sizeof(InT) + (double)M*N*sizeof(OutT) +
                 (double)N*sizeof(ScaleT) + (double)M*sizeof(ScaleT)) / (ms*1e-3) / 1e9;
    std::cout << "[SmoothQuant] M=" << M << " N=" << N
              << "  " << ms << " ms  " << bw << " GB/s\n";

    CHECK_HIP(hipFree(dX)); CHECK_HIP(hipFree(dY));
    CHECK_HIP(hipFree(dS)); CHECK_HIP(hipFree(dRS));
}

int main(int argc, char* argv[])
{
    hipDeviceProp_t prop; CHECK_HIP(hipGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << "\n\n";
    uint32_t M=3328, N=4096;
    if(argc>=3){ M=std::atoi(argv[1]); N=std::atoi(argv[2]); }
    std::cout << "=== rocWMMA SmoothQuant (rocWMMA port) ===\n";
    test_smoothquant(M, N);
    for(uint32_t n : {1024u, 2048u, 8192u}) test_smoothquant(M, n);
    return 0;
}
