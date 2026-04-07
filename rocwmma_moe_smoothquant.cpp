/*******************************************************************************
 * MIT License
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

/*
 * rocwmma_moe_smoothquant.cpp
 *
 * Description:
 *   MoE-aware Smooth Quantization ported from rocWMMA 14_moe_smoothquant.
 *   Extends smooth quantization to handle MoE routing: each token is assigned
 *   to topk experts, and smooth scales are per-expert (not global per-channel).
 *
 *   rocWMMA Optimizations Applied:
 *   - One block per (token, expert_assignment) pair
 *   - Topk expert IDs used to index into per-expert smooth scales
 *   - Smooth scale shape: [num_experts, hidden_size]
 *   - Output layout: [topk * tokens, hidden_size] (tokens replicated per expert)
 *   - Same two-pass quantization as smoothquant: find abs-max, then quantize
 *   - Vectorized loads along hidden_size (8x fp16)
 *   - Warp+block reduction for per-row abs-max
 *
 * Operation:
 *   For each (token t, expert assignment k in topk):
 *     expert_id = topk_ids[t, k]
 *     x_scaled  = x[t, :] * smooth_scale[expert_id, :]
 *     row_scale  = max(|x_scaled|) / 127
 *     qy[t*topk+k, :] = round(x_scaled / row_scale)
 *
 * Supported: all GPU targets
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
#include <random>
#include <set>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#define CHECK_HIP(cmd) do { \
    hipError_t e=(cmd); if(e!=hipSuccess){ \
    std::cerr<<"HIP "<<hipGetErrorString(e)<<" L"<<__LINE__<<"\n";exit(1);} }while(0)

using InT    = __half;
using ScaleT = float;
using QT     = int8_t;
using IdxT   = int32_t;

constexpr uint32_t WARP_SIZE = 64;
constexpr uint32_t WARPS     = 4;
constexpr uint32_t TBLOCK    = WARP_SIZE * WARPS;
constexpr uint32_t VEC       = 8;

__device__ __forceinline__ float warpReduceMax(float v)
{
    for(int off=WARP_SIZE/2;off>0;off>>=1) v=fmaxf(v,__shfl_down(v,off,WARP_SIZE));
    return v;
}

// ---------------------------------------------------------------------------
// MoE SmoothQuant kernel
// One block = one (token, topk_slot) pair
// Grid: (tokens * topk) blocks
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(TBLOCK)
kernel_moe_smoothquant(const InT*    __restrict__ x,           // [tokens, hidden]
                       const ScaleT* __restrict__ smooth_scale, // [experts, hidden]
                       const IdxT*   __restrict__ topk_ids,    // [tokens, topk]
                       QT*           __restrict__ qy,           // [topk*tokens, hidden]
                       ScaleT*       __restrict__ row_scale,    // [topk*tokens]
                       uint32_t tokens, uint32_t hidden, uint32_t topk,
                       uint32_t experts, float quant_max)
{
    __shared__ float smem[WARPS];
    __shared__ float s_rscale;

    uint32_t flat = blockIdx.x;           // flat index into [topk*tokens]
    uint32_t tok  = flat / topk;
    uint32_t slot = flat % topk;
    if(tok >= tokens) return;

    IdxT eid = topk_ids[tok * topk + slot];
    if(eid < 0 || (uint32_t)eid >= experts) return;

    const InT*    xrow  = x           + tok * hidden;
    const ScaleT* srow  = smooth_scale + (size_t)eid * hidden;
    QT*           qyrow = qy          + flat * hidden;

    // Pass 1: abs-max
    float amax = 0.f;
    for(uint32_t col=threadIdx.x*VEC; col+VEC<=hidden; col+=TBLOCK*VEC){
        const float4* px=reinterpret_cast<const float4*>(xrow+col);
        float4 vx=*px; const __half2* hx=reinterpret_cast<const __half2*>(&vx);
        for(int i=0;i<4;i++){
            float2 f=__half22float2(hx[i]);
            uint32_t c0=col+i*2, c1=c0+1;
            amax=fmaxf(amax,fabsf(f.x*srow[c0]));
            amax=fmaxf(amax,fabsf(f.y*srow[c1]));
        }
    }
    amax=warpReduceMax(amax);
    uint32_t wid=threadIdx.x/WARP_SIZE, lid=threadIdx.x%WARP_SIZE;
    if(lid==0) smem[wid]=amax;
    __syncthreads();
    if(wid==0){
        amax=(lid<WARPS)?smem[lid]:0.f;
        amax=warpReduceMax(amax);
        if(lid==0){ s_rscale=amax/quant_max; row_scale[flat]=s_rscale; }
    }
    __syncthreads();
    float inv_s=(s_rscale>0.f)?1.f/s_rscale:0.f;

    // Pass 2: quantize
    for(uint32_t col=threadIdx.x*VEC; col+VEC<=hidden; col+=TBLOCK*VEC){
        const float4* px=reinterpret_cast<const float4*>(xrow+col);
        float4 vx=*px; const __half2* hx=reinterpret_cast<const __half2*>(&vx);
        int8_t tmp[VEC];
        for(int i=0;i<4;i++){
            float2 f=__half22float2(hx[i]);
            uint32_t c0=col+i*2, c1=c0+1;
            float s0=f.x*srow[c0]*inv_s, s1=f.y*srow[c1]*inv_s;
            tmp[i*2]  =static_cast<int8_t>(fminf(fmaxf(rintf(s0),-quant_max),quant_max));
            tmp[i*2+1]=static_cast<int8_t>(fminf(fmaxf(rintf(s1),-quant_max),quant_max));
        }
        *reinterpret_cast<int64_t*>(qyrow+col)=*reinterpret_cast<int64_t*>(tmp);
    }
}

template<typename Fn>
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

void test_moe_smoothquant(uint32_t tokens, uint32_t hidden, uint32_t experts, uint32_t topk)
{
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> edist(0, experts-1);

    std::vector<InT>    hX(tokens*hidden, __float2half(0.5f));
    std::vector<ScaleT> hS(experts*hidden, 1.f);
    std::vector<IdxT>   hIds(tokens*topk);
    // Generate unique expert ids per token
    for(uint32_t t=0;t<tokens;t++){
        std::set<int> used;
        for(uint32_t k=0;k<topk;k++){
            int e=edist(rng);
            while(used.count(e)) e=edist(rng);
            used.insert(e); hIds[t*topk+k]=e;
        }
    }

    uint32_t out_rows = tokens * topk;
    std::vector<QT>     hQy(out_rows*hidden, 0);
    std::vector<ScaleT> hRS(out_rows, 0.f);

    InT *dX; ScaleT *dS, *dRS; IdxT *dIds; QT *dQy;
    CHECK_HIP(hipMalloc(&dX,   tokens*hidden*sizeof(InT)));
    CHECK_HIP(hipMalloc(&dS,   experts*hidden*sizeof(ScaleT)));
    CHECK_HIP(hipMalloc(&dIds, tokens*topk*sizeof(IdxT)));
    CHECK_HIP(hipMalloc(&dQy,  out_rows*hidden*sizeof(QT)));
    CHECK_HIP(hipMalloc(&dRS,  out_rows*sizeof(ScaleT)));
    CHECK_HIP(hipMemcpy(dX,   hX.data(),   tokens*hidden*sizeof(InT),    hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dS,   hS.data(),   experts*hidden*sizeof(ScaleT),hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dIds, hIds.data(), tokens*topk*sizeof(IdxT),     hipMemcpyHostToDevice));

    dim3 block(TBLOCK), grid(out_rows);
    auto fn=[&](){ hipLaunchKernelGGL(kernel_moe_smoothquant, grid, block, 0, 0,
                                      dX, dS, dIds, dQy, dRS,
                                      tokens, hidden, topk, experts, 127.f); };

    double ms=bench(fn);
    // Read: x[tokens,H] + smooth_scale[experts,H] + topk_ids + Write: qy[topk*tokens,H] + row_scale
    double bw=(tokens*hidden*sizeof(InT) + experts*hidden*sizeof(ScaleT) + tokens*topk*sizeof(IdxT)
               + out_rows*hidden*sizeof(QT) + out_rows*sizeof(ScaleT)) / (ms*1e-3) / 1e9;
    std::cout<<"[MoE-SmoothQuant] tokens="<<tokens<<" hidden="<<hidden
             <<" experts="<<experts<<" topk="<<topk
             <<"  "<<ms<<" ms  "<<bw<<" GB/s\n";

    CHECK_HIP(hipFree(dX)); CHECK_HIP(hipFree(dS)); CHECK_HIP(hipFree(dIds));
    CHECK_HIP(hipFree(dQy)); CHECK_HIP(hipFree(dRS));
}

int main(int argc, char* argv[])
{
    hipDeviceProp_t prop; CHECK_HIP(hipGetDeviceProperties(&prop, 0));
    std::cout<<"Device: "<<prop.name<<"\n\n";
    std::cout<<"=== rocWMMA MoE SmoothQuant (rocWMMA port) ===\n";
    // CK Tile 14_moe_smoothquant defaults: tokens=3328, hidden=4096, experts=32, topk=5
    test_moe_smoothquant(3328, 4096, 32, 5);
    test_moe_smoothquant(3328, 4096, 8,  2);
    test_moe_smoothquant(3328, 8192, 64, 5);
    return 0;
}
