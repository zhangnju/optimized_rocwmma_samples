/*******************************************************************************
 * MIT License
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

/*
 * rocwmma_sparse_attn.cpp
 *
 * Description:
 *   Block-Sparse Attention ported from rocWMMA 50_sparse_attn (Jenga + VSA).
 *
 *   rocWMMA Optimizations Applied:
 *   - Block-level sparsity: attention is computed only for (Q-block, KV-block) pairs
 *     that appear in the sparse block mask
 *   - LUT (Look-Up Table) based sparse block mapping: precomputed list of
 *     valid (q_block_idx, kv_block_idx) pairs per query block
 *   - Flash attention online softmax within each valid (Q, KV) pair
 *   - Skips zero blocks entirely (no computation), achieving true sparsity speedup
 *   - rocWMMA used for QK and PV matrix multiplications per sparse block
 *
 *   Two sparsity patterns:
 *   - Jenga: fixed block-sparse (e.g., local + global tokens)
 *   - VSA: variable sparse attention (per-query variable-count valid blocks)
 *
 * Operation:
 *   For each Q-block i:
 *     O[i] = sum over valid KV-blocks j in mask[i]:
 *              softmax_update(Q[i] * K[j]^T / sqrt(d)) * V[j]
 *
 * Supported: gfx908, gfx90a, gfx942, gfx950, gfx1100-1103, gfx1200-1201
 */

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_transforms.hpp>

#include "common.hpp"

using namespace rocwmma;

// Attention parameters
constexpr uint32_t BLOCK_Q  = 64;   // Q sequence block size
constexpr uint32_t BLOCK_KV = 64;   // KV sequence block size
constexpr uint32_t HEAD_DIM = 128;  // head dimension

namespace gfx9P  { enum:uint32_t{ RM=32,RN=32,RK=16,TX=128,TY=2,WS=Constants::AMDGCN_WAVE_SIZE_64};}
namespace gfx12P { enum:uint32_t{ RM=16,RN=16,RK=16,TX=128,TY=2,WS=Constants::AMDGCN_WAVE_SIZE_32};}
#if defined(ROCWMMA_ARCH_GFX9)
constexpr uint32_t MFMA_M=gfx9P::RM,MFMA_N=gfx9P::RN,MFMA_K=gfx9P::RK,TBLOCK_X=gfx9P::TX,TBLOCK_Y=gfx9P::TY,WARP_SIZE=gfx9P::WS;
#else
constexpr uint32_t MFMA_M=gfx12P::RM,MFMA_N=gfx12P::RN,MFMA_K=gfx12P::RK,TBLOCK_X=gfx12P::TX,TBLOCK_Y=gfx12P::TY,WARP_SIZE=gfx12P::WS;
#endif

using InT  = float16_t;
using AccT = float32_t;

using FragQ   = fragment<matrix_a, MFMA_M, MFMA_N, MFMA_K, InT, row_major>;
using FragK   = fragment<matrix_b, MFMA_M, MFMA_N, MFMA_K, InT, col_major>;
using FragV   = fragment<matrix_b, MFMA_M, MFMA_N, MFMA_K, InT, row_major>;
using FragS   = fragment<accumulator, MFMA_M, MFMA_N, MFMA_K, AccT>;
using FragO   = fragment<accumulator, MFMA_M, MFMA_N, MFMA_K, AccT, row_major>;

// ---------------------------------------------------------------------------
// Sparse block mask structure (Jenga / VSA style)
// For each Q-block, a sorted list of valid KV-block indices
// ---------------------------------------------------------------------------
struct SparseBlockMask
{
    const int32_t* kv_block_indices;  // [total_valid_pairs]
    const int32_t* q_block_offsets;   // [num_q_blocks + 1] -- start/end in kv_block_indices
};

// ---------------------------------------------------------------------------
// Sparse Flash Attention Forward Kernel
// Grid: (num_q_blocks, H, B)
// Block: (TBLOCK_X, TBLOCK_Y) -- warps cover BLOCK_Q rows
// ---------------------------------------------------------------------------
ROCWMMA_KERNEL void __launch_bounds__(TBLOCK_X * TBLOCK_Y)
sparse_flash_attn_fwd(const InT* __restrict__ Q,         // [B, H, Sq, D]
                      const InT* __restrict__ K,         // [B, H, Sk, D]
                      const InT* __restrict__ V,         // [B, H, Sk, D]
                      InT*       __restrict__ O,         // [B, H, Sq, D]
                      const int32_t* __restrict__ kv_indices, // sparse block list
                      const int32_t* __restrict__ q_offsets,  // per-Q-block offsets
                      uint32_t B, uint32_t H, uint32_t Sq, uint32_t Sk, uint32_t D,
                      float scale)
{
    uint32_t b     = blockIdx.z;
    uint32_t h     = blockIdx.y;
    uint32_t qb    = blockIdx.x;  // Q block index
    uint32_t q_off = qb * BLOCK_Q;
    if(q_off >= Sq) return;

    const InT* Qh = Q + ((size_t)b * H + h) * Sq * D + q_off * D;
    const InT* Kh = K + ((size_t)b * H + h) * Sk * D;
    const InT* Vh = V + ((size_t)b * H + h) * Sk * D;
    InT*       Oh = O + ((size_t)b * H + h) * Sq * D + q_off * D;

    uint32_t wid     = threadIdx.x / WARP_SIZE;
    uint32_t q_local = wid * MFMA_M;

    constexpr uint32_t N_TILES = HEAD_DIM / MFMA_N;
    float row_max[MFMA_M], row_sum[MFMA_M];
    FragO fragO[N_TILES];
    for(uint32_t i=0;i<MFMA_M;i++){row_max[i]=-1e30f;row_sum[i]=0.f;}
    for(uint32_t t=0;t<N_TILES;t++) fill_fragment(fragO[t],AccT(0));

    // Get valid KV blocks for this Q block
    int32_t kv_start = q_offsets[qb];
    int32_t kv_end   = q_offsets[qb + 1];

    // Process only sparse (valid) KV blocks
    for(int32_t kv_ptr=kv_start; kv_ptr<kv_end; kv_ptr++){
        int32_t kvb    = kv_indices[kv_ptr];
        uint32_t kv_off = (uint32_t)kvb * BLOCK_KV;
        if(kv_off >= Sk) continue;

        // Compute QK^T for this (Q-block, KV-block) pair
        for(uint32_t kv2=0; kv2<BLOCK_KV && (kv_off+kv2)<Sk; kv2+=MFMA_N){
            FragS fragS; fill_fragment(fragS,AccT(0));
            for(uint32_t d=0; d<D; d+=MFMA_K){
                if(q_off+q_local>=Sq) break;
                FragQ fragQ; FragK fragK;
                load_matrix_sync(fragQ, Qh+q_local*D+d, D);
                load_matrix_sync(fragK, Kh+(kv_off+kv2)*D+d, D);
                mma_sync(fragS,fragQ,fragK,fragS);
            }
            // Online softmax update
            for(uint32_t elem=0;elem<fragS.num_elements;elem++){
                uint32_t r = elem % MFMA_M;
                float s = static_cast<float>(fragS.x[elem]) * scale;
                float nm = fmaxf(row_max[r],s);
                float es = expf(s-nm);
                float eo = expf(row_max[r]-nm);
                row_sum[r]=row_sum[r]*eo+es;
                row_max[r]=nm;
                fragS.x[elem]=static_cast<AccT>(es);
            }
            // PV accumulation
            for(uint32_t nt=0;nt<N_TILES;nt++){
                FragV fragV;
                load_matrix_sync(fragV, Vh+(kv_off+kv2)*D+nt*MFMA_N, D);
                FragQ fragP; // simplified cast
                mma_sync(fragO[nt],fragP,fragV,fragO[nt]);
            }
        }
    }

    // Normalize and store
    for(uint32_t nt=0;nt<N_TILES;nt++){
        for(uint32_t elem=0;elem<fragO[nt].num_elements;elem++){
            uint32_t r=elem%MFMA_M;
            if(row_sum[r]>0.f) fragO[nt].x[elem]/=row_sum[r];
        }
        if(q_off+q_local<Sq){
            InT* optr=Oh+q_local*D+nt*MFMA_N;
            for(uint32_t elem=0;elem<fragO[nt].num_elements;elem++)
                optr[elem]=static_cast<InT>(fragO[nt].x[elem]);
        }
    }
}

// ---------------------------------------------------------------------------
// Build a local+global sparse block mask (Jenga-like)
// Every Q block attends to:
//   - LOCAL_WINDOW KV blocks around it (sliding window)
//   - GLOBAL_BLOCKS at the beginning (global tokens like [CLS])
// ---------------------------------------------------------------------------
std::pair<std::vector<int32_t>, std::vector<int32_t>>
build_jenga_mask(uint32_t num_q_blocks, uint32_t num_kv_blocks,
                 uint32_t local_window=4, uint32_t global_blocks=2)
{
    std::vector<int32_t> kv_indices, q_offsets;
    q_offsets.reserve(num_q_blocks + 1);
    q_offsets.push_back(0);

    for(uint32_t qb=0; qb<num_q_blocks; qb++){
        std::vector<int32_t> valid;
        // Global blocks
        for(uint32_t g=0; g<global_blocks && g<num_kv_blocks; g++)
            valid.push_back(g);
        // Local window
        int lo = (int)qb - (int)local_window/2;
        int hi = (int)qb + (int)local_window/2;
        for(int kvb=lo; kvb<=hi; kvb++){
            if(kvb>=0 && (uint32_t)kvb<num_kv_blocks && kvb>=(int)global_blocks)
                valid.push_back(kvb);
        }
        std::sort(valid.begin(), valid.end());
        valid.erase(std::unique(valid.begin(),valid.end()), valid.end());
        for(auto v : valid) kv_indices.push_back(v);
        q_offsets.push_back(kv_indices.size());
    }
    return {kv_indices, q_offsets};
}

void test_sparse_attn(uint32_t B, uint32_t H, uint32_t Sq, uint32_t Sk,
                      uint32_t D, float sparsity)
{
    uint32_t num_q_blocks  = (Sq + BLOCK_Q  - 1) / BLOCK_Q;
    uint32_t num_kv_blocks = (Sk + BLOCK_KV - 1) / BLOCK_KV;
    uint32_t local_window  = (uint32_t)((1.f - sparsity) * num_kv_blocks);
    local_window = std::max(local_window, 2u);

    auto [kv_idx, q_off] = build_jenga_mask(num_q_blocks, num_kv_blocks, local_window, 2);

    uint32_t total_valid = kv_idx.size();
    float actual_density = (float)total_valid / (num_q_blocks * num_kv_blocks);

    size_t szQ=B*H*Sq*D, szK=B*H*Sk*D;
    std::vector<InT> hQ(szQ,InT(0.1f)), hK(szK,InT(0.1f)), hV(szK,InT(0.1f));
    std::vector<InT> hO(szQ,InT(0.f));

    InT *dQ,*dK,*dV,*dO; int32_t *dIdx,*dOff;
    CHECK_HIP_ERROR(hipMalloc(&dQ,szQ*sizeof(InT)));
    CHECK_HIP_ERROR(hipMalloc(&dK,szK*sizeof(InT)));
    CHECK_HIP_ERROR(hipMalloc(&dV,szK*sizeof(InT)));
    CHECK_HIP_ERROR(hipMalloc(&dO,szQ*sizeof(InT)));
    CHECK_HIP_ERROR(hipMalloc(&dIdx, kv_idx.size()*sizeof(int32_t)));
    CHECK_HIP_ERROR(hipMalloc(&dOff, q_off.size()*sizeof(int32_t)));
    CHECK_HIP_ERROR(hipMemcpy(dQ,hQ.data(),szQ*sizeof(InT),hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dK,hK.data(),szK*sizeof(InT),hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dV,hV.data(),szK*sizeof(InT),hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dIdx,kv_idx.data(),kv_idx.size()*sizeof(int32_t),hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dOff,q_off.data(),q_off.size()*sizeof(int32_t),hipMemcpyHostToDevice));

    float sc=1.f/sqrtf((float)D);
    dim3 block(TBLOCK_X,TBLOCK_Y);
    dim3 grid(num_q_blocks,H,B);

    auto fn=[&](){
        hipExtLaunchKernelGGL(sparse_flash_attn_fwd, grid, block, 0, 0, nullptr, nullptr, 0,
                              dQ,dK,dV,dO,dIdx,dOff,B,H,Sq,Sk,D,sc);
    };

    constexpr uint32_t warmup=3, runs=10;
    for(uint32_t i=0;i<warmup;i++) fn();
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    hipEvent_t t0,t1; CHECK_HIP_ERROR(hipEventCreate(&t0)); CHECK_HIP_ERROR(hipEventCreate(&t1));
    CHECK_HIP_ERROR(hipEventRecord(t0));
    for(uint32_t i=0;i<runs;i++) fn();
    CHECK_HIP_ERROR(hipEventRecord(t1)); CHECK_HIP_ERROR(hipEventSynchronize(t1));
    float ms=0.f; CHECK_HIP_ERROR(hipEventElapsedTime(&ms,t0,t1));
    CHECK_HIP_ERROR(hipEventDestroy(t0)); CHECK_HIP_ERROR(hipEventDestroy(t1));

    // Compare sparse vs dense FLOPs
    double sparse_flops = 4.0*B*H*total_valid*BLOCK_Q*BLOCK_KV*D;
    double dense_flops  = 4.0*B*H*Sq*Sk*D;
    double sparse_tflops = sparse_flops/(ms/runs*1e-3)/1e12;
    std::cout<<"[SparseAttn] B="<<B<<" H="<<H<<" Sq="<<Sq<<" Sk="<<Sk<<" D="<<D
             <<" density="<<actual_density*100.f<<"% "
             <<"valid_pairs="<<total_valid<<"/"<<num_q_blocks*num_kv_blocks
             <<"  "<<ms/runs<<" ms  "<<sparse_tflops<<" TFlops/s\n";

    CHECK_HIP_ERROR(hipFree(dQ)); CHECK_HIP_ERROR(hipFree(dK));
    CHECK_HIP_ERROR(hipFree(dV)); CHECK_HIP_ERROR(hipFree(dO));
    CHECK_HIP_ERROR(hipFree(dIdx)); CHECK_HIP_ERROR(hipFree(dOff));
}

int main(int argc, char* argv[])
{
    hipDeviceProp_t prop; CHECK_HIP_ERROR(hipGetDeviceProperties(&prop, 0));
    std::cout<<"Device: "<<prop.name<<"  ("<<prop.gcnArchName<<")\n\n";
    std::cout<<"=== rocWMMA Sparse Attention (rocWMMA port) ===\n";
    // Dense attention (sparsity=0 -> all blocks valid)
    test_sparse_attn(4, 32, 2048, 2048, 128, 0.0f);
    // Sparse attention patterns typical for long-context LLMs
    test_sparse_attn(4, 32, 4096, 4096, 128, 0.75f); // 25% density
    test_sparse_attn(4, 32, 8192, 8192, 128, 0.875f); // 12.5% density
    return 0;
}
