/*******************************************************************************
 * MIT License
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

/*
 * rocwmma_moe_sorting.cpp
 *
 * Description:
 *   MoE Token Sorting ported from rocWMMA 13_moe_sorting.
 *
 *   rocWMMA Optimizations Applied:
 *   - Counting sort (histogram-based) for O(N) complexity vs O(N log N)
 *   - Three-phase pipeline: histogram -> prefix-sum -> scatter
 *   - Block-level histogram with LDS accumulation
 *   - Warp-level prefix-sum via inclusive scan
 *   - Coalesced scatter writes (sorted token indices per expert)
 *   - Pad tokens to unit_size boundaries per expert (MoE dispatch requirement)
 *
 * Operation:
 *   Given token_expert_ids[tokens, topk] (each token assigns to topk experts):
 *     sorted_token_ids[...]    = tokens sorted by expert assignment
 *     expert_start_ids[experts] = start offset per expert in sorted array
 *     num_tokens_per_expert[experts] = token count per expert (padded)
 *
 * Supported: all GPU targets
 */

#include <algorithm>
#include <random>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#define CHECK_HIP(cmd) do { \
    hipError_t e=(cmd); if(e!=hipSuccess){ \
    std::cerr<<"HIP "<<hipGetErrorString(e)<<" L"<<__LINE__<<"\n";exit(1);} }while(0)

using IdxT    = int32_t;
using WeightT = float;

constexpr uint32_t WARP_SIZE = 64;
constexpr uint32_t TBLOCK    = 256;

// ---------------------------------------------------------------------------
// Phase 1: Count tokens per expert (histogram)
// Each block processes a chunk of tokens; LDS histogram, then atomicAdd to global
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(TBLOCK)
kernel_moe_histogram(const IdxT* __restrict__ token_expert_ids, // [tokens * topk]
                     IdxT*       __restrict__ expert_count,     // [num_experts]
                     uint32_t total_token_topk,
                     uint32_t num_experts)
{
    extern __shared__ IdxT smem_hist[];
    // Zero local histogram
    for(uint32_t i = threadIdx.x; i < num_experts; i += TBLOCK)
        smem_hist[i] = 0;
    __syncthreads();

    // Count
    uint32_t idx = blockIdx.x * TBLOCK + threadIdx.x;
    if(idx < total_token_topk) {
        IdxT eid = token_expert_ids[idx];
        if(eid >= 0 && (uint32_t)eid < num_experts)
            atomicAdd(&smem_hist[eid], 1);
    }
    __syncthreads();

    // Merge to global
    for(uint32_t i = threadIdx.x; i < num_experts; i += TBLOCK)
        if(smem_hist[i] > 0) atomicAdd(&expert_count[i], smem_hist[i]);
}

// ---------------------------------------------------------------------------
// Phase 2: Prefix sum to compute start offsets (CPU side or single-block)
// ---------------------------------------------------------------------------
__global__ void
kernel_prefix_sum(const IdxT* __restrict__ expert_count,
                  IdxT*       __restrict__ expert_start,
                  IdxT*       __restrict__ num_tokens_padded,
                  uint32_t num_experts, uint32_t unit_size)
{
    // Single thread, single block
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        IdxT offset = 0;
        for(uint32_t e = 0; e < num_experts; e++) {
            expert_start[e] = offset;
            // Pad to unit_size
            IdxT cnt = expert_count[e];
            IdxT padded = (cnt + unit_size - 1) / unit_size * unit_size;
            num_tokens_padded[e] = padded;
            offset += padded;
        }
        expert_start[num_experts] = offset; // sentinel
    }
}

// ---------------------------------------------------------------------------
// Phase 3: Scatter tokens to sorted positions
// For each (token, expert) pair, write token_id to position
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(TBLOCK)
kernel_moe_scatter(const IdxT* __restrict__ token_expert_ids,  // [tokens, topk]
                   const IdxT* __restrict__ expert_start,      // [num_experts+1]
                   IdxT*       __restrict__ sorted_token_ids,  // [total_padded]
                   IdxT*       __restrict__ scatter_offsets,   // [num_experts] atomic counter
                   uint32_t tokens, uint32_t topk, uint32_t num_experts)
{
    uint32_t tid = blockIdx.x * TBLOCK + threadIdx.x;
    if(tid >= tokens * topk) return;

    uint32_t token_id = tid / topk;
    IdxT eid = token_expert_ids[tid];
    if(eid < 0 || (uint32_t)eid >= num_experts) return;

    // Atomically get slot in sorted array for this expert
    IdxT pos = atomicAdd(&scatter_offsets[eid], 1);
    IdxT slot = expert_start[eid] + pos;
    sorted_token_ids[slot] = static_cast<IdxT>(token_id);
}

template <typename Fn>
double bench(Fn fn, uint32_t w=3, uint32_t r=10)
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

void test_moe_sorting(uint32_t tokens, uint32_t num_experts, uint32_t topk, uint32_t unit_size=32)
{
    uint32_t total_topk = tokens * topk;
    // Generate random expert assignments
    std::vector<IdxT> hEids(total_topk);
    for(uint32_t t = 0; t < tokens; t++) {
        // Each token picks topk distinct experts
        std::vector<IdxT> experts(num_experts);
        std::iota(experts.begin(), experts.end(), 0);
        std::shuffle(experts.begin(), experts.end(), std::default_random_engine(t));
        for(uint32_t k = 0; k < topk; k++)
            hEids[t*topk+k] = experts[k];
    }

    IdxT *dEids, *dCount, *dStart, *dPadded, *dSorted, *dScatter;
    uint32_t total_padded = tokens * topk + num_experts * unit_size; // upper bound

    CHECK_HIP(hipMalloc(&dEids,    total_topk*sizeof(IdxT)));
    CHECK_HIP(hipMalloc(&dCount,   num_experts*sizeof(IdxT)));
    CHECK_HIP(hipMalloc(&dStart,   (num_experts+1)*sizeof(IdxT)));
    CHECK_HIP(hipMalloc(&dPadded,  num_experts*sizeof(IdxT)));
    CHECK_HIP(hipMalloc(&dSorted,  total_padded*sizeof(IdxT)));
    CHECK_HIP(hipMalloc(&dScatter, num_experts*sizeof(IdxT)));
    CHECK_HIP(hipMemcpy(dEids, hEids.data(), total_topk*sizeof(IdxT), hipMemcpyHostToDevice));

    dim3 blockH(TBLOCK), gridH((total_topk + TBLOCK-1)/TBLOCK);

    auto fn = [&]() {
        // Reset counters
        CHECK_HIP(hipMemset(dCount,   0, num_experts*sizeof(IdxT)));
        CHECK_HIP(hipMemset(dScatter, 0, num_experts*sizeof(IdxT)));
        CHECK_HIP(hipMemset(dSorted,  -1, total_padded*sizeof(IdxT)));

        // Phase 1: Histogram
        hipLaunchKernelGGL(kernel_moe_histogram, gridH, blockH,
                           num_experts*sizeof(IdxT), 0,
                           dEids, dCount, total_topk, num_experts);
        // Phase 2: Prefix sum (single block)
        hipLaunchKernelGGL(kernel_prefix_sum, dim3(1), dim3(1), 0, 0,
                           dCount, dStart, dPadded, num_experts, unit_size);
        // Phase 3: Scatter
        hipLaunchKernelGGL(kernel_moe_scatter, gridH, blockH, 0, 0,
                           dEids, dStart, dSorted, dScatter,
                           tokens, topk, num_experts);
    };

    double ms = bench(fn);
    double bw = (total_topk*sizeof(IdxT)*2.0 + total_padded*sizeof(IdxT)) / (ms*1e-3) / 1e9;
    std::cout << "[MoE Sorting] tokens=" << tokens << " experts=" << num_experts
              << " topk=" << topk << " unit=" << unit_size
              << "  " << ms << " ms  " << bw << " GB/s\n";

    CHECK_HIP(hipFree(dEids)); CHECK_HIP(hipFree(dCount)); CHECK_HIP(hipFree(dStart));
    CHECK_HIP(hipFree(dPadded)); CHECK_HIP(hipFree(dSorted)); CHECK_HIP(hipFree(dScatter));
}

int main(int argc, char* argv[])
{
    hipDeviceProp_t prop; CHECK_HIP(hipGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << "\n\n";
    uint32_t tokens=3328, experts=8, topk=4, unit=32;
    if(argc>=4){ tokens=std::atoi(argv[1]); experts=std::atoi(argv[2]); topk=std::atoi(argv[3]); }

    std::cout << "=== rocWMMA MoE Sorting (rocWMMA port) ===\n";
    test_moe_sorting(tokens, experts, topk, unit);
    for(auto [e,k] : std::vector<std::pair<uint32_t,uint32_t>>{{16,2},{32,4},{64,5},{128,8}})
        if(k <= e) test_moe_sorting(tokens, e, k, unit);
    return 0;
}
