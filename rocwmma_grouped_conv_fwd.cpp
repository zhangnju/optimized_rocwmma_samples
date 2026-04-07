/*******************************************************************************
 * MIT License
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

/*
 * rocwmma_grouped_conv_fwd.cpp
 *
 * Description:
 *   Grouped Convolution Forward ported from rocWMMA 20_grouped_convolution.
 *   Implements 2D grouped convolution via im2col + grouped GEMM using rocWMMA.
 *
 *   rocWMMA Optimizations Applied:
 *   - Group convolution decomposes to G independent GEMMs:
 *       Y[g,n,ho,wo,c_out/G] = sum_{c_in/G,kh,kw} X[g,n,hi,wi,c_in/G] * W[g,c_out/G,c_in/G,kh,kw]
 *   - Two-phase approach: im2col expands input, then GEMM per group
 *   - Im2col fused with group stride: each group's im2col reads a strided sub-tensor
 *   - rocWMMA double-buffer pipeline for the GEMM phase
 *   - Benchmark: forward pass only (bwd_weight / bwd_data are follow-on kernels)
 *
 * Operation (NHWC layout):
 *   Y[N, Ho, Wo, G, C_out/G] = Conv2D(X[N, H, W, G, C_in/G], W[G, C_out/G, C_in/G, Kh, Kw])
 *
 * Supported: all GPU targets (WMMA-compatible path for gfx11/gfx12)
 */

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_transforms.hpp>

#include "common.hpp"

using namespace rocwmma;

namespace gfx9P  { enum:uint32_t{ RM=32,RN=32,RK=16,BM=2,BN=2,TX=128,TY=2,WS=Constants::AMDGCN_WAVE_SIZE_64};}
namespace gfx12P { enum:uint32_t{ RM=16,RN=16,RK=16,BM=4,BN=4,TX=64, TY=2,WS=Constants::AMDGCN_WAVE_SIZE_32};}
#if defined(ROCWMMA_ARCH_GFX9)
constexpr uint32_t RM=gfx9P::RM,RN=gfx9P::RN,RK=gfx9P::RK,BM=gfx9P::BM,BN=gfx9P::BN,TX=gfx9P::TX,TY=gfx9P::TY,WS=gfx9P::WS;
#else
constexpr uint32_t RM=gfx12P::RM,RN=gfx12P::RN,RK=gfx12P::RK,BM=gfx12P::BM,BN=gfx12P::BN,TX=gfx12P::TX,TY=gfx12P::TY,WS=gfx12P::WS;
#endif
constexpr uint32_t ROCWMMA_M=RM,ROCWMMA_N=RN,ROCWMMA_K=RK,BLOCKS_M=BM,BLOCKS_N=BN,TBLOCK_X=TX,TBLOCK_Y=TY,WARP_SIZE=WS;
constexpr uint32_t WARP_TILE_M=BM*RM, WARP_TILE_N=BN*RN;
constexpr uint32_t MACRO_TILE_M=(TX/WS)*WARP_TILE_M, MACRO_TILE_N=TY*WARP_TILE_N, MACRO_TILE_K=RK;

using InT=float16_t; using OutT=float16_t; using AccT=float32_t;
using AL=col_major; using BL=row_major; using CL=row_major; using LL=col_major;
using MA=fragment<matrix_a,WARP_TILE_M,WARP_TILE_N,ROCWMMA_K,InT,AL>;
using MB=fragment<matrix_b,WARP_TILE_M,WARP_TILE_N,ROCWMMA_K,InT,BL>;
using MAcc=fragment<accumulator,WARP_TILE_M,WARP_TILE_N,ROCWMMA_K,AccT>;
using MO=fragment<accumulator,WARP_TILE_M,WARP_TILE_N,ROCWMMA_K,OutT,CL>;
using CoopS=fragment_scheduler::coop_row_major_2d<TBLOCK_X,TBLOCK_Y>;
using GRA=fragment<matrix_a,MACRO_TILE_M,MACRO_TILE_N,MACRO_TILE_K,InT,AL,CoopS>;
using GRB=fragment<matrix_b,MACRO_TILE_M,MACRO_TILE_N,MACRO_TILE_K,InT,BL,CoopS>;
using LWA=apply_data_layout_t<GRA,LL>;
using LWB=apply_data_layout_t<apply_transpose_t<GRB>,LL>;
using LRA=apply_data_layout_t<MA,LL>;
using LRB=apply_data_layout_t<apply_transpose_t<MB>,LL>;
constexpr uint32_t ldHA=GetIOShape_t<LWA>::BlockHeight, ldHB=GetIOShape_t<LWB>::BlockHeight;
constexpr uint32_t ldHt=ldHA+ldHB, ldWd=MACRO_TILE_K, szLds=ldHt*ldWd;
constexpr uint32_t ldsld=std::is_same_v<LL,row_major>?ldWd:ldHt;
ROCWMMA_DEVICE __forceinline__ auto toLWA(GRA const& g){return apply_data_layout<LL>(g);}
ROCWMMA_DEVICE __forceinline__ auto toLWB(GRB const& g){return apply_data_layout<LL>(apply_transpose(g));}
ROCWMMA_DEVICE __forceinline__ auto toMA(LRA const& l){return apply_data_layout<AL>(l);}
ROCWMMA_DEVICE __forceinline__ auto toMB(LRB const& l){return apply_data_layout<BL>(apply_transpose(l));}

// ---------------------------------------------------------------------------
// Im2col kernel (NHWGC layout, grouped)
// Expands X[N,H,W,G,C/G] -> Xcol[G, N*Ho*Wo, Kh*Kw*C/G]
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
kernel_grouped_im2col(const InT* __restrict__ X,
                      InT*       __restrict__ Xcol,
                      int N, int H, int W, int G, int C_per_g,
                      int Ho, int Wo, int Kh, int Kw,
                      int Sh, int Sw, int Ph, int Pw)
{
    // Each thread writes one element of Xcol[g, m, k]
    // m = n*Ho*Wo + ho*Wo + wo, k = kh*Kw*C_per_g + kw*C_per_g + c
    int g = blockIdx.z;
    int m_total = N * Ho * Wo;
    int K       = Kh * Kw * C_per_g;
    int idx     = blockIdx.x * 256 + threadIdx.x;
    if(idx >= m_total * K) return;

    int c   = idx % C_per_g; int r = idx / C_per_g;
    int kw  = r % Kw;        r /= Kw;
    int kh  = r % Kh;        r /= Kh;
    int wo  = r % Wo;        r /= Wo;
    int ho  = r % Ho;
    int n   = r / Ho;

    int ih = ho*Sh + kh - Ph, iw = wo*Sw + kw - Pw;
    InT val = (ih>=0 && ih<H && iw>=0 && iw<W)
              ? X[((n*H+ih)*W+iw)*G*C_per_g + g*C_per_g + c]
              : InT(0);

    // Xcol[g, m, k] row-major
    Xcol[(size_t)g * m_total * K + (size_t)(n*Ho*Wo+ho*Wo+wo) * K
         + kh*Kw*C_per_g + kw*C_per_g + c] = val;
}

// ---------------------------------------------------------------------------
// Per-group GEMM kernel (blockIdx.z = group)
// Xcol[g, M, K_conv] * W[g, K_conv, C_out/g] -> Y[g, M, C_out/g]
// ---------------------------------------------------------------------------
ROCWMMA_KERNEL void __launch_bounds__(TBLOCK_X*TBLOCK_Y)
grouped_conv_gemm_kernel(uint32_t M, uint32_t N_out, uint32_t K_conv,
                         const InT* __restrict__ Xcol,   // [G, M, K_conv] col-major M
                         const InT* __restrict__ W,      // [G, K_conv, N_out] row-major
                         OutT*      __restrict__ Y,       // [G, M, N_out]
                         uint32_t lda, uint32_t ldb, uint32_t ldc)
{
    uint32_t g   = blockIdx.z;
    const InT* A = Xcol + (size_t)g * M * K_conv;
    const InT* B = W    + (size_t)g * K_conv * N_out;
    OutT*      C = Y    + (size_t)g * M * N_out;

    constexpr auto wTile=make_coord2d(WARP_TILE_M,WARP_TILE_N);
    auto lWC=make_coord2d(threadIdx.x/WARP_SIZE,threadIdx.y);
    auto lWOff=lWC*wTile;
    auto mTC=make_coord2d(blockIdx.x,blockIdx.y)*make_coord2d(MACRO_TILE_M,MACRO_TILE_N);
    auto wTC=mTC+lWOff;
    if(get<0>(wTC)+WARP_TILE_M>M || get<1>(wTC)+WARP_TILE_N>N_out) return;

    using GRMapA=GetDataLayout_t<GRA>; using GRMapB=GetDataLayout_t<GRB>;
    auto rA=GRMapA::fromMatrixCoord(make_coord2d(get<0>(mTC),0u),lda);
    auto rB=GRMapB::fromMatrixCoord(make_coord2d(0u,get<1>(mTC)),ldb);
    auto kA=GRMapA::fromMatrixCoord(make_coord2d(0u,MACRO_TILE_K),lda);
    auto kB=GRMapB::fromMatrixCoord(make_coord2d(MACRO_TILE_K,0u),ldb);

    HIP_DYNAMIC_SHARED(void*,lmem);
    auto lOA=0u, lOB=GetDataLayout_t<LWA>::fromMatrixCoord(make_coord2d(ldHA,0u),ldsld);
    auto* lLo=reinterpret_cast<InT*>(lmem), *lHi=lLo+szLds;
    auto lRA=lOA+GetDataLayout_t<LRA>::fromMatrixCoord(make_coord2d(get<0>(lWOff),0u),ldsld);
    auto lRB=lOB+GetDataLayout_t<LRB>::fromMatrixCoord(make_coord2d(get<1>(lWOff),0u),ldsld);

    GRA grA; GRB grB;
    load_matrix_sync(grA,A+rA,lda); load_matrix_sync(grB,B+rB,ldb);
    rA+=kA; rB+=kB;
    store_matrix_sync(lLo+lOA,toLWA(grA),ldsld); store_matrix_sync(lLo+lOB,toLWB(grB),ldsld);
    MAcc fAcc; fill_fragment(fAcc,AccT(0));
    synchronize_workgroup();

    for(uint32_t ks=MACRO_TILE_K;ks<K_conv;ks+=MACRO_TILE_K){
        LRA lrA; LRB lrB;
        load_matrix_sync(lrA,lLo+lRA,ldsld); load_matrix_sync(lrB,lLo+lRB,ldsld);
        load_matrix_sync(grA,A+rA,lda); load_matrix_sync(grB,B+rB,ldb);
        rA+=kA; rB+=kB;
        mma_sync(fAcc,toMA(lrA),toMB(lrB),fAcc);
        store_matrix_sync(lHi+lOA,toLWA(grA),ldsld); store_matrix_sync(lHi+lOB,toLWB(grB),ldsld);
        synchronize_workgroup();
        auto* t=lLo; lLo=lHi; lHi=t;
    }
    LRA lrA; LRB lrB;
    load_matrix_sync(lrA,lLo+lRA,ldsld); load_matrix_sync(lrB,lLo+lRB,ldsld);
    mma_sync(fAcc,toMA(lrA),toMB(lrB),fAcc);

    MO fOut;
    for(uint32_t i=0;i<fAcc.num_elements;i++) fOut.x[i]=static_cast<OutT>(fAcc.x[i]);
    store_matrix_sync(C+GetDataLayout_t<MO>::fromMatrixCoord(wTC,ldc), fOut, ldc);
}

void test_grouped_conv_fwd(int N, int H, int W, int G, int C_in, int C_out,
                            int Kh, int Kw, int Sh=1, int Sw=1)
{
    int Ho = (H-Kh)/Sh+1, Wo = (W-Kw)/Sw+1;
    int C_per_g_in  = C_in  / G;
    int C_per_g_out = C_out / G;
    int M    = N * Ho * Wo;
    int K_cv = Kh * Kw * C_per_g_in;
    // Align
    int M_pad = ((M+MACRO_TILE_M-1)/MACRO_TILE_M)*MACRO_TILE_M;
    int N_pad = ((C_per_g_out+MACRO_TILE_N-1)/MACRO_TILE_N)*MACRO_TILE_N;
    int K_pad = ((K_cv+MACRO_TILE_K-1)/MACRO_TILE_K)*MACRO_TILE_K;

    size_t szX = (size_t)N*H*W*G*C_per_g_in;
    size_t szW = (size_t)G*K_pad*N_pad;
    size_t szXcol = (size_t)G*M_pad*K_pad;
    size_t szY = (size_t)G*M_pad*N_pad;

    std::vector<InT>  hX(szX,InT(0.1f)), hW(szW,InT(0.1f));
    InT *dX,*dWf,*dXcol; OutT *dY;
    CHECK_HIP_ERROR(hipMalloc(&dX,    szX   *sizeof(InT)));
    CHECK_HIP_ERROR(hipMalloc(&dWf,   szW   *sizeof(InT)));
    CHECK_HIP_ERROR(hipMalloc(&dXcol, szXcol*sizeof(InT)));
    CHECK_HIP_ERROR(hipMalloc(&dY,    szY   *sizeof(OutT)));
    CHECK_HIP_ERROR(hipMemcpy(dX,  hX.data(), szX*sizeof(InT),  hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dWf, hW.data(), szW*sizeof(InT),  hipMemcpyHostToDevice));

    int total_im2col = M * K_cv;
    dim3 blk256(256); dim3 gridIm2col((total_im2col+255)/256, 1, G);
    dim3 blkG(TBLOCK_X,TBLOCK_Y);
    dim3 gridG((M_pad+MACRO_TILE_M-1)/MACRO_TILE_M, (N_pad+MACRO_TILE_N-1)/MACRO_TILE_N, G);
    uint32_t ldsB=2u*sizeof(InT)*szLds;

    auto fn=[&](){
        hipLaunchKernelGGL(kernel_grouped_im2col, gridIm2col, blk256, 0, 0,
                           dX, dXcol, N, H, W, G, C_per_g_in, Ho, Wo, Kh, Kw, Sh, Sw, 0, 0);
        hipExtLaunchKernelGGL(grouped_conv_gemm_kernel, gridG, blkG, ldsB, 0,
                              nullptr, nullptr, 0,
                              (uint32_t)M_pad, (uint32_t)N_pad, (uint32_t)K_pad,
                              dXcol, dWf, dY, (uint32_t)M_pad, (uint32_t)N_pad, (uint32_t)N_pad);
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

    double flops = 2.0*N*Ho*Wo*C_out*Kh*Kw*C_per_g_in;
    double tflops = flops/(ms/runs*1e-3)/1e12;
    std::cout << "[GroupedConvFwd] N=" << N << " H=" << H << " W=" << W
              << " G=" << G << " Cin=" << C_in << " Cout=" << C_out
              << " K=" << Kh << "x" << Kw
              << "  " << ms/runs << " ms  " << tflops << " TFlops/s\n";

    CHECK_HIP_ERROR(hipFree(dX)); CHECK_HIP_ERROR(hipFree(dWf));
    CHECK_HIP_ERROR(hipFree(dXcol)); CHECK_HIP_ERROR(hipFree(dY));
}

int main(int argc, char* argv[])
{
    hipDeviceProp_t prop; CHECK_HIP_ERROR(hipGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << "  (" << prop.gcnArchName << ")\n\n";
    std::cout << "=== rocWMMA Grouped Conv Fwd (rocWMMA port) ===\n";
    // CK Tile 20_grouped_convolution typical configs (ResNet-like)
    test_grouped_conv_fwd(8,  56, 56,  1, 64,  64,  3, 3);
    test_grouped_conv_fwd(8,  28, 28,  1, 128, 128, 3, 3);
    test_grouped_conv_fwd(8,  14, 14,  1, 256, 256, 3, 3);
    test_grouped_conv_fwd(8,  7,  7,   1, 512, 512, 3, 3);
    // Grouped (G=4)
    test_grouped_conv_fwd(8,  28, 28,  4, 128, 128, 3, 3);
    test_grouped_conv_fwd(8,  14, 14,  4, 256, 256, 3, 3);
    return 0;
}
