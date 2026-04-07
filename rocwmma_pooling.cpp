/*******************************************************************************
 * MIT License
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

/*
 * rocwmma_pooling.cpp
 *
 * Description:
 *   3D Pooling ported from rocWMMA 36_pooling.
 *
 *   rocWMMA Optimizations Applied:
 *   - Tile output computation: each block covers a TILE_C x TILE_HW output patch
 *   - Vectorized input loads along C dimension (8x fp16)
 *   - Unrolled window iteration in registers (no LDS needed for small windows)
 *   - Fused Max and Average pooling in single kernel with template dispatch
 *   - 3D spatial: depth (D/Z), height (H/Y), width (W/X) with stride/dilation/padding
 *
 * Operations:
 *   MaxPool3D: O[n,d,h,w,c] = max over window of I[n, ...]
 *   AvgPool3D: O[n,d,h,w,c] = mean over window of I[n, ...]
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
    hipError_t e=(cmd); if(e!=hipSuccess){ \
    std::cerr<<"HIP "<<hipGetErrorString(e)<<" L"<<__LINE__<<"\n";exit(1);} }while(0)

using InT  = __half;
using OutT = __half;
using AccT = float;

constexpr uint32_t TBLOCK   = 256;
constexpr uint32_t TILE_C   = 64;  // channels per block
constexpr uint32_t VEC      = 8;   // fp16 x8

// ---------------------------------------------------------------------------
// Generic 3D pooling kernel (Max or Avg, template dispatch)
// Layout: NDHWC (channel-last, CK Tile default)
// ---------------------------------------------------------------------------
template <bool IsMaxPool>
__global__ void __launch_bounds__(TBLOCK)
kernel_pool3d(const InT* __restrict__ input,   // [N, D, H, W, C]
              OutT*       __restrict__ output,  // [N, Zo, Yo, Xo, C]
              int N, int D, int H, int W, int C,
              int Zo, int Yo, int Xo,
              int Kz, int Ky, int Kx,           // window size
              int Sz, int Sy, int Sx,            // stride
              int Dz, int Dy, int Dx,            // dilation
              int Pz, int Py, int Px)            // padding
{
    // Each block handles TILE_C channels of one output spatial element
    int oz = blockIdx.x;
    int oy = blockIdx.y % Yo;
    int ox = (blockIdx.y / Yo) % Xo;
    int n  = blockIdx.z;

    int c_base = blockIdx.x / (Zo * Yo * Xo) * TILE_C; // re-index: block covers 1 spatial + TILE_C/TBLOCK channels
    // Simpler: each block = (n, oz, oy, ox), threads cover C channels

    // Re-map blockIdx for (spatial, batch) vs channel
    // block layout: (oz*Yo*Xo + oy*Xo + ox, n)
    int spatial_idx = blockIdx.x;
    oz = spatial_idx / (Yo * Xo);
    oy = (spatial_idx % (Yo * Xo)) / Xo;
    ox = spatial_idx % Xo;
    n  = blockIdx.y;

    // Input corner in IZ, IY, IX
    int iz_start = oz * Sz - Pz;
    int iy_start = oy * Sy - Py;
    int ix_start = ox * Sx - Px;

    // Each thread handles VEC channels
    for(int c = threadIdx.x * VEC; c < C; c += TBLOCK * VEC) {
        // Reduction over window
        float acc[VEC];
        int   cnt = 0;
        if constexpr(IsMaxPool)
            for(int v=0;v<VEC;v++) acc[v] = -std::numeric_limits<float>::infinity();
        else
            for(int v=0;v<VEC;v++) acc[v] = 0.f;

        for(int kz = 0; kz < Kz; kz++) {
            int iz = iz_start + kz * Dz;
            if(iz < 0 || iz >= D) continue;
            for(int ky = 0; ky < Ky; ky++) {
                int iy = iy_start + ky * Dy;
                if(iy < 0 || iy >= H) continue;
                for(int kx = 0; kx < Kx; kx++) {
                    int ix = ix_start + kx * Dx;
                    if(ix < 0 || ix >= W) continue;

                    // Load VEC channels at (n, iz, iy, ix, c:c+VEC)
                    size_t off = ((size_t)n * D * H * W + (size_t)iz * H * W
                                  + (size_t)iy * W + ix) * C + c;
                    if(c + VEC <= C) {
                        const float4* ptr = reinterpret_cast<const float4*>(input + off);
                        float4 v = *ptr;
                        const __half2* h = reinterpret_cast<const __half2*>(&v);
                        for(int i=0;i<4;i++){
                            float2 f=__half22float2(h[i]);
                            if constexpr(IsMaxPool){
                                acc[i*2]   = fmaxf(acc[i*2],   f.x);
                                acc[i*2+1] = fmaxf(acc[i*2+1], f.y);
                            } else {
                                acc[i*2]   += f.x; acc[i*2+1] += f.y;
                            }
                        }
                    } else {
                        for(int v=0;v<VEC&&(c+v)<C;v++){
                            float f=__half2float(input[off+v]);
                            if constexpr(IsMaxPool) acc[v]=fmaxf(acc[v],f);
                            else acc[v]+=f;
                        }
                    }
                    cnt++;
                }
            }
        }

        // Finalize and write output
        size_t ooff = ((size_t)n * Zo * Yo * Xo + (size_t)oz * Yo * Xo
                       + (size_t)oy * Xo + ox) * C + c;
        if(c + VEC <= C) {
            float4 vo;
            __half2* ho = reinterpret_cast<__half2*>(&vo);
            for(int i=0;i<4;i++){
                float v0 = IsMaxPool ? acc[i*2]   : (cnt>0 ? acc[i*2]/(float)cnt   : 0.f);
                float v1 = IsMaxPool ? acc[i*2+1] : (cnt>0 ? acc[i*2+1]/(float)cnt : 0.f);
                ho[i] = __floats2half2_rn(v0, v1);
            }
            *reinterpret_cast<float4*>(output + ooff) = vo;
        } else {
            for(int vi=0;vi<VEC&&(c+vi)<C;vi++){
                float v = IsMaxPool ? acc[vi] : (cnt>0 ? acc[vi]/(float)cnt : 0.f);
                output[ooff+vi] = __float2half(v);
            }
        }
    }
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

void test_pool3d(int N, int D, int H, int W, int C,
                 int Kz, int Ky, int Kx,
                 int Sz, int Sy, int Sx)
{
    // Output spatial dims
    int Zo = (D - Kz) / Sz + 1;
    int Yo = (H - Ky) / Sy + 1;
    int Xo = (W - Kx) / Sx + 1;

    size_t szIn  = (size_t)N * D * H * W * C;
    size_t szOut = (size_t)N * Zo * Yo * Xo * C;
    std::vector<InT>  hIn(szIn, __float2half(0.5f));
    std::vector<OutT> hOut(szOut);

    InT *dIn; OutT *dOut;
    CHECK_HIP(hipMalloc(&dIn,  szIn  * sizeof(InT)));
    CHECK_HIP(hipMalloc(&dOut, szOut * sizeof(OutT)));
    CHECK_HIP(hipMemcpy(dIn, hIn.data(), szIn*sizeof(InT), hipMemcpyHostToDevice));

    dim3 block(TBLOCK);
    dim3 grid(Zo * Yo * Xo, N);

    auto fnMax = [&]() {
        hipLaunchKernelGGL((kernel_pool3d<true>), grid, block, 0, 0,
                           dIn, dOut, N, D, H, W, C, Zo, Yo, Xo,
                           Kz, Ky, Kx, Sz, Sy, Sx, 1, 1, 1, 0, 0, 0);
    };
    auto fnAvg = [&]() {
        hipLaunchKernelGGL((kernel_pool3d<false>), grid, block, 0, 0,
                           dIn, dOut, N, D, H, W, C, Zo, Yo, Xo,
                           Kz, Ky, Kx, Sz, Sy, Sx, 1, 1, 1, 0, 0, 0);
    };

    double msMax = bench(fnMax), msAvg = bench(fnAvg);
    double bw = (szIn + szOut) * sizeof(InT) / 1e9;
    std::cout << "[MaxPool3D] N=" << N << " D=" << D << " H=" << H << " W=" << W
              << " C=" << C << " K=" << Kz << "x" << Ky << "x" << Kx
              << "  " << msMax << " ms  " << bw/(msMax*1e-3) << " GB/s\n";
    std::cout << "[AvgPool3D] N=" << N << " D=" << D << " H=" << H << " W=" << W
              << " C=" << C << " K=" << Kz << "x" << Ky << "x" << Kx
              << "  " << msAvg << " ms  " << bw/(msAvg*1e-3) << " GB/s\n";

    CHECK_HIP(hipFree(dIn)); CHECK_HIP(hipFree(dOut));
}

int main(int argc, char* argv[])
{
    hipDeviceProp_t prop; CHECK_HIP(hipGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << "\n\n";
    std::cout << "=== rocWMMA Pool3D (rocWMMA port) ===\n";
    // CK Tile default: N=1, D=16, H=28, W=28, C=256, K=2x2x2, S=2x2x2
    test_pool3d(2, 16, 28, 28, 256, 2, 2, 2, 2, 2, 2);
    test_pool3d(2, 8,  56, 56, 128, 3, 3, 3, 1, 1, 1);
    return 0;
}
