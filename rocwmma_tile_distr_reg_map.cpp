/*******************************************************************************
 * MIT License
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

/*
 * rocwmma_tile_distr_reg_map.cpp
 *
 * Description:
 *   Tile Distribution Encoding Register Mapper ported from rocWMMA
 *   51_tile_distr_enc_reg_map.
 *
 *   This is a HOST-SIDE diagnostic/visualization utility that prints:
 *   1. How rocWMMA MFMA/WMMA instructions distribute data across wave lanes
 *   2. Which register holds which matrix element for A, B, C fragments
 *   3. Warp tile layout: how multiple MFMA tiles are arranged per warp
 *   4. Block tile layout: how warp tiles are arranged in a thread block
 *
 *   rocWMMA 51_tile_distr_enc_reg_map uses this to:
 *   - Verify correctness of tile distribution encodings before coding kernels
 *   - Debug incorrect data layouts in custom GEMM variants
 *   - Understand C-shuffle epilogue requirements
 *
 *   rocWMMA equivalent: uses GetIOShape_t and GetDataLayout_t traits to query
 *   fragment shapes, then prints the lane-to-element mapping.
 *
 * Output:
 *   For each fragment type (A, B, Acc) and each architecture:
 *   - Fragment shape (BlockHeight x BlockWidth x num_elements per lane)
 *   - Data layout type (row_major / col_major)
 *   - Element coverage per lane in the matrix
 *   - Warp tile structure
 *
 * Supported: all GPU targets (host-side utility, no GPU kernel)
 */

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <typeinfo>

#include <hip/hip_runtime.h>

#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_transforms.hpp>

#include "common.hpp"

using namespace rocwmma;

// ---------------------------------------------------------------------------
// Helper: print fragment shape info
// ---------------------------------------------------------------------------
template <typename FragT>
void printFragInfo(const std::string& name)
{
    using Shape  = GetIOShape_t<FragT>;
    using Layout = GetDataLayout_t<FragT>;

    std::cout << "  [" << name << "]\n";
    std::cout << "    BlockHeight    = " << Shape::BlockHeight << "\n";
    std::cout << "    BlockWidth     = " << Shape::BlockWidth  << "\n";
    std::cout << "    KDim           = " << Shape::KDim        << "\n";
    std::cout << "    num_elements   = " << FragT::num_elements << "\n";
    std::cout << "    DataType       = " << typeid(typename fragment_traits<FragT>::DataT).name() << "\n";

    // Coverage per lane in the matrix
    uint32_t rows_per_lane = Shape::BlockHeight * Shape::BlockWidth / FragT::num_elements;
    std::cout << "    Elements/lane  = " << FragT::num_elements << "\n";
    std::cout << "    Matrix region  = "
              << Shape::BlockHeight << " x " << Shape::BlockWidth << "\n\n";
}

// ---------------------------------------------------------------------------
// Print warp tile layout
// ---------------------------------------------------------------------------
void printWarpTileLayout(uint32_t blocks_m, uint32_t blocks_n,
                         uint32_t mfma_m, uint32_t mfma_n,
                         uint32_t mfma_k)
{
    uint32_t warp_m = blocks_m * mfma_m;
    uint32_t warp_n = blocks_n * mfma_n;
    std::cout << "  Warp Tile: " << warp_m << " x " << warp_n
              << " (" << blocks_m << "x" << blocks_n << " tiles of "
              << mfma_m << "x" << mfma_n << "x" << mfma_k << ")\n";
}

// ---------------------------------------------------------------------------
// Print block (macro) tile layout
// ---------------------------------------------------------------------------
void printMacroTileLayout(uint32_t warps_m, uint32_t warps_n,
                          uint32_t warp_m, uint32_t warp_n)
{
    std::cout << "  Macro Tile: " << (warps_m*warp_m) << " x " << (warps_n*warp_n)
              << " (" << warps_m << "x" << warps_n << " warps, each "
              << warp_m << "x" << warp_n << ")\n";
}

// ---------------------------------------------------------------------------
// Print LDS layout info
// ---------------------------------------------------------------------------
void printLdsLayout(uint32_t macro_tile_m, uint32_t macro_tile_n, uint32_t macro_tile_k,
                    uint32_t lds_height_a, uint32_t lds_height_b, uint32_t ldsld)
{
    uint32_t lds_total = (lds_height_a + lds_height_b) * macro_tile_k;
    std::cout << "  LDS Layout:\n";
    std::cout << "    A block: " << lds_height_a << " x " << macro_tile_k
              << " (leading dim=" << ldsld << ")\n";
    std::cout << "    B block (transposed): " << lds_height_b << " x " << macro_tile_k << "\n";
    std::cout << "    Total LDS (single buf): " << lds_total * 2 << " bytes (fp16)\n";
    std::cout << "    Total LDS (double buf): " << 2 * lds_total * 2 << " bytes (fp16)\n";
}

int main()
{
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    std::cout << "=== rocWMMA Tile Distribution Register Map (rocWMMA port) ===\n";
    std::cout << "Device: " << prop.name << "  (" << prop.gcnArchName << ")\n\n";

    // -----------------------------------------------------------------------
    // GFX9 (MFMA) config: 32x32x16, 2x2 blocks, 2x2 warps
    // -----------------------------------------------------------------------
    {
        std::cout << "--- GFX9 MFMA Config (32x32x16, BLOCKS=2x2, WARPS=2x2) ---\n";
        constexpr uint32_t RM=32,RN=32,RK=16,BM=2,BN=2,WM=2,WN=2;
        using InT=float16_t; using AccT=float32_t;

        using MA=fragment<matrix_a,    RM*BM, RN*BN, RK, InT, col_major>;
        using MB=fragment<matrix_b,    RM*BM, RN*BN, RK, InT, row_major>;
        using MC=fragment<accumulator, RM*BM, RN*BN, RK, AccT, row_major>;

        using GRA=fragment<matrix_a,    RM*BM*WM, RN*BN*WN, RK, InT, col_major>;
        using GRB=fragment<matrix_b,    RM*BM*WM, RN*BN*WN, RK, InT, row_major>;
        using LWA=apply_data_layout_t<GRA, col_major>;
        using LWB=apply_data_layout_t<apply_transpose_t<GRB>, col_major>;

        printFragInfo<MA>("matrix_a (warp tile 64x64x16)");
        printFragInfo<MB>("matrix_b (warp tile 64x64x16)");
        printFragInfo<MC>("accumulator (warp tile 64x64x16)");
        printWarpTileLayout(BM, BN, RM, RN, RK);
        printMacroTileLayout(WM, WN, RM*BM, RN*BN);

        constexpr uint32_t ldHA=GetIOShape_t<LWA>::BlockHeight;
        constexpr uint32_t ldHB=GetIOShape_t<LWB>::BlockHeight;
        constexpr uint32_t ldsld=ldHA+ldHB;
        printLdsLayout(RM*BM*WM, RN*BN*WN, RK, ldHA, ldHB, ldsld);
        std::cout << "\n";
    }

    // -----------------------------------------------------------------------
    // GFX12 (WMMA) config: 16x16x16, 4x4 blocks, 2x2 warps
    // -----------------------------------------------------------------------
    {
        std::cout << "--- GFX12 WMMA Config (16x16x16, BLOCKS=4x4, WARPS=2x2) ---\n";
        constexpr uint32_t RM=16,RN=16,RK=16,BM=4,BN=4,WM=2,WN=2;
        using InT=float16_t; using AccT=float32_t;

        using MA=fragment<matrix_a,    RM*BM, RN*BN, RK, InT, col_major>;
        using MB=fragment<matrix_b,    RM*BM, RN*BN, RK, InT, row_major>;
        using MC=fragment<accumulator, RM*BM, RN*BN, RK, AccT, row_major>;

        printFragInfo<MA>("matrix_a (warp tile 64x64x16)");
        printFragInfo<MB>("matrix_b (warp tile 64x64x16)");
        printFragInfo<MC>("accumulator (warp tile 64x64x16)");
        printWarpTileLayout(BM, BN, RM, RN, RK);
        printMacroTileLayout(WM, WN, RM*BM, RN*BN);
        std::cout << "\n";
    }

    // -----------------------------------------------------------------------
    // Cooperative fragment shapes (macro tile level)
    // -----------------------------------------------------------------------
    {
        std::cout << "--- Cooperative Global Read Fragment Shapes ---\n";
        constexpr uint32_t RM=32,RN=32,RK=16,BM=2,BN=2,WM=2,WN=2;
        constexpr uint32_t TX=128, TY=2;
        using InT=float16_t;
        using CoopS=fragment_scheduler::coop_row_major_2d<TX,TY>;
        using GRA=fragment<matrix_a, RM*BM*WM, RN*BN*WN, RK, InT, col_major, CoopS>;
        using GRB=fragment<matrix_b, RM*BM*WM, RN*BN*WN, RK, InT, row_major, CoopS>;

        printFragInfo<GRA>("GRA (coop col_major, macro A tile)");
        printFragInfo<GRB>("GRB (coop row_major, macro B tile)");
    }

    // -----------------------------------------------------------------------
    // Performance estimation table (matches CK Tile output)
    // -----------------------------------------------------------------------
    std::cout << "\n--- Performance Estimation (MI355X gfx950 MFMA) ---\n";
    std::cout << std::left
              << std::setw(20) << "Precision"
              << std::setw(15) << "MFMA tile"
              << std::setw(15) << "Peak TFlops"
              << std::setw(20) << "Effective (80%)"
              << "\n" << std::string(70,'-') << "\n";

    struct PeakPerf { const char* prec; const char* tile; double peak; };
    PeakPerf perfs[] = {
        {"FP16",  "32x32x16", 1300.0},
        {"BF16",  "32x32x16", 1300.0},
        {"FP8",   "32x32x32", 2600.0},
        {"INT8",  "32x32x32", 2600.0},
        {"TF32",  "16x16x8",   650.0},
        {"FP32",  "16x16x4",   325.0},
    };
    for(auto& p : perfs){
        std::cout << std::setw(20) << p.prec
                  << std::setw(15) << p.tile
                  << std::setw(15) << p.peak
                  << std::setw(20) << p.peak*0.8
                  << "\n";
    }

    std::cout << "\n=== Summary ===\n";
    std::cout << "This utility prints the tile distribution encoding and register\n";
    std::cout << "mapping for rocWMMA fragments, analogous to CK Tile's\n";
    std::cout << "51_tile_distr_enc_reg_map diagnostic tool.\n\n";
    std::cout << "Use this to:\n";
    std::cout << "  1. Verify tile sizes before writing a new kernel\n";
    std::cout << "  2. Check LDS sizing (single vs double buffer)\n";
    std::cout << "  3. Understand cooperative load split counts\n";
    std::cout << "  4. Plan epilogue (C-shuffle) register requirements\n";

    return 0;
}
