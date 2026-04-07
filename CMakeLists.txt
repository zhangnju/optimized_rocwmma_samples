###############################################################################
# CMakeLists.txt  —  optimized_rocwmma_samples (standalone build)
#
# Standalone build of all rocWMMA community samples (CK Tile port).
# Does NOT require being inside the rocWMMA source tree.
#
# Usage:
#   cmake -S . -B build \
#         -DCMAKE_BUILD_TYPE=Release \
#         -DGPU_TARGETS="gfx950" \
#         [-DROCWMMA_INCLUDE_DIR=/path/to/rocwmma/library/include]
#   cmake --build build -j$(nproc)
###############################################################################

cmake_minimum_required(VERSION 3.16)

project(optimized_rocwmma_samples
        VERSION 1.0.0
        LANGUAGES CXX)

# ---------------------------------------------------------------------------
# C++ standard
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# GPU target setup
# ---------------------------------------------------------------------------
if(NOT DEFINED GPU_TARGETS)
    execute_process(
        COMMAND bash -c "rocminfo 2>/dev/null | grep -oP 'amdgcn-amd-amdhsa--\\Kgfx[0-9]+' | sort -u | tr '\\n' ';' | sed 's/;$//'"
        OUTPUT_VARIABLE DETECTED_GPU_TARGETS
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(DETECTED_GPU_TARGETS)
        set(GPU_TARGETS "${DETECTED_GPU_TARGETS}")
    else()
        set(GPU_TARGETS "gfx950")
    endif()
endif()
message(STATUS "GPU_TARGETS = ${GPU_TARGETS}")

# ---------------------------------------------------------------------------
# Locate rocWMMA headers
#
# rocWMMA is header-only. Search in priority order:
#   1. ROCWMMA_INCLUDE_DIR  (explicit cmake var)
#   2. ROCWMMA_ROOT/library/include  (source tree)
#   3. ROCWMMA_ROOT/include          (install tree)
#   4. Well-known fallback paths
# ---------------------------------------------------------------------------
set(ROCWMMA_HEADER_SEARCH_PATHS "")

if(DEFINED ROCWMMA_INCLUDE_DIR)
    list(APPEND ROCWMMA_HEADER_SEARCH_PATHS "${ROCWMMA_INCLUDE_DIR}")
endif()

if(DEFINED ROCWMMA_ROOT)
    list(APPEND ROCWMMA_HEADER_SEARCH_PATHS
        "${ROCWMMA_ROOT}/library/include"
        "${ROCWMMA_ROOT}/include")
endif()

list(APPEND ROCWMMA_HEADER_SEARCH_PATHS
    "/home/rocm-libraries/projects/rocwmma/library/include"
    "/opt/rocm/include"
    "/opt/rocm-7.2.0/include")

find_path(ROCWMMA_FOUND_INCLUDE_DIR
    NAMES rocwmma/rocwmma.hpp
    PATHS ${ROCWMMA_HEADER_SEARCH_PATHS}
    NO_DEFAULT_PATH
    NO_CMAKE_SYSTEM_PATH)

if(NOT ROCWMMA_FOUND_INCLUDE_DIR)
    find_path(ROCWMMA_FOUND_INCLUDE_DIR
        NAMES rocwmma/rocwmma.hpp
        PATHS /usr/include /usr/local/include)
endif()

if(NOT ROCWMMA_FOUND_INCLUDE_DIR)
    message(FATAL_ERROR
        "rocwmma/rocwmma.hpp not found.\n"
        "Set -DROCWMMA_INCLUDE_DIR=/path/to/rocwmma/library/include\n"
        "or  -DROCWMMA_ROOT=/path/to/rocwmma/source")
endif()

message(STATUS "rocwmma headers: ${ROCWMMA_FOUND_INCLUDE_DIR}")

# Interface target for rocWMMA (header-only)
add_library(rocwmma_headers INTERFACE)
target_include_directories(rocwmma_headers INTERFACE "${ROCWMMA_FOUND_INCLUDE_DIR}")

# ---------------------------------------------------------------------------
# Locate common.hpp (from rocwmma/samples/ or this directory)
# ---------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/common.hpp")
    set(SAMPLES_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
else()
    # Try to find it next to rocwmma source
    get_filename_component(_rw_src "${ROCWMMA_FOUND_INCLUDE_DIR}" DIRECTORY)
    find_path(SAMPLES_INCLUDE_DIR common.hpp
        PATHS
            "${_rw_src}/samples"
            "/home/rocm-libraries/projects/rocwmma/samples"
        NO_DEFAULT_PATH)
    if(NOT SAMPLES_INCLUDE_DIR)
        message(FATAL_ERROR "common.hpp not found. Copy it to: ${CMAKE_CURRENT_SOURCE_DIR}/")
    endif()
endif()
message(STATUS "samples include (common.hpp): ${SAMPLES_INCLUDE_DIR}")

# ---------------------------------------------------------------------------
# Find HIP runtime (for hiprtc, hipEvent_t, etc.)
# ---------------------------------------------------------------------------
find_package(HIP REQUIRED)

# HIP runtime library (amdhip64)
find_library(HIP_RUNTIME_LIB amdhip64
    HINTS /opt/rocm/lib /opt/rocm-7.2.0/lib /usr/lib
    REQUIRED)
add_library(hip_runtime_lib UNKNOWN IMPORTED)
set_target_properties(hip_runtime_lib PROPERTIES IMPORTED_LOCATION "${HIP_RUNTIME_LIB}")
message(STATUS "HIP runtime: ${HIP_RUNTIME_LIB}")

# hiprtc library
find_library(HIPRTC_LIB hiprtc
    HINTS /opt/rocm/lib /opt/rocm-7.2.0/lib /usr/lib)
if(HIPRTC_LIB)
    add_library(hiprtc_lib UNKNOWN IMPORTED)
    set_target_properties(hiprtc_lib PROPERTIES IMPORTED_LOCATION "${HIPRTC_LIB}")
    message(STATUS "hiprtc: ${HIPRTC_LIB}")
else()
    message(WARNING "hiprtc library not found — some link steps may fail")
    add_library(hiprtc_lib INTERFACE)
endif()

# ---------------------------------------------------------------------------
# OpenMP (optional)
# ---------------------------------------------------------------------------
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP: ${OpenMP_CXX_VERSION}")
    set(OPENMP_LIB OpenMP::OpenMP_CXX)
else()
    set(OPENMP_LIB "")
endif()

# ---------------------------------------------------------------------------
# Aggregate target
# ---------------------------------------------------------------------------
add_custom_target(all_samples)

# ---------------------------------------------------------------------------
# Helper: add one sample executable
# ---------------------------------------------------------------------------
function(add_sample TARGET_NAME SOURCE_FILE)
    set(SRC "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}")
    if(NOT EXISTS "${SRC}")
        message(WARNING "Source not found, skipping: ${SOURCE_FILE}")
        return()
    endif()

    add_executable(${TARGET_NAME} "${SRC}")

    # rocWMMA headers (header-only)
    target_link_libraries(${TARGET_NAME} PRIVATE rocwmma_headers)

    # HIP runtime + hiprtc
    target_link_libraries(${TARGET_NAME} PRIVATE hip_runtime_lib hiprtc_lib)

    # OpenMP
    if(OPENMP_LIB)
        target_link_libraries(${TARGET_NAME} PRIVATE ${OPENMP_LIB})
    endif()

    # Include: this directory (common.hpp + source) + rocwmma headers
    target_include_directories(${TARGET_NAME} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${SAMPLES_INCLUDE_DIR}"
        "${ROCWMMA_FOUND_INCLUDE_DIR}")

    # HIP AMD platform defines (matching rocwmma build system)
    target_compile_definitions(${TARGET_NAME} PRIVATE
        USE_PROF_API=1               # enables HIP profiling API for accurate kernel timing
        __HIP_PLATFORM_AMD__=1
        __HIP_PLATFORM_HCC__=1
        __HIP_HCC_COMPAT_MODE__=1)

    # Core HIP compilation flags (matching the rocwmma build system flags.make)
    target_compile_options(${TARGET_NAME} PRIVATE
        -x hip                                   # treat as HIP source
        --driver-mode=g++
        -Xclang -fallow-half-arguments-and-returns
        -D__HIP_HCC_COMPAT_MODE__=1
        -fclang-abi-compat=17
        -Wno-format-nonliteral
        -parallel-jobs=4)                        # parallel compilation units

    # HIP device compilation: one --offload-arch per GPU target
    foreach(_tgt IN LISTS GPU_TARGETS)
        if(_tgt)
            target_compile_options(${TARGET_NAME} PRIVATE
                --offload-arch=${_tgt})
            target_link_options(${TARGET_NAME} PRIVATE
                --offload-arch=${_tgt})
        endif()
    endforeach()

    # Link flags matching the rocwmma build system
    target_link_options(${TARGET_NAME} PRIVATE
        --driver-mode=g++
        -Xclang -fallow-half-arguments-and-returns
        -D__HIP_HCC_COMPAT_MODE__=1
        -fclang-abi-compat=17
        --hip-link
        -Wl,-rpath,/opt/rocm/lib:/opt/rocm/llvm/lib
        -lm -ldl)

    set_target_properties(${TARGET_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

    add_dependencies(all_samples ${TARGET_NAME})
endfunction()

# ---------------------------------------------------------------------------
# Register all 26 samples
# ---------------------------------------------------------------------------

# ── GEMM variants ────────────────────────────────────────────────────────────
add_sample(rocwmma_perf_gemm              rocwmma_perf_gemm.cpp)
add_sample(rocwmma_gemm_splitk            rocwmma_gemm_splitk.cpp)
add_sample(rocwmma_gemm_quantized         rocwmma_gemm_quantized.cpp)
add_sample(rocwmma_batched_gemm           rocwmma_batched_gemm.cpp)
add_sample(rocwmma_gemm_multi_d           rocwmma_gemm_multi_d.cpp)
add_sample(rocwmma_batched_contraction    rocwmma_batched_contraction.cpp)
add_sample(rocwmma_grouped_gemm           rocwmma_grouped_gemm.cpp)
add_sample(rocwmma_streamk_gemm           rocwmma_streamk_gemm.cpp)
add_sample(rocwmma_flatmm                 rocwmma_flatmm.cpp)
add_sample(rocwmma_mx_gemm                rocwmma_mx_gemm.cpp)

# ── Attention ────────────────────────────────────────────────────────────────
add_sample(rocwmma_fmha_fwd               rocwmma_fmha_fwd.cpp)
add_sample(rocwmma_sparse_attn            rocwmma_sparse_attn.cpp)

# ── Normalization & Quantization ─────────────────────────────────────────────
add_sample(rocwmma_layernorm2d            rocwmma_layernorm2d.cpp)
add_sample(rocwmma_rmsnorm2d              rocwmma_rmsnorm2d.cpp)
add_sample(rocwmma_smoothquant            rocwmma_smoothquant.cpp)
add_sample(rocwmma_moe_smoothquant        rocwmma_moe_smoothquant.cpp)
add_sample(rocwmma_fused_moe              rocwmma_fused_moe.cpp)

# ── Reduction & Elementwise ───────────────────────────────────────────────────
add_sample(rocwmma_reduce                 rocwmma_reduce.cpp)
add_sample(rocwmma_elementwise            rocwmma_elementwise.cpp)

# ── MoE Infrastructure ───────────────────────────────────────────────────────
add_sample(rocwmma_topk_softmax           rocwmma_topk_softmax.cpp)
add_sample(rocwmma_moe_sorting            rocwmma_moe_sorting.cpp)

# ── Convolution & Data Layout ────────────────────────────────────────────────
add_sample(rocwmma_img2col_gemm           rocwmma_img2col_gemm.cpp)
add_sample(rocwmma_grouped_conv_fwd       rocwmma_grouped_conv_fwd.cpp)
add_sample(rocwmma_pooling                rocwmma_pooling.cpp)
add_sample(rocwmma_permute                rocwmma_permute.cpp)

# ── Diagnostic (host-side) ───────────────────────────────────────────────────
add_sample(rocwmma_tile_distr_reg_map     rocwmma_tile_distr_reg_map.cpp)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "============================================================")
message(STATUS " optimized_rocwmma_samples configured")
message(STATUS "------------------------------------------------------------")
message(STATUS " Build dir    : ${CMAKE_BINARY_DIR}")
message(STATUS " GPU targets  : ${GPU_TARGETS}")
message(STATUS " rocwmma hdrs : ${ROCWMMA_FOUND_INCLUDE_DIR}")
message(STATUS " common.hpp   : ${SAMPLES_INCLUDE_DIR}")
message(STATUS " C++ standard : ${CMAKE_CXX_STANDARD}")
message(STATUS " Build type   : ${CMAKE_BUILD_TYPE}")
message(STATUS "------------------------------------------------------------")
message(STATUS " Build all    : cmake --build build -j\$(nproc) --target all_samples")
message(STATUS " Single target: cmake --build build --target rocwmma_perf_gemm")
message(STATUS "============================================================")
message(STATUS "")
