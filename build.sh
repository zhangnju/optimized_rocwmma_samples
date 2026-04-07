#!/usr/bin/env bash
# =============================================================================
# build.sh  —  Build optimized rocWMMA samples (standalone, CK Tile port)
#
# Usage:
#   ./build.sh [OPTIONS]
#
# Options:
#   -g, --gpu-targets     GPU target(s), semicolon-separated (default: auto-detect)
#   -t, --target          CMake target to build (default: all_samples)
#   -j, --jobs            Parallel jobs (default: nproc)
#   -b, --build-dir       Build directory (default: ./build)
#   -i, --include-dir     rocwmma include dir (default: auto-search)
#   -c, --clean           Wipe build dir before configuring
#   -d, --debug           Build in Debug mode (default: Release)
#   -v, --verbose         Verbose make output
#   -h, --help            Show this help
#
# Examples:
#   ./build.sh                                         # auto-detect GPU, build all
#   ./build.sh -g gfx950                               # MI355X only
#   ./build.sh -g "gfx950;gfx1200;gfx1201"             # MI355X + RDNA4 fat binary
#   ./build.sh -t rocwmma_perf_gemm                    # single sample
#   ./build.sh -c -g gfx950                            # clean rebuild
#   ./build.sh -i /opt/rocm/include -g gfx1200         # custom include path
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_TARGETS=""; BUILD_TARGET="all_samples"; JOBS="$(nproc)"
BUILD_DIR="${SCRIPT_DIR}/build"; ROCWMMA_INCLUDE=""; CLEAN=0
BUILD_TYPE="Release"; VERBOSE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -g|--gpu-targets)  GPU_TARGETS="$2";       shift 2 ;;
        -t|--target)       BUILD_TARGET="$2";      shift 2 ;;
        -j|--jobs)         JOBS="$2";              shift 2 ;;
        -b|--build-dir)    BUILD_DIR="$2";         shift 2 ;;
        -i|--include-dir)  ROCWMMA_INCLUDE="$2";   shift 2 ;;
        -c|--clean)        CLEAN=1;                shift ;;
        -d|--debug)        BUILD_TYPE="Debug";     shift ;;
        -v|--verbose)      VERBOSE=1;              shift ;;
        -h|--help)         sed -n '2,25p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "[ERROR] Unknown option: $1"; exit 1 ;;
    esac
done

# ── Auto-detect GPU target ────────────────────────────────────────────────────
if [[ -z "${GPU_TARGETS}" ]]; then
    if command -v rocminfo &>/dev/null; then
        GPU_TARGETS=$(rocminfo 2>/dev/null \
            | grep -oP 'amdgcn-amd-amdhsa--\K(gfx\d+)' \
            | sort -u | tr '\n' ';' | sed 's/;$//')
    fi
    [[ -z "${GPU_TARGETS}" ]] && GPU_TARGETS="gfx950" && \
        echo "[WARN] Could not detect GPU via rocminfo. Using gfx950."
fi

# ── Auto-search rocwmma include dir ──────────────────────────────────────────
if [[ -z "${ROCWMMA_INCLUDE}" ]]; then
    for candidate in \
        "/home/rocm-libraries/projects/rocwmma/library/include" \
        "/opt/rocm/include" \
        "/opt/rocm-7.2.0/include"; do
        if [[ -f "${candidate}/rocwmma/rocwmma.hpp" ]]; then
            ROCWMMA_INCLUDE="${candidate}"; break
        fi
    done
    if [[ -z "${ROCWMMA_INCLUDE}" ]]; then
        echo "[ERROR] Cannot find rocwmma/rocwmma.hpp. Use -i to specify include dir."
        exit 1
    fi
fi

# ── Detect HIP compiler ───────────────────────────────────────────────────────
CXX_COMPILER=""
for cxx in amdclang++ /opt/rocm/bin/amdclang++ /opt/rocm-7.2.0/bin/amdclang++; do
    if command -v "$cxx" &>/dev/null || [[ -x "$cxx" ]]; then
        CXX_COMPILER="$cxx"; break
    fi
done
[[ -z "${CXX_COMPILER}" ]] && { echo "[ERROR] amdclang++ not found in PATH or /opt/rocm/bin"; exit 1; }

echo "============================================================"
echo " optimized_rocwmma_samples Build"
echo "============================================================"
echo " Source dir  : ${SCRIPT_DIR}"
echo " Build dir   : ${BUILD_DIR}"
echo " Build type  : ${BUILD_TYPE}"
echo " GPU targets : ${GPU_TARGETS}"
echo " CMake target: ${BUILD_TARGET}"
echo " rocwmma hdrs: ${ROCWMMA_INCLUDE}"
echo " HIP compiler: ${CXX_COMPILER}"
echo " Jobs        : ${JOBS}"
echo "============================================================"

[[ "${CLEAN}" -eq 1 && -d "${BUILD_DIR}" ]] && { echo "[INFO] Cleaning..."; rm -rf "${BUILD_DIR}"; }
mkdir -p "${BUILD_DIR}"

echo ""
echo "[INFO] Configuring..."
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    "-DGPU_TARGETS=${GPU_TARGETS}" \
    -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
    "-DROCWMMA_INCLUDE_DIR=${ROCWMMA_INCLUDE}" \
    2>&1 | grep -E "^--|error:|GPU_TARGETS|rocwmma|hiprtc|HIP runtime"
[[ -f "${BUILD_DIR}/CMakeCache.txt" ]] || { echo "[ERROR] CMake configuration failed (no CMakeCache.txt). Re-run cmake without grep to see errors."; exit 1; }

echo ""
echo "[INFO] Building: ${BUILD_TARGET} (-j${JOBS})..."
cmake --build "${BUILD_DIR}" \
    --target "${BUILD_TARGET}" \
    -- -j"${JOBS}" \
    $([[ "${VERBOSE}" -eq 1 ]] && echo "VERBOSE=1" || true)

echo ""
echo "============================================================"
echo " Build complete! Binaries in: ${BUILD_DIR}/"
echo "============================================================"
ls -1 "${BUILD_DIR}"/rocwmma_* 2>/dev/null | grep -v '\.' | sed 's|.*/||' | column
