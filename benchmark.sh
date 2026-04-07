#!/usr/bin/env bash
# =============================================================================
# benchmark.sh  —  Run all optimized rocWMMA sample benchmarks
#
# Usage:
#   ./benchmark.sh [OPTIONS]
#
# Options:
#   -b, --build-dir   Build directory (default: ./build)
#   -o, --output      Output report file (default: ./benchmark_results.txt)
#   -s, --samples     Comma-separated sample names to run (default: all)
#   -g, --gpu         GPU device index (default: 0)
#   --csv             Also write CSV results (benchmark_results.csv)
#   --timeout         Per-sample timeout in seconds (default: 120)
#   -h, --help        Show this help
#
# Examples:
#   ./benchmark.sh                                    # run all, GPU 0
#   ./benchmark.sh -g 2                               # use GPU 2
#   ./benchmark.sh -s rocwmma_perf_gemm               # single sample
#   ./benchmark.sh -s "rocwmma_layernorm2d,rocwmma_rmsnorm2d"
#   ./benchmark.sh --csv -o results_mi355x.txt        # full report + CSV
#   HIP_VISIBLE_DEVICES=3 ./benchmark.sh              # env override
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BUILD_DIR="${SCRIPT_DIR}/build"
OUTPUT_FILE="${SCRIPT_DIR}/benchmark_results.txt"
SAMPLES_FILTER=""
GPU_INDEX=0
WRITE_CSV=0
TIMEOUT_SEC=120

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--build-dir)  BUILD_DIR="$2";    shift 2 ;;
        -o|--output)     OUTPUT_FILE="$2";  shift 2 ;;
        -s|--samples)    SAMPLES_FILTER="$2"; shift 2 ;;
        -g|--gpu)        GPU_INDEX="$2";    shift 2 ;;
        --csv)           WRITE_CSV=1;       shift ;;
        --timeout)       TIMEOUT_SEC="$2";  shift 2 ;;
        -h|--help)       sed -n '2,22p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "[ERROR] Unknown option: $1"; exit 1 ;;
    esac
done

[[ -d "${BUILD_DIR}" ]] || {
    echo "[ERROR] Build dir not found: ${BUILD_DIR}"
    echo "        Run ./build.sh first."
    exit 1
}

export HIP_VISIBLE_DEVICES="${GPU_INDEX}"
CSV_FILE="${OUTPUT_FILE%.txt}.csv"

GPU_NAME=$(rocminfo 2>/dev/null | grep "Marketing Name" | head -1 | awk -F: '{print $2}' | xargs || echo "unknown")
GPU_ARCH=$(rocminfo 2>/dev/null | grep -oP 'gfx\d+' | head -1 || echo "unknown")
ROCM_VER=$(cat /opt/rocm/.info/version 2>/dev/null || rocm-smi --version 2>/dev/null | grep -oP '[\d.]+' | head -1 || echo "unknown")
TS=$(date '+%Y-%m-%d %H:%M:%S')

# ── All sample definitions: "binary|args|description" ─────────────────────
declare -a ALL_SAMPLES=(
    "rocwmma_perf_gemm||HGEMM Pipeline V1 vs V2 (double-buffer)"
    "rocwmma_perf_gemm|3840 4096 4096|HGEMM 3840x4096x4096"
    "rocwmma_perf_gemm|4096 4096 4096|HGEMM 4096^3"
    "rocwmma_perf_gemm|8192 8192 8192|HGEMM 8192^3"
    "rocwmma_gemm_splitk||Split-K GEMM (two-stage, FP32 workspace)"
    "rocwmma_gemm_quantized||FP8 Quantized GEMM (tensor scale)"
    "rocwmma_batched_gemm|4 1024 1024 1024|Batched GEMM batch=4 1024^3"
    "rocwmma_batched_gemm|8 512 512 2048|Batched GEMM batch=8"
    "rocwmma_gemm_multi_d||GEMM + fused bias/scale/ReLU epilogue"
    "rocwmma_batched_contraction||Batched Tensor Contraction"
    "rocwmma_grouped_gemm||Grouped GEMM (variable-size groups)"
    "rocwmma_streamk_gemm||Stream-K Load-Balanced GEMM"
    "rocwmma_flatmm||FlatMM (B bypasses LDS)"
    "rocwmma_mx_gemm||MX-format GEMM (E8M0 block scale)"
    "rocwmma_fmha_fwd||Flash Attention Forward (online softmax)"
    "rocwmma_sparse_attn||Block-Sparse Attention (Jenga LUT)"
    "rocwmma_layernorm2d|3328 4096|LayerNorm2D M=3328 N=4096"
    "rocwmma_layernorm2d|3328 8192|LayerNorm2D M=3328 N=8192"
    "rocwmma_rmsnorm2d|3328 4096|RMSNorm2D + Add+RMSNorm fused"
    "rocwmma_smoothquant|3328 4096|SmoothQuant FP16->INT8 N=4096"
    "rocwmma_smoothquant|3328 8192|SmoothQuant FP16->INT8 N=8192"
    "rocwmma_moe_smoothquant||MoE SmoothQuant (per-expert scale)"
    "rocwmma_fused_moe||Fused MoE (sort+GEMM+SiLU+GEMM)"
    "rocwmma_reduce|3328 4096|Row Reduce Sum/Max N=4096"
    "rocwmma_reduce|3328 8192|Row Reduce Sum/Max N=8192"
    "rocwmma_elementwise|3840 4096|Elementwise Add/Square/Transpose"
    "rocwmma_topk_softmax||TopK Softmax (warp-per-token)"
    "rocwmma_moe_sorting||MoE Token Sorting (counting sort)"
    "rocwmma_img2col_gemm||Im2Col + GEMM (conv-as-GEMM)"
    "rocwmma_grouped_conv_fwd||Grouped Convolution Forward"
    "rocwmma_pooling||3D Max/Avg Pooling (NDHWC)"
    "rocwmma_permute||Tensor Permute / NCHW<->NHWC"
    "rocwmma_tile_distr_reg_map||Tile Distribution Register Map (diagnostic)"
)

# Filter samples if -s specified
if [[ -n "${SAMPLES_FILTER}" ]]; then
    IFS=',' read -ra FILTER_ARR <<< "${SAMPLES_FILTER}"
    declare -a FILTERED=()
    for entry in "${ALL_SAMPLES[@]}"; do
        bin="${entry%%|*}"
        for f in "${FILTER_ARR[@]}"; do
            [[ "${bin}" == *"${f}"* ]] && FILTERED+=("${entry}") && break
        done
    done
    ALL_SAMPLES=("${FILTERED[@]}")
fi

TOTAL=${#ALL_SAMPLES[@]}

# Header
HEADER="$(cat <<EOF
================================================================================
 optimized_rocwmma_samples Benchmark Report
================================================================================
 Date       : ${TS}
 GPU        : ${GPU_NAME} (${GPU_ARCH})  [device ${GPU_INDEX}]
 ROCm       : ${ROCM_VER}
 Build dir  : ${BUILD_DIR}
 Timeout    : ${TIMEOUT_SEC}s per sample
================================================================================
EOF
)"
echo "${HEADER}"
echo "${HEADER}" > "${OUTPUT_FILE}"
[[ "${WRITE_CSV}" -eq 1 ]] && echo "sample,description,metric,value,unit" > "${CSV_FILE}"

PASS=0; FAIL=0; SKIP=0

for entry in "${ALL_SAMPLES[@]}"; do
    IFS='|' read -r BIN ARGS DESC <<< "${entry}"
    BINARY="${BUILD_DIR}/${BIN}"
    IDX=$((PASS+FAIL+SKIP+1))

    printf "\n%s\n" "────────────────────────────────────────────────────────────────────────────────"
    printf " [%2d/%d] %-35s  %s\n" "${IDX}" "${TOTAL}" "${BIN}" "${DESC}"
    printf "%s\n" "────────────────────────────────────────────────────────────────────────────────"
    printf "\n[%d/%d] %s | %s\n  Args: %s\n" \
        "${IDX}" "${TOTAL}" "${BIN}" "${DESC}" "${ARGS:-<defaults>}" >> "${OUTPUT_FILE}"

    if [[ ! -x "${BINARY}" ]]; then
        echo "  [SKIP] Binary not found: ${BINARY}"
        echo "  [SKIP] Binary not found" >> "${OUTPUT_FILE}"
        ((SKIP++)); continue
    fi

    EXIT_CODE=0
    RESULT=$(timeout "${TIMEOUT_SEC}s" "${BINARY}" ${ARGS} 2>&1) || EXIT_CODE=$?

    if [[ ${EXIT_CODE} -eq 124 ]]; then
        echo "  [TIMEOUT] Killed after ${TIMEOUT_SEC}s"
        echo "  [TIMEOUT]" >> "${OUTPUT_FILE}"
        ((FAIL++)); continue
    fi
    if [[ ${EXIT_CODE} -ne 0 ]]; then
        echo "  [FAIL] Exit code ${EXIT_CODE}"
        echo "${RESULT}" | tail -3
        printf "  [FAIL] exit=%d\n%s\n" "${EXIT_CODE}" "$(echo "${RESULT}" | tail -3)" >> "${OUTPUT_FILE}"
        ((FAIL++)); continue
    fi

    echo "${RESULT}"
    echo "${RESULT}" >> "${OUTPUT_FILE}"
    ((PASS++))

    # Extract to CSV
    if [[ "${WRITE_CSV}" -eq 1 ]]; then
        while IFS= read -r line; do
            label=$(echo "${line}" | grep -oP '^\[.*?\]' | head -1 | tr -d '[]')
            [[ -z "${label}" ]] && label="${DESC}"
            tf=$(echo "${line}" | grep -oP 'TFlops/s=\K[\d.]+' | head -1) || true
            [[ -n "${tf:-}" ]] && echo "${BIN},\"${DESC}\",\"${label}\",${tf},TFlops/s" >> "${CSV_FILE}"
            tf2=$(echo "${line}" | grep -oP '[\d.]+(?=\s+TFlops/s)' | head -1) || true
            [[ -n "${tf2:-}" ]] && echo "${BIN},\"${DESC}\",\"${label}\",${tf2},TFlops/s" >> "${CSV_FILE}"
            bw=$(echo "${line}" | grep -oP '[\d.]+(?=\s+GB/s)' | head -1) || true
            [[ -n "${bw:-}" ]] && echo "${BIN},\"${DESC}\",\"${label} bw\",${bw},GB/s" >> "${CSV_FILE}"
        done <<< "${RESULT}"
    fi
done

SUMMARY="$(cat <<EOF

================================================================================
 Benchmark Summary
================================================================================
 Total    : ${TOTAL}
 Passed   : ${PASS}
 Failed   : ${FAIL}
 Skipped  : ${SKIP}
 Output   : ${OUTPUT_FILE}
EOF
)"
[[ "${WRITE_CSV}" -eq 1 ]] && SUMMARY+="
 CSV      : ${CSV_FILE}"
SUMMARY+="
================================================================================"

echo "${SUMMARY}"
echo "${SUMMARY}" >> "${OUTPUT_FILE}"
