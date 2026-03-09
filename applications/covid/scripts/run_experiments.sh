#!/bin/bash
# COVID severity classification: submit training + benchmark SLURM array jobs.
#
# Usage:
#   bash run_experiments.sh           # submit training, then benchmarks after
#   bash run_experiments.sh --train   # only submit training jobs
#   bash run_experiments.sh --bench   # only submit benchmark jobs
#   bash run_experiments.sh --debug   # submit 1 training + 1 benchmark job

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

mkdir -p "${PROJECT_DIR}/slurm_logs/covid"
cd "${PROJECT_DIR}"

MODE="${1:---all}"
TOTAL_TASKS=10   # 4 datasets x 5 folds x 10 group_sizes
CHUNK_SIZE=8      # max array size allowed by SLURM manager

# Submit a script in chunks of CHUNK_SIZE, returning colon-separated job IDs.
# Usage: submit_chunked SCRIPT [extra sbatch args...]
submit_chunked() {
    local script="$1"; shift
    local n_tasks="${TOTAL_TASKS}"
    if [ "${MODE}" = "--debug" ]; then
        n_tasks=1
    fi
    local job_ids=()
    local start=1
    while [ "${start}" -le "${n_tasks}" ]; do
        local end=$(( start + CHUNK_SIZE - 1 ))
        if [ "${end}" -gt "${n_tasks}" ]; then
            end="${n_tasks}"
        fi
        local jid
        jid=$(sbatch --parsable --array="${start}-${end}" "$@" "${script}")
        job_ids+=("${jid}")
        echo "  Submitted array ${start}-${end} -> ${jid}" >&2
        start=$(( end + 1 ))
    done
    # Return colon-separated list (for use in --dependency)
    local IFS=":"
    echo "${job_ids[*]}"
}

case "${MODE}" in
    --train)
        echo "Submitting training jobs..."
        submit_chunked "${SCRIPT_DIR}/train_array.sh" > /dev/null
        ;;
    --bench)
        echo "Submitting benchmark jobs..."
        submit_chunked "${SCRIPT_DIR}/benchmark_array.sh" > /dev/null
        ;;
    --debug|--all)
        echo "Submitting training jobs..."
        TRAIN_IDS=$(submit_chunked "${SCRIPT_DIR}/train_array.sh")

        echo "Submitting benchmark jobs (depend on training)..."
        submit_chunked "${SCRIPT_DIR}/benchmark_array.sh" \
            --dependency="afterok:${TRAIN_IDS}" > /dev/null

        echo ""
        echo "Monitor with: squeue -u \$USER"
        echo "After completion: python figures/plot_results.py --results_dir outputs --output_dir figures"
        ;;
    *)
        echo "Usage: bash run_experiments.sh [--all|--train|--bench|--debug]"
        exit 1
        ;;
esac
