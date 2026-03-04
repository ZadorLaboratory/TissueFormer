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
ARRAY_OVERRIDE=()
if [ "${MODE}" = "--debug" ]; then
    ARRAY_OVERRIDE=(--array=1-1)
    echo "DEBUG MODE: submitting 1 task per job"
fi

submit_train() {
    sbatch --parsable "${ARRAY_OVERRIDE[@]}" "$@" "${SCRIPT_DIR}/train_array.sh"
}

submit_bench() {
    sbatch --parsable "${ARRAY_OVERRIDE[@]}" "$@" "${SCRIPT_DIR}/benchmark_array.sh"
}

case "${MODE}" in
    --train)
        TRAIN_ID=$(submit_train)
        echo "Submitted training array job: ${TRAIN_ID}"
        ;;
    --bench)
        BENCH_ID=$(submit_bench)
        echo "Submitted benchmark array job: ${BENCH_ID}"
        ;;
    --debug|--all)
        TRAIN_ID=$(submit_train)
        echo "Submitted training array job: ${TRAIN_ID}"

        BENCH_ID=$(submit_bench --dependency=afterok:${TRAIN_ID})
        echo "Submitted benchmark array job: ${BENCH_ID} (depends on ${TRAIN_ID})"

        echo ""
        echo "Monitor with: squeue -u \$USER"
        echo "After completion: python figures/plot_results.py --results_dir outputs --output_dir figures"
        ;;
    *)
        echo "Usage: bash run_experiments.sh [--all|--train|--bench|--debug]"
        exit 1
        ;;
esac
