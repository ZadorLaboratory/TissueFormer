#!/bin/bash
#SBATCH --job-name=covid-bench
#SBATCH --output=slurm_logs/covid/bench_%A_%a.out
#SBATCH --error=slurm_logs/covid/bench_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --partition=cpuq
#SBATCH --time=12:00:00

# 3 datasets x 5 folds x 10 benchmark_group_sizes = 150 tasks

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID not set. Use: sbatch --array=N-M scripts/benchmark_array.sh" >&2
    exit 1
fi

eval "$(micromamba shell hook --shell bash)"
micromamba activate brain_annotation2

DATASETS=(stevenson)
N_FOLDS=5
BENCHMARK_GROUP_SIZES=(2 4 8 16 32 64 128 256 512 all)
N_BENCHMARK_GROUP_SIZES=${#BENCHMARK_GROUP_SIZES[@]}

# Decode flat index -> (dataset, fold, group_size)
idx=$(( SLURM_ARRAY_TASK_ID - 1 ))
N_DATASETS=${#DATASETS[@]}
fold=$(( idx / (N_DATASETS * N_BENCHMARK_GROUP_SIZES) ))
remainder=$(( idx % (N_DATASETS * N_BENCHMARK_GROUP_SIZES) ))
ds_idx=$(( remainder / N_BENCHMARK_GROUP_SIZES ))
gs_idx=$(( remainder % N_BENCHMARK_GROUP_SIZES ))

dataset="${DATASETS[$ds_idx]}"
gs="${BENCHMARK_GROUP_SIZES[$gs_idx]}"

echo "=== Benchmark Task ${SLURM_ARRAY_TASK_ID}: dataset=${dataset} fold=${fold} gs=${gs} ==="

cd "$SLURM_SUBMIT_DIR"

python benchmarks.py \
    dataset_name="${dataset}" \
    data.group_size="${gs}" \
    data.h5ad_path="/grid/zador/data_norepl/Ari/transcriptomics/covid/${dataset}_processed.h5ad" \
    data.splits_path="/grid/zador/data_norepl/Ari/transcriptomics/covid/${dataset}_donor_splits.json" \
    data.cv_fold="${fold}" \
    run_classical=true \
    wandb.tags=[with_val]
