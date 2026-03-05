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

DATASETS=(combat ren stevenson)
N_FOLDS=5
BENCHMARK_GROUP_SIZES=(2 4 8 16 32 64 128 256 512 all)
N_BENCHMARK_GROUP_SIZES=${#BENCHMARK_GROUP_SIZES[@]}

# Decode flat index -> (dataset, fold, group_size)
idx=$(( SLURM_ARRAY_TASK_ID - 1 ))
ds_idx=$(( idx / (N_FOLDS * N_BENCHMARK_GROUP_SIZES) ))
remainder=$(( idx % (N_FOLDS * N_BENCHMARK_GROUP_SIZES) ))
fold=$(( remainder / N_BENCHMARK_GROUP_SIZES ))
gs_idx=$(( remainder % N_BENCHMARK_GROUP_SIZES ))

dataset="${DATASETS[$ds_idx]}"
gs="${BENCHMARK_GROUP_SIZES[$gs_idx]}"

echo "=== Benchmark Task ${SLURM_ARRAY_TASK_ID}: dataset=${dataset} fold=${fold} gs=${gs} ==="

cd "$SLURM_SUBMIT_DIR"

python benchmarks.py \
    dataset_name="${dataset}" \
    data.group_size="${gs}" \
    data.h5ad_path="applications/covid/data/${dataset}_processed.h5ad" \
    data.splits_path="applications/covid/data/${dataset}_donor_splits.json" \
    data.cv_fold="${fold}"
