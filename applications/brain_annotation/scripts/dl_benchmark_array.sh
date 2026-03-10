#!/bin/bash
#SBATCH --job-name=dl_benchmark
#SBATCH --array=1-45
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --qos=slow_nice
#SBATCH --partition=gpuq
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/dl_benchmark_%A_%a.out
#SBATCH --error=slurm_logs/dl_benchmark_%A_%a.err

# Run DL benchmarks (CellCnn, scAGG, ScRAT) across all fold × group_size combinations.
# 9 group sizes × 5 folds = 45 array tasks.

set -euo pipefail

module purge
module load cuda12.3/toolkit/12.3.2
module load cudnn8.6-cuda11.8/8.6.0.163

# Single-GPU DeepSpeed needs these env vars to avoid MPI fallback
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

# --- Index decoding ---
BATCH_SIZES=(1 2 4 8 16 32 64 128 256)
FOLDS=(0 1 2 3 test_enucleated)

N_BATCH_SIZES=${#BATCH_SIZES[@]}  # 9
N_FOLDS=${#FOLDS[@]}              # 5

idx=$(( SLURM_ARRAY_TASK_ID - 1 ))
batch_idx=$(( idx / N_FOLDS ))
fold_idx=$(( idx % N_FOLDS ))

batch_size="${BATCH_SIZES[$batch_idx]}"
fold="${FOLDS[$fold_idx]}"

echo "=== Task ${SLURM_ARRAY_TASK_ID}/45: fold=${fold} group_size=${batch_size} ==="

# --- Dataset path ---
DATA_ROOT="/grid/zador/data_norepl/Ari/transcriptomics/barseq/Chen2023"

if [ "$fold" = "test_enucleated" ]; then
    dataset_path="${DATA_ROOT}/train_test_barseq_all_exhausted_test_enucleated.dataset"
else
    dataset_path="${DATA_ROOT}/train_test_barseq_all_exhausted_fold${fold}.dataset"
fi

# --- Run benchmarks ---
cd "$(dirname "$0")/.."

python benchmarks.py \
    data.group_size="${batch_size}" \
    data.dataset_path="${dataset_path}" \
    run_bulk_expression_rf=false \
    run_bulk_expression_lr=false \
    run_h3type_rf=false \
    run_h3type_lr=false \
    run_cellcnn=true \
    run_scagg=true \
    run_scrat=true \
    wandb.name="dl_benchmark_fold${fold}_gs${batch_size}" \
    wandb.tags="[dl_benchmark]"
