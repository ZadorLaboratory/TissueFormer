#!/bin/bash
#SBATCH --job-name=covid-train
#SBATCH --output=slurm_logs/covid/train_%A_%a.out
#SBATCH --error=slurm_logs/covid/train_%A_%a.err
#SBATCH --array=1-150
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=gpuq
#SBATCH --time=1-00:00:00

# 3 datasets x 5 folds x 10 group_sizes = 150 tasks

module purge
module load cuda12.3/toolkit/12.3.2
module load cudnn8.6-cuda11.8/8.6.0.163

DATASETS=(combat ren stevenson)
N_FOLDS=5
GROUP_SIZES=(1 2 4 8 16 32 64 128 256 512)
N_GROUP_SIZES=${#GROUP_SIZES[@]}

# Decode flat index -> (dataset, fold, group_size)
idx=$(( SLURM_ARRAY_TASK_ID - 1 ))
ds_idx=$(( idx / (N_FOLDS * N_GROUP_SIZES) ))
remainder=$(( idx % (N_FOLDS * N_GROUP_SIZES) ))
fold=$(( remainder / N_GROUP_SIZES ))
gs_idx=$(( remainder % N_GROUP_SIZES ))

dataset="${DATASETS[$ds_idx]}"
gs="${GROUP_SIZES[$gs_idx]}"

echo "=== Task ${SLURM_ARRAY_TASK_ID}: dataset=${dataset} fold=${fold} gs=${gs} ==="

cd "$SLURM_SUBMIT_DIR"

if [ "$gs" -eq 1 ]; then
    python train.py \
        dataset_name="${dataset}" \
        data.group_size=1 \
        data.dataset_path="applications/covid/data/${dataset}.dataset" \
        data.splits_path="applications/covid/data/${dataset}_donor_splits.json" \
        data.cv_fold="${fold}" \
        model.pretrained_type="single-cell" \
        training.remove_unused_columns=true \
        seed="${fold}"
else
    python train.py \
        dataset_name="${dataset}" \
        data.group_size="${gs}" \
        data.dataset_path="applications/covid/data/${dataset}.dataset" \
        data.splits_path="applications/covid/data/${dataset}_donor_splits.json" \
        data.cv_fold="${fold}" \
        seed="${fold}"
fi
