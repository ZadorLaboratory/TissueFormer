#!/bin/bash
#SBATCH --job-name=covid-bsweep
#SBATCH --output=slurm_logs/covid/bsweep_%A_%a.out
#SBATCH --error=slurm_logs/covid/bsweep_%A_%a.err
#SBATCH --array=1-8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=gpuq
#SBATCH --time=00:30:00

# Sweep per_device_train_batch_size for group_size=1 (geneformer).
# Cancel once you see which sizes OOM.

module purge
module load cuda12.3/toolkit/12.3.2
module load cudnn8.6-cuda11.8/8.6.0.163

BATCH_SIZES=(4 8 16 32 64 128 256 512)
bs="${BATCH_SIZES[$((SLURM_ARRAY_TASK_ID - 1))]}"

echo "=== Task ${SLURM_ARRAY_TASK_ID}: batch_size=${bs} ==="

cd "$SLURM_SUBMIT_DIR"

python train.py \
    dataset_name="combat" \
    data.group_size=1 \
    data.dataset_path="/grid/zador/data_norepl/Ari/transcriptomics/covid/combat.dataset" \
    data.splits_path="/grid/zador/data_norepl/Ari/transcriptomics/covid/combat_donor_splits.json" \
    data.cv_fold=0 \
    model.pretrained_type="single-cell" \
    training.remove_unused_columns=true \
    training.per_device_train_batch_size="${bs}" \
    training.per_device_eval_batch_size="${bs}" \
    training.max_steps=5 \
    seed=0
