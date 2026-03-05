#!/bin/bash
#SBATCH --job-name=covid-train
#SBATCH --output=slurm_logs/covid/train_%A_%a.out
#SBATCH --error=slurm_logs/covid/train_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=80G
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=gpuq
#SBATCH --time=1-00:00:00
#SBATCH --qos=slow_nice
#SBATCH --requeue
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --array=1-150%8

# 3 datasets x 5 folds x 10 group_sizes = 150 tasks

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID not set. Use: sbatch --array=N-M scripts/train_array.sh" >&2
    exit 1
fi

module purge
module load cuda12.3/toolkit/12.3.2
module load cudnn8.6-cuda11.8/8.6.0.163

eval "$(micromamba shell hook --shell bash)"
micromamba activate brain_annotation2

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

# Forward SIGUSR1 to Python process for graceful preemption checkpoint
trap 'kill -USR1 "$CHILD_PID"; wait "$CHILD_PID"' SIGUSR1

# Single-GPU DeepSpeed needs these env vars to avoid MPI fallback
export MASTER_ADDR=localhost
export MASTER_PORT=$(( 29500 + SLURM_ARRAY_TASK_ID ))
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

if [ "$gs" -eq 1 ]; then
    python train.py \
        dataset_name="${dataset}" \
        data.group_size=1 \
        data.dataset_path="/grid/zador/data_norepl/Ari/transcriptomics/covid/${dataset}.dataset" \
        data.splits_path="/grid/zador/data_norepl/Ari/transcriptomics/covid/${dataset}_donor_splits.json" \
        model.bert_path_or_name="/grid/zador/data_norepl/Ari/transcriptomics/geneformer_models/base_human_geneformer" \
        data.cv_fold="${fold}" \
        model.pretrained_type="single-cell" \
        training.remove_unused_columns=true \
        seed="${fold}" \
        training.fp16=false \
        +training.bf16=true \
        training.per_device_train_batch_size=512 \
        training.per_device_eval_batch_size=512 &
    CHILD_PID=$!
    wait "$CHILD_PID"
else
    python train.py \
        dataset_name="${dataset}" \
        data.group_size="${gs}" \
        data.dataset_path="/grid/zador/data_norepl/Ari/transcriptomics/covid/${dataset}.dataset" \
        data.splits_path="/grid/zador/data_norepl/Ari/transcriptomics/covid//${dataset}_donor_splits.json" \
        model.bert_path_or_name="/grid/zador/data_norepl/Ari/transcriptomics/geneformer_models/base_human_geneformer" \
        data.cv_fold="${fold}" \
        seed="${fold}" \
        training.fp16=true \
        +training.bf16=false \
        training.per_device_train_batch_size=512 \
        training.per_device_eval_batch_size=512 &
    CHILD_PID=$!
    wait "$CHILD_PID"
fi
