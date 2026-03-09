#!/bin/bash
#SBATCH --job-name=covid-lrsweep
#SBATCH --output=slurm_logs/covid/lrsweep_%A_%a.out
#SBATCH --error=slurm_logs/covid/lrsweep_%A_%a.err
#SBATCH --array=1-18%8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=80G
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=gpuq
#SBATCH --time=2-00:00:00
#SBATCH --qos=bio_ai
#SBATCH --requeue
#SBATCH --signal=B:SIGUSR1@120

# Sweep learning rate: 3 datasets x 6 LRs = 18 tasks, fold 0 only.

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID not set. Use: sbatch scripts/sweep_lr.sh" >&2
    exit 1
fi

module purge
module load cuda12.3/toolkit/12.3.2
module load cudnn8.6-cuda11.8/8.6.0.163

eval "$(micromamba shell hook --shell bash)"
micromamba activate brain_annotation2

DATASETS=(combat ren stevenson)
LEARNING_RATES=(1e-5 5e-5 1e-4 3e-4 5e-4 1e-3)
N_LRS=${#LEARNING_RATES[@]}

# Decode flat index -> (dataset, lr)
idx=$(( SLURM_ARRAY_TASK_ID - 1 ))
ds_idx=$(( idx / N_LRS ))
lr_idx=$(( idx % N_LRS ))

dataset="${DATASETS[$ds_idx]}"
lr="${LEARNING_RATES[$lr_idx]}"

gs=64
# n = ceil(gs/16): scale grad accumulation and epochs for large group sizes
n=$(( (gs + 15) / 16 ))
grad_accum=$n
base_epochs=10
epochs=$(( base_epochs * n ))

echo "=== Task ${SLURM_ARRAY_TASK_ID}: dataset=${dataset} lr=${lr} n=${n} ==="

cd "$SLURM_SUBMIT_DIR"

# Forward SIGUSR1 to Python process for graceful preemption checkpoint
trap 'kill -USR1 "$CHILD_PID"; wait "$CHILD_PID"' SIGUSR1

python train.py \
    dataset_name="${dataset}" \
    data.group_size="${gs}" \
    data.dataset_path="/grid/zador/data_norepl/Ari/transcriptomics/covid/${dataset}.dataset" \
    data.splits_path="/grid/zador/data_norepl/Ari/transcriptomics/covid/${dataset}_donor_splits.json" \
    model.bert_path_or_name="/grid/zador/data_norepl/Ari/transcriptomics/geneformer_models/base_human_geneformer" \
    data.cv_fold=0 \
    seed=0 \
    training.learning_rate="${lr}" \
    training.fp16=true \
    +training.bf16=false \
    training.per_device_train_batch_size=1024 \
    training.per_device_eval_batch_size=1024 \
    training.gradient_accumulation_steps="${grad_accum}" \
    training.num_train_epochs="${epochs}" \
    wandb.tags=[lr_sweep] &
CHILD_PID=$!
wait "$CHILD_PID"
