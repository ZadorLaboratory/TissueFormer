#!/bin/bash
#SBATCH --job-name=covid-dl-bench
#SBATCH --output=slurm_logs/covid/dl_bench_%A_%a.out
#SBATCH --error=slurm_logs/covid/dl_bench_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=30G
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=gpuq
#SBATCH --time=1-00:00:00
#SBATCH --qos=bio_ai
#SBATCH --requeue
#SBATCH --array=1-10

# 3 datasets x 5 folds x 10 group_sizes = 150 tasks, spread across 8 array slots.
# Each slot runs ceil(remaining/8) configurations serially.
# Each configuration runs all 3 DL methods (CellCnn, scAGG, ScRAT) sequentially.
# Set FIRST_CONFIG to skip already-completed configs (1-based, default 1).
#   FIRST_CONFIG=11 sbatch scripts/dl_benchmark_array.sh   # skip first 10

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID not set. Use: sbatch scripts/dl_benchmark_array.sh" >&2
    exit 1
fi

module purge
module load cuda12.3/toolkit/12.3.2
module load cudnn8.6-cuda11.8/8.6.0.163

eval "$(micromamba shell hook --shell bash)"
micromamba activate brain_annotation2

# Single-GPU DeepSpeed needs these env vars to avoid MPI fallback
export MASTER_ADDR=localhost
export MASTER_PORT=$(( 29500 + SLURM_ARRAY_TASK_ID ))
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

DATASETS=(combat ren stevenson combined)
N_FOLDS=5
N_DATASETS=${#DATASETS[@]}
BENCHMARK_GROUP_SIZES=(2 4 8 16 32 64 128 256 512 all)
N_BENCHMARK_GROUP_SIZES=${#BENCHMARK_GROUP_SIZES[@]}
TOTAL_TASKS=$(( ${#DATASETS[@]} * N_FOLDS * N_BENCHMARK_GROUP_SIZES ))  # 150
MAX_ARRAY=10
FIRST_CONFIG=${FIRST_CONFIG:-1}
first_idx=$(( FIRST_CONFIG - 1 ))

# Compute which flat indices (0-based) this array element handles
remaining=$(( TOTAL_TASKS - first_idx ))
tasks_per_slot=$(( (remaining + MAX_ARRAY - 1) / MAX_ARRAY ))
slot=$(( SLURM_ARRAY_TASK_ID - 1 ))
start_idx=$(( first_idx + slot * tasks_per_slot ))
end_idx=$(( start_idx + tasks_per_slot ))
if [ "$end_idx" -gt "$TOTAL_TASKS" ]; then
    end_idx=$TOTAL_TASKS
fi

cd "$SLURM_SUBMIT_DIR"

for (( idx=start_idx; idx<end_idx; idx++ )); do
    # Decode flat index -> (dataset, fold, group_size)

    fold=$(( idx / (N_DATASETS * N_BENCHMARK_GROUP_SIZES) ))                                                                                        
    remainder=$(( idx % (N_DATASETS * N_BENCHMARK_GROUP_SIZES) ))
    ds_idx=$(( remainder / N_BENCHMARK_GROUP_SIZES ))
    gs_idx=$(( remainder % N_BENCHMARK_GROUP_SIZES ))

    dataset="${DATASETS[$ds_idx]}"
    gs="${BENCHMARK_GROUP_SIZES[$gs_idx]}"

    echo "=== Slot ${SLURM_ARRAY_TASK_ID}, config $(( idx + 1 ))/${TOTAL_TASKS}: dataset=${dataset} fold=${fold} gs=${gs} ==="

    python benchmarks.py \
        dataset_name="${dataset}" \
        data.group_size="${gs}" \
        data.h5ad_path="/grid/zador/data_norepl/Ari/transcriptomics/covid/${dataset}_processed.h5ad" \
        data.splits_path="/grid/zador/data_norepl/Ari/transcriptomics/covid/${dataset}_donor_splits.json" \
        data.cv_fold="${fold}" \
        run_classical=false \
        run_cellcnn=true \
        run_scagg=true \
        run_scrat=true
done
