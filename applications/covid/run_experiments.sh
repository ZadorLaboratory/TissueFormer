#!/bin/bash
# COVID severity classification: full experiment sweep.
# Runs tokenization, TissueFormer training, and benchmarks across
# all datasets, CV folds, and group sizes.

set -euo pipefail

cd "$(dirname "$0")"

DATASETS=(combat ren stevenson)
N_FOLDS=5
GROUP_SIZES=(1 2 4 8 16 32 64 128 256 512)
BENCHMARK_GROUP_SIZES=(2 4 8 16 32 64 128 256 512 all)

for dataset in "${DATASETS[@]}"; do
    echo "=== Dataset: ${dataset} ==="
    H5AD_PATH="data/${dataset}_processed.h5ad"

    if [ ! -f "$H5AD_PATH" ]; then
        echo "WARNING: ${H5AD_PATH} not found, skipping ${dataset}"
        continue
    fi

    for fold in $(seq 0 $((N_FOLDS - 1))); do
        echo "--- Fold ${fold} ---"
        PREFIX="${dataset}_fold${fold}"

        # Tokenize (once per dataset per fold)
        if [ ! -d "data/${PREFIX}.dataset" ]; then
            echo "Tokenizing ${dataset} fold ${fold}..."
            python data/tokenize_cells.py \
                --h5ad_path "$H5AD_PATH" \
                --output_directory data \
                --output_prefix "$PREFIX" \
                --cv-fold "$fold" \
                --raw-counts \
                --nproc 8
        else
            echo "Dataset data/${PREFIX}.dataset already exists, skipping tokenization"
        fi

        # TissueFormer: sweep group sizes with HP tuning via Hydra multirun
        for gs in "${GROUP_SIZES[@]}"; do
            echo "TissueFormer gs=${gs}..."
            if [ "$gs" -eq 1 ]; then
                python train.py \
                    dataset_name="${dataset}" \
                    data.group_size=1 \
                    data.dataset_path="data/${PREFIX}.dataset" \
                    model.pretrained_type="single-cell" \
                    training.remove_unused_columns=true \
                    seed="${fold}"
            else
                python train.py \
                    dataset_name="${dataset}" \
                    data.group_size="${gs}" \
                    data.dataset_path="data/${PREFIX}.dataset" \
                    seed="${fold}"
            fi
        done

        # Benchmarks: group-level (matched group sizes) + whole-donor
        for gs in "${BENCHMARK_GROUP_SIZES[@]}"; do
            echo "Benchmarks gs=${gs}..."
            python benchmarks.py \
                dataset_name="${dataset}" \
                data.group_size="${gs}" \
                data.dataset_path="data/${PREFIX}.dataset"
        done
    done
done

echo "=== All experiments complete ==="
echo "Run: python figures/plot_results.py --results_dir outputs --output_dir figures"
