#!/usr/bin/env python
"""
Create a combined COVID dataset by concatenating combat, ren, and stevenson.

Produces:
  - combined.dataset          (HF Dataset, concatenation of tokenized datasets)
  - combined_processed.h5ad   (AnnData, concatenation of processed h5ad files)
  - combined_donor_splits.json (5-fold StratifiedGroupKFold over all donors)
  - combined_label_map.json   (identical to individual: control/mild/severe)
"""

import argparse
import gc
import json
import os

import anndata as ad
from datasets import concatenate_datasets, load_from_disk

from tokenize_cells import get_donor_splits, LABEL_MAP_3CLASS


INDIVIDUAL_DATASETS = ["combat", "ren", "stevenson"]


def create_combined_dataset(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # --- Concatenate HF Datasets ---
    print("Loading tokenized datasets...")
    hf_datasets = []
    for name in INDIVIDUAL_DATASETS:
        path = os.path.join(data_dir, f"{name}.dataset")
        print(f"  Loading {path}")
        hf_datasets.append(load_from_disk(path))

    combined_hf = concatenate_datasets(hf_datasets)
    print(f"Combined HF dataset: {len(combined_hf)} cells")

    save_path = os.path.join(output_dir, "combined.dataset")
    combined_hf.save_to_disk(save_path)
    print(f"Saved {save_path}")
    del combined_hf, hf_datasets
    gc.collect()

    # --- Concatenate h5ad files ---
    print("\nLoading h5ad files...")
    adatas = []
    for name in INDIVIDUAL_DATASETS:
        path = os.path.join(data_dir, f"{name}_processed.h5ad")
        print(f"  Loading {path}")
        adatas.append(ad.read_h5ad(path))

    combined_adata = ad.concat(adatas, join="outer")
    print(f"Combined AnnData: {combined_adata.n_obs} cells, "
          f"{combined_adata.obs['donor_id'].nunique()} donors")

    # Free memory from individual adatas
    del adatas
    gc.collect()

    h5ad_path = os.path.join(output_dir, "combined_processed.h5ad")
    combined_adata.write_h5ad(h5ad_path)
    print(f"Saved {h5ad_path}")

    # --- Generate donor splits ---
    print("\nGenerating donor splits...")
    splits = get_donor_splits(combined_adata, n_splits=5, seed=42)

    split_info = {
        "n_splits": 5,
        "seed": 42,
        "folds": {
            str(i): {
                "train_donors": s[0],
                "test_donors": s[1],
            }
            for i, s in enumerate(splits)
        },
    }
    splits_path = os.path.join(output_dir, "combined_donor_splits.json")
    with open(splits_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"Saved {splits_path}")

    # --- Save label map ---
    label_map_path = os.path.join(output_dir, "combined_label_map.json")
    with open(label_map_path, "w") as f:
        json.dump(LABEL_MAP_3CLASS, f, indent=2)
    print(f"Saved {label_map_path}")

    del combined_adata
    gc.collect()

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create combined COVID dataset")
    parser.add_argument("--data-dir", type=str, default=".",
                        help="Directory containing individual dataset files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: same as data-dir)")
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir else args.data_dir
    create_combined_dataset(args.data_dir, output_dir)
