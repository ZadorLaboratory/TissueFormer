#!/usr/bin/env python
"""
Preprocess raw COVID h5ad files into a standardized format.

For each dataset, this script:
  1. Loads the raw h5ad
  2. Maps original severity labels to {control, mild, severe}
  3. Sets donor_id (using sample_id where donors have longitudinal samples)
  4. Removes donors with fewer than 1000 cells
  5. Saves the processed h5ad

This is a scriptified version of raw_data_standardization.ipynb.
"""

import argparse
import anndata
import numpy as np

MIN_CELLS_PER_DONOR = 1000

# ── Dataset-specific configurations ──────────────────────────────────────────

DATASET_CONFIGS = {
    "combat": {
        "label_column": "Source",
        "label_mapping": {
            ("COVID_SEV", "COVID_CRIT"): "severe",
            ("COVID_MILD", "COVID_HCW_MILD"): "mild",
            ("HV",): "control",
        },
    },
    "ren": {
        "label_column": "CoVID-19 severity",
        "label_mapping": {
            ("severe/critical",): "severe",
            ("mild/moderate",): "mild",
            ("control",): "control",
        },
    },
    "stevenson": {
        "label_column": "Status_on_day_collection_summary",
        "label_mapping": {
            ("Mild", "Moderate"): "mild",
            ("Severe", "Critical"): "severe",
            ("Healthy",): "control",
        },
        # Some donors were sampled at multiple timepoints with different
        # severity statuses (e.g. Moderate on D0, Critical on D12).
        # Since we classify current blood state, each sample is independent.
        "donor_id_column": "sample_id",
    },
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def apply_label_map(adata, column_name, mapping_dict):
    """Map values in column_name to standardized labels via mapping_dict."""
    def mapper(row):
        value = row[column_name]
        for key, mapped_value in mapping_dict.items():
            if value in key:
                return mapped_value
        return np.nan

    adata.obs["label"] = adata.obs.apply(mapper, axis=1)
    adata = adata[~adata.obs["label"].isna()]
    return adata


def remove_low_count_donors(adata, min_cells=MIN_CELLS_PER_DONOR):
    """Remove donors with fewer than min_cells cells."""
    donor_counts = adata.obs["donor_id"].value_counts()
    keep = donor_counts[donor_counts >= min_cells].index
    return adata[adata.obs["donor_id"].isin(keep)]


def preprocess_dataset(input_path, output_path, config):
    """Load, label, filter, and save a single dataset."""
    print(f"Loading {input_path}...")
    adata = anndata.read_h5ad(input_path)
    print(f"  {len(adata)} cells loaded")

    # Apply label mapping
    adata = apply_label_map(adata, config["label_column"], config["label_mapping"])
    print(f"  {len(adata)} cells after label mapping")

    # Override donor_id if configured (e.g. for longitudinal datasets)
    if "donor_id_column" in config:
        col = config["donor_id_column"]
        print(f"  Using '{col}' as donor_id")
        adata.obs["donor_id"] = adata.obs[col]

    # Remove low-count donors
    adata = remove_low_count_donors(adata)
    print(f"  {len(adata)} cells after removing donors with <{MIN_CELLS_PER_DONOR} cells")

    # Verify label integrity
    donor_labels = adata.obs.groupby("donor_id", observed=True)["label"].nunique()
    multi = donor_labels[donor_labels > 1]
    if len(multi) > 0:
        for d in multi.index:
            labels = adata.obs.loc[adata.obs["donor_id"] == d, "label"].unique().tolist()
            print(f"  WARNING: Donor {d} has multiple labels: {labels}")
        raise ValueError(
            f"{len(multi)} donor(s) have multiple labels. "
            "Consider using a sample-level donor_id_column in the config."
        )

    # Summary
    donors_per_label = adata.obs.groupby("label", observed=True)["donor_id"].nunique()
    print(f"  Donors per label:\n{donors_per_label.to_string()}")

    adata.write_h5ad(output_path)
    print(f"  Saved to {output_path}")
    return adata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw COVID h5ad files")
    parser.add_argument("--dataset", choices=list(DATASET_CONFIGS.keys()),
                        help="Dataset to preprocess. If omitted, processes all.")
    parser.add_argument("--data-dir", default=".",
                        help="Directory containing raw h5ad files (default: .)")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(DATASET_CONFIGS.keys())

    for name in datasets:
        import os
        input_path = os.path.join(args.data_dir, f"{name}.h5ad")
        output_path = os.path.join(args.data_dir, f"{name}_processed.h5ad")
        if not os.path.exists(input_path):
            print(f"Skipping {name}: {input_path} not found")
            continue
        preprocess_dataset(input_path, output_path, DATASET_CONFIGS[name])
        print()
