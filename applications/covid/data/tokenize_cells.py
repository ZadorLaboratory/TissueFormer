#!/usr/bin/env python
"""
Tokenize COVID scRNA-seq h5ad files with donor-stratified cross-validation splits.

Adapts the brain annotation tokenizer for COVID severity classification.
Key differences from brain annotation:
  - Labels are severity levels (control/mild/severe) instead of brain regions
  - Train/test splits are by donor (StratifiedGroupKFold) instead of by animal
  - No spatial filtering or coordinate processing
  - Uses default Geneformer token dictionary (Ensembl gene IDs)

Tokenization is done once for the entire dataset.  CV fold splitting
happens downstream at training / benchmark time using the saved
donor_splits.json metadata.
"""

from tissueformer.tokenizer import TranscriptomeTokenizer
import time
import argparse
import os
import json
import numpy as np
import anndata as ad
from sklearn.model_selection import StratifiedGroupKFold
from pathlib import Path


def _find_geneformer_token_dict():
    """Locate the Geneformer token dictionary from the installed package."""
    try:
        import geneformer
        pkg_dir = Path(geneformer.__file__).parent
        # Try the standard token_dictionary.pkl first
        candidates = [
            pkg_dir / "token_dictionary.pkl",
            pkg_dir / "util" / "files" / "tokens" / "geneformer_token_dict.pkl",
        ]
        for p in candidates:
            if p.exists():
                return str(p)
    except ImportError:
        pass
    raise FileNotFoundError(
        "Could not find Geneformer token dictionary. "
        "Install geneformer or provide --token-dictionary-file."
    )


LABEL_MAP_3CLASS = {"control": 0, "mild": 1, "severe": 2}
LABEL_MAP_2CLASS = {"mild": 0, "severe": 1}


def get_label_map(adata):
    """Determine label map based on unique labels in the dataset."""
    unique_labels = set(adata.obs["label"].unique())
    if "control" in unique_labels:
        return LABEL_MAP_3CLASS
    return LABEL_MAP_2CLASS


def get_donor_splits(adata, n_splits=5, seed=42):
    """
    Create donor-stratified CV splits using StratifiedGroupKFold.

    Returns list of (train_donors, test_donors) tuples, one per fold.
    All cells from a donor stay in the same split.
    """
    # Get one row per donor for stratification
    donor_df = adata.obs.groupby("donor_id", observed=True)["label"].first().reset_index()
    donors = donor_df["donor_id"].values
    labels = donor_df["label"].values

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # groups = donors for StratifiedGroupKFold (each donor is its own group)
    splits = []
    for train_idx, test_idx in sgkf.split(donors, labels, groups=donors):
        train_donors = donors[train_idx].tolist()
        test_donors = donors[test_idx].tolist()
        splits.append((train_donors, test_donors))

    return splits


def tokenize_dataset(h5ad_path, output_directory, output_prefix,
                     nproc=8, raw_counts=False, n_splits=5, seed=42,
                     token_dictionary_file=None):
    """Tokenize an h5ad file and save CV split metadata.

    All cells are tokenized once into a single HF Dataset.  Fold
    splitting is deferred to training / benchmark time.
    """
    t0 = time.time()

    # Resolve token dictionary
    if token_dictionary_file is None:
        token_dictionary_file = _find_geneformer_token_dict()
    print(f"Using token dictionary: {token_dictionary_file}")

    print(f"Loading {h5ad_path}...")
    adata = ad.read_h5ad(h5ad_path)
    print(f"Loaded {len(adata)} cells")

    # Determine label map
    label_map = get_label_map(adata)
    print(f"Label map: {label_map}")

    # Filter to labels in the map (drop any unexpected labels)
    valid_labels = set(label_map.keys())
    mask = adata.obs["label"].isin(valid_labels)
    if mask.sum() < len(adata):
        print(f"Filtering from {len(adata)} to {mask.sum()} cells (removing labels not in {valid_labels})")
        adata = adata[mask].copy()

    # Compute donor splits and save metadata
    splits = get_donor_splits(adata, n_splits=n_splits, seed=seed)

    split_info = {
        "n_splits": n_splits,
        "seed": seed,
        "folds": {
            str(i): {
                "train_donors": s[0],
                "test_donors": s[1],
            }
            for i, s in enumerate(splits)
        }
    }
    os.makedirs(output_directory, exist_ok=True)
    splits_path = os.path.join(output_directory, f"{output_prefix}_donor_splits.json")
    with open(splits_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"Saved donor splits to {splits_path}")

    # Save label map
    label_map_path = os.path.join(output_directory, f"{output_prefix}_label_map.json")
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)

    # Metadata columns to preserve during tokenization
    label_dict = {
        "donor_id": "donor_id",
        "label": "label",
        "cell_type": "cell_type",
    }

    # Tokenize all cells at once
    print(f"\nTokenizing all {len(adata)} cells...")

    # Write temporary h5ad for tokenizer
    tmp_path = os.path.join(output_directory, "_tmp_all.h5ad")
    adata.write_h5ad(tmp_path)

    tk = TranscriptomeTokenizer(
        label_dict,
        nproc=nproc,
        token_dictionary_file=token_dictionary_file,
        retain_counts=raw_counts,
        prepend_cls=False,  # Default Geneformer token dict has no <cls> token
    )

    tokenized_dataset, _ = tk.tokenize_data(
        tmp_path, output_directory, "_tmp_all", save=False
    )

    # Clean up temp file
    os.remove(tmp_path)

    # Map string labels to integers
    tokenized_dataset = tokenized_dataset.map(
        lambda x: {"label": label_map[x["label"]]}
    )

    # Verify label integrity: all cells for a donor should have the same label
    donor_ids = np.array(tokenized_dataset["donor_id"])
    labels = np.array(tokenized_dataset["label"])
    for donor in np.unique(donor_ids):
        donor_labels = labels[donor_ids == donor]
        assert len(np.unique(donor_labels)) == 1, \
            f"Donor {donor} has multiple labels: {np.unique(donor_labels)}"

    # Verify labels are in valid range
    unique_labels = np.unique(labels)
    assert unique_labels.min() >= 0, f"Negative label found: {unique_labels.min()}"
    assert unique_labels.max() < len(label_map), \
        f"Label {unique_labels.max()} >= num_labels {len(label_map)}"

    print(f"  {len(tokenized_dataset)} cells tokenized")

    # Save as single Dataset
    save_path = os.path.join(output_directory, f"{output_prefix}.dataset")
    tokenized_dataset.save_to_disk(save_path)
    print(f"\nSaved dataset to {save_path}")
    print(f"Total time: {time.time() - t0:.1f}s")

    return tokenized_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize COVID scRNA-seq data")
    parser.add_argument("--h5ad_path", type=str, required=True,
                        help="Path to processed h5ad file")
    parser.add_argument("--output_directory", type=str, default=".",
                        help="Output directory")
    parser.add_argument("--output_prefix", type=str, default="covid",
                        help="Output prefix for files")
    parser.add_argument("--nproc", type=int, default=8,
                        help="Number of processes for tokenization")
    parser.add_argument("--raw-counts", action="store_true",
                        help="Retain raw counts in tokenized dataset")
    parser.add_argument("--n-splits", type=int, default=5,
                        help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for splits")
    parser.add_argument("--token-dictionary-file", type=str, default=None,
                        help="Path to token dictionary pickle. If None, uses Geneformer default.")
    args = parser.parse_args()

    tokenize_dataset(
        h5ad_path=args.h5ad_path,
        output_directory=args.output_directory,
        output_prefix=args.output_prefix,
        nproc=args.nproc,
        raw_counts=args.raw_counts,
        n_splits=args.n_splits,
        seed=args.seed,
        token_dictionary_file=args.token_dictionary_file,
    )
