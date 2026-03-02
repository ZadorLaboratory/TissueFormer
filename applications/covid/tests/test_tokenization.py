"""Tests for COVID tokenization pipeline."""

import pytest
import os
import json
import tempfile
import numpy as np
import anndata as ad
import scipy.sparse


def make_synthetic_h5ad(path, n_cells=50, n_donors=3, n_genes=100, seed=42):
    """Create a minimal synthetic h5ad file for testing."""
    rng = np.random.RandomState(seed)

    # Random count matrix
    X = scipy.sparse.random(n_cells, n_genes, density=0.3, random_state=rng,
                            format="csr") * 10
    X.data = np.abs(X.data).astype(np.float32)

    # Create gene names as fake Ensembl IDs
    var_names = [f"ENSG{i:011d}" for i in range(n_genes)]

    # Assign donors and labels
    donors = [f"donor_{i % n_donors}" for i in range(n_cells)]
    label_map = {0: "control", 1: "mild", 2: "severe"}
    # Each donor gets a consistent label
    donor_labels = {f"donor_{i}": label_map[i % 3] for i in range(n_donors)}
    labels = [donor_labels[d] for d in donors]
    cell_types = [f"type_{rng.randint(0, 5)}" for _ in range(n_cells)]

    obs = {
        "donor_id": donors,
        "label": labels,
        "cell_type": cell_types,
    }

    import pandas as pd
    obs_df = pd.DataFrame(obs, index=[f"cell_{i}" for i in range(n_cells)])
    var_df = pd.DataFrame(index=var_names)

    adata = ad.AnnData(X=X, obs=obs_df, var=var_df)
    adata.write_h5ad(path)
    return adata


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


class TestTokenization:

    def test_donor_isolation(self, tmp_dir):
        """No donor_id should appear in both train and test."""
        from applications.covid.data.tokenize_cells import tokenize_and_split

        h5ad_path = os.path.join(tmp_dir, "test.h5ad")
        make_synthetic_h5ad(h5ad_path, n_cells=180, n_donors=12)

        ds = tokenize_and_split(
            h5ad_path, tmp_dir, "test_covid", cv_fold=0,
            nproc=1, n_splits=3
        )

        train_donors = set(ds["train"]["donor_id"])
        test_donors = set(ds["test"]["donor_id"])
        assert len(train_donors & test_donors) == 0, \
            f"Donors in both splits: {train_donors & test_donors}"

    def test_label_integrity(self, tmp_dir):
        """All cells for a donor should have the same integer label."""
        from applications.covid.data.tokenize_cells import tokenize_and_split

        h5ad_path = os.path.join(tmp_dir, "test.h5ad")
        make_synthetic_h5ad(h5ad_path, n_cells=180, n_donors=12)

        ds = tokenize_and_split(
            h5ad_path, tmp_dir, "test_covid", cv_fold=0,
            nproc=1, n_splits=3
        )

        for split in ["train", "test"]:
            donors = np.array(ds[split]["donor_id"])
            labels = np.array(ds[split]["label"])
            for donor in np.unique(donors):
                donor_labels = labels[donors == donor]
                assert len(np.unique(donor_labels)) == 1, \
                    f"Donor {donor} has inconsistent labels in {split}"

    def test_splits_have_expected_columns(self, tmp_dir):
        """Output should have input_ids, donor_id, label, cell_type."""
        from applications.covid.data.tokenize_cells import tokenize_and_split

        h5ad_path = os.path.join(tmp_dir, "test.h5ad")
        make_synthetic_h5ad(h5ad_path, n_cells=180, n_donors=12)

        ds = tokenize_and_split(
            h5ad_path, tmp_dir, "test_covid", cv_fold=0,
            nproc=1, n_splits=3
        )

        for split in ["train", "test"]:
            cols = ds[split].column_names
            assert "input_ids" in cols
            assert "donor_id" in cols
            assert "label" in cols
            assert "cell_type" in cols

    def test_label_map_saved(self, tmp_dir):
        """label_map.json should be saved and round-trip correctly."""
        from applications.covid.data.tokenize_cells import tokenize_and_split

        h5ad_path = os.path.join(tmp_dir, "test.h5ad")
        make_synthetic_h5ad(h5ad_path, n_cells=180, n_donors=12)

        tokenize_and_split(
            h5ad_path, tmp_dir, "test_covid", cv_fold=0,
            nproc=1, n_splits=3
        )

        map_path = os.path.join(tmp_dir, "test_covid_label_map.json")
        assert os.path.exists(map_path)
        with open(map_path) as f:
            label_map = json.load(f)
        assert "control" in label_map
        assert label_map["control"] == 0

    def test_donor_splits_saved(self, tmp_dir):
        """donor_splits.json should record fold assignments."""
        from applications.covid.data.tokenize_cells import tokenize_and_split

        h5ad_path = os.path.join(tmp_dir, "test.h5ad")
        make_synthetic_h5ad(h5ad_path, n_cells=180, n_donors=12)

        tokenize_and_split(
            h5ad_path, tmp_dir, "test_covid", cv_fold=0,
            nproc=1, n_splits=3
        )

        splits_path = os.path.join(tmp_dir, "test_covid_donor_splits.json")
        assert os.path.exists(splits_path)
        with open(splits_path) as f:
            splits = json.load(f)
        assert "folds" in splits
        assert "0" in splits["folds"]
        assert "train_donors" in splits["folds"]["0"]
        assert "test_donors" in splits["folds"]["0"]
