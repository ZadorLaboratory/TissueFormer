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

    def test_returns_single_dataset(self, tmp_dir):
        """tokenize_dataset should return a single Dataset, not DatasetDict."""
        from applications.covid.data.tokenize_cells import tokenize_dataset
        from datasets import Dataset

        h5ad_path = os.path.join(tmp_dir, "test.h5ad")
        make_synthetic_h5ad(h5ad_path, n_cells=180, n_donors=12)

        ds = tokenize_dataset(
            h5ad_path, tmp_dir, "test_covid",
            nproc=1, n_splits=3
        )

        assert isinstance(ds, Dataset), \
            f"Expected Dataset, got {type(ds)}"

    def test_donor_isolation_in_splits(self, tmp_dir):
        """No donor_id should appear in both train and test for any fold."""
        from applications.covid.data.tokenize_cells import tokenize_dataset

        h5ad_path = os.path.join(tmp_dir, "test.h5ad")
        make_synthetic_h5ad(h5ad_path, n_cells=180, n_donors=12)

        tokenize_dataset(
            h5ad_path, tmp_dir, "test_covid",
            nproc=1, n_splits=3
        )

        splits_path = os.path.join(tmp_dir, "test_covid_donor_splits.json")
        with open(splits_path) as f:
            splits = json.load(f)

        for fold_key, fold in splits["folds"].items():
            train_donors = set(fold["train_donors"])
            test_donors = set(fold["test_donors"])
            assert len(train_donors & test_donors) == 0, \
                f"Fold {fold_key}: donors in both splits: {train_donors & test_donors}"

    def test_label_integrity(self, tmp_dir):
        """All cells for a donor should have the same integer label."""
        from applications.covid.data.tokenize_cells import tokenize_dataset

        h5ad_path = os.path.join(tmp_dir, "test.h5ad")
        make_synthetic_h5ad(h5ad_path, n_cells=180, n_donors=12)

        ds = tokenize_dataset(
            h5ad_path, tmp_dir, "test_covid",
            nproc=1, n_splits=3
        )

        donors = np.array(ds["donor_id"])
        labels = np.array(ds["label"])
        for donor in np.unique(donors):
            donor_labels = labels[donors == donor]
            assert len(np.unique(donor_labels)) == 1, \
                f"Donor {donor} has inconsistent labels"

    def test_dataset_has_expected_columns(self, tmp_dir):
        """Output should have input_ids, donor_id, label, cell_type."""
        from applications.covid.data.tokenize_cells import tokenize_dataset

        h5ad_path = os.path.join(tmp_dir, "test.h5ad")
        make_synthetic_h5ad(h5ad_path, n_cells=180, n_donors=12)

        ds = tokenize_dataset(
            h5ad_path, tmp_dir, "test_covid",
            nproc=1, n_splits=3
        )

        cols = ds.column_names
        assert "input_ids" in cols
        assert "donor_id" in cols
        assert "label" in cols
        assert "cell_type" in cols

    def test_label_map_saved(self, tmp_dir):
        """label_map.json should be saved and round-trip correctly."""
        from applications.covid.data.tokenize_cells import tokenize_dataset

        h5ad_path = os.path.join(tmp_dir, "test.h5ad")
        make_synthetic_h5ad(h5ad_path, n_cells=180, n_donors=12)

        tokenize_dataset(
            h5ad_path, tmp_dir, "test_covid",
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
        from applications.covid.data.tokenize_cells import tokenize_dataset

        h5ad_path = os.path.join(tmp_dir, "test.h5ad")
        make_synthetic_h5ad(h5ad_path, n_cells=180, n_donors=12)

        tokenize_dataset(
            h5ad_path, tmp_dir, "test_covid",
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

    def test_splits_only_mode(self, tmp_dir):
        """splits_only=True should generate JSON files without tokenizing."""
        from applications.covid.data.tokenize_cells import tokenize_dataset

        h5ad_path = os.path.join(tmp_dir, "test.h5ad")
        make_synthetic_h5ad(h5ad_path, n_cells=180, n_donors=12)

        result = tokenize_dataset(
            h5ad_path, tmp_dir, "test_covid",
            nproc=1, n_splits=3, splits_only=True
        )

        # Should return None (no dataset created)
        assert result is None

        # Splits and label map should exist
        splits_path = os.path.join(tmp_dir, "test_covid_donor_splits.json")
        label_map_path = os.path.join(tmp_dir, "test_covid_label_map.json")
        assert os.path.exists(splits_path)
        assert os.path.exists(label_map_path)

        # Dataset directory should NOT exist
        dataset_path = os.path.join(tmp_dir, "test_covid.dataset")
        assert not os.path.exists(dataset_path)

        # Splits should be valid
        with open(splits_path) as f:
            splits = json.load(f)
        assert len(splits["folds"]) == 3
