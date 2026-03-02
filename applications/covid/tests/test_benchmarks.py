"""Tests for COVID benchmark feature aggregation and classifiers."""

import pytest
import os
import tempfile
import numpy as np
import anndata as ad
import scipy.sparse
import pandas as pd


def make_benchmark_h5ad(path, n_donors=6, cells_per_donor=50, n_genes=20, seed=42):
    """Create a synthetic h5ad for benchmark testing."""
    rng = np.random.RandomState(seed)
    n_cells = n_donors * cells_per_donor
    X = rng.rand(n_cells, n_genes).astype(np.float32)

    donors = []
    labels = []
    cell_types = []
    for i in range(n_donors):
        donor = f"donor_{i}"
        label = ["control", "mild", "severe"][i % 3]
        for _ in range(cells_per_donor):
            donors.append(donor)
            labels.append(label)
            cell_types.append(f"type_{rng.randint(0, 4)}")

    obs = pd.DataFrame({
        "donor_id": donors,
        "label": labels,
        "cell_type": cell_types,
    }, index=[f"cell_{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.write_h5ad(path)
    return adata


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


class TestAggregation:

    def test_pseudobulk_shape(self, tmp_dir):
        """Pseudobulk features should have shape (n_donors, n_genes)."""
        from applications.covid.benchmarks import aggregate_donor_features

        path = os.path.join(tmp_dir, "test.h5ad")
        make_benchmark_h5ad(path, n_donors=4, n_genes=20)

        donors = [f"donor_{i}" for i in range(4)]
        features, labels, donor_arr = aggregate_donor_features(
            path, donors, feature_type="pseudobulk"
        )
        assert features.shape == (4, 20)
        assert labels.shape == (4,)
        assert donor_arr.shape == (4,)

    def test_pseudobulk_is_mean(self, tmp_dir):
        """Pseudobulk should be the mean of cell expression per donor."""
        from applications.covid.benchmarks import aggregate_donor_features

        path = os.path.join(tmp_dir, "test.h5ad")
        adata = make_benchmark_h5ad(path, n_donors=2, cells_per_donor=10, n_genes=5)

        features, _, _ = aggregate_donor_features(
            path, ["donor_0"], feature_type="pseudobulk"
        )
        # Manually compute mean for donor_0
        adata = ad.read_h5ad(path)
        donor_mask = adata.obs["donor_id"] == "donor_0"
        expected = np.mean(adata.X[donor_mask], axis=0)
        np.testing.assert_allclose(features[0], expected, rtol=1e-5)

    def test_cell_type_histogram_normalized(self, tmp_dir):
        """Cell type histogram should sum to 1."""
        from applications.covid.benchmarks import aggregate_donor_features

        path = os.path.join(tmp_dir, "test.h5ad")
        make_benchmark_h5ad(path, n_donors=3)

        features, _, _ = aggregate_donor_features(
            path, ["donor_0", "donor_1"], feature_type="cell_type_histogram"
        )
        for row in features:
            np.testing.assert_almost_equal(row.sum(), 1.0)

    def test_group_size_subsampling(self, tmp_dir):
        """With group_size, should subsample cells per donor."""
        from applications.covid.benchmarks import aggregate_donor_features

        path = os.path.join(tmp_dir, "test.h5ad")
        make_benchmark_h5ad(path, n_donors=2, cells_per_donor=100, n_genes=10)

        # Different group sizes should give different features (stochastic)
        f1, _, _ = aggregate_donor_features(
            path, ["donor_0"], feature_type="pseudobulk", group_size=10, seed=1
        )
        f2, _, _ = aggregate_donor_features(
            path, ["donor_0"], feature_type="pseudobulk", group_size=10, seed=2
        )
        # With different seeds, means over 10 cells should differ
        assert not np.allclose(f1, f2)


class TestClassifiers:

    def test_rf_runs(self, tmp_dir):
        """Random Forest should run without error on synthetic data."""
        from applications.covid.benchmarks import aggregate_donor_features, create_classifier

        path = os.path.join(tmp_dir, "test.h5ad")
        make_benchmark_h5ad(path, n_donors=12, cells_per_donor=30, n_genes=10)

        train_donors = [f"donor_{i}" for i in range(9)]
        test_donors = [f"donor_{i}" for i in range(9, 12)]

        train_feat, train_labels, _ = aggregate_donor_features(
            path, train_donors, "pseudobulk"
        )
        test_feat, test_labels, _ = aggregate_donor_features(
            path, test_donors, "pseudobulk"
        )

        clf = create_classifier("random_forest", seed=42)
        clf.fit(train_feat, train_labels)
        preds = clf.predict(test_feat)
        assert preds.shape == (3,)

    def test_lr_runs(self, tmp_dir):
        """Logistic Regression should run without error on synthetic data."""
        from applications.covid.benchmarks import aggregate_donor_features, create_classifier

        path = os.path.join(tmp_dir, "test.h5ad")
        make_benchmark_h5ad(path, n_donors=12, cells_per_donor=30, n_genes=10)

        train_donors = [f"donor_{i}" for i in range(9)]
        test_donors = [f"donor_{i}" for i in range(9, 12)]

        train_feat, train_labels, _ = aggregate_donor_features(
            path, train_donors, "pseudobulk"
        )
        test_feat, test_labels, _ = aggregate_donor_features(
            path, test_donors, "pseudobulk"
        )

        clf = create_classifier("logistic_regression", seed=42)
        clf.fit(train_feat, train_labels)
        preds = clf.predict(test_feat)
        probs = clf.predict_proba(test_feat)
        assert preds.shape == (3,)
        assert probs.shape[0] == 3
