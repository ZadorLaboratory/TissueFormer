"""
End-to-end integration test for the COVID severity pipeline.

Tests the full flow: synthetic h5ad -> tokenization -> training -> benchmarks.
Uses tiny synthetic data for fast execution.
"""

import pytest
import os
import tempfile
import numpy as np
import anndata as ad
import scipy.sparse
import pandas as pd


def make_tiny_h5ad(path, n_donors=6, cells_per_donor=30, n_genes=50, seed=42):
    """Create a tiny h5ad suitable for end-to-end testing."""
    rng = np.random.RandomState(seed)
    n_cells = n_donors * cells_per_donor
    X = scipy.sparse.random(n_cells, n_genes, density=0.5, random_state=rng,
                            format="csr") * 100
    X.data = np.abs(X.data).astype(np.float32)

    # Use fake Ensembl IDs
    var_names = [f"ENSG{i:011d}" for i in range(n_genes)]

    donors = []
    labels = []
    cell_types = []
    for i in range(n_donors):
        donor = f"donor_{i}"
        label = ["control", "mild", "severe"][i % 3]
        for _ in range(cells_per_donor):
            donors.append(donor)
            labels.append(label)
            cell_types.append(f"type_{rng.randint(0, 3)}")

    obs = pd.DataFrame({
        "donor_id": donors,
        "label": labels,
        "cell_type": cell_types,
    }, index=[f"cell_{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=var_names)

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.write_h5ad(path)
    return adata


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


class TestEndToEnd:

    def test_tokenize_and_benchmark(self, tmp_dir):
        """Tokenize, then run benchmarks on the same splits."""
        from applications.covid.data.tokenize_cells import tokenize_and_split
        from applications.covid.benchmarks import aggregate_donor_features
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        # 1. Create synthetic h5ad (12 donors = 4 per class, enough for CV)
        h5ad_path = os.path.join(tmp_dir, "test.h5ad")
        make_tiny_h5ad(h5ad_path, n_donors=12, cells_per_donor=20)

        # 2. Tokenize with 2-fold CV
        ds = tokenize_and_split(
            h5ad_path, tmp_dir, "e2e_test", cv_fold=0,
            nproc=1, n_splits=2
        )

        assert "train" in ds
        assert "test" in ds
        assert len(ds["train"]) > 0
        assert len(ds["test"]) > 0

        # 3. Run benchmarks
        train_donors = list(set(ds["train"]["donor_id"]))
        test_donors = list(set(ds["test"]["donor_id"]))

        for feat_type in ["pseudobulk", "cell_type_histogram"]:
            train_feat, train_labels, _ = aggregate_donor_features(
                h5ad_path, train_donors, feat_type
            )
            test_feat, test_labels, _ = aggregate_donor_features(
                h5ad_path, test_donors, feat_type
            )

            # 4. Train and predict (use simple classifiers, not GridSearchCV,
            #    since tiny test data has too few samples for inner CV)
            classifiers = [
                ("rf", RandomForestClassifier(n_estimators=10, random_state=42)),
                ("lr", LogisticRegression(max_iter=200, random_state=42)),
            ]
            for clf_name, clf in classifiers:
                clf.fit(train_feat, train_labels)
                preds = clf.predict(test_feat)
                probs = clf.predict_proba(test_feat)

                # 5. Verify outputs
                assert preds.shape == test_labels.shape
                assert probs.shape[0] == len(test_labels)

                # Metrics in valid range
                acc = (preds == test_labels).mean()
                assert 0.0 <= acc <= 1.0
                assert not np.any(np.isnan(preds))
                assert not np.any(np.isnan(probs))

    def test_donor_sampler_with_tokenized_data(self, tmp_dir):
        """DonorGroupSampler should work with tokenized dataset."""
        from applications.covid.data.tokenize_cells import tokenize_and_split
        from tissueformer.samplers import DonorGroupSampler

        h5ad_path = os.path.join(tmp_dir, "test.h5ad")
        make_tiny_h5ad(h5ad_path, n_donors=6)

        ds = tokenize_and_split(
            h5ad_path, tmp_dir, "e2e_test", cv_fold=0,
            nproc=1, n_splits=2
        )

        # Test with group_size=4
        sampler = DonorGroupSampler(
            ds["train"], batch_size=16, group_size=4, seed=0
        )
        indices = list(sampler)
        assert len(indices) > 0
        assert len(indices) % 4 == 0

        # Verify donor purity
        donor_ids = np.array(ds["train"]["donor_id"])
        for start in range(0, min(len(indices), 40), 4):
            group = indices[start:start + 4]
            donors = set(donor_ids[group])
            assert len(donors) == 1

        # Test with group_size=1
        sampler1 = DonorGroupSampler(
            ds["train"], batch_size=16, group_size=1, seed=0
        )
        indices1 = list(sampler1)
        assert len(indices1) == len(ds["train"])
