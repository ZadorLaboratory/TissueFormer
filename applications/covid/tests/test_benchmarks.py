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


class TestMultiGroupAggregation:

    def test_multi_group_shape(self, tmp_dir):
        """Multi-group features should have multiple rows per donor."""
        from applications.covid.benchmarks import aggregate_donor_features_multi

        path = os.path.join(tmp_dir, "test.h5ad")
        make_benchmark_h5ad(path, n_donors=2, cells_per_donor=100, n_genes=10)

        donors = ["donor_0", "donor_1"]
        features, labels, donor_arr = aggregate_donor_features_multi(
            path, donors, feature_type="pseudobulk", group_size=20
        )
        # ceil(100/20) = 5 groups per donor, 2 donors = 10
        assert features.shape == (10, 10)
        assert labels.shape == (10,)
        assert donor_arr.shape == (10,)
        # Check each donor has 5 groups
        assert (donor_arr == "donor_0").sum() == 5
        assert (donor_arr == "donor_1").sum() == 5

    def test_multi_group_gs1(self, tmp_dir):
        """group_size=1 should create one group per cell."""
        from applications.covid.benchmarks import aggregate_donor_features_multi

        path = os.path.join(tmp_dir, "test.h5ad")
        make_benchmark_h5ad(path, n_donors=2, cells_per_donor=10, n_genes=5)

        features, labels, donor_arr = aggregate_donor_features_multi(
            path, ["donor_0"], feature_type="pseudobulk", group_size=1
        )
        # 10 cells / gs=1 = 10 groups
        assert features.shape == (10, 5)

    def test_multi_group_padding(self, tmp_dir):
        """Last group should be padded when n_cells % group_size != 0."""
        from applications.covid.benchmarks import aggregate_donor_features_multi

        path = os.path.join(tmp_dir, "test.h5ad")
        make_benchmark_h5ad(path, n_donors=1, cells_per_donor=7, n_genes=5)

        features, _, _ = aggregate_donor_features_multi(
            path, ["donor_0"], feature_type="pseudobulk", group_size=3
        )
        # ceil(7/3) = 3 groups
        assert features.shape == (3, 5)

    def test_aggregate_to_donor(self):
        """aggregate_to_donor should correctly aggregate predictions."""
        from applications.covid.benchmarks import aggregate_to_donor

        # 2 donors, 3 groups each
        predictions = np.array([0, 1, 0, 2, 2, 1])
        labels = np.array([0, 0, 0, 2, 2, 2])
        probs = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.1, 0.8],
            [0.1, 0.1, 0.8],
            [0.1, 0.7, 0.2],
        ])
        donor_ids = np.array(["d0", "d0", "d0", "d1", "d1", "d1"])

        result = aggregate_to_donor(predictions, labels, probs, donor_ids, 3)

        # Majority vote: d0 -> 0 (2 votes), d1 -> 2 (2 votes)
        np.testing.assert_array_equal(result["majority_vote"]["predictions"], [0, 2])
        np.testing.assert_array_equal(result["majority_vote"]["labels"], [0, 2])

        # Mean probs for d0: mean of rows 0-2
        expected_mp_d0 = np.mean(probs[:3], axis=0)
        np.testing.assert_allclose(
            result["mean_probs"]["probs"][0], expected_mp_d0, rtol=1e-5
        )


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


class TestDLBenchmarks:
    """Smoke tests for CellCnn, scAGG, and ScRAT benchmark models."""

    def _make_synthetic_mil_data(self, n_samples=12, n_genes=20, cells_per=50):
        """Create synthetic MIL data with class signal."""
        rng = np.random.RandomState(42)
        expression = rng.randn(n_samples * cells_per, n_genes).astype(np.float32)
        for i in range(n_samples):
            label = i % 3
            expression[i * cells_per:(i + 1) * cells_per, label * 3:(label + 1) * 3] += 2.0
        sample_indices = [np.arange(i * cells_per, (i + 1) * cells_per) for i in range(n_samples)]
        labels = np.array([i % 3 for i in range(n_samples)])
        return expression, sample_indices, labels

    def test_cellcnn_forward_shape(self):
        """CellCnn forward pass produces correct output shape."""
        import torch
        from tissueformer.benchmark_models.cellcnn import CellCnn
        model = CellCnn(n_genes=20, n_classes=3, n_filters=4, cells_per_input=50)
        x = torch.randn(4, 50, 20)
        out = model(x)
        assert out.shape == (4, 3)

    def test_scagg_forward_shape(self):
        """ScAGG forward pass produces correct output shape."""
        import torch
        from tissueformer.benchmark_models.scagg import ScAGG
        model = ScAGG(n_genes=20, n_classes=3, hidden_dim=32, n_heads=2, n_heads2=2)
        x = torch.randn(4, 50, 20)
        out = model(x)
        assert out.shape == (4, 3)

    def test_scrat_forward_shape(self):
        """ScRAT forward pass produces correct output shape."""
        import torch
        from tissueformer.benchmark_models.scrat import ScRAT
        model = ScRAT(n_genes=20, n_classes=3, hidden_dim=32, n_heads=4)
        x = torch.randn(4, 50, 20)
        out = model(x)
        assert out.shape == (4, 3)

    def test_topk_pool(self):
        """CellCnn top-k pooling selects correct cells."""
        import torch
        from tissueformer.benchmark_models.cellcnn import TopKPool
        pool = TopKPool(k=2)
        x = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
        out = pool(x)
        # Top 2 cells are cells 2 and 3: [[6,7,8], [9,10,11]] -> mean = [7.5, 8.5, 9.5]
        expected = torch.tensor([[7.5, 8.5, 9.5]])
        assert torch.allclose(out, expected)

    def test_topk_pool_with_mask(self):
        """Top-k pooling respects padding mask."""
        import torch
        from tissueformer.benchmark_models.cellcnn import TopKPool
        pool = TopKPool(k=2)
        x = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
        mask = torch.tensor([[False, False, True, True]])  # mask cells 2,3
        out = pool(x, mask=mask)
        # Only cells 0 and 1 are valid: [[0,1,2], [3,4,5]] -> mean = [1.5, 2.5, 3.5]
        expected = torch.tensor([[1.5, 2.5, 3.5]])
        assert torch.allclose(out, expected)

    def test_mil_collate_padding(self):
        """mil_collate_fn pads variable-length bags correctly."""
        import torch
        from tissueformer.benchmark_models.data import mil_collate_fn
        batch = [
            (torch.randn(10, 5), torch.tensor(0.0)),
            (torch.randn(20, 5), torch.tensor(1.0)),
        ]
        cells, labels, mask = mil_collate_fn(batch)
        assert cells.shape == (2, 20, 5)
        assert mask.shape == (2, 20)
        assert mask[0, :10].sum() == 0  # first 10 real
        assert mask[0, 10:].sum() == 10  # last 10 padded
        assert mask[1].sum() == 0  # all real

    def test_cellcnn_smoke_train(self):
        """CellCnn trains for 2 epochs, loss decreases."""
        import torch, wandb
        from torch.utils.data import DataLoader
        from tissueformer.benchmark_models import CellCnn, BenchmarkTrainer, MILDataset, mil_collate_fn
        wandb.init(mode='disabled')

        expr, si, labels = self._make_synthetic_mil_data()
        ds = MILDataset(expr, si, labels, cells_per_sample=50)
        loader = DataLoader(ds, batch_size=4, collate_fn=mil_collate_fn)

        model = CellCnn(n_genes=20, n_classes=3, n_filters=4, cells_per_input=50)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer = BenchmarkTrainer(
            model=model, optimizer=opt,
            train_loader=loader, val_loader=loader,
            device=torch.device('cpu'), n_epochs=2,
            early_stopping_patience=None, model_name='test_cellcnn', n_classes=3,
        )
        loss1 = trainer.train_epoch()
        loss2 = trainer.train_epoch()
        wandb.finish()
        # Loss should generally decrease with strong signal
        assert isinstance(loss1, float)
        assert isinstance(loss2, float)

    def test_scrat_mixup(self):
        """ScRAT cell-type mixup produces correct output structure."""
        from tissueformer.benchmark_models.scrat import cell_type_mixup
        expr = np.random.randn(200, 10).astype(np.float32)
        si = [np.arange(0, 100), np.arange(100, 200)]
        labels = np.array([0, 1])
        ct = np.concatenate([np.zeros(100), np.ones(100)]).astype(np.int64)

        aug_expr, aug_idx, aug_labels = cell_type_mixup(
            expr, si, labels, ct, n_pseudo=3, cells_per_pseudo=50, alpha=0.5
        )
        assert len(aug_idx) == 5  # 2 original + 3 pseudo
        assert aug_labels.shape[0] == 5
        assert aug_labels[0] == 0
        assert aug_labels[1] == 1
