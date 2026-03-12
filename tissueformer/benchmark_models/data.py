"""
MIL (Multiple Instance Learning) datasets for benchmark models.

Each sample is a "bag" of cells belonging to one donor/patient/spatial group.
The models aggregate cell-level features internally to make a single prediction
per bag.

Expression matrices (X) are kept sparse throughout loading and preprocessing
to avoid materializing huge dense arrays.  Densification happens only at
__getitem__ time on small per-crop slices.
"""

import numpy as np
import scipy.sparse
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Preprocessing functions (sparse-aware)
# ---------------------------------------------------------------------------

def fit_zscore_params(X, cell_indices: np.ndarray | None = None):
    """Compute mean and scale from (optionally a subset of) X.

    Works on both sparse and dense X.  Returns (mean, scale) as 1-D float32
    arrays suitable for lazy per-crop normalization.
    """
    if cell_indices is not None:
        subset = X[cell_indices]
    else:
        subset = X
    if scipy.sparse.issparse(subset):
        # Use sparse-friendly stats
        mean = np.asarray(subset.mean(axis=0)).ravel().astype(np.float32)
        # var = E[X^2] - E[X]^2
        mean_sq = np.asarray(subset.multiply(subset).mean(axis=0)).ravel()
        var = (mean_sq - mean ** 2).clip(min=0)
        scale = np.sqrt(var).astype(np.float32)
    else:
        mean = np.mean(subset, axis=0).astype(np.float32)
        scale = np.std(subset, axis=0).astype(np.float32)
    scale[scale == 0] = 1.0
    return mean, scale


def preprocess_zscore(X: np.ndarray, scaler: StandardScaler | None = None):
    """Z-score normalization (CellCnn default).  Dense path only."""
    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    return X, scaler


def preprocess_cp10k_log1p(X):
    """CP10k + log1p normalization.  Sparse-aware."""
    if scipy.sparse.issparse(X):
        X = X.astype(np.float32, copy=True)
        row_sums = np.asarray(X.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        # Efficient in-place row scaling for CSR
        if not scipy.sparse.isspmatrix_csr(X):
            X = X.tocsr()
        diag = scipy.sparse.diags(1e4 / row_sums)
        X = diag @ X
        X = X.log1p()
        return X
    # Dense fallback
    total = X.sum(axis=1, keepdims=True)
    total = np.where(total == 0, 1, total)
    X = X / total * 1e4
    X = np.log1p(X)
    return X


def select_hvgs(X, n_hvgs: int = 1000) -> tuple:
    """Select top n_hvgs highly variable genes by variance.

    Works on both sparse and dense X.  Returns (X_subset, gene_indices).
    """
    if X.shape[1] <= n_hvgs:
        return X, np.arange(X.shape[1])
    if scipy.sparse.issparse(X):
        mean = np.asarray(X.mean(axis=0)).ravel()
        mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
        var = (mean_sq - mean ** 2).clip(min=0)
    else:
        var = np.var(X, axis=0)
    top_idx = np.argsort(var)[-n_hvgs:]
    top_idx = np.sort(top_idx)
    return X[:, top_idx], top_idx


# ---------------------------------------------------------------------------
# MIL Dataset
# ---------------------------------------------------------------------------

def _densify(X_row):
    """Convert a slice to a dense float32 numpy array."""
    if scipy.sparse.issparse(X_row):
        return np.asarray(X_row.toarray(), dtype=np.float32)
    return np.asarray(X_row, dtype=np.float32)


class MILDataset(Dataset):
    """Multiple-instance learning dataset for donor/sample-level classification.

    Each item is a bag of cells for one sample, with a single label.
    Expression may be sparse; densification happens per-crop.

    Args:
        expression: (n_total_cells, n_genes) expression matrix (sparse or dense).
        sample_indices: List of arrays, each containing cell indices for one sample.
        labels: (n_samples,) integer or float labels.
        cells_per_sample: If set, randomly crop this many cells per sample.
            If None, use all cells (bags will be of variable size).
        cell_types: (n_total_cells,) cell type IDs (optional, for ScRAT mixup).
        zscore_mean: Optional (n_genes,) mean for lazy z-score normalization.
        zscore_scale: Optional (n_genes,) scale for lazy z-score normalization.
    """

    def __init__(
        self,
        expression,
        sample_indices: list[np.ndarray],
        labels: np.ndarray,
        cells_per_sample: int | None = None,
        cell_types: np.ndarray | None = None,
        zscore_mean: np.ndarray | None = None,
        zscore_scale: np.ndarray | None = None,
    ):
        self.expression = expression
        self.sample_indices = sample_indices
        self.labels = np.asarray(labels)
        self.cells_per_sample = cells_per_sample
        self.cell_types = cell_types
        self.zscore_mean = zscore_mean
        self.zscore_scale = zscore_scale

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        cell_idx = self.sample_indices[idx]
        label = self.labels[idx]

        if self.cells_per_sample is not None and len(cell_idx) > self.cells_per_sample:
            chosen = np.random.choice(len(cell_idx), self.cells_per_sample, replace=False)
            cell_idx = cell_idx[chosen]
        elif self.cells_per_sample is not None and len(cell_idx) < self.cells_per_sample:
            chosen = np.random.choice(len(cell_idx), self.cells_per_sample, replace=True)
            cell_idx = cell_idx[chosen]

        cells = _densify(self.expression[cell_idx])
        if self.zscore_mean is not None:
            cells = (cells - self.zscore_mean) / self.zscore_scale
        return torch.from_numpy(cells).float(), torch.tensor(label).float()


class CroppedMILDataset(Dataset):
    """Dataset that draws multiple fixed-size crops per sample (for ScRAT).

    Each sample is expanded into `crops_per_sample` items, each a random
    crop of `cells_per_crop` cells from that sample.
    Expression may be sparse; densification happens per-crop.

    Args:
        expression: (n_total_cells, n_genes) expression matrix (sparse or dense).
        sample_indices: List of arrays, each containing cell indices for one sample.
        labels: (n_samples,) labels.
        cells_per_crop: Number of cells per crop.
        crops_per_sample: Number of crops to generate per sample.
        zscore_mean: Optional (n_genes,) mean for lazy z-score normalization.
        zscore_scale: Optional (n_genes,) scale for lazy z-score normalization.
    """

    def __init__(
        self,
        expression,
        sample_indices: list[np.ndarray],
        labels: np.ndarray,
        cells_per_crop: int = 500,
        crops_per_sample: int = 20,
        zscore_mean: np.ndarray | None = None,
        zscore_scale: np.ndarray | None = None,
    ):
        self.expression = expression
        self.sample_indices = sample_indices
        self.labels = np.asarray(labels)
        self.cells_per_crop = cells_per_crop
        self.crops_per_sample = crops_per_sample
        self.zscore_mean = zscore_mean
        self.zscore_scale = zscore_scale

    def __len__(self):
        return len(self.sample_indices) * self.crops_per_sample

    def __getitem__(self, idx):
        sample_idx = idx // self.crops_per_sample
        cell_idx = self.sample_indices[sample_idx]
        label = self.labels[sample_idx]

        if len(cell_idx) >= self.cells_per_crop:
            chosen = np.random.choice(len(cell_idx), self.cells_per_crop, replace=False)
        else:
            chosen = np.random.choice(len(cell_idx), self.cells_per_crop, replace=True)
        cell_idx = cell_idx[chosen]

        cells = _densify(self.expression[cell_idx])
        if self.zscore_mean is not None:
            cells = (cells - self.zscore_mean) / self.zscore_scale
        return torch.from_numpy(cells).float(), torch.tensor(label).float()


def mil_collate_fn(batch):
    """Collate variable-length bags by padding to max size in the batch.

    Returns:
        cells: (batch_size, max_cells, n_genes) padded tensor.
        labels: (batch_size,) label tensor.
        mask: (batch_size, max_cells) boolean mask (True = padded).
    """
    cells_list, labels_list = zip(*batch)

    max_cells = max(c.size(0) for c in cells_list)
    n_genes = cells_list[0].size(1)
    batch_size = len(cells_list)

    cells = torch.zeros(batch_size, max_cells, n_genes)
    mask = torch.ones(batch_size, max_cells, dtype=torch.bool)

    for i, c in enumerate(cells_list):
        n = c.size(0)
        cells[i, :n] = c
        mask[i, :n] = False

    labels = torch.stack(labels_list)
    return cells, labels, mask


def load_covid_mil_data(
    h5ad_path: str,
    donor_ids: list[str],
    label_map: dict[str, int],
    donor_key: str = "donor_id",
    label_key: str = "label",
    cell_type_key: str | None = "cell_type",
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray | None, list[str]]:
    """Load COVID h5ad and organize by donor for MIL.

    Returns:
        expression: (n_cells, n_genes) dense array.
        sample_indices: List of index arrays, one per donor.
        labels: (n_donors,) integer labels.
        cell_types: (n_cells,) integer cell type IDs, or None.
        donor_order: List of donor IDs in output order.
    """
    import anndata as ad

    adata = ad.read_h5ad(h5ad_path)
    mask = adata.obs[donor_key].isin(donor_ids)
    adata = adata[mask]

    # Extract expression — keep sparse to avoid OOM on large datasets
    X = adata.X
    if scipy.sparse.issparse(X):
        X = X.astype(np.float32, copy=False)
        if not scipy.sparse.isspmatrix_csr(X):
            X = X.tocsr()
    else:
        X = np.asarray(X, dtype=np.float32)

    # Build per-donor indices
    sample_indices = []
    labels = []
    donor_order = []
    for donor in sorted(set(donor_ids) & set(adata.obs[donor_key].unique())):
        donor_mask = (adata.obs[donor_key] == donor).values
        idx = np.where(donor_mask)[0]
        sample_indices.append(idx)
        donor_label = label_map[adata.obs[label_key].iloc[idx[0]]]
        labels.append(donor_label)
        donor_order.append(donor)

    labels = np.array(labels, dtype=np.int64)

    # Cell types
    cell_types = None
    if cell_type_key and cell_type_key in adata.obs.columns:
        ct_unique = sorted(adata.obs[cell_type_key].unique())
        ct_map = {ct: i for i, ct in enumerate(ct_unique)}
        cell_types = np.array(
            [ct_map[ct] for ct in adata.obs[cell_type_key]], dtype=np.int64
        )

    return X, sample_indices, labels, cell_types, donor_order
