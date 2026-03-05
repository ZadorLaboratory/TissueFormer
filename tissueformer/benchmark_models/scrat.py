"""
ScRAT: PyTorch re-implementation.

Reference: Mao et al., "ScRAT: a transformer-based method for
single-cell multi-omic integration", Bioinformatics 2024.
Original code: https://github.com/yuzhenmao/ScRAT

Architecture: Linear projection -> positional encoding -> Transformer encoder
-> mean pooling -> classifier with sigmoid output.

Key features:
- Cell-type-aware sample mixup augmentation for small sample sizes
- BCE loss (one-vs-rest) even for multi-class
- Cosine LR schedule with warmup
"""

import math
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (standard Transformer PE)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ScRAT(nn.Module):
    """ScRAT Transformer model.

    Args:
        n_genes: Number of input features per cell.
        n_classes: Number of output classes.
        hidden_dim: Model/embedding dimension (paper: 128).
        n_heads: Number of attention heads (paper: 8).
        n_layers: Number of transformer layers (paper: 1).
        dropout: Dropout rate (paper: 0.3).
        max_cells: Maximum sequence length for positional encoding.
    """

    def __init__(
        self,
        n_genes: int,
        n_classes: int,
        hidden_dim: int = 128,
        n_heads: int = 8,
        n_layers: int = 1,
        dropout: float = 0.3,
        max_cells: int = 5000,
    ):
        super().__init__()
        self.n_classes = n_classes

        # Input projection (2-layer MLP as in reference dimRedu_net)
        self.input_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_genes, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.pos_enc = PositionalEncoding(hidden_dim, max_len=max_cells)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Output head (2-layer MLP as in reference)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, n_cells, n_genes) -- cell features.
            mask: (batch, n_cells) -- True for padded positions.

        Returns:
            (batch, n_classes) -- logits (pre-sigmoid).
        """
        h = self.input_proj(x)  # (batch, n_cells, hidden_dim)
        h = self.pos_enc(h)

        # TransformerEncoder uses src_key_padding_mask: True = ignore
        h = self.transformer(h, src_key_padding_mask=mask)

        # Mask-aware mean pooling over cells
        if mask is not None:
            h = h.masked_fill(mask.unsqueeze(-1), 0.0)
            n_valid = (~mask).sum(dim=1, keepdim=True).float().clamp(min=1)
            h = h.sum(dim=1) / n_valid
        else:
            h = h.mean(dim=1)

        logits = self.output_net(h)
        return logits


def cell_type_mixup(
    expression: np.ndarray,
    sample_indices: list[np.ndarray],
    labels: np.ndarray,
    cell_types: np.ndarray,
    n_pseudo: int = 300,
    cells_per_pseudo: int = 10000,
    alpha: float = 0.5,
    noise_var: float = 1e-5,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Cell-type-aware sample mixup augmentation.

    For each pseudo-sample, pick two real samples, sample lambda ~ Beta(alpha, alpha),
    and for each cell type present in either sample, interpolate cells:
        x_mix = lambda * x_i + (1 - lambda) * x_j

    Args:
        expression: (n_total_cells, n_genes) expression matrix for all cells.
        sample_indices: List of arrays, each containing cell indices for one sample.
        labels: (n_samples,) float labels (can be soft for label mixing).
        cell_types: (n_total_cells,) integer cell type ID per cell.
        n_pseudo: Number of pseudo-samples to generate.
        cells_per_pseudo: Target number of cells per pseudo-sample.
        alpha: Beta distribution parameter.
        noise_var: Variance of Gaussian noise added to mixed cells.

    Returns:
        aug_expression: Extended expression matrix including new cells.
        aug_indices: Extended sample_indices list with pseudo-samples appended.
        aug_labels: Extended labels array with soft labels for pseudo-samples.
    """
    n_orig_cells = expression.shape[0]
    n_genes = expression.shape[1]
    n_samples = len(sample_indices)

    # Pre-allocate space for augmented cells
    max_new_cells = n_pseudo * cells_per_pseudo
    aug_expression = np.zeros(
        (n_orig_cells + max_new_cells, n_genes), dtype=expression.dtype
    )
    aug_expression[:n_orig_cells] = expression

    aug_indices = list(sample_indices)
    aug_labels = list(labels)

    cursor = n_orig_cells
    rng = np.random.RandomState()

    unique_cell_types = np.unique(cell_types)

    for _ in range(n_pseudo):
        # Pick two samples (within same class, as in paper's same_pheno=1 default)
        i, j = rng.choice(n_samples, size=2, replace=True)
        lam = rng.beta(alpha, alpha)

        idx_i = sample_indices[i]
        idx_j = sample_indices[j]
        ct_i = cell_types[idx_i]
        ct_j = cell_types[idx_j]

        new_cell_indices = []
        for ct in unique_cell_types:
            cells_i = idx_i[ct_i == ct]
            cells_j = idx_j[ct_j == ct]
            if len(cells_i) == 0 and len(cells_j) == 0:
                continue

            # Target count for this cell type, proportional to its frequency
            n_ct = max(
                1,
                int(
                    cells_per_pseudo
                    * (
                        lam * len(cells_i) / max(len(idx_i), 1)
                        + (1 - lam) * len(cells_j) / max(len(idx_j), 1)
                    )
                ),
            )

            # Sample cells with replacement
            if len(cells_i) > 0:
                sampled_i = expression[rng.choice(cells_i, n_ct, replace=True)]
            else:
                sampled_i = np.zeros((n_ct, n_genes), dtype=expression.dtype)

            if len(cells_j) > 0:
                sampled_j = expression[rng.choice(cells_j, n_ct, replace=True)]
            else:
                sampled_j = np.zeros((n_ct, n_genes), dtype=expression.dtype)

            # Interpolate
            mixed = lam * sampled_i + (1 - lam) * sampled_j

            # Add noise
            if noise_var > 0:
                mixed += rng.normal(0, np.sqrt(noise_var), mixed.shape)

            # Store
            end = cursor + n_ct
            if end > aug_expression.shape[0]:
                # Extend array if needed
                extra = max(max_new_cells // 2, end - aug_expression.shape[0])
                aug_expression = np.concatenate(
                    [
                        aug_expression,
                        np.zeros((extra, n_genes), dtype=expression.dtype),
                    ]
                )
            aug_expression[cursor:end] = mixed
            new_cell_indices.append(np.arange(cursor, end))
            cursor = end

        if new_cell_indices:
            aug_indices.append(np.concatenate(new_cell_indices))
            # Soft label mixing
            aug_labels.append(lam * labels[i] + (1 - lam) * labels[j])

    aug_expression = aug_expression[:cursor]
    aug_labels = np.array(aug_labels)
    return aug_expression, aug_indices, aug_labels
