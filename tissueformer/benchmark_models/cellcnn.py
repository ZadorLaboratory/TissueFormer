"""
CellCnn: PyTorch re-implementation.

Reference: Arvaniti & Claassen, "Sensitive detection of rare disease-associated
cell subsets via representation learning", Nature Communications 2017.
Original code: https://github.com/eiriniar/CellCnn (Keras/TensorFlow)

Architecture: Conv1D(kernel_size=1) -> top-k mean pooling -> dense classifier.
The 1x1 convolution acts as a per-cell linear transform. Top-k pooling selects
the k cells with highest activation per filter and averages them, enabling
detection of rare cell subsets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKPool(nn.Module):
    """Top-k mean pooling: for each filter, select the top k cells by
    activation value, then average them.

    This is the core contribution of CellCnn -- it allows the model to focus
    on rare cell subsets rather than averaging over all cells.

    Args:
        k: Number of top cells to select per filter. If k >= n_cells,
           this degenerates to standard mean pooling.
    """

    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (batch, n_cells, n_filters) -- conv output per cell.
            mask: (batch, n_cells) -- True for padded positions.

        Returns:
            (batch, n_filters) -- pooled representation.
        """
        if mask is not None:
            # Set padded positions to -inf so they sort to the bottom
            x = x.masked_fill(mask.unsqueeze(-1), float("-inf"))

        # Sort descending along cell dimension, take top k, average
        sorted_x, _ = torch.sort(x, dim=1, descending=True)
        k = min(self.k, x.size(1))
        top_k = sorted_x[:, :k, :]

        # If mask was applied, some top-k entries could still be -inf
        # (when a sample has fewer than k real cells). Replace with 0.
        top_k = top_k.masked_fill(top_k == float("-inf"), 0.0)

        # Count real (non-masked) cells in top-k for correct averaging
        if mask is not None:
            sorted_mask, _ = torch.sort(mask.float(), dim=1)  # 0s first
            top_k_mask = sorted_mask[:, :k].unsqueeze(-1)  # (batch, k, 1)
            n_valid = k - top_k_mask.sum(dim=1)  # (batch, 1)
            n_valid = n_valid.clamp(min=1)
            return top_k.sum(dim=1) / n_valid
        else:
            return top_k.mean(dim=1)


class CellCnn(nn.Module):
    """CellCnn model.

    Args:
        n_genes: Number of input features (genes/markers) per cell.
        n_classes: Number of output classes.
        n_filters: Number of conv filters (paper samples from 3-9).
        maxpool_percentage: Percentage of cells to keep in top-k pooling.
        dropout: Dropout rate (paper uses 0.5 when n_filters > 5).
    """

    def __init__(
        self,
        n_genes: int,
        n_classes: int,
        n_filters: int = 6,
        maxpool_percentage: float = 5.0,
        dropout: float = 0.5,
        cells_per_input: int = 200,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.n_classes = n_classes

        # Conv1D with kernel_size=1: equivalent to a shared linear layer per cell
        self.conv = nn.Conv1d(
            in_channels=n_genes,
            out_channels=n_filters,
            kernel_size=1,
        )

        # Top-k pooling
        k = max(1, int(maxpool_percentage / 100.0 * cells_per_input))
        self.pool = TopKPool(k)

        # Dropout (paper: only if n_filters > 5)
        self.dropout = nn.Dropout(dropout) if n_filters > 5 else nn.Identity()

        # Output classifier
        self.classifier = nn.Linear(n_filters, n_classes)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, n_cells, n_genes) -- raw cell features.
            mask: (batch, n_cells) -- True for padded positions.

        Returns:
            (batch, n_classes) -- logits.
        """
        # Conv1d expects (batch, channels, length) = (batch, n_genes, n_cells)
        h = self.conv(x.transpose(1, 2))  # -> (batch, n_filters, n_cells)
        h = F.relu(h)
        h = h.transpose(1, 2)  # -> (batch, n_cells, n_filters)

        # Top-k pooling -> (batch, n_filters)
        h = self.pool(h, mask=mask)

        h = self.dropout(h)
        logits = self.classifier(h)  # -> (batch, n_classes)
        return logits

    def l1_loss(self, l1_coeff: float = 1e-4) -> torch.Tensor:
        """L1 regularization on conv and classifier weights (as in original)."""
        l1 = torch.tensor(0.0, device=self.conv.weight.device)
        l1 += self.conv.weight.abs().sum()
        l1 += self.classifier.weight.abs().sum()
        return l1_coeff * l1
