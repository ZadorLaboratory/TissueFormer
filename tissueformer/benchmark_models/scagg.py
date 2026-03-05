"""
scAGG: PyTorch re-implementation (base MLP + mean pooling variant).

Reference: Verlaan et al., "scAGG: single-cell gene expression analysis
using graph-based aggregation", CSBJ 2025.
Original code: https://github.com/timoverlaan/scAGG

The paper showed the simple MLP + mean pooling variant performs as well as
or better than the GAT/attention variants, so we implement only the base model.

Architecture: 2-layer MLP cell encoder -> mean pooling -> linear classifier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScAGG(nn.Module):
    """scAGG base model (NoGraph variant with mean pooling).

    Args:
        n_genes: Number of input features per cell.
        n_classes: Number of output classes.
        hidden_dim: Hidden dimension (paper default: 128).
        n_heads: Number of heads for first layer expansion (paper: 8).
        n_heads2: Number of heads for second layer (paper: 4).
        dropout: Dropout rate (paper default: 0.1 for base model).
    """

    def __init__(
        self,
        n_genes: int,
        n_classes: int,
        hidden_dim: int = 128,
        n_heads: int = 8,
        n_heads2: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Two-layer MLP cell encoder
        self.layer1 = nn.Linear(n_genes, hidden_dim * n_heads)
        self.layer2 = nn.Linear(hidden_dim * n_heads, hidden_dim * n_heads2)
        self.dropout = dropout

        # Classifier
        self.classifier = nn.Linear(hidden_dim * n_heads2, n_classes)

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
            (batch, n_classes) -- logits.
        """
        # Cell encoder
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = F.elu(self.layer1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.elu(self.layer2(h))

        # Mask-aware mean pooling
        if mask is not None:
            h = h.masked_fill(mask.unsqueeze(-1), 0.0)
            n_valid = (~mask).sum(dim=1, keepdim=True).float().clamp(min=1)
            h = h.sum(dim=1) / n_valid  # (batch, hidden)
        else:
            h = h.mean(dim=1)  # (batch, hidden)

        logits = self.classifier(h)
        return logits
