"""Tests for SpatialGroupCollator with donor (non-spatial) data."""

import pytest
import torch
import numpy as np
from tissueformer.samplers import SpatialGroupCollator


@pytest.fixture
def collator():
    return SpatialGroupCollator(
        group_size=4,
        label_key="labels",
        feature_keys=["input_ids"],
        relative_positions=False,
    )


def make_features(n_cells, group_size, seq_len=10, label=1):
    """Create a list of feature dicts simulating collator input."""
    features = []
    for _ in range(n_cells):
        features.append({
            "input_ids": list(range(1, seq_len + 1)),
            "labels": label,
            "attention_mask": [1] * seq_len,
        })
    return features


class TestSpatialGroupCollator:

    def test_output_shape(self, collator):
        """Output tensors should have shape (num_groups, group_size, seq_len)."""
        features = make_features(8, group_size=4, seq_len=10)
        batch = collator(features)

        assert batch["input_ids"].shape == (2, 4, 10)  # 8 cells / 4 per group = 2 groups
        assert batch["attention_mask"].shape == (2, 4, 10)

    def test_majority_label(self, collator):
        """When all cells in group share label, majority label should match."""
        features = make_features(4, group_size=4, label=2)
        batch = collator(features)

        assert batch["labels"].item() == 2

    @pytest.mark.parametrize("group_size", [2, 4, 8])
    def test_various_group_sizes(self, group_size):
        col = SpatialGroupCollator(
            group_size=group_size,
            label_key="labels",
            relative_positions=False,
        )
        features = make_features(group_size * 3, group_size=group_size, seq_len=8)
        batch = col(features)

        assert batch["input_ids"].shape[0] == 3
        assert batch["input_ids"].shape[1] == group_size

    def test_no_relative_positions(self, collator):
        """relative_positions=False should not produce position tensors."""
        features = make_features(4, group_size=4)
        batch = collator(features)

        assert "relative_positions" not in batch

    def test_single_cell_labels(self, collator):
        """Single-cell labels should be preserved."""
        features = make_features(4, group_size=4, label=1)
        batch = collator(features)

        assert "single_cell_labels" in batch
        assert batch["single_cell_labels"].shape == (1, 4)
        assert (batch["single_cell_labels"] == 1).all()

    def test_variable_seq_lengths(self):
        """Collator should handle variable sequence lengths via padding."""
        col = SpatialGroupCollator(
            group_size=2, label_key="labels", relative_positions=False
        )
        features = [
            {"input_ids": [1, 2, 3], "labels": 0, "attention_mask": [1, 1, 1]},
            {"input_ids": [4, 5, 6, 7, 8], "labels": 0, "attention_mask": [1, 1, 1, 1, 1]},
        ]
        batch = col(features)
        # Should pad shorter sequence to length 5
        assert batch["input_ids"].shape == (1, 2, 5)
