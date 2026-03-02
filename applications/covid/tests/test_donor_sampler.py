"""Tests for DonorGroupSampler."""

import pytest
import numpy as np
from collections import Counter
from datasets import Dataset

from tissueformer.samplers import DonorGroupSampler


def make_synthetic_dataset(n_donors=5, cells_per_donor=100, seed=42):
    """Create a synthetic dataset with known donor structure."""
    rng = np.random.RandomState(seed)
    donor_ids = []
    labels = []
    input_ids = []

    for i in range(n_donors):
        donor_id = f"donor_{i}"
        label = i % 3  # cycle through 0, 1, 2
        for _ in range(cells_per_donor):
            donor_ids.append(donor_id)
            labels.append(label)
            input_ids.append(rng.randint(1, 100, size=10).tolist())

    return Dataset.from_dict({
        "donor_id": donor_ids,
        "label": labels,
        "input_ids": input_ids,
    })


@pytest.fixture
def dataset():
    return make_synthetic_dataset()


@pytest.fixture
def small_donor_dataset():
    """Dataset where one donor has very few cells."""
    donor_ids = ["big_donor"] * 100 + ["small_donor"] * 3
    labels = [0] * 100 + [1] * 3
    rng = np.random.RandomState(0)
    input_ids = [rng.randint(1, 100, size=10).tolist() for _ in range(103)]
    return Dataset.from_dict({
        "donor_id": donor_ids,
        "label": labels,
        "input_ids": input_ids,
    })


class TestDonorGroupSampler:

    def test_donor_purity(self, dataset):
        """Every group should contain cells from exactly one donor."""
        sampler = DonorGroupSampler(
            dataset, batch_size=64, group_size=8, seed=0
        )
        donor_ids = np.array(dataset["donor_id"])
        indices = list(sampler)

        # Group indices by group_size
        for start in range(0, len(indices), 8):
            group = indices[start:start + 8]
            if len(group) < 8:
                break
            group_donors = set(donor_ids[group])
            assert len(group_donors) == 1, \
                f"Group has cells from multiple donors: {group_donors}"

    @pytest.mark.parametrize("group_size", [4, 8, 16])
    def test_group_size_correctness(self, dataset, group_size):
        """Every group should have exactly group_size cells."""
        sampler = DonorGroupSampler(
            dataset, batch_size=64, group_size=group_size, seed=0
        )
        indices = list(sampler)
        assert len(indices) % group_size == 0

    def test_small_donor_replacement(self, small_donor_dataset):
        """Donor with fewer cells than group_size should sample with replacement."""
        sampler = DonorGroupSampler(
            small_donor_dataset, batch_size=16, group_size=8, seed=42,
            iterate_all_donors=True
        )
        indices = list(sampler)
        # big_donor: ceil(100/8)*8 = 104, small_donor: ceil(3/8)*8 = 8
        assert len(indices) == 104 + 8

    def test_eval_mode_uses_all_cells(self, dataset):
        """iterate_all_donors=True should create multiple groups per donor using all cells."""
        group_size = 8
        cells_per_donor = 100  # from make_synthetic_dataset
        n_donors = 5
        import math
        groups_per_donor = math.ceil(cells_per_donor / group_size)  # 13

        sampler = DonorGroupSampler(
            dataset, batch_size=64, group_size=group_size, seed=42,
            iterate_all_donors=True
        )
        donor_ids = np.array(dataset["donor_id"])
        indices = list(sampler)

        # Each donor should have ceil(100/8) = 13 groups, 13*8 = 104 indices
        expected_total = n_donors * groups_per_donor * group_size
        assert len(indices) == expected_total

        # Check donor group counts
        counts = sampler.get_eval_donor_group_counts()
        assert len(counts) == n_donors
        for donor, n_groups in counts.items():
            assert n_groups == groups_per_donor

        # Check all groups are donor-pure
        for start in range(0, len(indices), group_size):
            group = indices[start:start + group_size]
            group_donors = set(donor_ids[group])
            assert len(group_donors) == 1

    def test_eval_mode_coverage(self, dataset):
        """iterate_all_donors=True should visit every donor."""
        sampler = DonorGroupSampler(
            dataset, batch_size=64, group_size=8, seed=42,
            iterate_all_donors=True
        )
        donor_ids = np.array(dataset["donor_id"])
        indices = list(sampler)

        seen_donors = set()
        for start in range(0, len(indices), 8):
            group = indices[start:start + 8]
            seen_donors.add(donor_ids[group[0]])

        assert len(seen_donors) == 5

    def test_train_mode_diversity(self, dataset):
        """Over many iterations, all donors should be sampled."""
        sampler = DonorGroupSampler(
            dataset, batch_size=64, group_size=8, seed=0,
            num_samples=1000 * 8  # many groups
        )
        donor_ids = np.array(dataset["donor_id"])
        indices = list(sampler)

        seen_donors = set()
        for start in range(0, len(indices), 8):
            group = indices[start:start + 8]
            seen_donors.add(donor_ids[group[0]])

        assert len(seen_donors) == 5, f"Only saw {len(seen_donors)} donors"

    def test_seed_reproducibility(self, dataset):
        """Same seed should produce same iteration order."""
        sampler1 = DonorGroupSampler(
            dataset, batch_size=64, group_size=8, seed=123
        )
        sampler2 = DonorGroupSampler(
            dataset, batch_size=64, group_size=8, seed=123
        )
        assert list(sampler1) == list(sampler2)

    def test_group_size_one(self, dataset):
        """group_size=1 should yield individual cell indices."""
        sampler = DonorGroupSampler(
            dataset, batch_size=64, group_size=1, seed=0
        )
        indices = list(sampler)
        # Should be a permutation of dataset indices
        assert len(indices) == len(dataset)
        assert set(indices) == set(range(len(dataset)))

    def test_set_epoch(self, dataset):
        """Different epochs should produce different orderings."""
        sampler = DonorGroupSampler(
            dataset, batch_size=64, group_size=8, seed=0
        )
        indices_e0 = list(sampler)
        sampler.set_epoch(1)
        indices_e1 = list(sampler)
        # Very unlikely to be the same
        assert indices_e0 != indices_e1
