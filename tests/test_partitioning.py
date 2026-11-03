"""Unit tests for the splitting algorithms (data_splitting/partitioning.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_splitting"))
from partitioning import (  # noqa: E402
    apply_label_noise,
    apply_sensor_skew,
    dirichlet_split,
    group_split,
    iid_split,
    pathological_split,
    quantity_split,
)

LABELS = np.random.default_rng(0).integers(0, 5, size=1000)


def assert_partition(parts, n):
    """Every sample in exactly one client."""
    joined = np.concatenate(parts)
    assert len(joined) == n
    assert len(np.unique(joined)) == n


@pytest.mark.parametrize("split", [
    lambda: iid_split(LABELS, 4),
    lambda: dirichlet_split(LABELS, 4, alpha=0.3, min_samples=10),
    lambda: pathological_split(LABELS, 4, classes_per_client=2),
    lambda: quantity_split(LABELS, 4, alpha=0.5, min_samples=10),
])
def test_splits_cover_every_sample_once(split):
    parts = split()
    assert len(parts) == 4
    assert_partition(parts, len(LABELS))


def test_same_seed_same_split():
    a = dirichlet_split(LABELS, 3, alpha=0.3, seed=7)
    b = dirichlet_split(LABELS, 3, alpha=0.3, seed=7)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_pathological_limits_classes_per_client():
    for part in pathological_split(LABELS, 5, classes_per_client=2):
        assert len(np.unique(LABELS[part])) <= 2


def test_group_split_keeps_groups_together():
    groups = np.repeat(np.arange(10), 100)
    parts = group_split(groups, 3)
    assert_partition(parts, len(groups))
    owners = [set(groups[p]) for p in parts]
    for i in range(3):
        for j in range(i + 1, 3):
            assert not owners[i] & owners[j]


def test_label_noise_rate():
    noisy = apply_label_noise(LABELS, 0.2, num_classes=5, seed=0)
    assert 0.15 < np.mean(noisy != LABELS) < 0.25
    assert np.array_equal(apply_label_noise(LABELS, 0.0, 5), LABELS)


def test_sensor_skew_leaves_client_0_unchanged():
    values = np.random.default_rng(0).normal(size=(50, 6))
    assert np.allclose(apply_sensor_skew(values, 0, 3), values)
    assert not np.allclose(apply_sensor_skew(values, 2, 3), values)
