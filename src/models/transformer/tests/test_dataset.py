"""Tests for SPYWindowDataset."""

import pytest
import numpy as np
import torch
from src.data.dataset import SPYWindowDataset

N_ROWS = 100
W = 20
N_FEAT = 10  # d_feat + d_sent


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_dataset(n: int = N_ROWS, window: int = W, seed: int = 0) -> SPYWindowDataset:
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((n, N_FEAT)).astype(np.float32)
    regime_labels = rng.integers(0, 4, n).astype(np.int64)
    dir_labels = rng.integers(0, 2, n).astype(np.int64)
    return SPYWindowDataset(features, regime_labels, dir_labels, window_size=window)


# ── Length ────────────────────────────────────────────────────────────────────

def test_dataset_length():
    ds = _make_dataset(n=100, window=20)
    assert len(ds) == 100 - 20


def test_dataset_length_different_window():
    ds = _make_dataset(n=120, window=40)
    assert len(ds) == 120 - 40


# ── Item shapes ───────────────────────────────────────────────────────────────

def test_dataset_x_shape():
    ds = _make_dataset()
    x, y_dir, y_reg = ds[0]
    assert x.shape == (W, N_FEAT)


def test_dataset_x_dtype():
    ds = _make_dataset()
    x, _, _ = ds[0]
    assert x.dtype == torch.float32


def test_dataset_y_dir_is_scalar():
    ds = _make_dataset()
    _, y_dir, _ = ds[0]
    assert isinstance(y_dir, int)


def test_dataset_y_reg_is_scalar():
    ds = _make_dataset()
    _, _, y_reg = ds[0]
    assert isinstance(y_reg, int)


# ── Label value ranges ────────────────────────────────────────────────────────

def test_dataset_y_dir_in_valid_range():
    ds = _make_dataset()
    for i in range(len(ds)):
        _, y_dir, _ = ds[i]
        assert 0 <= y_dir <= 1


def test_dataset_y_reg_in_valid_range():
    ds = _make_dataset()
    for i in range(len(ds)):
        _, _, y_reg = ds[i]
        assert 0 <= y_reg <= 3


# ── Chronological alignment ───────────────────────────────────────────────────

def test_dataset_windows_overlap_by_w_minus_1():
    """
    Consecutive windows should share W-1 rows.
    ds[0] covers rows 0..W-1; ds[1] covers rows 1..W.
    """
    features = np.arange(N_ROWS * N_FEAT, dtype=np.float32).reshape(N_ROWS, N_FEAT)
    regime = np.zeros(N_ROWS, dtype=np.int64)
    dirs = np.zeros(N_ROWS, dtype=np.int64)
    ds = SPYWindowDataset(features, regime, dirs, window_size=W)
    x0, _, _ = ds[0]
    x1, _, _ = ds[1]
    assert torch.allclose(x0[1:], x1[:-1])


def test_dataset_first_window_starts_at_row_0():
    features = np.arange(N_ROWS * N_FEAT, dtype=np.float32).reshape(N_ROWS, N_FEAT)
    regime = np.zeros(N_ROWS, dtype=np.int64)
    dirs = np.zeros(N_ROWS, dtype=np.int64)
    ds = SPYWindowDataset(features, regime, dirs, window_size=W)
    x0, _, _ = ds[0]
    assert torch.allclose(x0[0], torch.from_numpy(features[0]))


def test_dataset_regime_label_from_last_window_row():
    """y_reg for window idx should be regime_labels[idx + W - 1]."""
    n = 50
    features = np.zeros((n, N_FEAT), dtype=np.float32)
    regime = np.arange(n, dtype=np.int64) % 4
    dirs = np.zeros(n, dtype=np.int64)
    ds = SPYWindowDataset(features, regime, dirs, window_size=W)
    for idx in range(len(ds)):
        _, _, y_reg = ds[idx]
        expected = int(regime[idx + W - 1])
        assert y_reg == expected, f"idx={idx}: y_reg={y_reg}, expected={expected}"


def test_dataset_dir_label_from_last_window_row():
    """y_dir for window idx should be dir_labels[idx + W - 1]."""
    n = 50
    features = np.zeros((n, N_FEAT), dtype=np.float32)
    regime = np.zeros(n, dtype=np.int64)
    dirs = np.arange(n, dtype=np.int64) % 2
    ds = SPYWindowDataset(features, regime, dirs, window_size=W)
    for idx in range(len(ds)):
        _, y_dir, _ = ds[idx]
        expected = int(dirs[idx + W - 1])
        assert y_dir == expected


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_dataset_last_valid_index():
    ds = _make_dataset(n=N_ROWS, window=W)
    last_idx = len(ds) - 1
    x, y_dir, y_reg = ds[last_idx]
    assert x.shape == (W, N_FEAT)


def test_dataset_torch_tensor_type():
    ds = _make_dataset()
    x, _, _ = ds[0]
    assert isinstance(x, torch.Tensor)
