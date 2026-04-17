"""
get_fold_loaders — build train/val/test DataLoaders for one walk-forward fold.

All normalization statistics (scaler mean/std, direction thresholds, return
mean/std) are computed from the training split only and applied uniformly
to val and test — no future information leaks across splits.
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .dataset import SPYWindowDataset
from .features import (
    apply_normalization,
    derive_features,
    fit_scaler,
    make_direction_labels,
    standardize_returns,
)


def get_fold_loaders(
    fold_dir: str,
    window_size: int = 20,
    batch_size: int = 64,
    q_low: float = 0.40,
    q_high: float = 0.60,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads train/val/test CSVs from fold_dir, applies the full feature
    engineering + normalization pipeline, and returns three DataLoaders.

    Args:
        fold_dir    : path to a directory containing
                      spy_train_labeled.csv, spy_val_labeled.csv,
                      spy_test_labeled.csv
        window_size : W (lookback window)
        batch_size  : DataLoader batch size
        q_low       : lower quantile for direction neutral band
        q_high      : upper quantile for direction neutral band
        num_workers : DataLoader worker processes

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_df = pd.read_csv(
        os.path.join(fold_dir, "spy_train_labeled.csv"), index_col=0, parse_dates=True
    )
    val_df = pd.read_csv(
        os.path.join(fold_dir, "spy_val_labeled.csv"), index_col=0, parse_dates=True
    )
    test_df = pd.read_csv(
        os.path.join(fold_dir, "spy_test_labeled.csv"), index_col=0, parse_dates=True
    )

    # ── Fit normalization on training split only ──────────────────────────
    train_derived = derive_features(train_df)
    scaler = fit_scaler(train_derived)

    train_feat = apply_normalization(train_df, scaler)
    val_feat = apply_normalization(val_df, scaler)
    test_feat = apply_normalization(test_df, scaler)

    # ── Direction labels (quantile thresholds from training returns) ──────
    # shift(-1) gives next-day return; drop last NaN for threshold fitting
    train_next_ret = train_df["log_return"].shift(-1).iloc[:-1].values

    train_dir_all, lo, hi = make_direction_labels(
        train_next_ret, train_df["log_return"].shift(-1).values, q_low, q_high
    )
    val_dir_all, _, _ = make_direction_labels(
        train_next_ret, val_df["log_return"].shift(-1).values, q_low, q_high
    )
    test_dir_all, _, _ = make_direction_labels(
        train_next_ret, test_df["log_return"].shift(-1).values, q_low, q_high
    )

    # ── Standardized return targets ───────────────────────────────────────
    train_ret_std, _, _ = standardize_returns(
        train_next_ret, train_df["log_return"].shift(-1).values
    )
    val_ret_std, _, _ = standardize_returns(
        train_next_ret, val_df["log_return"].shift(-1).values
    )
    test_ret_std, _, _ = standardize_returns(
        train_next_ret, test_df["log_return"].shift(-1).values
    )

    # ── Regime labels ─────────────────────────────────────────────────────
    train_reg = train_df["regime_label"].values.astype(np.int64)
    val_reg = val_df["regime_label"].values.astype(np.int64)
    test_reg = test_df["regime_label"].values.astype(np.int64)

    # ── Build datasets and loaders ────────────────────────────────────────
    train_ds = SPYWindowDataset(train_feat, train_reg, train_dir_all, train_ret_std, window_size)
    val_ds = SPYWindowDataset(val_feat, val_reg, val_dir_all, val_ret_std, window_size)
    test_ds = SPYWindowDataset(test_feat, test_reg, test_dir_all, test_ret_std, window_size)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
