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
    SENT_COLS,
    UNBOUNDED_COLS,
    standardize_returns,
)

def _required_raw_columns() -> list[str]:
    """
    Columns required to (a) build model features and (b) build labels.

    Derived feature prerequisites come from `derive_features()` and the base
    unbounded feature columns from `UNBOUNDED_COLS`.
    Sentiment columns come from `SENT_COLS`.
    Regime labels are required for the multitask head target.
    """
    # derive_features prerequisites
    derived_prereqs = ["Adj Close", "ma_10", "ma_20", "ma_50", "rsi_14"]
    # feature inputs used by scaling + sentiment pass-through
    feature_inputs = list(UNBOUNDED_COLS) + list(SENT_COLS)
    # supervised labels
    label_inputs = ["regime_label", "log_return"]
    # de-duplicate while preserving order
    seen: set[str] = set()
    required: list[str] = []
    for col in [*derived_prereqs, *feature_inputs, *label_inputs]:
        if col not in seen:
            required.append(col)
            seen.add(col)
    return required


def _clean_split(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """
    Drop rows that cannot produce finite model features/labels.

    This prevents NaNs from indicator warmup periods or sparse sentiment rows
    from propagating into windows and producing NaN losses.
    """
    required_cols = _required_raw_columns()
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(
            f"{split_name} split is missing required columns: {missing_cols}"
        )

    cleaned = df.copy()
    # Treat +/-inf as missing, then drop any row where a required column is
    # NaN or Inf (covers indicator warmup rows and any sparse sentiment rows).
    cleaned[required_cols] = cleaned[required_cols].replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.dropna(subset=required_cols).copy()
    if cleaned.empty:
        raise ValueError(
            f"{split_name} split became empty after dropping rows with missing model inputs."
        )
    return cleaned


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
    train_df = _clean_split(train_df, "train")
    val_df = _clean_split(val_df, "val")
    test_df = _clean_split(test_df, "test")

    # ── Fit normalization on training split only ──────────────────────────
    train_derived = derive_features(train_df)
    scaler = fit_scaler(train_derived)

    train_feat = apply_normalization(train_df, scaler)
    val_feat = apply_normalization(val_df, scaler)
    test_feat = apply_normalization(test_df, scaler)
    for split_name, split_feat in (
        ("train", train_feat),
        ("val", val_feat),
        ("test", test_feat),
    ):
        if not np.isfinite(split_feat).all():
            bad = int(np.size(split_feat) - np.isfinite(split_feat).sum())
            raise ValueError(
                f"{split_name} features contain {bad} non-finite values after normalization."
            )

    # ── Direction labels (quantile thresholds from training returns) ──────
    # shift(-1) gives next-day return; drop last NaN for threshold fitting
    train_next_ret = train_df["log_return"].shift(-1).iloc[:-1].values
    train_next_ret = train_next_ret[np.isfinite(train_next_ret)]
    if train_next_ret.size == 0:
        raise ValueError("No finite training next-day returns available for label thresholds.")

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
    # shift(-1) produces NaN at the last position (no next-day return).
    # nan_to_num replaces that sentinel NaN with 0 so SPYWindowDataset never
    # hands a non-finite value to the DataLoader (the dataset math already
    # prevents the last position from being sampled, but this is a belt-and-
    # suspenders guard against any off-by-one edge case).
    train_ret_std, _, _ = standardize_returns(
        train_next_ret, train_df["log_return"].shift(-1).values
    )
    val_ret_std, _, _ = standardize_returns(
        train_next_ret, val_df["log_return"].shift(-1).values
    )
    test_ret_std, _, _ = standardize_returns(
        train_next_ret, test_df["log_return"].shift(-1).values
    )
    train_ret_std = np.nan_to_num(train_ret_std, nan=0.0, posinf=0.0, neginf=0.0)
    val_ret_std   = np.nan_to_num(val_ret_std,   nan=0.0, posinf=0.0, neginf=0.0)
    test_ret_std  = np.nan_to_num(test_ret_std,  nan=0.0, posinf=0.0, neginf=0.0)

    # ── Regime labels ─────────────────────────────────────────────────────
    train_reg = train_df["regime_label"].values.astype(np.int64)
    val_reg = val_df["regime_label"].values.astype(np.int64)
    test_reg = test_df["regime_label"].values.astype(np.int64)

    # ── Build datasets and loaders ────────────────────────────────────────
    train_ds = SPYWindowDataset(train_feat, train_reg, train_dir_all, train_ret_std, window_size)
    val_ds = SPYWindowDataset(val_feat, val_reg, val_dir_all, val_ret_std, window_size)
    test_ds = SPYWindowDataset(test_feat, test_reg, test_dir_all, test_ret_std, window_size)
    for split_name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        if len(ds) <= 0:
            raise ValueError(
                f"{split_name} dataset has no valid windows; lower --window-size or inspect fold data."
            )

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
