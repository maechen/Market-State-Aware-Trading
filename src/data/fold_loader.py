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
    REGIME_PROB_COLS,
    SENT_COLS,
    UNBOUNDED_COLS,
    standardize_returns,
)

def _required_raw_columns() -> list[str]:
    """
    Columns required to (a) build model features and (b) build labels.

    Includes OHLCV columns (High, Low, Volume) needed by the new momentum
    features (ATR, BB%B, volume ratio) computed in derive_features().
    """
    # derive_features prerequisites (OHLCV + MAs + RSI + MACD signal)
    derived_prereqs = [
        "Adj Close", "High", "Low", "Volume",
        "ma_10", "ma_20", "ma_50",
        "rsi_14", "macd", "macd_signal",
    ]
    # feature inputs: price/tech (unbounded) + regime probs + sentiment
    feature_inputs = list(UNBOUNDED_COLS) + list(REGIME_PROB_COLS) + list(SENT_COLS)
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


def _fwd_return(df: pd.DataFrame, n_forward: int) -> pd.Series:
    """
    Computes the n_forward-day cumulative log return for each row.

    ``_fwd_return(df, n)[t]`` = log_return[t+1] + ... + log_return[t+n],
    i.e. the total return over the next n trading days (the label horizon).
    The last n rows will be NaN (no complete n-day window ahead of them).

    Implementation note:
        shift(-n) maps log_return[t+n] to index t.
        .rolling(n).sum() then sums the n values ending at each index,
        which (after the shift) corresponds to log_return[t+1]..log_return[t+n].
    """
    if n_forward == 1:
        return df["log_return"].shift(-1)
    return df["log_return"].shift(-n_forward).rolling(n_forward).sum()


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
    shuffle_train: bool = True,
    n_dir_classes: int = 2,
    dir_n_forward: int = 5,
    ret_n_forward: int = 5,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads train/val/test CSVs from fold_dir, applies the full feature
    engineering + normalization pipeline, and returns three DataLoaders.

    Args:
        fold_dir      : path to a directory containing
                        spy_train_labeled.csv, spy_val_labeled.csv,
                        spy_test_labeled.csv
        window_size   : W (lookback window in trading days)
        batch_size    : DataLoader batch size
        q_low         : lower quantile for neutral band (3-class mode only)
        q_high        : upper quantile for neutral band (3-class mode only)
        num_workers   : DataLoader worker processes
        shuffle_train : randomise training batch order (recommended)
        n_dir_classes : 2 = binary Up/Down (default); 3 = Bear/Neutral/Bull
        dir_n_forward : label horizon for direction target (days ahead, default 5)
        ret_n_forward : label horizon for return target (days ahead, default 5)

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

    # ── Direction labels ──────────────────────────────────────────────────
    # dir_n_forward-day cumulative log return used as the direction signal.
    # The last dir_n_forward rows will be NaN (no complete forward window);
    # nan_to_num converts them to class 0, which affects at most dir_n_forward
    # samples out of ~1200+ — negligible contamination.
    train_fwd_dir = _fwd_return(train_df, dir_n_forward).values
    train_fwd_dir_finite = train_fwd_dir[np.isfinite(train_fwd_dir)]
    if train_fwd_dir_finite.size == 0:
        raise ValueError(
            f"No finite training forward returns for direction labels "
            f"(dir_n_forward={dir_n_forward})."
        )

    train_dir_all, lo, hi = make_direction_labels(
        train_fwd_dir_finite,
        train_fwd_dir,
        q_low, q_high, n_classes=n_dir_classes,
    )
    val_dir_all, _, _ = make_direction_labels(
        train_fwd_dir_finite,
        _fwd_return(val_df, dir_n_forward).values,
        q_low, q_high, n_classes=n_dir_classes,
    )
    test_dir_all, _, _ = make_direction_labels(
        train_fwd_dir_finite,
        _fwd_return(test_df, dir_n_forward).values,
        q_low, q_high, n_classes=n_dir_classes,
    )

    # ── Standardized return targets ───────────────────────────────────────
    # ret_n_forward-day cumulative log return standardised using training stats.
    # The last ret_n_forward rows will be NaN; nan_to_num replaces with 0.
    train_fwd_ret = _fwd_return(train_df, ret_n_forward).values
    train_fwd_ret_finite = train_fwd_ret[np.isfinite(train_fwd_ret)]

    train_ret_std, _, _ = standardize_returns(train_fwd_ret_finite, train_fwd_ret)
    val_ret_std, _, _   = standardize_returns(
        train_fwd_ret_finite, _fwd_return(val_df, ret_n_forward).values
    )
    test_ret_std, _, _  = standardize_returns(
        train_fwd_ret_finite, _fwd_return(test_df, ret_n_forward).values
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
        train_ds, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
