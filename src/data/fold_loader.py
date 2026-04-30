"""
get_fold_loaders — build train/val/test DataLoaders for one walk-forward fold.

All normalization statistics (scaler mean/std, direction thresholds) are
computed from the training split only and applied uniformly to val and test
— no future information leaks across splits.

The model predicts two targets:
    direction : binary Up/Down event label. In vol_threshold mode, ambiguous
                returns inside ±k * rolling_vol_20 * sqrt(dir_n_forward) are
                labeled -1 and ignored by direction CE.
    regime    : HMM state (0..K-1, K=4) from Viterbi decoding
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
    num_workers: int = 0,
    shuffle_train: bool = True,
    n_dir_classes: int = 2,
    dir_n_forward: int = 5,
    dir_label_mode: str = "sign",
    dir_vol_k: float = 0.50,
    dir_vol_col: str = "rolling_vol_20",
    dir_ignore_index: int = -1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads train/val/test CSVs from fold_dir, applies the full feature
    engineering + normalization pipeline, and returns three DataLoaders.

    The model predicts direction (binary Up/Down) and regime (HMM state).
    The auxiliary return regression head has been removed — after extensive
    experimentation it consistently achieved standardised MAE ≈ 0.9 (random
    predictions), causing gradient interference that degraded direction accuracy.

    Args:
        fold_dir      : path to a directory containing
                        spy_train_labeled.csv, spy_val_labeled.csv,
                        spy_test_labeled.csv
        window_size   : W (lookback window in trading days)
        batch_size    : DataLoader batch size
        num_workers   : DataLoader worker processes
        shuffle_train : randomise training batch order (recommended)
        n_dir_classes : 2 = binary Up/Down (default); 3 = Bear/Neutral/Bull
        dir_n_forward : label horizon for direction target (days ahead, default 5)

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
    # Last dir_n_forward rows do not have complete future returns.
    # In vol_threshold mode they become neutral/ignored labels (-1).
    # In sign mode they remain assigned by the legacy binary behavior.
    if dir_label_mode == "vol_threshold":
        for split_name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
            if dir_vol_col not in split_df.columns:
                raise KeyError(
                    f"{split_name} split is missing dir_vol_col={dir_vol_col!r} "
                    "required for vol_threshold direction labels."
                )
            finite = np.isfinite(split_df[dir_vol_col].replace([np.inf, -np.inf], np.nan))
            if not finite.any():
                raise ValueError(
                    f"{split_name} split has no finite values in dir_vol_col={dir_vol_col!r}."
                )

    train_fwd_dir = _fwd_return(train_df, dir_n_forward).values
    val_fwd_dir = _fwd_return(val_df, dir_n_forward).values
    test_fwd_dir = _fwd_return(test_df, dir_n_forward).values
    train_fwd_dir_finite = train_fwd_dir[np.isfinite(train_fwd_dir)]
    if train_fwd_dir_finite.size == 0:
        raise ValueError(
            f"No finite training forward returns for direction labels "
            f"(dir_n_forward={dir_n_forward})."
        )

    train_vol = train_df[dir_vol_col].values if dir_label_mode == "vol_threshold" else None
    val_vol = val_df[dir_vol_col].values if dir_label_mode == "vol_threshold" else None
    test_vol = test_df[dir_vol_col].values if dir_label_mode == "vol_threshold" else None
    vol_horizon_scale = float(np.sqrt(dir_n_forward)) if dir_label_mode == "vol_threshold" else 1.0
    train_dir_all, lo, hi = make_direction_labels(
        train_fwd_dir_finite,
        train_fwd_dir,
        n_classes=n_dir_classes,
        label_mode=dir_label_mode,
        eval_vol=train_vol,
        vol_k=dir_vol_k,
        vol_horizon_scale=vol_horizon_scale,
        ignore_index=dir_ignore_index,
    )
    val_dir_all, _, _ = make_direction_labels(
        train_fwd_dir_finite,
        val_fwd_dir,
        n_classes=n_dir_classes,
        label_mode=dir_label_mode,
        eval_vol=val_vol,
        vol_k=dir_vol_k,
        vol_horizon_scale=vol_horizon_scale,
        ignore_index=dir_ignore_index,
    )
    test_dir_all, _, _ = make_direction_labels(
        train_fwd_dir_finite,
        test_fwd_dir,
        n_classes=n_dir_classes,
        label_mode=dir_label_mode,
        eval_vol=test_vol,
        vol_k=dir_vol_k,
        vol_horizon_scale=vol_horizon_scale,
        ignore_index=dir_ignore_index,
    )

    # ── Regime labels ─────────────────────────────────────────────────────
    train_reg = train_df["regime_label"].values.astype(np.int64)
    val_reg = val_df["regime_label"].values.astype(np.int64)
    test_reg = test_df["regime_label"].values.astype(np.int64)

    # ── Build datasets and loaders ────────────────────────────────────────
    train_ds = SPYWindowDataset(train_feat, train_reg, train_dir_all, window_size)
    val_ds = SPYWindowDataset(val_feat, val_reg, val_dir_all, window_size)
    test_ds = SPYWindowDataset(test_feat, test_reg, test_dir_all, window_size)
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
