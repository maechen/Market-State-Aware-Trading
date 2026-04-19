"""
Feature engineering pipeline for the MarketTransformer.

Three-tier normalization strategy (from implementation plan V2, C9):
    1. Unbounded (log_return, rolling_vol_20, macd, relative MA distances):
       StandardScaler fit on training fold only.
    2. Bounded [0, 100] (rsi_14): divide by 100 → [0, 1]; no further scaling.
    3. Probability simplex (Negative, Neutral, Positive): pass-through.

Final feature vector (10 dimensions in FEATURE_ORDER):
    [log_return, rolling_vol_20, macd,
     close_to_ma10, close_to_ma20, ma10_to_ma50,
     rsi_norm,
     Negative, Neutral, Positive]
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── Column constants ──────────────────────────────────────────────────────────

UNBOUNDED_COLS: list[str] = ["log_return", "rolling_vol_20", "macd"]
DERIVED_COLS: list[str] = ["close_to_ma10", "close_to_ma20", "ma10_to_ma50"]
SCALE_COLS: list[str] = UNBOUNDED_COLS + DERIVED_COLS  # 6 z-scored columns
RSI_COL: str = "rsi_norm"
SENT_COLS: list[str] = ["Negative", "Neutral", "Positive"]

FEATURE_ORDER: list[str] = SCALE_COLS + [RSI_COL] + SENT_COLS  # length = 10


# ── Feature derivation ────────────────────────────────────────────────────────

def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes relative MA distance features and RSI normalization.
    Does NOT modify the original DataFrame.

    New columns added:
        close_to_ma10 = Adj Close / ma_10 - 1
        close_to_ma20 = Adj Close / ma_20 - 1
        ma10_to_ma50  = ma_10 / ma_50 - 1
        rsi_norm      = rsi_14 / 100
    """
    df = df.copy()
    df["close_to_ma10"] = df["Adj Close"] / df["ma_10"] - 1
    df["close_to_ma20"] = df["Adj Close"] / df["ma_20"] - 1
    df["ma10_to_ma50"] = df["ma_10"] / df["ma_50"] - 1
    df["rsi_norm"] = df["rsi_14"] / 100.0
    return df


# ── Scaler ────────────────────────────────────────────────────────────────────

def fit_scaler(train_df: pd.DataFrame) -> StandardScaler:
    """
    Fits a StandardScaler on SCALE_COLS from the training fold derived
    DataFrame (must have already had derive_features applied).
    """
    scaler = StandardScaler()
    scaler.fit(train_df[SCALE_COLS].values)
    return scaler


def apply_normalization(df: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    """
    Applies the three-tier normalization and returns a float32 array of shape
    (n_rows, 10) in FEATURE_ORDER.

    Args:
        df     : raw fold DataFrame (with original CSV columns)
        scaler : StandardScaler fitted on the training fold
    Returns:
        numpy array of shape (n_rows, 10), dtype float32
    """
    derived = derive_features(df)
    scaled = scaler.transform(derived[SCALE_COLS].values)   # (n, 6)  z-scored
    rsi = derived["rsi_norm"].values.reshape(-1, 1)          # (n, 1)  [0, 1]
    sent = df[SENT_COLS].values                              # (n, 3)  pass-through
    return np.concatenate([scaled, rsi, sent], axis=1).astype(np.float32)


# ── Direction labels ──────────────────────────────────────────────────────────

def make_direction_labels(
    train_ret: np.ndarray,
    eval_ret: np.ndarray,
    q_low: float = 0.40,
    q_high: float = 0.60,
) -> Tuple[np.ndarray, float, float]:
    """
    Builds 3-class direction labels using fold-specific quantile thresholds.

    The neutral band is [q_low, q_high] quantile of the *training* return
    distribution. The same lo/hi thresholds are applied to eval_ret without
    leaking any eval statistics.

    Args:
        train_ret : 1-D array of next-day log_returns from the training fold
                    (NaN-free; typically df["log_return"].shift(-1).iloc[:-1])
        eval_ret  : 1-D array of next-day log_returns for the target split
        q_low     : lower quantile bound for neutral band (default 0.40)
        q_high    : upper quantile bound for neutral band (default 0.60)
    Returns:
        labels : int64 array of 0 (Bear), 1 (Neutral), 2 (Bull)
        lo     : lower threshold value
        hi     : upper threshold value
    """
    lo = float(np.quantile(train_ret, q_low))
    hi = float(np.quantile(train_ret, q_high))
    labels = np.where(eval_ret < lo, 0, np.where(eval_ret > hi, 2, 1)).astype(np.int64)
    return labels, lo, hi


# ── Return standardization ────────────────────────────────────────────────────

def standardize_returns(
    train_ret: np.ndarray,
    eval_ret: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """
    Standardizes eval_ret using training fold mean and std.

    Args:
        train_ret : NaN-free training next-day returns
        eval_ret  : target split next-day returns (may contain NaN at last pos)
    Returns:
        standardized : float32 array
        mean         : training mean
        std          : training std (clamped to 1.0 if near zero)
    """
    mean = float(np.mean(train_ret))
    std = float(np.std(train_ret))
    if std < 1e-8:
        std = 1.0
    return ((eval_ret - mean) / std).astype(np.float32), mean, std
