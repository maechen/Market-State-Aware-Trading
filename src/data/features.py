"""
Feature engineering pipeline for the MarketTransformer.

Four-tier normalization strategy:
    1. Unbounded (log_return, rolling_vol_20, macd, relative MA distances):
       StandardScaler fit on training fold only.
    2. Bounded [0, 100] (rsi_14): divide by 100 → [0, 1]; no further scaling.
    3. Probability simplex — regime_prob_0..3 (HMM posterior) and
       Negative/Neutral/Positive (sentiment): pass-through as-is.

Final feature vector (14 dimensions in FEATURE_ORDER):
    Price/tech — first d_feat=11 columns (fed to CrossAttentionGate as K/V):
        [log_return, rolling_vol_20, macd,
         close_to_ma10, close_to_ma20, ma10_to_ma50,
         rsi_norm,
         regime_prob_0, regime_prob_1, regime_prob_2, regime_prob_3]

    Sentiment — last d_sent=3 columns (fed to CrossAttentionGate as Q):
        [Negative, Neutral, Positive]

Rationale for regime_prob inclusion:
    The HMM posterior probabilities capture high-level market state (crash,
    recovery, bull, high-vol) not fully encoded by individual technical
    indicators.  Regime state has strong directional predictability: crash
    regime → predominantly bear days; bull regime → predominantly bull days.
    Including them as price/tech features lets the CrossAttentionGate learn
    regime-conditioned price representations, directly addressing the gap
    between regime accuracy (≥85 %) and direction accuracy.
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
REGIME_PROB_COLS: list[str] = [
    "regime_prob_0", "regime_prob_1", "regime_prob_2", "regime_prob_3"
]
SENT_COLS: list[str] = ["Negative", "Neutral", "Positive"]

# 14 total: 11 price/tech (d_feat) + 3 sentiment (d_sent)
FEATURE_ORDER: list[str] = SCALE_COLS + [RSI_COL] + REGIME_PROB_COLS + SENT_COLS


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
    Applies the four-tier normalization and returns a float32 array of shape
    (n_rows, 14) in FEATURE_ORDER.

    Args:
        df     : raw fold DataFrame (with original CSV columns)
        scaler : StandardScaler fitted on the training fold
    Returns:
        numpy array of shape (n_rows, 14), dtype float32
    """
    derived = derive_features(df)
    scaled = scaler.transform(derived[SCALE_COLS].values)   # (n, 6)  z-scored
    rsi = derived["rsi_norm"].values.reshape(-1, 1)          # (n, 1)  [0, 1]
    regime = df[REGIME_PROB_COLS].values                     # (n, 4)  pass-through [0,1]
    sent = df[SENT_COLS].values                              # (n, 3)  pass-through [0,1]
    return np.concatenate([scaled, rsi, regime, sent], axis=1).astype(np.float32)


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
