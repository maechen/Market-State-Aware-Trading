"""
Feature engineering pipeline for the MarketTransformer.

Five-tier normalization strategy:
    1. Unbounded z-scored (SCALE_COLS, 10 cols):
       StandardScaler fit on training fold only.
    2. Bounded [0, 1] — rsi_norm: divide rsi_14 by 100.
    3. Bounded [0, 1] — bb_pct: Bollinger Band %B, clipped to [0, 1].
    4. Probability simplex — regime_prob_0..3 (HMM posterior): pass-through.
    5. Probability simplex — Negative/Neutral/Positive (sentiment): pass-through.

Final feature vector (19 dimensions in FEATURE_ORDER):
    Price/tech — first d_feat=16 columns (fed to CrossAttentionGate as K/V):
        z-scored (10):  log_return, rolling_vol_20, macd,
                        close_to_ma10, close_to_ma20, ma10_to_ma50,
                        roc_5, macd_hist, atr_ratio, vol_ratio
        bounded (1):    rsi_norm
        bounded (1):    bb_pct
        pass-through (4): regime_prob_0..3

    Sentiment — last d_sent=3 columns (fed to CrossAttentionGate as Q):
        [Negative, Neutral, Positive]

New features and their rationale (SOTA research, 2025):
    roc_5       5-day cumulative log return — captures medium-term momentum.
                Literature shows 5-day ROC is among the strongest short-horizon
                predictors of 5-day forward direction.
    macd_hist   MACD histogram (= macd − macd_signal), normalised by price.
                Captures trend strength changes and divergences.
    atr_ratio   ATR(14) / Adj Close — normalised volatility.  High ATR → wide
                daily range → regime-aware feature.
    bb_pct      Bollinger Band %B = (price − lower) / (upper − lower).
                Positions price within volatility-adjusted bands;
                >0.8 = overbought, <0.2 = oversold.
    vol_ratio   Volume / 20-day avg volume — detects institutional volume surges
                which are leading indicators of directional moves.
    regime_probs HMM posterior probabilities — high-level market state proxy
                that directly informs forward direction.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── Column constants ──────────────────────────────────────────────────────────

UNBOUNDED_COLS: list[str] = ["log_return", "rolling_vol_20", "macd"]
DERIVED_COLS: list[str] = ["close_to_ma10", "close_to_ma20", "ma10_to_ma50"]
MOMENTUM_COLS: list[str] = ["roc_5", "macd_hist", "atr_ratio", "vol_ratio"]
# All 10 columns that get z-scored by StandardScaler:
SCALE_COLS: list[str] = UNBOUNDED_COLS + DERIVED_COLS + MOMENTUM_COLS
RSI_COL: str = "rsi_norm"
BB_COL: str = "bb_pct"
REGIME_PROB_COLS: list[str] = [
    "regime_prob_0", "regime_prob_1", "regime_prob_2", "regime_prob_3"
]
SENT_COLS: list[str] = ["Negative", "Neutral", "Positive"]

# 19 total: 16 price/tech (d_feat) + 3 sentiment (d_sent)
FEATURE_ORDER: list[str] = SCALE_COLS + [RSI_COL, BB_COL] + REGIME_PROB_COLS + SENT_COLS


# ── Feature derivation ────────────────────────────────────────────────────────

def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all engineered features from the raw fold DataFrame.
    Does NOT modify the original DataFrame.

    Original columns (unchanged):
        close_to_ma10 = Adj Close / ma_10 - 1
        close_to_ma20 = Adj Close / ma_20 - 1
        ma10_to_ma50  = ma_10 / ma_50 - 1
        rsi_norm      = rsi_14 / 100

    New momentum/volatility columns:
        roc_5         5-day cumulative log return (past 5 days, NOT forward-looking).
                      Captures medium-term momentum as a feature; the model predicts
                      the FUTURE 5-day return as a target — these are independent.
        macd_hist     (macd − macd_signal) / Adj Close — price-normalised MACD
                      histogram measuring trend strength and divergence.
        atr_ratio     ATR(14) / Adj Close — normalised daily range volatility.
                      True Range = max(High−Low, |High−prevClose|, |Low−prevClose|).
        bb_pct        Bollinger Band %B = (Adj Close − lower) / (upper − lower)
                      where bands = ma_20 ± 2 × 20-day price rolling std.
                      Clipped to [0, 1].
        vol_ratio     Volume / Volume.rolling(20).mean() — detects institutional
                      volume surges that often precede directional moves.

    Requires columns: Adj Close, High, Low, Volume, ma_10, ma_20, ma_50,
                      rsi_14, macd, macd_signal, log_return.
    """
    df = df.copy()

    # ── Original relative-MA and RSI features ────────────────────────────────
    df["close_to_ma10"] = df["Adj Close"] / df["ma_10"] - 1
    df["close_to_ma20"] = df["Adj Close"] / df["ma_20"] - 1
    df["ma10_to_ma50"]  = df["ma_10"]     / df["ma_50"] - 1
    df["rsi_norm"]      = df["rsi_14"] / 100.0

    # ── 5-day past momentum (feature, not label) ──────────────────────────────
    df["roc_5"] = df["log_return"].rolling(5, min_periods=1).sum()

    # ── MACD histogram, price-normalised ─────────────────────────────────────
    price_safe = df["Adj Close"].clip(lower=1e-8)
    df["macd_hist"] = (df["macd"] - df["macd_signal"]) / price_safe

    # ── ATR(14) / price ───────────────────────────────────────────────────────
    prev_close = df["Adj Close"].shift(1)
    true_range = pd.concat(
        [
            (df["High"] - df["Low"]).abs(),
            (df["High"] - prev_close).abs(),
            (df["Low"]  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr_ratio"] = true_range.rolling(14, min_periods=1).mean() / price_safe

    # ── Bollinger Band %B ─────────────────────────────────────────────────────
    rolling_std = df["Adj Close"].rolling(20, min_periods=5).std().fillna(1e-8)
    bb_upper = df["ma_20"] + 2.0 * rolling_std
    bb_lower = df["ma_20"] - 2.0 * rolling_std
    band_width = (bb_upper - bb_lower).clip(lower=1e-8)
    df["bb_pct"] = ((df["Adj Close"] - bb_lower) / band_width).clip(0.0, 1.0)

    # ── Volume ratio (today / 20-day avg) ─────────────────────────────────────
    # min_periods=1 ensures no NaN even for the first row (early-window lookback
    # is shorter but finite); clamp denominator to 1 to prevent division by zero.
    avg_vol = df["Volume"].rolling(20, min_periods=1).mean().clip(lower=1.0)
    df["vol_ratio"] = df["Volume"] / avg_vol

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
    Applies the five-tier normalization and returns a float32 array of shape
    (n_rows, 19) in FEATURE_ORDER.

    Args:
        df     : raw fold DataFrame (with original CSV columns)
        scaler : StandardScaler fitted on the training fold
    Returns:
        numpy array of shape (n_rows, 19), dtype float32
    """
    derived = derive_features(df)
    scaled  = scaler.transform(derived[SCALE_COLS].values)  # (n, 10) z-scored
    rsi     = derived["rsi_norm"].values.reshape(-1, 1)      # (n, 1)  [0, 1]
    bb      = derived["bb_pct"].values.reshape(-1, 1)        # (n, 1)  [0, 1]
    regime  = df[REGIME_PROB_COLS].values                    # (n, 4)  pass-through [0,1]
    sent    = df[SENT_COLS].values                           # (n, 3)  pass-through [0,1]
    return np.concatenate([scaled, rsi, bb, regime, sent], axis=1).astype(np.float32)


# ── Direction labels ──────────────────────────────────────────────────────────

def make_direction_labels(
    train_ret: np.ndarray,
    eval_ret: np.ndarray,
    q_low: float = 0.40,
    q_high: float = 0.60,
    n_classes: int = 2,
) -> Tuple[np.ndarray, float, float]:
    """
    Builds direction labels for the target split.

    Two modes controlled by ``n_classes``:

    **Binary (n_classes=2, default):**
        Labels are 1 (Up) if the forward return is positive, 0 (Down) otherwise.
        No quantile thresholds are computed or used.  This eliminates the
        distribution-shift problem where training quantile thresholds produce
        heavily imbalanced (60-70 % neutral) classes at test time.  With a
        A 5-day forward horizon is less noisy than 1-day direction, but binary
        SPY direction remains difficult and should be evaluated against majority
        and walk-forward baselines.

    **Three-class (n_classes=3):**
        Bear = below q_low quantile of training returns.
        Neutral = within [q_low, q_high] quantile.
        Bull = above q_high quantile.
        Thresholds lo/hi are derived from train_ret and applied to eval_ret.

    Args:
        train_ret : 1-D NaN-free array of forward log_returns from training fold
                    (used only for quantile computation in 3-class mode)
        eval_ret  : 1-D array of forward log_returns for the target split
        q_low     : lower quantile bound (3-class only)
        q_high    : upper quantile bound (3-class only)
        n_classes : 2 = binary Up/Down; 3 = Bear/Neutral/Bull
    Returns:
        labels : int64 array of class indices
        lo     : lower threshold (0.0 for binary mode)
        hi     : upper threshold (0.0 for binary mode)
    """
    if n_classes == 2:
        labels = (eval_ret > 0).astype(np.int64)
        return labels, 0.0, 0.0

    lo = float(np.quantile(train_ret, q_low))
    hi = float(np.quantile(train_ret, q_high))
    labels = np.where(eval_ret < lo, 0, np.where(eval_ret > hi, 2, 1)).astype(np.int64)
    return labels, lo, hi


