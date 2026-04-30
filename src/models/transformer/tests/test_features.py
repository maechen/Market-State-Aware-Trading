"""Tests for src/data/features.py — feature engineering and label construction."""

import pytest
import numpy as np
import pandas as pd
from src.data.features import (
    derive_features,
    fit_scaler,
    apply_normalization,
    make_direction_labels,
    FEATURE_ORDER,
    SCALE_COLS,
    SENT_COLS,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_raw_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    """Minimal DataFrame matching the training CSV schema (includes OHLCV for new momentum features)."""
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame({
        "Adj Close": close,
        "High": close * rng.uniform(1.00, 1.02, n),
        "Low": close * rng.uniform(0.98, 1.00, n),
        "Volume": rng.uniform(1e6, 1e8, n),
        "log_return": rng.normal(0, 0.01, n),
        "rolling_vol_20": rng.uniform(0.005, 0.02, n),
        "rsi_14": rng.uniform(20, 80, n),
        "macd": rng.normal(0, 0.5, n),
        "macd_signal": rng.normal(0, 0.5, n),
        "ma_10": close * rng.uniform(0.98, 1.02, n),
        "ma_20": close * rng.uniform(0.97, 1.03, n),
        "ma_50": close * rng.uniform(0.95, 1.05, n),
        "regime_label": rng.integers(0, 4, n),
        "regime_prob_0": rng.uniform(0, 1, n),
        "regime_prob_1": rng.uniform(0, 1, n),
        "regime_prob_2": rng.uniform(0, 1, n),
        "regime_prob_3": rng.uniform(0, 1, n),
        "Negative": rng.dirichlet(np.ones(3), n)[:, 0],
        "Neutral": rng.dirichlet(np.ones(3), n)[:, 1],
        "Positive": rng.dirichlet(np.ones(3), n)[:, 2],
    })
    return df


# ── derive_features ───────────────────────────────────────────────────────────

def test_derive_features_adds_close_to_ma10():
    df = _make_raw_df()
    out = derive_features(df)
    assert "close_to_ma10" in out.columns


def test_derive_features_adds_close_to_ma20():
    df = _make_raw_df()
    out = derive_features(df)
    assert "close_to_ma20" in out.columns


def test_derive_features_adds_ma10_to_ma50():
    df = _make_raw_df()
    out = derive_features(df)
    assert "ma10_to_ma50" in out.columns


def test_derive_features_adds_rsi_norm():
    df = _make_raw_df()
    out = derive_features(df)
    assert "rsi_norm" in out.columns


def test_close_to_ma10_values():
    df = _make_raw_df()
    out = derive_features(df)
    expected = df["Adj Close"] / df["ma_10"] - 1
    pd.testing.assert_series_equal(out["close_to_ma10"], expected, check_names=False)


def test_rsi_norm_range():
    df = _make_raw_df()
    out = derive_features(df)
    assert (out["rsi_norm"] >= 0).all() and (out["rsi_norm"] <= 1).all()


def test_derive_features_does_not_modify_original():
    df = _make_raw_df()
    _ = derive_features(df)
    assert "close_to_ma10" not in df.columns


# ── fit_scaler ────────────────────────────────────────────────────────────────

def test_fit_scaler_returns_fitted_scaler():
    from sklearn.preprocessing import StandardScaler
    df = _make_raw_df()
    derived = derive_features(df)
    scaler = fit_scaler(derived)
    assert isinstance(scaler, StandardScaler)
    assert hasattr(scaler, "mean_")


def test_scaler_fit_on_scale_cols():
    df = _make_raw_df()
    derived = derive_features(df)
    scaler = fit_scaler(derived)
    assert len(scaler.mean_) == len(SCALE_COLS)


# ── apply_normalization ───────────────────────────────────────────────────────

def test_apply_normalization_output_shape():
    df = _make_raw_df(n=100)
    derived = derive_features(df)
    scaler = fit_scaler(derived)
    out = apply_normalization(df, scaler)
    assert out.shape == (100, 19)


def test_apply_normalization_dtype_float32():
    df = _make_raw_df()
    derived = derive_features(df)
    scaler = fit_scaler(derived)
    out = apply_normalization(df, scaler)
    assert out.dtype == np.float32


def test_apply_normalization_rsi_column_in_0_1():
    """RSI column (index 6 in FEATURE_ORDER) should be in [0, 1]."""
    df = _make_raw_df()
    derived = derive_features(df)
    scaler = fit_scaler(derived)
    out = apply_normalization(df, scaler)
    rsi_idx = FEATURE_ORDER.index("rsi_norm")
    rsi_col = out[:, rsi_idx]
    assert (rsi_col >= 0).all() and (rsi_col <= 1).all()


def test_apply_normalization_sentiment_passthrough():
    """Sentiment columns (Negative, Neutral, Positive) should be unchanged."""
    df = _make_raw_df(seed=42)
    derived = derive_features(df)
    scaler = fit_scaler(derived)
    out = apply_normalization(df, scaler)
    for col in SENT_COLS:
        idx = FEATURE_ORDER.index(col)
        np.testing.assert_allclose(out[:, idx], df[col].values, rtol=1e-5)


def test_scaler_fit_only_on_train():
    """
    Train-fit scaler applied to val must differ from val-fit scaler,
    confirming no information from val is used in normalization.
    """
    train_df = _make_raw_df(n=200, seed=0)
    val_df = _make_raw_df(n=50, seed=99)
    train_derived = derive_features(train_df)
    scaler_train = fit_scaler(train_derived)
    val_derived = derive_features(val_df)
    scaler_val = fit_scaler(val_derived)
    out_train_fit = apply_normalization(val_df, scaler_train)
    out_val_fit = apply_normalization(val_df, scaler_val)
    # They should generally differ because train and val have different statistics
    assert not np.allclose(out_train_fit, out_val_fit)


# ── make_direction_labels ─────────────────────────────────────────────────────

def test_make_direction_labels_binary_returns_valid_classes():
    """Default mode (n_classes=2) should return only 0 (Down) or 1 (Up)."""
    rng = np.random.default_rng(0)
    train_ret = rng.normal(0, 0.01, 500)
    eval_ret = rng.normal(0, 0.01, 100)
    labels, lo, hi = make_direction_labels(train_ret, eval_ret)
    assert set(labels).issubset({0, 1})


def test_make_direction_labels_binary_extreme_values():
    """Binary mode: clearly negative return → 0, clearly positive → 1."""
    rng = np.random.default_rng(0)
    train_ret = rng.normal(0, 0.01, 500)
    eval_ret = np.array([-1.0, 1.0])
    labels, lo, hi = make_direction_labels(train_ret, eval_ret)
    assert labels[0] == 0   # clearly Down
    assert labels[1] == 1   # clearly Up


def test_make_direction_labels_three_class_returns_valid_classes():
    """3-class mode should return only {0, 1, 2}."""
    rng = np.random.default_rng(0)
    train_ret = rng.normal(0, 0.01, 500)
    eval_ret = rng.normal(0, 0.01, 100)
    labels, lo, hi = make_direction_labels(train_ret, eval_ret, n_classes=3)
    assert set(labels).issubset({0, 1, 2})


def test_make_direction_labels_neutral_proportion_approx_20pct():
    """
    With q_low=0.40, q_high=0.60, ~20% of training returns should be Neutral
    when thresholds are applied back to the training set (3-class mode only).
    """
    rng = np.random.default_rng(0)
    train_ret = rng.normal(0, 0.01, 1000)
    labels, lo, hi = make_direction_labels(train_ret, train_ret, n_classes=3)
    neutral_frac = (labels == 1).mean()
    assert 0.15 <= neutral_frac <= 0.25, f"Neutral fraction {neutral_frac:.2f} outside [0.15, 0.25]"


def test_make_direction_labels_thresholds_from_train():
    rng = np.random.default_rng(0)
    train_ret = rng.normal(0, 0.01, 500)
    eval_ret = np.array([-1.0, 0.0, 1.0])
    labels, lo, hi = make_direction_labels(train_ret, eval_ret, n_classes=3)
    assert labels[0] == 0   # clearly Bear
    assert labels[2] == 2   # clearly Bull


def test_make_direction_labels_three_class_returns_lo_hi():
    """3-class mode must return quantile thresholds lo < hi."""
    rng = np.random.default_rng(0)
    train_ret = rng.normal(0, 0.01, 500)
    eval_ret = rng.normal(0, 0.01, 100)
    labels, lo, hi = make_direction_labels(train_ret, eval_ret, n_classes=3)
    assert lo < hi
    assert lo == np.quantile(train_ret, 0.40)
    assert hi == np.quantile(train_ret, 0.60)


def test_make_direction_labels_binary_lo_hi_are_zero():
    """Binary mode uses zero threshold; lo=hi=0.0."""
    rng = np.random.default_rng(0)
    train_ret = rng.normal(0, 0.01, 500)
    eval_ret = rng.normal(0, 0.01, 100)
    labels, lo, hi = make_direction_labels(train_ret, eval_ret, n_classes=2)
    assert lo == 0.0
    assert hi == 0.0


def test_make_direction_labels_dtype_is_int():
    rng = np.random.default_rng(0)
    train_ret = rng.normal(0, 0.01, 200)
    labels, _, _ = make_direction_labels(train_ret, train_ret)
    assert labels.dtype in (np.int32, np.int64)


