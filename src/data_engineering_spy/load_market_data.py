from typing import Tuple

import numpy as np
import pandas as pd


def load_data(path: str = "spy_market_data.csv") -> pd.DataFrame:
    """Load market data, enforce chronological order, and index by Date."""
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date", ascending=True)
    df = df.drop_duplicates(subset=["Date"], keep="last")
    df = df.set_index("Date")
    return df


def slice_data(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Return an inclusive date slice as a new DataFrame."""
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    mask = (df.index >= start_ts) & (df.index <= end_ts)
    return df.loc[mask].copy()


def create_walk_forward_split(
    df: pd.DataFrame,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/validation/test splits with pure date slicing."""
    train_df = slice_data(df, train_start, train_end)
    val_df = slice_data(df, val_start, val_end)
    test_df = slice_data(df, test_start, test_end)
    return train_df, val_df, test_df


def get_hmm_features(df: pd.DataFrame) -> np.ndarray:
    """
    Return HMM feature matrix with:
    - log_return
    - rolling_vol_20
    """
    feature_df = df[["log_return", "rolling_vol_20"]].dropna(
        subset=["log_return", "rolling_vol_20"]
    )
    return feature_df.to_numpy()


if __name__ == "__main__":
    full_df = load_data()
    print(f"Date range: {full_df.index.min().date()} to {full_df.index.max().date()}")

    train_df, val_df, test_df = create_walk_forward_split(
        full_df,
        train_start="2015-01-01",
        train_end="2019-12-31",
        val_start="2020-01-01",
        val_end="2020-12-31",
        test_start="2021-01-01",
        test_end="2021-12-31",
    )

    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")
