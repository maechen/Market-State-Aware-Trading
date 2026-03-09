from typing import Tuple

import numpy as np
import pandas as pd

"""
There are some values that are null till the 50th day. Specifically, the ma_50. rolling_vol_20 is null till the 20th day.
rsi_14 is null till the 14th day. ma_10 is null till the 10th day. ma_20 is null till the 20th day.
"""

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


def find_missing_data_points(df: pd.DataFrame, warmup_days: int = 50) -> pd.DataFrame:
    """
    Identifies specific Dates and Columns that are empty starting from the 50th day.
    Returns a DataFrame listing [Date, Column] for all gaps.
    """
    # 1. Slice the data to ignore the initial calculation period
    test_slice = df.iloc[warmup_days:]
    
    # 2. Find coordinates of all null values
    # .stack(dropna=False) turns columns into a secondary index
    # We then filter for only the null values
    null_mask = test_slice.isnull().stack()
    missing_points = null_mask[null_mask].index.to_list()
    
    # 3. Format into a readable DataFrame
    if not missing_points:
        print(f"✅ No missing values found after index {warmup_days}.")
        return pd.DataFrame(columns=["Date", "Column"])
    
    report_df = pd.DataFrame(missing_points, columns=["Date", "Column"])
    
    print(f"⚠️ Found {len(report_df)} missing data points:")
    print(report_df.to_string(index=False))
    
    return report_df


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
    print(find_missing_data_points(full_df))

    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")
