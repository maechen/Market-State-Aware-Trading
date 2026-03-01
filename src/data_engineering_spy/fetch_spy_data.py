import numpy as np
import pandas as pd
import yfinance as yf


START_DATE = "1993-01-01"
OUTPUT_PATH = "spy_market_data.csv"
FINAL_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "log_return",
    "rolling_vol_20",
    "rsi_14",
    "macd",
    "macd_signal",
    "ma_10",
    "ma_20",
    "ma_50",
]


def _flatten_columns_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance multi-index columns for a single ticker download."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df


def download_spy_data(start_date: str = START_DATE) -> pd.DataFrame:
    """Download daily SPY data from the given start date to today."""
    end_date = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download(
        tickers="SPY",
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if df.empty:
        raise ValueError("No SPY data was downloaded.")

    df = _flatten_columns_if_needed(df).reset_index()

    required_columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns from download: {missing}")

    df = df[required_columns].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].dt.tz is not None:
        df["Date"] = df["Date"].dt.tz_localize(None)

    df = df.sort_values("Date", ascending=True)
    df = df.drop_duplicates(subset=["Date"], keep="last")
    df = df.reset_index(drop=True)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add strictly backward-looking engineered features."""
    out = df.copy()
    adj_close = out["Adj Close"]

    # Daily log returns from adjusted close.
    out["log_return"] = np.log(adj_close / adj_close.shift(1))

    # 20-day rolling volatility of log returns.
    out["rolling_vol_20"] = out["log_return"].rolling(window=20, min_periods=20).std()

    # RSI(14), computed manually with rolling average gains/losses.
    delta = adj_close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD and signal line from adjusted close.
    ema_12 = adj_close.ewm(span=12, adjust=False).mean()
    ema_26 = adj_close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema_12 - ema_26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()

    # Backward-looking simple moving averages.
    out["ma_10"] = adj_close.rolling(window=10, min_periods=10).mean()
    out["ma_20"] = adj_close.rolling(window=20, min_periods=20).mean()
    out["ma_50"] = adj_close.rolling(window=50, min_periods=50).mean()
    return out


def finalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return final column set and order without dropping NaN rows."""
    return df[FINAL_COLUMNS].copy()


def export_to_csv(df: pd.DataFrame, output_path: str = OUTPUT_PATH) -> None:
    """Write final dataset to CSV, overwriting if file exists."""
    df.to_csv(output_path, index=False)


def print_summary(df: pd.DataFrame) -> None:
    """Print basic safety checks and dataset stats."""
    min_date = df["Date"].min()
    max_date = df["Date"].max()
    print(f"Date range: {min_date.date()} to {max_date.date()}")
    print(f"Total rows: {len(df)}")
    print(f"Nulls in log_return: {df['log_return'].isna().sum()}")
    print(f"Nulls in rolling_vol_20: {df['rolling_vol_20'].isna().sum()}")


def main() -> None:
    raw_df = download_spy_data(start_date=START_DATE)
    feature_df = add_features(raw_df)
    final_df = finalize_dataset(feature_df)
    export_to_csv(final_df, output_path=OUTPUT_PATH)
    print_summary(final_df)


if __name__ == "__main__":
    main()
