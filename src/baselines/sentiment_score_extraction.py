import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


# Financially oriented seed lexicons for lightweight sentiment scoring.
POSITIVE_WORDS = {
    "beat",
    "beats",
    "bullish",
    "buy",
    "buys",
    "breakout",
    "growth",
    "gains",
    "gain",
    "improve",
    "improved",
    "improves",
    "outperform",
    "outperforms",
    "outperformed",
    "profit",
    "profits",
    "record",
    "rebound",
    "strong",
    "surge",
    "surges",
    "upgrade",
    "upgrades",
    "upgraded",
    "upside",
    "rally",
    "raises",
    "raised",
    "rise",
    "rises",
}

NEGATIVE_WORDS = {
    "bearish",
    "cut",
    "cuts",
    "cutting",
    "decline",
    "declines",
    "declined",
    "downgrade",
    "downgrades",
    "downgraded",
    "drop",
    "drops",
    "dropped",
    "fall",
    "falls",
    "fell",
    "loss",
    "losses",
    "miss",
    "misses",
    "missed",
    "plunge",
    "plunges",
    "plunged",
    "risk",
    "risks",
    "sell",
    "sells",
    "slump",
    "slumps",
    "slumped",
    "weak",
    "warning",
    "warnings",
}

TOKEN_PATTERN = re.compile(r"[a-z]+")
MARKET_TZ = "America/New_York"
MARKET_OPEN_MINUTE = 9 * 60 + 30
MARKET_CLOSE_MINUTE = 16 * 60


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def sentiment_score_to_bucket(score: float) -> int:
    """
    Map sentiment score in [-1, 1] to an integer bucket in [0, 10].
    """
    bucket = int(round((score + 1.0) * 5))
    return max(0, min(10, bucket))


def compute_sentiment_score(text: str) -> float:
    """
    Compute a normalized sentiment score in [-1, 1] from finance-focused words.
    """
    if not text:
        return 0.0

    tokens = _tokenize(text)
    if not tokens:
        return 0.0

    pos_count = sum(token in POSITIVE_WORDS for token in tokens)
    neg_count = sum(token in NEGATIVE_WORDS for token in tokens)
    total_hits = pos_count + neg_count

    if total_hits == 0:
        return 0.0

    raw = (pos_count - neg_count) / total_hits
    return float(max(-1.0, min(1.0, raw)))


def _first_existing_column(columns: Iterable[str], preferred: list[str]) -> str | None:
    existing = set(columns)
    for col in preferred:
        if col in existing:
            return col
    return None


def _safe_series(df: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None:
        return pd.Series([""] * len(df), index=df.index, dtype="object")
    return df[column].fillna("").astype(str)


def extract_timestamped_sentiment(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv, low_memory=False)
    if df.empty:
        raise ValueError(f"Input file is empty: {input_csv}")

    date_col = _first_existing_column(df.columns, ["Date", "date", "timestamp", "Datetime"])
    if date_col is None:
        raise KeyError("Could not find a timestamp column. Expected one of: Date, date, timestamp, Datetime")

    symbol_col = _first_existing_column(df.columns, ["Stock_symbol", "stock_symbol", "Symbol", "Ticker"])
    title_col = _first_existing_column(df.columns, ["Article_title", "Title", "title", "headline"])
    article_col = _first_existing_column(df.columns, ["Article", "article", "content", "text"])
    summary_col = _first_existing_column(
        df.columns,
        ["Textrank_summary", "Lexrank_summary", "Lsa_summary", "Luhn_summary", "summary"],
    )
    url_col = _first_existing_column(df.columns, ["Url", "URL", "url", "link"])
    publisher_col = _first_existing_column(df.columns, ["Publisher", "publisher", "source"])

    timestamp = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    title = _safe_series(df, title_col)
    article = _safe_series(df, article_col)
    summary = _safe_series(df, summary_col)

    text_for_scoring = (title + " " + article + " " + summary).str.strip()
    sentiment_scores = text_for_scoring.map(compute_sentiment_score)
    sentiment_score_buckets = sentiment_scores.map(sentiment_score_to_bucket)

    result = pd.DataFrame(
        {
            "timestamp_utc": timestamp,
            "date": timestamp.dt.date,
            "stock_symbol": _safe_series(df, symbol_col).replace("", "UNKNOWN"),
            "sentiment_score": sentiment_scores,
            "sentiment_score_bucket": sentiment_score_buckets,
            "article_title": title,
            "url": _safe_series(df, url_col),
            "publisher": _safe_series(df, publisher_col),
            "text_length": text_for_scoring.str.len(),
        }
    )

    result = result.dropna(subset=["timestamp_utc"])
    result = result[result["text_length"] > 0]
    result = result.sort_values("timestamp_utc").reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract timestamped sentiment scores from FNSPID news data.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/fnspid/Stock_news/All_external.csv"),
        help="Path to the FNSPID news CSV input.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/fnspid_timestamped_sentiment.csv"),
        help="Path to output CSV with timestamped sentiment scores.",
    )
    parser.add_argument(
        "--daily-output",
        type=Path,
        default=Path("data/processed/fnspid_daily_sentiment.csv"),
        help="Optional daily aggregated sentiment output path.",
    )
    parser.add_argument(
        "--daily-news-stats-output",
        type=Path,
        default=Path("data/processed/fnspid_daily_news_stats.csv"),
        help="Path to output per-day histogram counts and market-hours news counts.",
    )
    args = parser.parse_args()

    extracted = extract_timestamped_sentiment(args.input, args.output)

    daily = (
        extracted.groupby(["date", "stock_symbol"], as_index=False)
        .agg(
            daily_sentiment_mean=("sentiment_score", "mean"),
            daily_sentiment_std=("sentiment_score", "std"),
            article_count=("sentiment_score", "count"),
        )
        .sort_values(["date", "stock_symbol"])
        .reset_index(drop=True)
    )
    daily["daily_sentiment_std"] = daily["daily_sentiment_std"].fillna(0.0)

    args.daily_output.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.daily_output, index=False)

    timestamps_market_tz = extracted["timestamp_utc"].dt.tz_convert(MARKET_TZ)
    market_minutes = timestamps_market_tz.dt.hour * 60 + timestamps_market_tz.dt.minute
    is_weekday = timestamps_market_tz.dt.dayofweek < 5
    is_market_hours = is_weekday & (market_minutes >= MARKET_OPEN_MINUTE) & (market_minutes <= MARKET_CLOSE_MINUTE)

    daily_base = extracted.assign(
        market_date=timestamps_market_tz.dt.date,
        is_market_hours=is_market_hours,
    )

    daily_histogram = (
        daily_base.groupby(["market_date", "sentiment_score_bucket"], as_index=False)
        .agg(score_count=("sentiment_score_bucket", "size"))
        .pivot(index="market_date", columns="sentiment_score_bucket", values="score_count")
        .reindex(columns=list(range(11)))
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    daily_histogram = daily_histogram.rename(columns={i: f"score_{i}_count" for i in range(11)})

    daily_market_counts = daily_base.groupby("market_date", as_index=False).agg(
        total_news_count=("sentiment_score", "size"),
        market_hours_news_count=("is_market_hours", "sum"),
    )
    daily_market_counts["market_hours_news_count"] = daily_market_counts["market_hours_news_count"].astype(int)
    daily_market_counts["outside_market_hours_news_count"] = (
        daily_market_counts["total_news_count"] - daily_market_counts["market_hours_news_count"]
    )

    daily_news_stats = daily_market_counts.merge(daily_histogram, on="market_date", how="left")

    histogram_columns = [f"score_{i}_count" for i in range(11)]
    daily_news_stats["sentiment_score_histogram"] = daily_news_stats[histogram_columns].apply(
        lambda row: ", ".join(f"{i}:{int(row[f'score_{i}_count'])}" for i in range(11) if int(row[f"score_{i}_count"]) > 0),
        axis=1,
    )

    args.daily_news_stats_output.parent.mkdir(parents=True, exist_ok=True)
    daily_news_stats.sort_values("market_date").to_csv(args.daily_news_stats_output, index=False)

    print(f"Wrote {len(extracted):,} timestamped rows to: {args.output}")
    print(f"Wrote {len(daily):,} daily rows to: {args.daily_output}")
    print(f"Wrote {len(daily_news_stats):,} daily news stats rows to: {args.daily_news_stats_output}")


if __name__ == "__main__":
    main()
