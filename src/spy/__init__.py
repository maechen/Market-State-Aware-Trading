"""
SPY-specific data utilities for the Market-State-Aware-Trading project.

This package provides:
- Raw data fetching and feature engineering from yfinance (fetch_spy_data).
- Loading, slicing, and HMM feature extraction for GHMM (market_data_utils).
"""

from .fetch_spy_data import (
    START_DATE,
    OUTPUT_PATH,
    download_spy_data,
    add_features,
    finalize_dataset,
    export_to_csv,
    print_summary,
)
from .market_data_utils import (
    load_data,
    slice_data,
    create_walk_forward_split,
    get_hmm_features,
    find_missing_data_points,
)

__all__ = [
    "START_DATE",
    "OUTPUT_PATH",
    "download_spy_data",
    "add_features",
    "finalize_dataset",
    "export_to_csv",
    "print_summary",
    "load_data",
    "slice_data",
    "create_walk_forward_split",
    "get_hmm_features",
    "find_missing_data_points",
]

