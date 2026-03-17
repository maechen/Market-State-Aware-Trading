"""
Market regime detection with GHMM on log returns and rolling volatility.

Fit on (log_return, rolling_vol_20d).
Outputs regime labels and posteriors.
BIC and persistence filtering for model selection.
"""

from .ghmm_selection import select_and_fit_ghmm, satisfies_persistence

__all__ = [
    "select_and_fit_ghmm",
    "satisfies_persistence",
]
