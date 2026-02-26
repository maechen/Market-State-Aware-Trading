"""
Market regime detection with GHMM on log returns and rolling volatility.

Fit on (log_return, rolling_vol_20d).
Outputs regime labels and posteriors.
BIC and persistence filtering for model selection.
"""

from .ghmm import RegimeGHMM, build_observation_matrix
from .selection import select_and_fit_ghmm, satisfies_persistence

__all__ = [
    "RegimeGHMM",
    "build_observation_matrix",
    "select_and_fit_ghmm",
    "satisfies_persistence",
]
