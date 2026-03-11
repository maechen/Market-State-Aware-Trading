"""
Market State Aware Trading source package
"""

# re-exporting regime detection API for convenience
from src.regimes import (
    satisfies_persistence,
    select_and_fit_ghmm,
)

__all__ = [
    "satisfies_persistence",
    "select_and_fit_ghmm",
]
