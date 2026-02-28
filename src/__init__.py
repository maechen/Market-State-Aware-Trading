"""
Market State Aware Trading source package
"""

# re-exporting regime detection API for convenience
from src.regimes import (
    RegimeGHMM,
    build_observation_matrix,
    satisfies_persistence,
    select_and_fit_ghmm,
)

__all__ = [
    "RegimeGHMM",
    "build_observation_matrix",
    "satisfies_persistence",
    "select_and_fit_ghmm",
]
