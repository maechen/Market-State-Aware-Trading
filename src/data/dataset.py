"""
SPYWindowDataset — sliding-window dataset built from one fold split.

Each sample is a W-day feature window plus two aligned labels for the
last day of that window:
    y_dir : n-day forward direction (0=Down, 1=Up, -1=Neutral/ignored)
    y_reg : HMM regime label at the last window day (0..K-1, K=4)

Label alignment note:
    For window index idx (rows [idx, idx+W)), both labels are derived
    from row idx+W-1 (the last day of the window).

    dir_labels[idx+W-1] is a forward-looking quantity computed by
    _fwd_return() in fold_loader.py using dir_n_forward (default 5)
    trading days ahead.  The last dir_n_forward rows have NaN labels
    (no complete forward window) and are replaced with 0 via nan_to_num
    — negligible contamination.

IMPORTANT — Viterbi look-ahead within training split:
    Regime labels on training rows are Viterbi-decoded over the entire
    training window, which introduces look-ahead bias within that window.
    Val/test labels are decoded using a model fit on prior data only and
    are genuinely out-of-sample. This is a known limitation shared with
    all HMM labeling pipelines and must be acknowledged in the final report.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class SPYWindowDataset(Dataset):
    """
    Args:
        features      : (n_rows, d_feat+d_sent) float32 — normalized features
        regime_labels : (n_rows,) int64 — HMM state per row
        dir_labels    : (n_rows,) int64 — n-day forward direction label per row.
                        In vol_threshold mode, -1 means neutral/no-event and is
                        ignored by direction CE.
        window_size   : W (number of consecutive days per sample)
    """

    def __init__(
        self,
        features: np.ndarray,
        regime_labels: np.ndarray,
        dir_labels: np.ndarray,
        window_size: int = 20,
    ) -> None:
        self.features = features
        self.regime_labels = regime_labels
        self.dir_labels = dir_labels
        self.W = window_size
        self.n = len(features) - self.W

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        """
        Returns:
            x     : (W, n_feat) float32 tensor — feature window
            y_dir : int — direction label at last window day (0=Down, 1=Up, -1=ignored)
            y_reg : int — regime label at last window day (0..K-1)
        """
        x = torch.from_numpy(self.features[idx : idx + self.W])  # (W, n_feat)
        last = idx + self.W - 1
        y_reg = int(self.regime_labels[last])
        y_dir = int(self.dir_labels[last])
        return x, y_dir, y_reg
