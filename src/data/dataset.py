"""
SPYWindowDataset — sliding-window dataset built from one fold split.

Each sample is a W-day feature window plus three aligned labels for the
last day of that window:
    y_dir     : next-day direction (0=Bear, 1=Neutral, 2=Bull)
    y_reg     : HMM regime label at the last window day
    y_ret_std : standardized next-day log_return at the last window day

Label alignment note:
    For window index idx (rows [idx, idx+W)), all three labels are derived
    from row idx+W-1 (the last day of the window).

    dir_labels[idx+W-1] and ret_std[idx+W-1] were computed as
    shift(-1) quantities, so they reference log_return[idx+W] which must
    exist. Therefore the valid range of idx is [0, n_rows - W - 1], giving
    n_rows - W total samples.

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
        dir_labels    : (n_rows,) int64 — direction label per row
                        dir_labels[i] = direction of log_return[i+1]
        ret_std       : (n_rows,) float32 — standardized next-day return
                        ret_std[i] = (log_return[i+1] - mean) / std
        window_size   : W (number of consecutive days per sample)
    """

    def __init__(
        self,
        features: np.ndarray,
        regime_labels: np.ndarray,
        dir_labels: np.ndarray,
        ret_std: np.ndarray,
        window_size: int = 20,
    ) -> None:
        self.features = features
        self.regime_labels = regime_labels
        self.dir_labels = dir_labels
        self.ret_std = ret_std
        self.W = window_size
        # last valid last-row index: n_rows - 2 (so that label shift is in-bounds)
        self.n = len(features) - self.W

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        """
        Returns:
            x        : (W, n_feat) float32 tensor — feature window
            y_dir    : int        — direction label at last window day
            y_reg    : int        — regime label at last window day
            y_ret    : float      — standardized return at last window day
        """
        x = torch.from_numpy(self.features[idx : idx + self.W])  # (W, n_feat)
        last = idx + self.W - 1
        y_reg = int(self.regime_labels[last])
        y_dir = int(self.dir_labels[last])
        y_ret = float(self.ret_std[last])
        return x, y_dir, y_reg, y_ret
