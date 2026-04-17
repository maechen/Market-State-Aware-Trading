"""Multi-task prediction heads that operate on the 32D latent state z."""

import torch
import torch.nn as nn


class DirectionHead(nn.Module):
    """
    3-class next-day direction head: Bear (0) / Neutral (1) / Bull (2).
    Returns raw logits — CrossEntropyLoss with label_smoothing is applied
    externally in model.compute_loss().
    """

    def __init__(self, d_z: int, n_classes: int = 3) -> None:
        super().__init__()
        self.linear = nn.Linear(d_z, n_classes)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, d_z) → logits: (B, n_classes)"""
        return self.linear(z)


class RegimeHead(nn.Module):
    """
    HMM regime classification head: states 0–3 (matching GHMM best_n=4).
    Returns raw logits.
    """

    def __init__(self, d_z: int, n_classes: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(d_z, n_classes)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, d_z) → logits: (B, n_classes)"""
        return self.linear(z)


class ReturnHead(nn.Module):
    """
    Next-day standardized return regression head.
    Trained with Huber loss (smooth L1) to be robust to outlier days.
    Returns a scalar prediction per sample.
    """

    def __init__(self, d_z: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_z, 1)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, d_z) → ret_pred: (B, 1)"""
        return self.linear(z)
