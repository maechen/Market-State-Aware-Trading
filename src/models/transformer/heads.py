"""
Linear classification and regression heads on the bottleneck latent vector z.
"""

import torch
import torch.nn as nn


class DirectionHead(nn.Module):
    """
    next-day direction logits (bear / neutral / bull)
    :param d_z: bottleneck dimension
    :param n_classes: number of direction classes (default 3)
    """

    def __init__(self, d_z: int, n_classes: int = 3) -> None:
        super().__init__()
        self.linear = nn.Linear(d_z, n_classes)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        map latent z to direction logits
        :param z: tensor (batch, d_z)
        :return: logits (batch, n_classes)
        """
        return self.linear(z)


class RegimeHead(nn.Module):
    """
    HMM regime logits (states 0..K-1)
    K should match GHMM hidden state count used for labels
    :param d_z: bottleneck dimension
    :param n_classes: number of regime classes (default 4 for GHMM with best_n=4)
    """

    def __init__(self, d_z: int, n_classes: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(d_z, n_classes)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        map latent z to regime logits
        :param z: tensor (batch, d_z)
        :return: logits (batch, n_classes)
        """
        return self.linear(z)


class ReturnHead(nn.Module):
    """
    scalar prediction of standardized next-day log return
    :param d_z: bottleneck dimension
    """

    def __init__(self, d_z: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_z, 1)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        map latent z to one return value per sample
        :param z: tensor (batch, d_z)
        :return: predictions (batch, 1)
        """
        return self.linear(z)
