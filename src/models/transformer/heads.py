"""
Classification and regression heads on the bottleneck latent vector z.
"""

import torch
import torch.nn as nn


class DirectionHead(nn.Module):
    """
    n-day forward direction logits (binary Up/Down by default).

    When hidden > 0 the head uses a two-layer MLP
    (Linear → GELU → Dropout → Linear) for greater expressivity.
    When hidden == 0 it degrades to a single linear map.

    :param d_z: input dimension (d_model when use_task_specific_heads=True, else d_z)
    :param n_classes: number of direction classes (2 = binary Up/Down default, 3 = Bear/Neutral/Bull)
    :param hidden: inner hidden size of the MLP; 0 = linear-only
    :param dropout: dropout rate inside the MLP (ignored when hidden == 0)
    """

    def __init__(
        self,
        d_z: int,
        n_classes: int = 2,
        hidden: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden > 0:
            self.net: nn.Module = nn.Sequential(
                nn.Linear(d_z, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, n_classes),
            )
            nn.init.xavier_uniform_(self.net[0].weight)  # type: ignore[index]
            nn.init.zeros_(self.net[0].bias)              # type: ignore[index]
            nn.init.xavier_uniform_(self.net[3].weight)  # type: ignore[index]
            nn.init.zeros_(self.net[3].bias)              # type: ignore[index]
        else:
            self.net = nn.Linear(d_z, n_classes)
            nn.init.xavier_uniform_(self.net.weight)      # type: ignore[union-attr]
            nn.init.zeros_(self.net.bias)                 # type: ignore[union-attr]

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        map latent z to direction logits
        :param z: tensor (batch, d_z)
        :return: logits (batch, n_classes)
        """
        return self.net(z)


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
    scalar prediction of standardized n-day forward log return (median, τ=0.5).
    :param d_z: input dimension (d_model when use_task_specific_heads=True, else d_z)
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
