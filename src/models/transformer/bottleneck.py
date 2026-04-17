"""Bottleneck projection: d_model → d_z via Linear → Tanh."""

import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    """
    Linear(d_model, d_z) → Tanh → z

    Returns both:
        z     (B, d_z)  — post-Tanh, bounded in (-1, 1), used by all heads
        z_pre (B, d_z)  — pre-Tanh, unbounded, available for the RL stage

    The supervised training loss always operates on z. The RL observation
    is selected by config.use_pre_tanh_z: if True, the RL agent receives
    z_pre; otherwise it receives z.

    If z_pre norms collapse near 0 during training, consider widening the
    bottleneck or switching to LayerNorm → Linear → SiLU as an ablation.
    """

    def __init__(self, d_model: int, d_z: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, d_z)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h : (B, d_model) — pooled hidden state from TemporalReadout
        Returns:
            z     : (B, d_z) — bounded latent state
            z_pre : (B, d_z) — unbounded pre-activation
        """
        z_pre = self.linear(h)
        z = torch.tanh(z_pre)
        return z, z_pre
