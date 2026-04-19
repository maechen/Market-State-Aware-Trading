"""
Linear bottleneck with Tanh for bounded latent states used by multitask heads and optional RL.
"""

import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    """
    map pooled hidden state to a low-dimensional latent via linear then tanh.
    :param d_model: input dimension (transformer hidden size)
    :param d_z: output dimension (latent bottleneck size, e.g. 32)
    """

    def __init__(self, d_model: int, d_z: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, d_z)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        apply linear then tanh
        expose both bounded and pre-activation vectors for heads and RL
        :param h: pooled hidden state tensor, shape (batch, d_model)
        :return:
          z: post-Tanh latent, shape (batch, d_z), in (-1, 1), used by supervised heads
          z_pre: pre-Tanh linear output, shape (batch, d_z), optional RL observation if use_pre_tanh_z
        """
        z_pre = self.linear(h)
        z = torch.tanh(z_pre)
        return z, z_pre
