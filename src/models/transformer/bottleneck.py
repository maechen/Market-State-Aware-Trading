"""
Linear bottleneck (no activation) for compact latent states used by the regime head and optional RL.

The original design used a tanh activation to bound z to (-1, 1), motivated by compressing a
large d_model=128 transformer into a small RL state.  With the model now at d_model=64, the tanh
is not needed for compression and actively hurts training: it saturates for large z_pre values,
killing gradients that flow back through the regime head.  The linear-only projection retains the
dimensionality reduction (d_model → d_z) without introducing saturation.

Both z and z_pre are returned for backward compatibility with code that reads `use_pre_tanh_z`;
they are now identical (the pre-activation IS the activation).
"""

import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    """
    map pooled hidden state to a low-dimensional latent via a plain linear projection.
    :param d_model: input dimension (transformer hidden size)
    :param d_z: output dimension (latent dimension for DRL and regime head, e.g. 16)
    """

    def __init__(self, d_model: int, d_z: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, d_z)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        apply linear projection (no activation) to the pooled hidden state.
        :param h: pooled hidden state tensor, shape (batch, d_model)
        :return:
          z:     linear projection, shape (batch, d_z); used by regime head and DRL
          z_pre: identical to z (kept for API compatibility with use_pre_tanh_z flag)
        """
        z = self.linear(h)
        return z, z
