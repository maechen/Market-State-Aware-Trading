"""
Sentiment gating modules.

SentimentGate  — default low-capacity gate (MASTER-style).
CrossAttentionGate — proposal-faithful cross-attention gate (Ablation A).

Neither module changes the number of input channels; only the CrossAttentionGate
projects into d_model space (the feature_proj layer in model.py is skipped when
this gate is active).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SentimentGate(nn.Module):
    """
    MASTER-style gating: produces a soft feature-weighting vector from the
    last-step sentiment and applies it element-wise to every price/tech timestep.

        gate_w = d_feat × softmax(Linear(d_sent, d_feat)(sent) / beta)
        out    = price_tech * gate_w.unsqueeze(1)          (B, W, d_feat)

    This is NOT the same as the proposal's cross-attention gate. It is the
    low-capacity default, appropriate for small single-asset datasets where
    overfitting is a primary risk.

    Adapted from: Li et al. (2024) MASTER.
    """

    def __init__(self, d_sent: int, d_feat: int, beta: float = 1.0) -> None:
        super().__init__()
        self.d_feat = d_feat
        self.beta = beta
        self.linear = nn.Linear(d_sent, d_feat)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, price_tech: torch.Tensor, sent_last: torch.Tensor) -> torch.Tensor:
        """
        Args:
            price_tech : (B, W, d_feat)
            sent_last  : (B, d_sent) — sentiment at the last window timestep
        Returns:
            gated      : (B, W, d_feat)
        """
        gate_w = self.d_feat * F.softmax(self.linear(sent_last) / self.beta, dim=-1)
        return price_tech * gate_w.unsqueeze(1)


class CrossAttentionGate(nn.Module):
    """
    Proposal-faithful gate: Q = f(sentiment), K/V = price/tech sequence.
    Uses nn.MultiheadAttention in cross-attention mode (separate Q, K, V).

    The sentiment vector at the last timestep is projected into query space.
    The price/tech sequence is projected into key/value space. The attended
    context is expanded back to (B, W, d_model) so the feature_proj step is
    skipped in model.py.

        Q   = Linear(d_sent, d_model)(sent_last).unsqueeze(1)   (B, 1, d_model)
        K,V = Linear(d_feat, d_model)(price_tech)               (B, W, d_model)
        ctx = MHA(Q, K, V)                                       (B, 1, d_model)
        out = ctx.expand(-1, W, -1)                              (B, W, d_model)

    Enabled via config.gate_mode = GateMode.CROSS_ATTN.
    """

    def __init__(
        self,
        d_sent: int,
        d_feat: int,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d_sent, d_model)
        self.kv_proj = nn.Linear(d_feat, d_model)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        nn.init.zeros_(self.kv_proj.bias)

    def forward(self, price_tech: torch.Tensor, sent_last: torch.Tensor) -> torch.Tensor:
        """
        Args:
            price_tech : (B, W, d_feat)
            sent_last  : (B, d_sent)
        Returns:
            out        : (B, W, d_model)  — already in model space
        """
        W = price_tech.size(1)
        Q = self.q_proj(sent_last).unsqueeze(1)   # (B, 1, d_model)
        KV = self.kv_proj(price_tech)              # (B, W, d_model)
        ctx, _ = self.mha(Q, KV, KV, need_weights=False)  # (B, 1, d_model)
        return ctx.expand(-1, W, -1)               # (B, W, d_model)
