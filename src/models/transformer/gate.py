"""
Sentiment-driven gating: MASTER-style feature scaling or cross-attention into model space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SentimentGate(nn.Module):
    """
    MASTER style gating produces a soft feature-weighting vector from the
    last-step sentiment and applies it element-wise to every price/tech timestep.
    :param d_sent: sentiment embedding dimension (e.g. 3 for Neg/Neu/Pos)
    :param d_feat: number of price/technical channels to gate
    :param beta: softmax temperature (larger beta => softer, more uniform weights)
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
        Element-wise multiply price/tech by d_feat * softmax(Linear(sent_last) / beta) broadcast over time.
        :param price_tech: tensor (batch, W, d_feat)
        :param sent_last: sentiment at last window timestep, shape (batch, d_sent)
        :return: gated tensor (batch, W, d_feat)
        """
        gate_w = self.d_feat * F.softmax(self.linear(sent_last) / self.beta, dim=-1)
        return price_tech * gate_w.unsqueeze(1)


class CrossAttentionGate(nn.Module):
    """
    cross-attention gate from proposal
    query from sentiment
    keys/values from price/tech sequence
    output is d_model space
    :param d_sent: sentiment dimension for query projection
    :param d_feat: price/tech feature dimension for key/value projection
    :param d_model: transformer hidden size (embed_dim for MultiheadAttention)
    :param n_heads: number of attention heads
    :param dropout: dropout on attention weights
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
        Project sentiment to one query token, price/tech to keys/values; expand cross-attn context to length W.
        :param price_tech: tensor (batch, W, d_feat)
        :param sent_last: tensor (batch, d_sent)
        :return: tensor (batch, W, d_model) ready for positional encoding (no separate Linear in model)
        """
        W = price_tech.size(1)
        Q = self.q_proj(sent_last).unsqueeze(1)   # (B, 1, d_model)
        KV = self.kv_proj(price_tech)              # (B, W, d_model)
        ctx, _ = self.mha(Q, KV, KV, need_weights=False)  # (B, 1, d_model)
        return ctx.expand(-1, W, -1)               # (B, W, d_model)
