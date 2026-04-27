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

    def forward(self, price_tech: torch.Tensor, sent_seq: torch.Tensor) -> torch.Tensor:
        """
        Element-wise multiply price/tech by d_feat * softmax(Linear(sent_last) / beta) broadcast over time.
        Uses only the last timestep's sentiment (MASTER paper convention).
        :param price_tech: tensor (batch, W, d_feat)
        :param sent_seq:   tensor (batch, W, d_sent) — full sentiment window; only last step is used
        :return: gated tensor (batch, W, d_feat)
        """
        sent_last = sent_seq[:, -1, :]  # (B, d_sent) — last-step sentiment only
        gate_w = self.d_feat * F.softmax(self.linear(sent_last) / self.beta, dim=-1)
        return price_tech * gate_w.unsqueeze(1)


class CrossAttentionGate(nn.Module):
    """
    Cross-attention gate: each sentiment token in the window acts as a query to
    reweight price/technical features across the same window.

    This matches the project proposal's description: "cross-attention where sentiment
    tokens act as a query to reweight price/technical features."  The full sentiment
    sequence (not just the last step) is projected into query space, while the
    price/tech sequence forms the keys and values.  Multi-head attention then produces,
    at every timestep t, a sentiment-weighted linear combination of all price/tech
    vectors — a true per-timestep reweighting of price features by sentiment context.

    Forward pass:
        1. q_proj(sent_seq)    → sentiment queries  (B, W, d_model)
        2. kv_proj(price_tech) → price keys/values  (B, W, d_model)
        3. cross-attn(Q, K, V) → reweighted output  (B, W, d_model)

    At each position t the output is a learned weighted sum over all price timesteps,
    where the weights are determined by how much sentiment_t aligns with each
    price_j — a proper feature-level reweighting rather than additive fusion.

    :param d_sent: sentiment dimension (e.g. 3 for Neg/Neu/Pos scores)
    :param d_feat: price/tech feature dimension for key/value projections
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
        self.q_proj = nn.Linear(d_sent, d_model)    # sentiment tokens → queries
        self.kv_proj = nn.Linear(d_feat, d_model)   # price/tech → keys & values
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        for proj in (self.q_proj, self.kv_proj):
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(self, price_tech: torch.Tensor, sent_seq: torch.Tensor) -> torch.Tensor:
        """
        Reweight price/tech features at every timestep using sentiment queries.
        :param price_tech: tensor (batch, W, d_feat) — price and technical features
        :param sent_seq:   tensor (batch, W, d_sent) — sentiment scores for every timestep
        :return: tensor (batch, W, d_model) — sentiment-reweighted price representation
        """
        Q = self.q_proj(sent_seq)                          # (B, W, d_model)
        KV = self.kv_proj(price_tech)                      # (B, W, d_model)
        reweighted, _ = self.mha(Q, KV, KV, need_weights=False)  # (B, W, d_model)
        return reweighted
