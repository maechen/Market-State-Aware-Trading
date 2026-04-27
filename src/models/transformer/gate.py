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
    Cross-attention gate: sentiment attends over the price/tech sequence to produce
    a global context vector, which is then added to each timestep's projected price
    features.  This preserves the full temporal structure of the price sequence while
    injecting sentiment-derived information at every position.

    Forward pass:
        1. q_proj(sent_last)  → single sentiment query   (B, 1, d_model)
        2. kv_proj(price_tech) → price keys/values        (B, W, d_model)
        3. cross-attn(Q, K, V) → sentiment context        (B, 1, d_model)
        4. price_proj(price_tech) → price residual        (B, W, d_model)
        5. output = price_residual + broadcast(ctx)       (B, W, d_model)

    Step 5 ensures that each timestep carries its own price content (step 4) plus
    a shared sentiment-weighted summary of the entire price history (step 3).  The
    original design (expand-only) collapsed all temporal information into a single
    vector; the additive residual corrects this.

    :param d_sent: sentiment dimension for query projection
    :param d_feat: price/tech feature dimension for key/value and residual projections
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
        self.q_proj = nn.Linear(d_sent, d_model)    # sentiment → query
        self.kv_proj = nn.Linear(d_feat, d_model)   # price → key/value
        self.price_proj = nn.Linear(d_feat, d_model) # price → residual (temporal path)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        for proj in (self.q_proj, self.kv_proj, self.price_proj):
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(self, price_tech: torch.Tensor, sent_last: torch.Tensor) -> torch.Tensor:
        """
        Produce a (B, W, d_model) sequence that retains full price temporal structure
        augmented by a sentiment-derived global context.
        :param price_tech: tensor (batch, W, d_feat)
        :param sent_last: tensor (batch, d_sent) — last-step sentiment
        :return: tensor (batch, W, d_model)
        """
        Q = self.q_proj(sent_last).unsqueeze(1)            # (B, 1, d_model)
        KV = self.kv_proj(price_tech)                      # (B, W, d_model)
        ctx, _ = self.mha(Q, KV, KV, need_weights=False)  # (B, 1, d_model)
        price_h = self.price_proj(price_tech)              # (B, W, d_model)
        return price_h + ctx                               # broadcast over W → (B, W, d_model)
