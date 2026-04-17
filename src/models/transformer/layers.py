"""
Core transformer building blocks.

All attention and normalization use PyTorch built-ins (nn.MultiheadAttention,
nn.LayerNorm) for correctness, hardware acceleration, and consistent
initialization (Xavier uniform via PyTorch defaults).

Pre-LN (norm_first) convention is followed throughout, matching
transformer.py's TransformerBlock pattern.
"""

import math

import torch
import torch.nn as nn

from .config import ReadoutMode


# ── Causal mask ───────────────────────────────────────────────────────────────

def make_causal_mask(W: int, device: torch.device) -> torch.Tensor:
    """
    Returns a (W, W) boolean attn_mask for nn.MultiheadAttention.

    PyTorch convention for bool attn_mask:
        True  = BLOCKED (position is masked out / set to -inf)
        False = ATTEND

    The upper triangle (j > i) is True so that position i can only attend
    to positions 0..i. This makes the last-step hidden state a causal summary
    of the entire sequence — the reason for using a causal mask here is
    representational, not leakage prevention (all W days are in the past at
    inference time).
    """
    return torch.triu(torch.ones(W, W, device=device), diagonal=1).bool()


# ── Positional Encoding ───────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al. 2017).
    PE buffer pattern from model.py: register_buffer with batch dimension.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, W, d_model) → (B, W, d_model)"""
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


# ── Transformer Block ─────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    One causal transformer block with Pre-LN residual connections.

    Pre-LN convention (from transformer.py):
        x = x + Dropout(Attn(LN(x)))
        x = x + Dropout(FFN(LN(x)))

    Uses:
        nn.MultiheadAttention — batch_first=True, fused SDPA backend
        nn.LayerNorm
        FFN: Linear → GELU → Dropout → Linear → Dropout
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x           : (B, W, d_model)
            causal_mask : (W, W) bool tensor — True = blocked
        Returns:
            x           : (B, W, d_model)
        """
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


# ── Transformer Stack ─────────────────────────────────────────────────────────

class TransformerStack(nn.Module):
    """N × TransformerBlock followed by a final LayerNorm."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        """x: (B, W, d_model) → (B, W, d_model)"""
        for block in self.blocks:
            x = block(x, causal_mask)
        return self.norm(x)


# ── Temporal Readout ──────────────────────────────────────────────────────────

class TemporalReadout(nn.Module):
    """
    Collapses (B, W, d_model) → (B, d_model).

    LAST:      h[:, -1, :]  — default; privileged last position under causal mask
    MEAN:      h.mean(dim=1)
    ATTN_POOL: MASTER-style learned temporal attention (TemporalAttention pattern):
                   proj  = W_q(h)                        (B, W, d_model)
                   query = proj[:, -1, :].unsqueeze(-1)  (B, d_model, 1)
                   lam   = softmax(proj @ query, dim=1)  (B, W, 1)
                   out   = (lam.transpose(1,2) @ h).squeeze(1)
    """

    def __init__(self, d_model: int, mode: ReadoutMode = ReadoutMode.LAST) -> None:
        super().__init__()
        self.mode = mode
        if mode == ReadoutMode.ATTN_POOL:
            self.W_q = nn.Linear(d_model, d_model, bias=False)
            nn.init.xavier_uniform_(self.W_q.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, W, d_model) → (B, d_model)"""
        if self.mode == ReadoutMode.LAST:
            return h[:, -1, :]
        elif self.mode == ReadoutMode.MEAN:
            return h.mean(dim=1)
        else:  # ATTN_POOL
            proj = self.W_q(h)                              # (B, W, d_model)
            query = proj[:, -1, :].unsqueeze(-1)            # (B, d_model, 1)
            scores = torch.bmm(proj, query).squeeze(-1)     # (B, W)
            weights = torch.softmax(scores, dim=1).unsqueeze(1)  # (B, 1, W)
            return torch.bmm(weights, h).squeeze(1)         # (B, d_model)
