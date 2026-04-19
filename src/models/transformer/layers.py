"""
Causal transformer stack, sinusoidal positional encoding, and temporal readout helpers.
"""

import math

import torch
import torch.nn as nn

from .config import ReadoutMode


def make_causal_mask(W: int, device: torch.device) -> torch.Tensor:
    """
    build attention mask so position i attends only to j <= i (lower triangle unmasked)
    :param W: sequence length (window size)
    :param device: torch device for the mask tensor
    :return: tensor of shape (W, W), dtype bool; True means masked (blocked), False means attend
    """
    return torch.triu(torch.ones(W, W, device=device), diagonal=1).bool()


class PositionalEncoding(nn.Module):
    """
    add fixed sinusoidal PE to token embeddings 
    dropout after adding
    :param d_model: model dimension
    :param dropout: dropout probability after adding PE
    :param max_len: precomputed PE rows (must be >= any W used)
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
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        add PE to x and apply dropout
        :param x: tensor (batch, W, d_model)
        :return: tensor (batch, W, d_model)
        """
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """
    one causal self-attention block with pre-LayerNorm and GELU FFN (residual connections).
    :param d_model: hidden size
    :param n_heads: attention heads
    :param d_ff: FFN inner dimension
    :param dropout: dropout for attention and FFN
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
        self-attention with causal mask, then position-wise FFN
        both with residual add
        :param x: tensor (batch, W, d_model)
        :param causal_mask: bool tensor (W, W), True where attention is blocked
        :return: tensor (batch, W, d_model)
        """
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerStack(nn.Module):
    """
    stack of causal TransformerBlocks followed by final LayerNorm
    :param d_model: hidden size
    :param n_heads: attention heads per block
    :param d_ff: FFN inner size per block
    :param n_layers: number of blocks
    :param dropout: dropout probability
    """

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
        """
        run the stack with the same causal mask at every layer
        :param x: tensor (batch, W, d_model)
        :param causal_mask: bool tensor (W, W) for self-attention
        :return: tensor (batch, W, d_model)
        """
        for block in self.blocks:
            x = block(x, causal_mask)
        return self.norm(x)


class TemporalReadout(nn.Module):
    """
    pool (batch, W, d_model) down to (batch, d_model) for the bottleneck
    :param d_model: hidden size (used for ATTN_POOL linear)
    :param mode: LAST (h[:, -1]), MEAN, or ATTN_POOL (softmax weights over time)
    """

    def __init__(self, d_model: int, mode: ReadoutMode = ReadoutMode.LAST) -> None:
        super().__init__()
        self.mode = mode
        if mode == ReadoutMode.ATTN_POOL:
            self.W_q = nn.Linear(d_model, d_model, bias=False)
            nn.init.xavier_uniform_(self.W_q.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        apply LAST, MEAN, or ATTN_POOL pooling over the time dimension
        :param h: tensor (batch, W, d_model)
        :return: tensor (batch, d_model)
        """
        if self.mode == ReadoutMode.LAST:
            return h[:, -1, :]
        elif self.mode == ReadoutMode.MEAN:
            return h.mean(dim=1)
        else:
            proj = self.W_q(h)
            query = proj[:, -1, :].unsqueeze(-1)
            scores = torch.bmm(proj, query).squeeze(-1)
            weights = torch.softmax(scores, dim=1).unsqueeze(1)
            return torch.bmm(weights, h).squeeze(1)
