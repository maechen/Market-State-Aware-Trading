"""Tests for PositionalEncoding, TransformerBlock, TransformerStack,
TemporalReadout, and the causal mask helper."""

import pytest
import torch
from src.models.transformer.config import ReadoutMode
from src.models.transformer.layers import (
    PositionalEncoding,
    TransformerBlock,
    TransformerStack,
    TemporalReadout,
    make_causal_mask,
)

B, W, D_MODEL, N_HEADS, D_FF = 4, 20, 128, 4, 512


# ── Causal mask ───────────────────────────────────────────────────────────────

def test_causal_mask_shape():
    mask = make_causal_mask(W, torch.device("cpu"))
    assert mask.shape == (W, W)


def test_causal_mask_dtype_is_bool():
    mask = make_causal_mask(W, torch.device("cpu"))
    assert mask.dtype == torch.bool


def test_causal_mask_lower_triangle_is_false():
    """False = attend. Lower triangle (j <= i) must all be False."""
    mask = make_causal_mask(W, torch.device("cpu"))
    for i in range(W):
        for j in range(i + 1):
            assert not mask[i, j], f"mask[{i},{j}] should be False (attend)"


def test_causal_mask_upper_triangle_is_true():
    """True = blocked. Upper triangle (j > i) must all be True."""
    mask = make_causal_mask(W, torch.device("cpu"))
    for i in range(W):
        for j in range(i + 1, W):
            assert mask[i, j], f"mask[{i},{j}] should be True (block)"


# ── PositionalEncoding ────────────────────────────────────────────────────────

def test_positional_encoding_output_shape():
    pe = PositionalEncoding(D_MODEL, dropout=0.0)
    x = torch.zeros(B, W, D_MODEL)
    out = pe(x)
    assert out.shape == (B, W, D_MODEL)


def test_positional_encoding_different_positions():
    """Sinusoidal PE must produce different vectors at different positions."""
    pe = PositionalEncoding(D_MODEL, dropout=0.0)
    x = torch.zeros(1, W, D_MODEL)
    out = pe(x)
    assert not torch.allclose(out[0, 0, :], out[0, 1, :])


def test_positional_encoding_adds_to_input():
    """With dropout=0, output should differ from all-zero input."""
    pe = PositionalEncoding(D_MODEL, dropout=0.0)
    x = torch.zeros(B, W, D_MODEL)
    out = pe(x)
    assert not torch.all(out == 0)


def test_positional_encoding_no_gradient_on_pe():
    """The PE buffer should not accumulate gradients."""
    pe = PositionalEncoding(D_MODEL, dropout=0.0)
    x = torch.randn(B, W, D_MODEL, requires_grad=True)
    out = pe(x)
    out.sum().backward()
    assert pe.pe.grad is None


# ── TransformerBlock ──────────────────────────────────────────────────────────

def _causal_mask():
    return make_causal_mask(W, torch.device("cpu"))


def test_transformer_block_output_shape():
    block = TransformerBlock(D_MODEL, N_HEADS, D_FF, dropout=0.0)
    x = torch.randn(B, W, D_MODEL)
    out = block(x, _causal_mask())
    assert out.shape == (B, W, D_MODEL)


def test_transformer_block_gradient_flows():
    block = TransformerBlock(D_MODEL, N_HEADS, D_FF, dropout=0.0)
    x = torch.randn(B, W, D_MODEL, requires_grad=True)
    out = block(x, _causal_mask())
    out.sum().backward()
    assert x.grad is not None


def test_transformer_block_causal_property():
    """
    With a causal mask, changing positions t=5..W-1 must NOT change
    the output at positions 0..4.
    """
    block = TransformerBlock(D_MODEL, N_HEADS, D_FF, dropout=0.0)
    block.eval()
    mask = _causal_mask()

    x1 = torch.randn(B, W, D_MODEL)
    x2 = x1.clone()
    x2[:, 5:, :] = torch.randn(B, W - 5, D_MODEL)

    with torch.no_grad():
        out1 = block(x1, mask)
        out2 = block(x2, mask)

    assert torch.allclose(out1[:, :5, :], out2[:, :5, :], atol=1e-5), \
        "Causal mask violated: early positions affected by future changes"


# ── TransformerStack ──────────────────────────────────────────────────────────

def test_transformer_stack_output_shape():
    stack = TransformerStack(D_MODEL, N_HEADS, D_FF, n_layers=4, dropout=0.0)
    x = torch.randn(B, W, D_MODEL)
    out = stack(x, _causal_mask())
    assert out.shape == (B, W, D_MODEL)


def test_transformer_stack_n_layers_respected():
    stack = TransformerStack(D_MODEL, N_HEADS, D_FF, n_layers=2, dropout=0.0)
    assert len(stack.blocks) == 2


def test_transformer_stack_gradient_flows():
    stack = TransformerStack(D_MODEL, N_HEADS, D_FF, n_layers=2, dropout=0.0)
    x = torch.randn(B, W, D_MODEL, requires_grad=True)
    out = stack(x, _causal_mask())
    out.sum().backward()
    assert x.grad is not None


# ── TemporalReadout ───────────────────────────────────────────────────────────

def test_temporal_readout_last_shape():
    readout = TemporalReadout(D_MODEL, ReadoutMode.LAST)
    h = torch.randn(B, W, D_MODEL)
    out = readout(h)
    assert out.shape == (B, D_MODEL)


def test_temporal_readout_last_value():
    readout = TemporalReadout(D_MODEL, ReadoutMode.LAST)
    h = torch.randn(B, W, D_MODEL)
    out = readout(h)
    assert torch.allclose(out, h[:, -1, :])


def test_temporal_readout_mean_shape():
    readout = TemporalReadout(D_MODEL, ReadoutMode.MEAN)
    h = torch.randn(B, W, D_MODEL)
    out = readout(h)
    assert out.shape == (B, D_MODEL)


def test_temporal_readout_mean_value():
    readout = TemporalReadout(D_MODEL, ReadoutMode.MEAN)
    h = torch.randn(B, W, D_MODEL)
    out = readout(h)
    assert torch.allclose(out, h.mean(dim=1), atol=1e-6)


def test_temporal_readout_attn_pool_shape():
    readout = TemporalReadout(D_MODEL, ReadoutMode.ATTN_POOL)
    h = torch.randn(B, W, D_MODEL)
    out = readout(h)
    assert out.shape == (B, D_MODEL)


def test_temporal_readout_attn_pool_gradient_flows():
    readout = TemporalReadout(D_MODEL, ReadoutMode.ATTN_POOL)
    h = torch.randn(B, W, D_MODEL, requires_grad=True)
    out = readout(h)
    out.sum().backward()
    assert h.grad is not None


def test_temporal_readout_modes_produce_different_outputs():
    """The three readout modes should produce different numerical outputs."""
    h = torch.randn(B, W, D_MODEL)
    last = TemporalReadout(D_MODEL, ReadoutMode.LAST)(h)
    mean = TemporalReadout(D_MODEL, ReadoutMode.MEAN)(h)
    # last and mean should differ (they read different parts of h)
    assert not torch.allclose(last, mean)
