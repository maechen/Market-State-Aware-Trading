"""Tests for SentimentGate and CrossAttentionGate."""

import pytest
import torch
from src.models.transformer.gate import SentimentGate, CrossAttentionGate

B, W, D_FEAT, D_SENT, D_MODEL, N_HEADS = 4, 20, 7, 3, 128, 4


# ── SentimentGate ─────────────────────────────────────────────────────────────

def _make_sentiment_gate():
    return SentimentGate(d_sent=D_SENT, d_feat=D_FEAT, beta=1.0)


def test_sentiment_gate_output_shape():
    gate = _make_sentiment_gate()
    price_tech = torch.randn(B, W, D_FEAT)
    sent_last = torch.randn(B, D_SENT)
    out = gate(price_tech, sent_last)
    assert out.shape == (B, W, D_FEAT)


def test_sentiment_gate_output_is_nonnegative():
    """Gate weights are d_feat × softmax(…), so all values must be ≥ 0."""
    gate = _make_sentiment_gate()
    price_tech = torch.ones(B, W, D_FEAT)
    sent_last = torch.randn(B, D_SENT)
    out = gate(price_tech, sent_last)
    assert (out >= 0).all()


def test_sentiment_gate_weights_sum_to_d_feat():
    """
    gate_w = d_feat × softmax(…) sums to d_feat along the last dim.
    When price_tech is all-ones, the output at each timestep equals the gate
    weights, so each row should sum to d_feat.
    """
    gate = _make_sentiment_gate()
    price_tech = torch.ones(B, W, D_FEAT)
    sent_last = torch.randn(B, D_SENT)
    out = gate(price_tech, sent_last)
    row_sums = out[:, 0, :].sum(dim=-1)  # sum over features at first timestep
    expected = torch.full((B,), float(D_FEAT))
    assert torch.allclose(row_sums, expected, atol=1e-5)


def test_sentiment_gate_different_sentiment_produces_different_output():
    gate = _make_sentiment_gate()
    price_tech = torch.ones(B, W, D_FEAT)
    sent1 = torch.zeros(B, D_SENT)
    sent2 = torch.ones(B, D_SENT)
    out1 = gate(price_tech, sent1)
    out2 = gate(price_tech, sent2)
    assert not torch.allclose(out1, out2)


def test_sentiment_gate_broadcasts_over_time():
    """All W timesteps receive the same gating (gate is time-independent)."""
    gate = _make_sentiment_gate()
    price_tech = torch.ones(B, W, D_FEAT)
    sent_last = torch.randn(B, D_SENT)
    out = gate(price_tech, sent_last)
    # Since price_tech is all-ones, out[:, t, :] should be identical for all t
    for t in range(1, W):
        assert torch.allclose(out[:, 0, :], out[:, t, :], atol=1e-6)


def test_sentiment_gate_gradient_flows():
    gate = _make_sentiment_gate()
    price_tech = torch.randn(B, W, D_FEAT, requires_grad=True)
    sent_last = torch.randn(B, D_SENT, requires_grad=True)
    out = gate(price_tech, sent_last)
    loss = out.sum()
    loss.backward()
    assert price_tech.grad is not None
    assert sent_last.grad is not None


# ── CrossAttentionGate ────────────────────────────────────────────────────────

def _make_cross_attn_gate():
    return CrossAttentionGate(
        d_sent=D_SENT, d_feat=D_FEAT, d_model=D_MODEL, n_heads=N_HEADS
    )


def test_cross_attn_gate_output_shape():
    gate = _make_cross_attn_gate()
    price_tech = torch.randn(B, W, D_FEAT)
    sent_last = torch.randn(B, D_SENT)
    out = gate(price_tech, sent_last)
    assert out.shape == (B, W, D_MODEL)


def test_cross_attn_gate_different_sentiment_produces_different_output():
    gate = _make_cross_attn_gate()
    price_tech = torch.randn(B, W, D_FEAT)
    sent1 = torch.zeros(B, D_SENT)
    sent2 = torch.ones(B, D_SENT) * 5.0
    out1 = gate(price_tech, sent1)
    out2 = gate(price_tech, sent2)
    assert not torch.allclose(out1, out2)


def test_cross_attn_gate_gradient_flows():
    gate = _make_cross_attn_gate()
    price_tech = torch.randn(B, W, D_FEAT, requires_grad=True)
    sent_last = torch.randn(B, D_SENT, requires_grad=True)
    out = gate(price_tech, sent_last)
    loss = out.sum()
    loss.backward()
    assert price_tech.grad is not None
    assert sent_last.grad is not None
