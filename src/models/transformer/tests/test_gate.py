"""Tests for SentimentGate and CrossAttentionGate."""

import pytest
import torch
from src.models.transformer.gate import SentimentGate, CrossAttentionGate

B, W, D_FEAT, D_SENT, D_MODEL, N_HEADS = 4, 20, 16, 3, 64, 4


# ── SentimentGate ─────────────────────────────────────────────────────────────

def _make_sentiment_gate():
    return SentimentGate(d_sent=D_SENT, d_feat=D_FEAT, beta=1.0)


def test_sentiment_gate_output_shape():
    gate = _make_sentiment_gate()
    price_tech = torch.randn(B, W, D_FEAT)
    sent_seq = torch.randn(B, W, D_SENT)
    out = gate(price_tech, sent_seq)
    assert out.shape == (B, W, D_FEAT)


def test_sentiment_gate_output_is_nonnegative():
    """Gate weights are d_feat × softmax(…), so all values must be ≥ 0."""
    gate = _make_sentiment_gate()
    price_tech = torch.ones(B, W, D_FEAT)
    sent_seq = torch.randn(B, W, D_SENT)
    out = gate(price_tech, sent_seq)
    assert (out >= 0).all()


def test_sentiment_gate_weights_sum_to_d_feat():
    """
    gate_w = d_feat × softmax(…) sums to d_feat along the last dim.
    When price_tech is all-ones, the output at each timestep equals the gate
    weights, so each row should sum to d_feat.
    """
    gate = _make_sentiment_gate()
    price_tech = torch.ones(B, W, D_FEAT)
    sent_seq = torch.randn(B, W, D_SENT)
    out = gate(price_tech, sent_seq)
    row_sums = out[:, 0, :].sum(dim=-1)  # sum over features at first timestep
    expected = torch.full((B,), float(D_FEAT))
    assert torch.allclose(row_sums, expected, atol=1e-5)


def test_sentiment_gate_different_last_sentiment_produces_different_output():
    """Only the last timestep's sentiment affects output (MASTER convention)."""
    gate = _make_sentiment_gate()
    price_tech = torch.ones(B, W, D_FEAT)
    # Same early timesteps, different last step → different gating
    sent1 = torch.zeros(B, W, D_SENT)
    sent2 = torch.zeros(B, W, D_SENT)
    sent2[:, -1, :] = 1.0
    out1 = gate(price_tech, sent1)
    out2 = gate(price_tech, sent2)
    assert not torch.allclose(out1, out2)


def test_sentiment_gate_only_last_step_matters():
    """Changing early (non-last) sentiment timesteps must NOT change the output."""
    gate = _make_sentiment_gate()
    price_tech = torch.ones(B, W, D_FEAT)
    sent1 = torch.zeros(B, W, D_SENT)
    sent2 = torch.zeros(B, W, D_SENT)
    sent2[:, :-1, :] = torch.randn(B, W - 1, D_SENT)  # only early steps differ
    out1 = gate(price_tech, sent1)
    out2 = gate(price_tech, sent2)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_sentiment_gate_broadcasts_over_time():
    """All W timesteps receive the same gating (gate is time-independent)."""
    gate = _make_sentiment_gate()
    price_tech = torch.ones(B, W, D_FEAT)
    sent_seq = torch.randn(B, W, D_SENT)
    out = gate(price_tech, sent_seq)
    # Since price_tech is all-ones, out[:, t, :] should be identical for all t
    for t in range(1, W):
        assert torch.allclose(out[:, 0, :], out[:, t, :], atol=1e-6)


def test_sentiment_gate_gradient_flows():
    gate = _make_sentiment_gate()
    price_tech = torch.randn(B, W, D_FEAT, requires_grad=True)
    sent_seq = torch.randn(B, W, D_SENT, requires_grad=True)
    out = gate(price_tech, sent_seq)
    loss = out.sum()
    loss.backward()
    assert price_tech.grad is not None
    assert sent_seq.grad is not None


# ── CrossAttentionGate ────────────────────────────────────────────────────────

def _make_cross_attn_gate():
    return CrossAttentionGate(
        d_sent=D_SENT, d_feat=D_FEAT, d_model=D_MODEL, n_heads=N_HEADS
    )


def test_cross_attn_gate_output_shape():
    """Output is (B, W, D_MODEL) — one sentiment-reweighted price vector per timestep."""
    gate = _make_cross_attn_gate()
    price_tech = torch.randn(B, W, D_FEAT)
    sent_seq = torch.randn(B, W, D_SENT)
    out = gate(price_tech, sent_seq)
    assert out.shape == (B, W, D_MODEL)


def test_cross_attn_gate_different_sentiment_produces_different_output():
    """Different sentiment token sequences should produce different reweightings."""
    gate = _make_cross_attn_gate()
    price_tech = torch.randn(B, W, D_FEAT)
    sent1 = torch.zeros(B, W, D_SENT)
    sent2 = torch.ones(B, W, D_SENT) * 5.0
    out1 = gate(price_tech, sent1)
    out2 = gate(price_tech, sent2)
    assert not torch.allclose(out1, out2)


def test_cross_attn_gate_different_price_produces_different_output():
    """Different price sequences should produce different outputs (keys/values change)."""
    gate = _make_cross_attn_gate()
    sent_seq = torch.randn(B, W, D_SENT)
    price1 = torch.zeros(B, W, D_FEAT)
    price2 = torch.ones(B, W, D_FEAT) * 5.0
    out1 = gate(price1, sent_seq)
    out2 = gate(price2, sent_seq)
    assert not torch.allclose(out1, out2)


def test_cross_attn_gate_each_timestep_can_differ():
    """With different sentiment at each timestep, outputs should vary across W."""
    gate = _make_cross_attn_gate()
    gate.eval()
    price_tech = torch.randn(B, W, D_FEAT)
    sent_seq = torch.randn(B, W, D_SENT)  # distinct sentiment per timestep
    with torch.no_grad():
        out = gate(price_tech, sent_seq)
    # Not all timestep outputs should be identical (unlike SentimentGate broadcast)
    all_same = all(torch.allclose(out[:, 0, :], out[:, t, :], atol=1e-5) for t in range(1, W))
    assert not all_same, "CrossAttentionGate should produce distinct outputs per timestep"


def test_cross_attn_gate_gradient_flows():
    gate = _make_cross_attn_gate()
    price_tech = torch.randn(B, W, D_FEAT, requires_grad=True)
    sent_seq = torch.randn(B, W, D_SENT, requires_grad=True)
    out = gate(price_tech, sent_seq)
    loss = out.sum()
    loss.backward()
    assert price_tech.grad is not None
    assert sent_seq.grad is not None
