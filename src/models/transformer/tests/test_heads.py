"""Tests for DirectionHead and RegimeHead."""

import torch
from src.models.transformer.heads import DirectionHead, RegimeHead

B, D_Z = 4, 32


def test_direction_head_output_shape():
    head = DirectionHead(d_z=D_Z, n_classes=3)
    z = torch.randn(B, D_Z)
    out = head(z)
    assert out.shape == (B, 3)


def test_direction_head_raw_logits_not_softmaxed():
    """Raw logits can be negative; softmax output would be positive."""
    head = DirectionHead(d_z=D_Z, n_classes=3)
    z = torch.randn(B, D_Z)
    out = head(z)
    assert (out < 0).any(), "Expected some negative raw logits"


def test_direction_head_gradient_flows():
    head = DirectionHead(d_z=D_Z, n_classes=3)
    z = torch.randn(B, D_Z, requires_grad=True)
    out = head(z)
    out.sum().backward()
    assert z.grad is not None


def test_regime_head_output_shape():
    head = RegimeHead(d_z=D_Z, n_classes=4)
    z = torch.randn(B, D_Z)
    out = head(z)
    assert out.shape == (B, 4)


def test_regime_head_custom_n_classes():
    head = RegimeHead(d_z=D_Z, n_classes=3)
    z = torch.randn(B, D_Z)
    assert head(z).shape == (B, 3)


def test_regime_head_gradient_flows():
    head = RegimeHead(d_z=D_Z, n_classes=4)
    z = torch.randn(B, D_Z, requires_grad=True)
    out = head(z)
    out.sum().backward()
    assert z.grad is not None


def test_heads_share_no_parameters():
    """Direction and regime heads should have independent parameters."""
    dir_head = DirectionHead(D_Z, 3)
    reg_head = RegimeHead(D_Z, 4)
    dir_params = set(id(p) for p in dir_head.parameters())
    reg_params = set(id(p) for p in reg_head.parameters())
    assert len(dir_params & reg_params) == 0
