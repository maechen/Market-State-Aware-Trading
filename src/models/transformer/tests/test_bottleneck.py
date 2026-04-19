"""Tests for the Bottleneck module."""

import pytest
import torch
from src.models.transformer.bottleneck import Bottleneck

B, D_MODEL, D_Z = 4, 128, 32


def _make_bottleneck():
    return Bottleneck(d_model=D_MODEL, d_z=D_Z)


def test_bottleneck_z_shape():
    bn = _make_bottleneck()
    h = torch.randn(B, D_MODEL)
    z, z_pre = bn(h)
    assert z.shape == (B, D_Z)


def test_bottleneck_z_pre_shape():
    bn = _make_bottleneck()
    h = torch.randn(B, D_MODEL)
    z, z_pre = bn(h)
    assert z_pre.shape == (B, D_Z)


def test_z_is_bounded_in_minus1_plus1():
    """Post-Tanh output must lie strictly in (-1, 1)."""
    bn = _make_bottleneck()
    # Use large inputs to push z_pre to extreme values
    h = torch.randn(B, D_MODEL) * 100.0
    z, _ = bn(h)
    assert (z >= -1.0).all() and (z <= 1.0).all()


def test_z_equals_tanh_z_pre():
    bn = _make_bottleneck()
    h = torch.randn(B, D_MODEL)
    z, z_pre = bn(h)
    assert torch.allclose(z, torch.tanh(z_pre), atol=1e-6)


def test_z_pre_can_exceed_tanh_bounds():
    """z_pre should be able to take values outside [-1, 1]."""
    bn = _make_bottleneck()
    # With large inputs, z_pre should have values far from ±1
    h = torch.randn(B, D_MODEL) * 100.0
    _, z_pre = bn(h)
    assert (z_pre.abs() > 1.0).any(), "z_pre should have values outside [-1, 1]"


def test_bottleneck_returns_tuple_of_two():
    bn = _make_bottleneck()
    h = torch.randn(B, D_MODEL)
    result = bn(h)
    assert isinstance(result, tuple) and len(result) == 2


def test_bottleneck_gradient_flows_through_z():
    bn = _make_bottleneck()
    h = torch.randn(B, D_MODEL, requires_grad=True)
    z, z_pre = bn(h)
    z.sum().backward()
    assert h.grad is not None


def test_bottleneck_gradient_flows_through_z_pre():
    bn = _make_bottleneck()
    h = torch.randn(B, D_MODEL, requires_grad=True)
    z, z_pre = bn(h)
    z_pre.sum().backward()
    assert h.grad is not None


def test_bottleneck_different_inputs_different_outputs():
    bn = _make_bottleneck()
    h1 = torch.randn(B, D_MODEL)
    h2 = torch.randn(B, D_MODEL)
    z1, _ = bn(h1)
    z2, _ = bn(h2)
    assert not torch.allclose(z1, z2)
