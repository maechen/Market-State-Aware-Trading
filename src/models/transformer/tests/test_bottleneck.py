"""Tests for the Bottleneck module."""

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


def test_bottleneck_returns_tuple_of_two():
    bn = _make_bottleneck()
    h = torch.randn(B, D_MODEL)
    result = bn(h)
    assert isinstance(result, tuple) and len(result) == 2


def test_z_and_z_pre_are_identical():
    """Without tanh, z and z_pre are the same linear projection."""
    bn = _make_bottleneck()
    h = torch.randn(B, D_MODEL)
    z, z_pre = bn(h)
    assert torch.allclose(z, z_pre), "z and z_pre should be identical (no activation)"


def test_z_equals_linear_of_h():
    """z must be exactly the linear projection of h (no activation applied)."""
    bn = _make_bottleneck()
    h = torch.randn(B, D_MODEL)
    z, _ = bn(h)
    with torch.no_grad():
        expected = bn.linear(h)
    assert torch.allclose(z, expected, atol=1e-6)


def test_z_is_unbounded():
    """Without tanh the projection can take values outside [-1, 1]."""
    bn = _make_bottleneck()
    h = torch.randn(B, D_MODEL) * 100.0
    z, _ = bn(h)
    assert (z.abs() > 1.0).any(), "z should have values outside [-1, 1] for large inputs"


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
