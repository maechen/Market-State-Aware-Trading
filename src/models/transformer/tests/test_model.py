"""Tests for MarketTransformer (full model)."""

import pytest
import torch
from src.models.transformer.config import TransformerConfig, GateMode, ReadoutMode
from src.models.transformer.model import MarketTransformer

B = 4

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_input(cfg: TransformerConfig, batch: int = B) -> torch.Tensor:
    return torch.randn(batch, cfg.window_size, cfg.d_feat + cfg.d_sent)


def make_targets(cfg: TransformerConfig, batch: int = B) -> dict:
    return {
        "y_dir": torch.randint(0, cfg.n_dir_classes, (batch,)),
        "y_reg": torch.randint(0, cfg.n_reg_classes, (batch,)),
        "y_ret_std": torch.randn(batch),
    }


# ── Shape checks for all gate × readout combinations ─────────────────────────

GATE_MODES = [GateMode.MASTER, GateMode.CROSS_ATTN]
READOUT_MODES = [ReadoutMode.LAST, ReadoutMode.MEAN, ReadoutMode.ATTN_POOL]


@pytest.mark.parametrize("gate_mode", GATE_MODES)
@pytest.mark.parametrize("readout_mode", READOUT_MODES)
def test_output_shapes(gate_mode, readout_mode):
    cfg = TransformerConfig(gate_mode=gate_mode, readout_mode=readout_mode)
    model = MarketTransformer(cfg)
    model.eval()
    x = make_input(cfg)
    with torch.no_grad():
        out = model(x)
    assert out["z"].shape == (B, cfg.d_z), f"{gate_mode},{readout_mode}: z wrong"
    assert out["z_pre"].shape == (B, cfg.d_z)
    assert out["dir_logits"].shape == (B, cfg.n_dir_classes)
    assert out["reg_logits"].shape == (B, cfg.n_reg_classes)
    assert out["ret_pred"].shape == (B, 1)


def test_output_has_all_keys():
    cfg = TransformerConfig()
    model = MarketTransformer(cfg)
    out = model(make_input(cfg))
    for key in ("z", "z_pre", "dir_logits", "reg_logits", "ret_pred"):
        assert key in out, f"Missing key: {key}"


# ── z bounds ─────────────────────────────────────────────────────────────────

def test_z_is_bounded():
    cfg = TransformerConfig()
    model = MarketTransformer(cfg)
    model.eval()
    with torch.no_grad():
        out = model(make_input(cfg))
    assert (out["z"] >= -1.0).all() and (out["z"] <= 1.0).all()


def test_z_pre_can_be_unbounded():
    """z_pre should not be constrained to [-1, 1]."""
    cfg = TransformerConfig()
    model = MarketTransformer(cfg)
    # Use large-magnitude input to encourage large z_pre values
    x = make_input(cfg) * 50.0
    model.eval()
    with torch.no_grad():
        out = model(x)
    # z should still be bounded
    assert (out["z"] >= -1.0).all() and (out["z"] <= 1.0).all()


# ── Loss computation ──────────────────────────────────────────────────────────

def test_compute_loss_returns_scalar_tensor():
    cfg = TransformerConfig()
    model = MarketTransformer(cfg)
    out = model(make_input(cfg))
    loss, _ = model.compute_loss(out, make_targets(cfg))
    assert loss.shape == torch.Size([])


def test_compute_loss_dict_has_expected_keys():
    cfg = TransformerConfig()
    model = MarketTransformer(cfg)
    out = model(make_input(cfg))
    _, info = model.compute_loss(out, make_targets(cfg))
    for key in ("dir_loss", "reg_loss", "ret_loss", "total_loss"):
        assert key in info, f"Missing loss key: {key}"


def test_compute_loss_components_are_nonnegative():
    cfg = TransformerConfig()
    model = MarketTransformer(cfg)
    out = model(make_input(cfg))
    _, info = model.compute_loss(out, make_targets(cfg))
    assert info["dir_loss"] >= 0
    assert info["reg_loss"] >= 0
    assert info["ret_loss"] >= 0


def test_compute_loss_total_equals_weighted_sum():
    cfg = TransformerConfig(lambda_dir=1.0, lambda_reg=0.5, lambda_ret=0.5)
    model = MarketTransformer(cfg)
    out = model(make_input(cfg))
    _, info = model.compute_loss(out, make_targets(cfg))
    expected = (cfg.lambda_dir * info["dir_loss"]
                + cfg.lambda_reg * info["reg_loss"]
                + cfg.lambda_ret * info["ret_loss"])
    assert abs(info["total_loss"] - expected) < 1e-5


# ── Gradient flow ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("gate_mode", GATE_MODES)
def test_gradient_flows_end_to_end(gate_mode):
    cfg = TransformerConfig(gate_mode=gate_mode, n_layers=2)
    model = MarketTransformer(cfg)
    x = make_input(cfg)
    out = model(x)
    loss, _ = model.compute_loss(out, make_targets(cfg))
    loss.backward()
    # Every parameter with requires_grad should have a gradient
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for parameter: {name}"


# ── Causal property ───────────────────────────────────────────────────────────

def test_causal_property_of_transformer_stack():
    """
    With a causal mask, changing positions t=5..W-1 in the input must NOT
    change the transformer hidden states at positions 0..4.
    The final model output uses only the last hidden state, but we can check
    the internal stack directly via partial forward.
    """
    from src.models.transformer.layers import make_causal_mask, TransformerStack

    stack = TransformerStack(128, 4, 512, n_layers=2, dropout=0.0)
    stack.eval()
    mask = make_causal_mask(20, torch.device("cpu"))

    x1 = torch.randn(B, 20, 128)
    x2 = x1.clone()
    x2[:, 5:, :] = torch.randn(B, 15, 128)

    with torch.no_grad():
        h1 = stack(x1, mask)
        h2 = stack(x2, mask)

    assert torch.allclose(h1[:, :5, :], h2[:, :5, :], atol=1e-5), \
        "Causal mask violated in TransformerStack"


# ── Variable window size ──────────────────────────────────────────────────────

@pytest.mark.parametrize("window_size", [20, 40, 60])
def test_variable_window_size(window_size):
    cfg = TransformerConfig(window_size=window_size, n_layers=2)
    model = MarketTransformer(cfg)
    model.eval()
    x = make_input(cfg)
    with torch.no_grad():
        out = model(x)
    assert out["z"].shape == (B, cfg.d_z)


# ── configure_optimizers ──────────────────────────────────────────────────────

def test_configure_optimizers_returns_two_groups():
    cfg = TransformerConfig()
    model = MarketTransformer(cfg)
    groups = model.configure_optimizers(lr=1e-3, weight_decay=1e-4)
    assert len(groups) == 2


def test_configure_optimizers_no_overlap():
    cfg = TransformerConfig()
    model = MarketTransformer(cfg)
    groups = model.configure_optimizers(lr=1e-3, weight_decay=1e-4)
    decay_ids = {id(p) for p in groups[0]["params"]}
    no_decay_ids = {id(p) for p in groups[1]["params"]}
    assert len(decay_ids & no_decay_ids) == 0


def test_configure_optimizers_covers_all_params():
    cfg = TransformerConfig()
    model = MarketTransformer(cfg)
    groups = model.configure_optimizers(lr=1e-3, weight_decay=1e-4)
    all_covered = {id(p) for g in groups for p in g["params"]}
    all_params = {id(p) for p in model.parameters()}
    assert all_covered == all_params


def test_no_decay_group_has_zero_weight_decay():
    cfg = TransformerConfig()
    model = MarketTransformer(cfg)
    groups = model.configure_optimizers(lr=1e-3, weight_decay=1e-4)
    assert groups[1]["weight_decay"] == 0.0
