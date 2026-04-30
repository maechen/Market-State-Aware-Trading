"""Tests for TransformerConfig dataclass and enums."""

import pytest
from src.models.transformer.config import TransformerConfig, GateMode, ReadoutMode


def test_default_d_feat():
    cfg = TransformerConfig()
    assert cfg.d_feat == 7


def test_default_d_sent():
    cfg = TransformerConfig()
    assert cfg.d_sent == 3


def test_total_input_width():
    cfg = TransformerConfig()
    assert cfg.d_feat + cfg.d_sent == 10


def test_default_d_model():
    cfg = TransformerConfig()
    assert cfg.d_model == 64


def test_default_n_heads_divides_d_model():
    cfg = TransformerConfig()
    assert cfg.d_model % cfg.n_heads == 0


def test_default_d_ff_is_2x_d_model():
    """d_ff is 2× d_model (reduced from 4× to cut params on small dataset)."""
    cfg = TransformerConfig()
    assert cfg.d_ff == 2 * cfg.d_model


def test_default_d_z():
    cfg = TransformerConfig()
    assert cfg.d_z == 16


def test_default_n_layers():
    cfg = TransformerConfig()
    assert cfg.n_layers == 2


def test_default_n_heads():
    cfg = TransformerConfig()
    assert cfg.n_heads == 4


def test_default_gate_mode_is_cross_attn():
    cfg = TransformerConfig()
    assert cfg.gate_mode == GateMode.CROSS_ATTN


def test_default_readout_mode_is_last():
    cfg = TransformerConfig()
    assert cfg.readout_mode == ReadoutMode.LAST


def test_default_n_dir_classes():
    cfg = TransformerConfig()
    assert cfg.n_dir_classes == 3


def test_default_n_reg_classes():
    cfg = TransformerConfig()
    assert cfg.n_reg_classes == 4


def test_gate_mode_enum_values():
    assert GateMode.MASTER == "master"
    assert GateMode.CROSS_ATTN == "cross_attn"


def test_readout_mode_enum_values():
    assert ReadoutMode.LAST == "last"
    assert ReadoutMode.MEAN == "mean"
    assert ReadoutMode.ATTN_POOL == "attn_pool"


def test_config_override():
    cfg = TransformerConfig(d_model=64, n_heads=4, n_layers=2)
    assert cfg.d_model == 64
    assert cfg.n_heads == 4
    assert cfg.n_layers == 2


def test_cross_attn_gate_mode_selectable():
    cfg = TransformerConfig(gate_mode=GateMode.CROSS_ATTN)
    assert cfg.gate_mode == GateMode.CROSS_ATTN


def test_lambda_defaults():
    cfg = TransformerConfig()
    assert cfg.lambda_dir == 2.0
    assert cfg.lambda_reg == 0.3
    assert cfg.lambda_ret == 0.5


def test_use_task_specific_heads_default():
    """Direction and return heads bypass tanh bottleneck by default."""
    cfg = TransformerConfig()
    assert cfg.use_task_specific_heads is True


def test_focal_gamma_default():
    """Focal loss exponent defaults to 2.0 for direction head."""
    cfg = TransformerConfig()
    assert cfg.focal_gamma == 2.0


def test_task_specific_heads_can_be_disabled():
    cfg = TransformerConfig(use_task_specific_heads=False)
    assert cfg.use_task_specific_heads is False


def test_focal_gamma_zero_is_allowed():
    cfg = TransformerConfig(focal_gamma=0.0)
    assert cfg.focal_gamma == 0.0


def test_dir_quantile_defaults():
    """Neutral band widened to 33/67 for balanced 33/33/33 direction classes."""
    cfg = TransformerConfig()
    assert cfg.dir_q_low == 0.33
    assert cfg.dir_q_high == 0.67
    assert cfg.dir_q_low < cfg.dir_q_high


def test_dir_head_hidden_default():
    """Direction head defaults to a two-layer MLP with hidden dim 32."""
    cfg = TransformerConfig()
    assert cfg.dir_head_hidden == 32


def test_dir_label_smoothing_default():
    """Label smoothing disabled by default to avoid raising the loss floor."""
    cfg = TransformerConfig()
    assert cfg.dir_label_smoothing == 0.0
