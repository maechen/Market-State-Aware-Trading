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
    assert cfg.d_model == 128


def test_default_n_heads_divides_d_model():
    cfg = TransformerConfig()
    assert cfg.d_model % cfg.n_heads == 0


def test_default_d_ff_is_4x_d_model():
    cfg = TransformerConfig()
    assert cfg.d_ff == 4 * cfg.d_model


def test_default_d_z():
    cfg = TransformerConfig()
    assert cfg.d_z == 32


def test_default_n_layers():
    cfg = TransformerConfig()
    assert cfg.n_layers == 4


def test_default_n_heads():
    cfg = TransformerConfig()
    assert cfg.n_heads == 4


def test_default_gate_mode_is_master():
    cfg = TransformerConfig()
    assert cfg.gate_mode == GateMode.MASTER


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
    assert cfg.lambda_dir == 1.0
    assert cfg.lambda_reg == 0.5
    assert cfg.lambda_ret == 0.5


def test_dir_quantile_defaults():
    cfg = TransformerConfig()
    assert cfg.dir_q_low == 0.40
    assert cfg.dir_q_high == 0.60
    assert cfg.dir_q_low < cfg.dir_q_high
