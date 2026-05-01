"""Tests for TransformerConfig dataclass and enums."""

import pytest
from src.models.transformer.config import TransformerConfig, GateMode, ReadoutMode


def test_default_d_feat():
    """16 price/tech features: 10 z-scored + 1 RSI + 1 BB%B + 4 regime probs."""
    cfg = TransformerConfig()
    assert cfg.d_feat == 16


def test_default_d_sent():
    cfg = TransformerConfig()
    assert cfg.d_sent == 3


def test_total_input_width():
    """19 total: 16 price/tech + 3 sentiment."""
    cfg = TransformerConfig()
    assert cfg.d_feat + cfg.d_sent == 19


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
    assert cfg.n_layers == 3


def test_default_n_heads():
    cfg = TransformerConfig()
    assert cfg.n_heads == 4


def test_default_gate_mode_is_cross_attn():
    cfg = TransformerConfig()
    assert cfg.gate_mode == GateMode.CROSS_ATTN


def test_default_readout_mode_is_attn_pool():
    cfg = TransformerConfig()
    assert cfg.readout_mode == ReadoutMode.ATTN_POOL


def test_default_n_dir_classes():
    cfg = TransformerConfig()
    assert cfg.n_dir_classes == 2


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
    """Loss is direction focal-CE + regime CE with no auxiliary tasks."""
    cfg = TransformerConfig()
    assert cfg.lambda_dir == 0.25
    assert cfg.lambda_reg == 1.0


def test_use_task_specific_heads_default():
    """Direction head bypasses the 64→16 bottleneck projection by default."""
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


def test_dir_entropy_coeff_default():
    """Entropy regularisation defaults to 0.3 to overcome fold5/7/8 directional bias."""
    cfg = TransformerConfig()
    assert cfg.dir_entropy_coeff == 0.3


def test_dir_entropy_coeff_can_be_disabled():
    cfg = TransformerConfig(dir_entropy_coeff=0.0)
    assert cfg.dir_entropy_coeff == 0.0


def test_d_feat_default_includes_momentum_and_regime():
    """d_feat=16: 10 scaled + 1 RSI + 1 BB%B + 4 regime_prob pass-through."""
    cfg = TransformerConfig()
    assert cfg.d_feat == 16


def test_n_dir_classes_default_binary():
    """Default direction head is binary (up/down) to avoid quantile distribution shift."""
    cfg = TransformerConfig()
    assert cfg.n_dir_classes == 2


def test_n_dir_classes_can_be_three():
    cfg = TransformerConfig(n_dir_classes=3)
    assert cfg.n_dir_classes == 3


