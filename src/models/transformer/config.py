"""
enums and dataclass configuration for the MarketTransformer (gates, readout, losses, window size).
"""

from dataclasses import dataclass
from enum import Enum


class GateMode(str, Enum):
    """
    which gating block precedes the transformer stack.
    MASTER: softmax feature gate from last-step sentiment (default).
    CROSS_ATTN: cross-attention with sentiment as query over price sequence (for ablation).
    """
    MASTER = "master"
    CROSS_ATTN = "cross_attn"


class ReadoutMode(str, Enum):
    """
    how to pool sequence dimension (B, W, d_model) to (B, d_model) before the bottleneck.
    LAST: last timestep
    MEAN: mean over W
    ATTN_POOL: learned attention over time
    """
    LAST = "last"
    MEAN = "mean"
    ATTN_POOL = "attn_pool"


@dataclass
class TransformerConfig:
    """
    Hyperparameters for MarketTransformer training and inference.
    :param d_feat: number of price/technical feature channels per timestep (after engineering)
    :param d_sent: number of sentiment channels per timestep (gate input)
    :param window_size: lookback window length W in days
    :param d_model: transformer embedding dimension
    :param d_ff: feedforward hidden size (typically 4 * d_model)
    :param d_z: bottleneck latent dimension (e.g. 32 for RL state)
    :param n_layers: number of stacked causal transformer blocks
    :param n_heads: number of multi-head attention heads (d_model must divide evenly)
    :param dropout: dropout probability for attention and FFN
    :param gate_beta: temperature for MASTER softmax gate (softmax(linear/beta))
    :param gate_mode: MASTER or CROSS_ATTN gating variant
    :param readout_mode: LAST, MEAN, or ATTN_POOL temporal pooling
    :param use_pre_tanh_z: if True, downstream RL uses z_pre instead of z
    :param n_dir_classes: direction head classes (Bear/Neutral/Bull = 3)
    :param n_reg_classes: regime head classes (GHMM states, typically 4)
    :param lambda_dir: loss weight on direction cross-entropy
    :param lambda_reg: loss weight on regime cross-entropy
    :param lambda_ret: loss weight on Huber return regression
    :param dir_label_smoothing: label smoothing for direction cross-entropy
    :param dir_q_low: lower quantile of train returns defining neutral band for direction labels
    :param dir_q_high: upper quantile of train returns defining neutral band for direction labels
    """
    d_feat: int = 7
    d_sent: int = 3
    window_size: int = 20

    d_model: int = 128
    d_ff: int = 512
    d_z: int = 32
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1

    gate_beta: float = 1.0
    gate_mode: GateMode = GateMode.MASTER
    readout_mode: ReadoutMode = ReadoutMode.LAST
    use_pre_tanh_z: bool = False # if true, RL obs uses z_pre instead of z

    n_dir_classes: int = 3
    n_reg_classes: int = 4
    lambda_dir: float = 1.0
    lambda_reg: float = 0.5
    lambda_ret: float = 0.5
    dir_label_smoothing: float = 0.1
    dir_q_low: float = 0.40
    dir_q_high: float = 0.60
