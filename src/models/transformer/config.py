from dataclasses import dataclass
from enum import Enum


class GateMode(str, Enum):
    MASTER = "master" # default: MASTER softmax feature gate
    CROSS_ATTN = "cross_attn" # for ablation: cross-attention gate


class ReadoutMode(str, Enum):
    LAST = "last" # default: h[:, -1, :]
    MEAN = "mean" # mean-pool over W timesteps
    ATTN_POOL = "attn_pool" # MASTER temporal attention pooling


@dataclass
class TransformerConfig:
    # input dimensions
    d_feat: int = 7 # number of price features
    d_sent: int = 3 # number of sentiment features
    window_size: int = 20 # W lookback window (days)

    # architecture
    d_model: int = 128 # transformer hidden dimension
    d_ff: int = 512 # FFN hidden size (4 × d_model)
    d_z: int = 32 # bottleneck dimension
    n_layers: int = 4 # number of transformer blocks
    n_heads: int = 4 # attention heads (d_model / n_heads = 32)
    dropout: float = 0.1

    # gate & readout
    gate_beta: float = 1.0
    gate_mode: GateMode = GateMode.MASTER
    readout_mode: ReadoutMode = ReadoutMode.LAST
    use_pre_tanh_z: bool = False  # if True, RL obs uses z_pre instead of z

    # supervision
    n_dir_classes: int = 3
    n_reg_classes: int = 4
    lambda_dir: float = 1.0
    lambda_reg: float = 0.5
    lambda_ret: float = 0.5
    dir_label_smoothing: float = 0.1
    dir_q_low: float = 0.40   # quantile lower bound for neutral direction band
    dir_q_high: float = 0.60  # quantile upper bound for neutral direction band
