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
    :param d_ff: feedforward hidden size (typically 2–4 × d_model)
    :param d_z: bottleneck latent dimension (e.g. 16 for RL state)
    :param n_layers: number of stacked causal transformer blocks
    :param n_heads: number of multi-head attention heads (d_model must divide evenly)
    :param dropout: dropout probability for attention and FFN
    :param gate_beta: temperature for MASTER softmax gate (softmax(linear/beta))
    :param gate_mode: MASTER or CROSS_ATTN gating variant
    :param readout_mode: LAST, MEAN, or ATTN_POOL temporal pooling
    :param use_pre_tanh_z: if True, downstream RL uses z_pre instead of z
    :param n_dir_classes: direction head classes (2 = binary Up/Down, default;
                           3 = Bear/Neutral/Bull, legacy — suffers quantile drift)
    :param n_reg_classes: regime head classes (GHMM states; BIC selects K=4)
    :param lambda_dir: loss weight on direction cross-entropy
    :param lambda_reg: loss weight on regime cross-entropy
    :param lambda_ret: loss weight on Quantile (pinball, τ=0.5) return regression
    :param dir_label_smoothing: label smoothing ε for direction cross-entropy (0 = disabled)
    :param dir_q_low: lower quantile of train returns defining neutral band for direction labels
    :param dir_q_high: upper quantile of train returns defining neutral band for direction labels
    :param dir_head_hidden: hidden dim for the two-layer direction MLP head (0 = linear only)
    """
    # 16 price/tech features:
    #   z-scored (10): log_return, rolling_vol_20, macd, close_to_ma10,
    #                  close_to_ma20, ma10_to_ma50, roc_5, macd_hist,
    #                  atr_ratio, vol_ratio
    #   bounded (1):   rsi_norm
    #   bounded (1):   bb_pct (Bollinger Band %B)
    #   pass-through (4): regime_prob_0..3
    d_feat: int = 16
    d_sent: int = 3
    window_size: int = 30

    # Reduced from d_model=128/d_ff=512/n_layers=4/n_heads=8 to match ~1240 samples/fold.
    # Original ~800 K params (645 params/sample) caused extreme overparameterisation;
    # new config targets ~50–80 K params (~40–65 params/sample).
    d_model: int = 64
    d_ff: int = 128
    d_z: int = 16
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.2

    gate_beta: float = 1.0
    gate_mode: GateMode = GateMode.CROSS_ATTN
    readout_mode: ReadoutMode = ReadoutMode.LAST
    use_pre_tanh_z: bool = False  # if true, RL obs uses z_pre instead of z

    # Binary direction (2) is the default.  The 3-class (Bear/Neutral/Bull)
    # approach suffers from quantile-threshold distribution shift between
    # training and test periods, causing 60-70 % of test labels to be Neutral
    # and inducing mode collapse.  Binary up/down (sign of n-day forward return)
    # eliminates this problem entirely.
    n_dir_classes: int = 2
    n_reg_classes: int = 4

    # lambda_dir raised to 2.0 so direction gets priority in the gradient budget.
    # lambda_reg raised from 0.2 to 0.3: the CrossAttentionGate is less efficient
    # at preserving regime-discriminative features than the multiplicative MASTER gate;
    # a slightly stronger regime signal helps recover the ~6 pp accuracy regression.
    # lambda_ret reduced to 0.2: the 20-day return target is genuinely noisier than
    # direction/regime, so its gradient contribution is capped to avoid polluting the
    # shared encoder.  The return head still provides an auxiliary regularising signal
    # that shapes the latent space toward return-relevant features (SOTA: arxiv:2411.01456).
    lambda_dir: float = 2.0
    lambda_reg: float = 0.3
    lambda_ret: float = 0.2

    # Label smoothing disabled (was 0.1): added ~0.11 nats to the floor of an already-
    # failing direction head, making optimisation harder with no benefit at this stage.
    dir_label_smoothing: float = 0.0

    # Widened neutral band 40/60 → 33/67 to produce balanced 33/33/33 direction classes.
    # The original 40/60 split left neutral as a 20% minority with no class weighting,
    # causing the direction head to ignore it entirely.
    dir_q_low: float = 0.33
    dir_q_high: float = 0.67

    # Two-layer MLP direction head; 0 = single linear (disabled).
    dir_head_hidden: int = 32

    # When True, direction and return heads bypass the 16-dim bottleneck and read
    # directly from h_pooled (64-dim).  The tanh activation was already removed from
    # the bottleneck (fixing gradient saturation), but the linear 64→16 projection
    # still discards information.  Task-specific heads give dir/ret 4× more capacity
    # and decouple their gradients from the regime→z path.  Set False only to ablate.
    use_task_specific_heads: bool = True

    # Focal loss exponent γ for the direction cross-entropy.  γ=0 reduces to
    # standard cross-entropy; γ=2 down-weights ambiguous boundary samples (returns
    # very close to the neutral-band thresholds) and focuses gradients on harder,
    # more clearly-labelled examples.
    focal_gamma: float = 2.0

    # Entropy regularisation coefficient for the direction head.
    # Subtracts `dir_entropy_coeff * H(softmax(dir_logits))` from dir_loss to
    # MAXIMISE prediction entropy, preventing the head from collapsing to always
    # predicting the same class (mode collapse).  Empirically 0.05–0.15 is a good
    # range; set to 0.0 to disable.
    dir_entropy_coeff: float = 0.1

    # Return head: variance regularisation coefficient.
    # Adds `ret_var_coeff * ReLU(0.3 - std(ret_pred))` to ret_loss to penalise
    # near-constant predictions and encourage the return head to spread its
    # outputs.  The threshold 0.3 is relative to standardised scale (≈ 0.3σ).
    # Set to 0.0 to disable.
    ret_var_coeff: float = 0.1
