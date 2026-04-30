"""
MarketTransformer: sentiment gate, causal transformer, bottleneck, and two
supervised heads — direction (binary Up/Down) and regime (HMM state).

The auxiliary return regression head has been permanently removed.  After
extensive experimentation across 5-day and 20-day horizons the head
consistently achieved standardised MAE ≈ 0.9 (essentially random predictions).
Gradient interference from this failing auxiliary task degraded the shared
encoder (MTL task-interference literature, MT2ST OpenReview 2024).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TransformerConfig, GateMode
from .gate import SentimentGate, CrossAttentionGate
from .layers import PositionalEncoding, TransformerStack, TemporalReadout, make_causal_mask
from .bottleneck import Bottleneck
from .heads import DirectionHead, RegimeHead


class MarketTransformer(nn.Module):
    """
    End-to-end model: sentiment gate → causal transformer → bottleneck → two heads.

    Direction head: predicts n-day forward Up/Down (binary classification).
    Regime head:    predicts HMM market state (0..K-1, K=4).

    Loss = λ_dir × focal_CE(direction) + λ_reg × CE(regime)

    :param config: TransformerConfig with dimensions, gate mode, readout, and loss weights
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

        if config.gate_mode == GateMode.MASTER:
            self.gate = SentimentGate(config.d_sent, config.d_feat, config.gate_beta)
            self.feature_proj: Optional[nn.Linear] = nn.Linear(config.d_feat, config.d_model)
            nn.init.xavier_uniform_(self.feature_proj.weight)
            nn.init.zeros_(self.feature_proj.bias)
        else:
            # CrossAttentionGate outputs d_model space directly; skip feature_proj
            self.gate = CrossAttentionGate(
                config.d_sent, config.d_feat, config.d_model, config.n_heads, config.dropout
            )
            self.feature_proj = None

        self.pos_enc = PositionalEncoding(config.d_model, config.dropout)
        self.transformer_stack = TransformerStack(
            config.d_model, config.n_heads, config.d_ff, config.n_layers, config.dropout
        )
        self.readout = TemporalReadout(config.d_model, config.readout_mode)
        self.bottleneck = Bottleneck(config.d_model, config.d_z)

        # Regime head reads from z (compact 16-dim bottleneck output).
        # z is intentionally small so DRL can use it as a low-dimensional state.
        self.reg_head = RegimeHead(config.d_z, config.n_reg_classes)

        # Direction head reads from h_pooled (d_model=64) when use_task_specific_heads=True.
        # Bypassing the 64→16 bottleneck projection gives the direction head 4× more
        # representational capacity and decouples its gradients from the regime→z path.
        if config.use_task_specific_heads:
            self.dir_head = DirectionHead(
                config.d_model, config.n_dir_classes, hidden=config.dir_head_hidden
            )
        else:
            self.dir_head = DirectionHead(
                config.d_z, config.n_dir_classes, hidden=config.dir_head_hidden
            )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Encode a batch of W-day windows into latent z and two-head predictions.

        :param x: tensor (batch, W, d_feat + d_sent)
        :return:
          dict with keys: z, z_pre, dir_logits, reg_logits
        """
        B, W, _ = x.shape
        price_tech = x[..., : self.config.d_feat]
        sent_seq = x[:, :, self.config.d_feat :]   # (B, W, d_sent)

        gated = self.gate(price_tech, sent_seq)

        if self.feature_proj is not None:
            h = self.feature_proj(gated)
        else:
            h = gated

        h = self.pos_enc(h)
        causal_mask = make_causal_mask(W, x.device)
        h = self.transformer_stack(h, causal_mask)

        h_pooled = self.readout(h)
        z, z_pre = self.bottleneck(h_pooled)

        dir_logits = self.dir_head(h_pooled if self.config.use_task_specific_heads else z)

        return {
            "z": z,
            "z_pre": z_pre,
            "dir_logits": dir_logits,
            "reg_logits": self.reg_head(z),
        }

    def compute_loss(
        self, out: dict, targets: dict
    ) -> tuple[torch.Tensor, dict]:
        """
        Loss = λ_dir × focal_CE(direction) + λ_reg × CE(regime)

        :param out:     forward() output dict (z, dir_logits, reg_logits)
        :param targets: dict with y_dir (int64), y_reg (int64); optional dir_weights
        :return:
          total : scalar loss tensor for backprop
          info  : dict of float component losses (dir_loss, reg_loss, total_loss)
        """
        cfg = self.config

        dir_loss = F.cross_entropy(
            out["dir_logits"],
            targets["y_dir"],
            label_smoothing=cfg.dir_label_smoothing,
        )

        reg_loss = F.cross_entropy(out["reg_logits"], targets["y_reg"])

        total = cfg.lambda_dir * dir_loss + cfg.lambda_reg * reg_loss

        return total, {
            "dir_loss": dir_loss.item(),
            "reg_loss": reg_loss.item(),
            "total_loss": total.item(),
        }

    def configure_optimizers(self, lr: float, weight_decay: float) -> list[dict]:
        """
        Split parameters into AdamW-style decay (linear/MHA weights) vs no-decay (bias, LayerNorm).
        :param lr: base learning rate
        :param weight_decay: L2 coefficient for the decay group
        :return: list of two optimizer param groups
        """
        decay: set[str] = set()
        no_decay: set[str] = set()

        for module_name, module in self.named_modules():
            for param_name, _ in module.named_parameters():
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                if param_name.endswith("bias"):
                    no_decay.add(full_name)
                elif param_name.endswith("weight") and isinstance(module, nn.Linear):
                    decay.add(full_name)
                elif param_name.endswith("weight") and isinstance(
                    module, nn.MultiheadAttention
                ):
                    decay.add(full_name)
                elif param_name.endswith("weight") and isinstance(
                    module, (nn.LayerNorm, nn.Embedding)
                ):
                    no_decay.add(full_name)

        param_dict = {n: p for n, p in self.named_parameters()}
        assert len(decay & no_decay) == 0, "Parameter overlap between decay/no-decay sets"
        assert len(param_dict.keys() - (decay | no_decay)) == 0, \
            "Some parameters not assigned to decay or no-decay"

        return [
            {
                "params": [param_dict[n] for n in sorted(decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[n] for n in sorted(no_decay)],
                "weight_decay": 0.0,
            },
        ]
