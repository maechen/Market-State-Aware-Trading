"""
MarketTransformer: sentiment gate, causal transformer, bottleneck, and multitask heads (direction, regime, return).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import TransformerConfig, GateMode
from .gate import SentimentGate, CrossAttentionGate
from .layers import PositionalEncoding, TransformerStack, TemporalReadout, make_causal_mask
from .bottleneck import Bottleneck
from .heads import DirectionHead, RegimeHead, ReturnHead


class MarketTransformer(nn.Module):
    """
    end-to-end model from windowed features to latent z and three supervised predictions
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
        self.dir_head = DirectionHead(
            config.d_z, config.n_dir_classes, hidden=config.dir_head_hidden
        )
        self.reg_head = RegimeHead(config.d_z, config.n_reg_classes)
        self.ret_head = ReturnHead(config.d_z)

    def forward(self, x: torch.Tensor) -> dict:
        """
        encode a batch of W-day windows into latent z and multitask logits
        :param x: tensor (batch, W, d_feat + d_sent); first d_feat columns are price/tech, rest sentiment
        :return:
          dict with keys z, z_pre, dir_logits, reg_logits, ret_pred (shapes per TransformerConfig)
        """
        B, W, _ = x.shape
        price_tech = x[..., : self.config.d_feat]
        sent_seq = x[:, :, self.config.d_feat :]   # (B, W, d_sent) full sentiment window

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

        return {
            "z": z,
            "z_pre": z_pre,
            "dir_logits": self.dir_head(z),
            "reg_logits": self.reg_head(z),
            "ret_pred": self.ret_head(z),
        }

    def compute_loss(
        self, out: dict, targets: dict
    ) -> tuple[torch.Tensor, dict]:
        """
        weighted sum of direction CE, regime CE, and huber return loss.
        :param out: forward() output dict
        :param targets: dict with y_dir, y_reg, y_ret_std (batch tensors); optional dir_weights for CE
        :return:
          total: scalar loss tensor for backprop
          info: dict of float component losses (dir_loss, reg_loss, ret_loss, total_loss)
        """
        cfg = self.config

        dir_loss = F.cross_entropy(
            out["dir_logits"],
            targets["y_dir"],
            weight=targets.get("dir_weights"),
            label_smoothing=cfg.dir_label_smoothing,
        )
        reg_loss = F.cross_entropy(out["reg_logits"], targets["y_reg"])
        ret_loss = F.huber_loss(out["ret_pred"].squeeze(-1), targets["y_ret_std"].float())

        total = cfg.lambda_dir * dir_loss + cfg.lambda_reg * reg_loss + cfg.lambda_ret * ret_loss

        return total, {
            "dir_loss": dir_loss.item(),
            "reg_loss": reg_loss.item(),
            "ret_loss": ret_loss.item(),
            "total_loss": total.item(),
        }

    def configure_optimizers(self, lr: float, weight_decay: float) -> list[dict]:
        """
        split parameters into AdamW-style decay (linear / MHA weights) vs no-decay (bias, LayerNorm).
        :param lr: base learning rate (stored in returned dicts only if caller uses it; here unused)
        :param weight_decay: L2 coefficient for the decay group
        :return:
          list of two optimizer param groups: high decay, zero decay
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
                    # Fused in_proj_weight / separate q/k/v weights are not on nn.Linear
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
