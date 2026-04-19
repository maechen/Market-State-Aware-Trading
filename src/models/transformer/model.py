"""
MarketTransformer — full sentiment-guided transformer with multitask heads.

Pipeline:
    SentimentGate → FeatureProj → PositionalEncoding → TransformerStack
    → TemporalReadout → Bottleneck → [DirectionHead, RegimeHead, ReturnHead]

Input:  x of shape (B, W, d_feat + d_sent)
        columns 0 .. d_feat-1  = price/tech features (gated)
        columns d_feat .. end  = sentiment features (gate input only)

Output: dict with keys
        'z'          (B, d_z)        — bounded latent state (Tanh)
        'z_pre'      (B, d_z)        — unbounded pre-activation
        'dir_logits' (B, n_dir)      — direction raw logits
        'reg_logits' (B, n_reg)      — regime raw logits
        'ret_pred'   (B, 1)          — return regression prediction
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
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

        # ── Gate ─────────────────────────────────────────────────────────
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

        # ── Transformer core ──────────────────────────────────────────────
        self.pos_enc = PositionalEncoding(config.d_model, config.dropout)
        self.transformer_stack = TransformerStack(
            config.d_model, config.n_heads, config.d_ff, config.n_layers, config.dropout
        )
        self.readout = TemporalReadout(config.d_model, config.readout_mode)

        # ── Bottleneck + heads ────────────────────────────────────────────
        self.bottleneck = Bottleneck(config.d_model, config.d_z)
        self.dir_head = DirectionHead(config.d_z, config.n_dir_classes)
        self.reg_head = RegimeHead(config.d_z, config.n_reg_classes)
        self.ret_head = ReturnHead(config.d_z)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x : (B, W, d_feat + d_sent)
        Returns:
            dict of z, z_pre, dir_logits, reg_logits, ret_pred
        """
        B, W, _ = x.shape
        price_tech = x[..., : self.config.d_feat]   # (B, W, d_feat)
        sent_last = x[:, -1, self.config.d_feat :]  # (B, d_sent)

        gated = self.gate(price_tech, sent_last)     # (B, W, d_feat or d_model)

        if self.feature_proj is not None:
            h = self.feature_proj(gated)             # (B, W, d_model)
        else:
            h = gated                                # (B, W, d_model) already

        h = self.pos_enc(h)

        causal_mask = make_causal_mask(W, x.device)
        h = self.transformer_stack(h, causal_mask)   # (B, W, d_model)

        h_pooled = self.readout(h)                   # (B, d_model)
        z, z_pre = self.bottleneck(h_pooled)         # (B, d_z), (B, d_z)

        return {
            "z": z,
            "z_pre": z_pre,
            "dir_logits": self.dir_head(z),          # (B, n_dir)
            "reg_logits": self.reg_head(z),          # (B, n_reg)
            "ret_pred": self.ret_head(z),            # (B, 1)
        }

    # ── Loss ──────────────────────────────────────────────────────────────

    def compute_loss(
        self, out: dict, targets: dict
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            out     : dict from forward()
            targets : dict with keys
                      'y_dir'       (B,)   int long
                      'y_reg'       (B,)   int long
                      'y_ret_std'   (B,)   float
                      'dir_weights' (n_dir_classes,) optional class weights
        Returns:
            total_loss : scalar tensor
            info       : dict of per-component float losses
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

    # ── Optimizer ─────────────────────────────────────────────────────────

    def configure_optimizers(self, lr: float, weight_decay: float) -> list[dict]:
        """
        Separates parameters into decay (nn.Linear weights) and no-decay
        (biases, LayerNorm weights) groups, following the pattern from
        transformer.py's configure_optimizers.
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
