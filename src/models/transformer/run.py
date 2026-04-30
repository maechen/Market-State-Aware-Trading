"""
CLI runner for transformer training across walk-forward folds.

Supports three variants:
- gating
- no_gating
- no_sentiment

Also exports latent vectors (`z`) for train/val/test/all as `.npy` arrays with
aligned date arrays for downstream DRL usage.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

# Keep local module names (e.g., bottleneck.py) from shadowing site packages.
THIS_DIR = Path(__file__).resolve().parent
sys.path = [
    path_entry
    for path_entry in sys.path
    if Path(path_entry or ".").resolve() != THIS_DIR
]

# Allow running as `python src/models/transformer/run.py` from repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from configs.walkforward_folds import FOLDS
from src.data.fold_loader import get_fold_loaders
from src.models.transformer import GateMode, MarketTransformer, ReadoutMode, TransformerConfig

VARIANT_CHOICES = ("gating", "no_gating", "no_sentiment")
SPLIT_NAMES = ("train", "val", "test")


class FeatureWindowDataset(Dataset):
    """Inference-only dataset that emits all valid windows from feature rows."""

    def __init__(self, features: np.ndarray, window_size: int) -> None:
        if features.ndim != 2:
            raise ValueError(f"Expected 2D features, got shape {features.shape}.")
        if window_size <= 0:
            raise ValueError("window_size must be >= 1.")
        self.features = np.asarray(features, dtype=np.float32)
        self.window_size = window_size
        self.n_windows = max(0, self.features.shape[0] - window_size + 1)

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> torch.Tensor:
        window = self.features[idx : idx + self.window_size]
        return torch.from_numpy(window)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_arg: str) -> torch.device:
    requested = device_arg.strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def _resolve_folds(selected_folds: str | None, max_folds: int | None) -> list[dict[str, str]]:
    """Resolve fold list from `--folds` and `--max-folds`."""
    if selected_folds:
        selected_names = {name.strip() for name in selected_folds.split(",") if name.strip()}
        resolved = [fold for fold in FOLDS if fold["name"] in selected_names]
        missing = selected_names - {fold["name"] for fold in resolved}
        if missing:
            raise ValueError(f"Unknown fold names: {sorted(missing)}")
    else:
        resolved = list(FOLDS)

    if max_folds is not None:
        if max_folds <= 0:
            raise ValueError("--max-folds must be >= 1.")
        resolved = resolved[:max_folds]

    if not resolved:
        raise ValueError("No folds selected.")
    return resolved


def _transformer_config_to_dict(cfg: TransformerConfig) -> dict[str, Any]:
    cfg_dict = asdict(cfg)
    cfg_dict["gate_mode"] = cfg.gate_mode.value
    cfg_dict["readout_mode"] = cfg.readout_mode.value
    return cfg_dict


def _build_transformer_config(args: argparse.Namespace) -> TransformerConfig:
    gate_mode = GateMode(args.gate_mode)
    readout_mode = ReadoutMode(args.readout_mode)

    if args.variant == "no_gating":
        gate_mode = GateMode.MASTER

    return TransformerConfig(
        d_feat=args.d_feat,
        d_sent=args.d_sent,
        window_size=args.window_size,
        d_model=args.d_model,
        d_ff=args.d_ff,
        d_z=args.d_z,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        gate_beta=args.gate_beta,
        gate_mode=gate_mode,
        readout_mode=readout_mode,
        n_dir_classes=args.n_dir_classes,
        n_reg_classes=args.n_reg_classes,
        lambda_dir=args.lambda_dir,
        lambda_reg=args.lambda_reg,
        dir_label_smoothing=args.dir_label_smoothing,
        dir_head_hidden=args.dir_head_hidden,
        use_task_specific_heads=args.use_task_specific_heads,
        dir_label_mode=args.dir_label_mode,
        dir_vol_k=args.dir_vol_k,
        dir_vol_col=args.dir_vol_col,
        dir_ignore_index=args.dir_ignore_index,
    )


def _apply_variant_to_inputs(x: torch.Tensor, variant: str, d_feat: int) -> torch.Tensor:
    if variant == "no_sentiment":
        x = x.clone()
        x[..., d_feat:] = 0.0
    return x


def _prepare_model(config: TransformerConfig, variant: str, device: torch.device) -> MarketTransformer:
    model = MarketTransformer(config).to(device)

    if variant == "no_gating":
        if not hasattr(model.gate, "linear"):
            raise ValueError(
                "no_gating variant requires MASTER-style SentimentGate (with linear layer)."
            )
        with torch.no_grad():
            model.gate.linear.weight.zero_()
            model.gate.linear.bias.zero_()
        model.gate.linear.weight.requires_grad_(False)
        model.gate.linear.bias.requires_grad_(False)

    return model


def _run_epoch(
    model: MarketTransformer,
    dataloader: DataLoader,
    device: torch.device,
    variant: str,
    d_feat: int,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    totals = {
        "total_loss": 0.0,
        "dir_loss": 0.0,
        "reg_loss": 0.0,
        "dir_correct": 0.0,
        "dir_event_count": 0.0,
        "dir_total_count": 0.0,
        "reg_correct": 0.0,
        "num_samples": 0,
    }

    for batch_idx, (x, y_dir, y_reg) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = x.to(device, non_blocking=True)
        x = _apply_variant_to_inputs(x, variant=variant, d_feat=d_feat)
        if not torch.isfinite(x).all():
            raise ValueError(
                f"Non-finite values detected in input batch at index {batch_idx}."
            )
        targets: dict[str, Any] = {
            "y_dir": y_dir.to(device=device, dtype=torch.long, non_blocking=True),
            "y_reg": y_reg.to(device=device, dtype=torch.long, non_blocking=True),
        }

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            out = model(x)
            loss, info = model.compute_loss(out, targets)
            if not torch.isfinite(loss):
                raise ValueError(
                    f"Non-finite loss encountered at batch {batch_idx}."
                )
            if is_train:
                loss.backward()
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

        batch_size = int(x.shape[0])
        totals["num_samples"] += batch_size
        totals["total_loss"] += info["total_loss"] * batch_size
        dir_events_for_loss = int(info.get("dir_event_count", 0.0))
        if dir_events_for_loss > 0:
            totals["dir_loss"] += info["dir_loss"] * dir_events_for_loss
        totals["reg_loss"] += info["reg_loss"] * batch_size
        dir_mask = targets["y_dir"] != model.config.dir_ignore_index
        dir_events = int(dir_mask.sum().item())
        totals["dir_event_count"] += dir_events
        totals["dir_total_count"] += int(targets["y_dir"].numel())
        if dir_events > 0:
            dir_pred = out["dir_logits"].argmax(dim=1)
            totals["dir_correct"] += ((dir_pred[dir_mask] == targets["y_dir"][dir_mask]).sum().item())
        totals["reg_correct"] += (
            (out["reg_logits"].argmax(dim=1) == targets["y_reg"]).sum().item()
        )

    if totals["num_samples"] == 0:
        raise ValueError("Epoch completed with zero samples. Check data and batch limits.")

    n = float(totals["num_samples"])
    dir_event_count = float(totals["dir_event_count"])
    dir_total_count = float(totals["dir_total_count"])
    return {
        "total_loss": totals["total_loss"] / n,
        "dir_loss": (totals["dir_loss"] / dir_event_count if dir_event_count > 0 else float("nan")),
        "reg_loss": totals["reg_loss"] / n,
        "dir_acc": (totals["dir_correct"] / dir_event_count if dir_event_count > 0 else float("nan")),
        "dir_event_count": dir_event_count,
        "dir_coverage": (dir_event_count / dir_total_count if dir_total_count > 0 else float("nan")),
        "reg_acc": totals["reg_correct"] / n,
        "num_samples": float(totals["num_samples"]),
    }


def _collect_split_predictions(
    model: MarketTransformer,
    dataloader: DataLoader,
    device: torch.device,
    variant: str,
    d_feat: int,
) -> dict[str, np.ndarray]:
    """
    Collect per-sample predictions/targets for a split in dataloader order.

    Returns:
        dict with y_dir_true (includes -1 ignored labels), y_dir_pred, y_reg_true, y_reg_pred.
    """
    model.eval()
    y_dir_true: list[np.ndarray] = []
    y_dir_pred: list[np.ndarray] = []
    y_reg_true: list[np.ndarray] = []
    y_reg_pred: list[np.ndarray] = []

    with torch.no_grad():
        for x, y_dir, y_reg in dataloader:
            x = x.to(device, non_blocking=True)
            x = _apply_variant_to_inputs(x, variant=variant, d_feat=d_feat)
            out = model(x)

            y_dir_true.append(y_dir.cpu().numpy().astype(np.int64))
            y_dir_pred.append(out["dir_logits"].argmax(dim=1).cpu().numpy().astype(np.int64))
            y_reg_true.append(y_reg.cpu().numpy().astype(np.int64))
            y_reg_pred.append(out["reg_logits"].argmax(dim=1).cpu().numpy().astype(np.int64))

    def _concat_or_empty(parts: list[np.ndarray], dtype: np.dtype) -> np.ndarray:
        if not parts:
            return np.empty((0,), dtype=dtype)
        return np.concatenate(parts, axis=0).astype(dtype)

    return {
        "y_dir_true": _concat_or_empty(y_dir_true, np.int64),
        "y_dir_pred": _concat_or_empty(y_dir_pred, np.int64),
        "y_reg_true": _concat_or_empty(y_reg_true, np.int64),
        "y_reg_pred": _concat_or_empty(y_reg_pred, np.int64),
    }


def _direction_event_metrics(y_true: np.ndarray, y_pred: np.ndarray, ignore_index: int = -1) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != ignore_index
    n_total = int(y_true.shape[0])
    n_event = int(mask.sum())
    if n_event == 0:
        return {"dir_event_count": 0.0, "dir_coverage": 0.0, "dir_acc_event": float("nan"), "dir_mae_event": float("nan"), "dir_up_rate_event": float("nan"), "dir_pred_up_rate_event": float("nan"), "dir_majority_acc_event": float("nan"), "dir_acc_minus_majority_event": float("nan")}
    yt = y_true[mask].astype(np.int64)
    yp = y_pred[mask].astype(np.int64)
    up_rate = float(np.mean(yt == 1))
    acc = float(np.mean(yt == yp))
    return {"dir_event_count": float(n_event), "dir_coverage": float(n_event / n_total) if n_total > 0 else float("nan"), "dir_acc_event": acc, "dir_mae_event": float(np.mean(np.abs(yt - yp))), "dir_up_rate_event": up_rate, "dir_pred_up_rate_event": float(np.mean(yp == 1)), "dir_majority_acc_event": float(max(up_rate, 1.0 - up_rate)), "dir_acc_minus_majority_event": float(acc - max(up_rate, 1.0 - up_rate))}

def _export_split_predictions(
    model: MarketTransformer,
    split_name: str,
    dataloader: DataLoader,
    fold_dir: Path,
    output_dir: Path,
    config: TransformerConfig,
    variant: str,
    device: torch.device,
) -> tuple[int, dict[str, float]]:
    preds = _collect_split_predictions(model=model, dataloader=dataloader, device=device, variant=variant, d_feat=config.d_feat)
    n_samples = int(preds["y_dir_true"].shape[0])
    split_dates = _load_split_dates(fold_dir=fold_dir)
    aligned_dates = split_dates[split_name][config.window_size - 1 : -1].to_numpy(dtype="datetime64[ns]")
    if aligned_dates.shape[0] != n_samples:
        raise RuntimeError(f"Prediction/date mismatch for {split_name} split: {n_samples} vs {aligned_dates.shape[0]}")
    pred_df = pd.DataFrame({"date": pd.to_datetime(aligned_dates), "y_dir_true": preds["y_dir_true"], "y_dir_pred": preds["y_dir_pred"], "is_dir_event": preds["y_dir_true"] != config.dir_ignore_index, "y_reg_true": preds["y_reg_true"], "y_reg_pred": preds["y_reg_pred"]})
    pred_df.to_csv(output_dir / f"{split_name}_predictions.csv", index=False)
    return n_samples, _direction_event_metrics(preds["y_dir_true"], preds["y_dir_pred"], ignore_index=config.dir_ignore_index)


def _save_checkpoint(
    path: Path,
    model: MarketTransformer,
    optimizer: torch.optim.Optimizer,
    config: TransformerConfig,
    fold_name: str,
    variant: str,
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "fold": fold_name,
        "variant": variant,
        "config": _transformer_config_to_dict(config),
        "metrics": metrics,
    }
    torch.save(payload, path)


def _load_split_dates(fold_dir: Path) -> dict[str, pd.DatetimeIndex]:
    split_dates: dict[str, pd.DatetimeIndex] = {}
    for split in SPLIT_NAMES:
        split_path = fold_dir / f"spy_{split}_labeled.csv"
        split_df = pd.read_csv(split_path, index_col=0, parse_dates=True)
        split_dates[split] = pd.DatetimeIndex(split_df.index)
    return split_dates


def _extract_latent_matrix(
    model: MarketTransformer,
    features: np.ndarray,
    window_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    variant: str,
    d_feat: int,
) -> np.ndarray:
    dataset = FeatureWindowDataset(features=features, window_size=window_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_latents: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch_x in dataloader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_x = _apply_variant_to_inputs(batch_x, variant=variant, d_feat=d_feat)
            out = model(batch_x)
            all_latents.append(out["z"].detach().cpu().numpy().astype(np.float32))

    if not all_latents:
        return np.empty((0, model.config.d_z), dtype=np.float32)
    return np.concatenate(all_latents, axis=0)


def _export_fold_latents(
    model: MarketTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    fold_dir: Path,
    output_dir: Path,
    config: TransformerConfig,
    variant: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, int]:
    split_to_loader = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
    split_dates = _load_split_dates(fold_dir=fold_dir)

    all_latents: list[np.ndarray] = []
    all_dates: list[np.ndarray] = []
    row_counts: dict[str, int] = {}

    for split in SPLIT_NAMES:
        split_loader = split_to_loader[split]
        split_features = np.asarray(split_loader.dataset.features, dtype=np.float32)
        split_latents = _extract_latent_matrix(
            model=model,
            features=split_features,
            window_size=config.window_size,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            variant=variant,
            d_feat=config.d_feat,
        )

        aligned_dates = split_dates[split][config.window_size - 1 :].to_numpy(dtype="datetime64[ns]")
        if split_latents.shape[0] != aligned_dates.shape[0]:
            raise RuntimeError(
                f"Latent/date mismatch for {split}: {split_latents.shape[0]} vs {aligned_dates.shape[0]}"
            )

        np.save(output_dir / f"latents_{split}.npy", split_latents)
        np.save(output_dir / f"dates_{split}.npy", aligned_dates)

        row_counts[f"latent_rows_{split}"] = int(split_latents.shape[0])

        if split_latents.shape[0] > 0:
            all_latents.append(split_latents)
            all_dates.append(aligned_dates)

    if all_latents:
        latents_all = np.concatenate(all_latents, axis=0).astype(np.float32)
        dates_all = np.concatenate(all_dates, axis=0).astype("datetime64[ns]")
    else:
        latents_all = np.empty((0, config.d_z), dtype=np.float32)
        dates_all = np.empty((0,), dtype="datetime64[ns]")

    np.save(output_dir / "latents_all.npy", latents_all)
    np.save(output_dir / "dates_all.npy", dates_all)
    row_counts["latent_rows_all"] = int(latents_all.shape[0])

    metadata = {
        "variant": variant,
        "window_size": config.window_size,
        "d_z": config.d_z,
        "split_order_for_all": list(SPLIT_NAMES),
        **row_counts,
    }
    with open(output_dir / "latent_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return row_counts


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be >= 1.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be >= 1.")
    if args.lr <= 0.0:
        raise ValueError("--lr must be > 0.")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be >= 0.")
    if args.window_size <= 0:
        raise ValueError("--window-size must be >= 1.")
    if args.d_feat <= 0:
        raise ValueError("--d-feat must be >= 1.")
    if args.d_sent < 0:
        raise ValueError("--d-sent must be >= 0.")
    if args.d_model <= 0 or args.d_ff <= 0 or args.d_z <= 0:
        raise ValueError("--d-model/--d-ff/--d-z must be >= 1.")
    if args.n_layers <= 0 or args.n_heads <= 0:
        raise ValueError("--n-layers and --n-heads must be >= 1.")
    if args.dropout < 0.0 or args.dropout >= 1.0:
        raise ValueError("--dropout must be in [0, 1).")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be >= 0.")
    if args.max_train_batches is not None and args.max_train_batches <= 0:
        raise ValueError("--max-train-batches must be >= 1 when provided.")
    if args.max_eval_batches is not None and args.max_eval_batches <= 0:
        raise ValueError("--max-eval-batches must be >= 1 when provided.")
    if args.variant == "no_gating" and args.gate_mode != GateMode.MASTER.value:
        print(
            "[Info] no_gating selected; overriding gate_mode to 'master' "
            "for identity-gate emulation.",
            flush=True,
        )


def run_training(args: argparse.Namespace) -> pd.DataFrame:
    _validate_args(args)

    device = _resolve_device(args.device)
    _set_seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    config = _build_transformer_config(args)
    selected_folds = _resolve_folds(selected_folds=args.folds, max_folds=args.max_folds)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    run_config = {
        "variant": args.variant,
        "fold_root": args.fold_root,
        "output_dir": args.output_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": str(device),
        "checkpoint_every": args.checkpoint_every,
        "max_train_batches": args.max_train_batches,
        "max_eval_batches": args.max_eval_batches,
        "folds": [fold["name"] for fold in selected_folds],
        "transformer": _transformer_config_to_dict(config),
        "vol_threshold_horizon_scaled": bool(config.dir_label_mode == "vol_threshold"),
        "vol_horizon_scale": float(np.sqrt(args.dir_n_forward)) if config.dir_label_mode == "vol_threshold" else 1.0,
    }
    with open(output_root / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)

    fold_rows: list[dict[str, float | int | str]] = []

    for fold in selected_folds:
        fold_name = fold["name"]
        fold_dir = Path(args.fold_root) / fold_name
        if not fold_dir.exists():
            raise FileNotFoundError(f"Fold directory not found: {fold_dir}")

        print(f"\n=== Running {fold_name} ({args.variant}) ===", flush=True)
        train_loader, val_loader, test_loader = get_fold_loaders(
            fold_dir=str(fold_dir),
            window_size=config.window_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle_train=args.shuffle_train,
            n_dir_classes=config.n_dir_classes,
            dir_n_forward=args.dir_n_forward,
            dir_label_mode=config.dir_label_mode,
            dir_vol_k=config.dir_vol_k,
            dir_vol_col=config.dir_vol_col,
            dir_ignore_index=config.dir_ignore_index,
        )

        feature_dim = int(train_loader.dataset.features.shape[1])
        expected_dim = config.d_feat + config.d_sent
        if feature_dim != expected_dim:
            raise ValueError(
                f"Feature dimension mismatch: dataset has {feature_dim}, "
                f"config expects {expected_dim} (d_feat + d_sent)."
            )

        model = _prepare_model(config=config, variant=args.variant, device=device)
        optimizer = torch.optim.AdamW(
            model.configure_optimizers(lr=args.lr, weight_decay=args.weight_decay),
            lr=args.lr,
        )
        warmup_epochs = max(1, int(args.epochs * args.warmup_frac)) if args.warmup_frac > 0 else 0
        if warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.epochs - warmup_epochs), eta_min=1e-6
            )
            scheduler: torch.optim.lr_scheduler.LRScheduler = (
                torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_epochs],
                )
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=1e-6
            )

        fold_output = output_root / fold_name
        fold_output.mkdir(parents=True, exist_ok=True)
        best_ckpt_path = fold_output / "best_model.pt"

        # Early stopping tracks the sum of direction + regime validation losses.
        # With the return head removed, this is equivalent to tracking the two
        # supervised objectives used by the current model.
        best_val_primary_loss = float("inf")  # dir_loss + reg_loss only
        best_val_loss = float("inf")           # total (for reporting in fold_summary.json)
        best_epoch = -1
        epoch_rows: list[dict[str, float | int]] = []

        for epoch in range(1, args.epochs + 1):
            train_metrics = _run_epoch(
                model=model,
                dataloader=train_loader,
                device=device,
                variant=args.variant,
                d_feat=config.d_feat,
                optimizer=optimizer,
                max_batches=args.max_train_batches,
                grad_clip=args.grad_clip,
            )
            val_metrics = _run_epoch(
                model=model,
                dataloader=val_loader,
                device=device,
                variant=args.variant,
                d_feat=config.d_feat,
                optimizer=None,
                max_batches=args.max_eval_batches,
            )

            scheduler.step()

            print(
                f"Epoch {epoch:03d} | "
                f"train={train_metrics['total_loss']:.4f} "
                f"(dir={train_metrics['dir_loss']:.4f} acc={train_metrics['dir_acc']:.3f} "
                f"reg={train_metrics['reg_loss']:.4f} acc={train_metrics['reg_acc']:.3f}) | "
                f"val={val_metrics['total_loss']:.4f} "
                f"(dir={val_metrics['dir_loss']:.4f} acc={val_metrics['dir_acc']:.3f} "
                f"reg={val_metrics['reg_loss']:.4f} acc={val_metrics['reg_acc']:.3f}) "
                f"lr={scheduler.get_last_lr()[0]:.2e}",
                flush=True,
            )

            epoch_rows.append(
                {
                    "epoch": epoch,
                    "train_total_loss": float(train_metrics["total_loss"]),
                    "train_dir_loss": float(train_metrics["dir_loss"]),
                    "train_reg_loss": float(train_metrics["reg_loss"]),
                    "train_dir_acc": float(train_metrics["dir_acc"]),
                    "train_reg_acc": float(train_metrics["reg_acc"]),
                    "train_num_samples": int(train_metrics["num_samples"]),
                    "val_total_loss": float(val_metrics["total_loss"]),
                    "val_dir_loss": float(val_metrics["dir_loss"]),
                    "val_reg_loss": float(val_metrics["reg_loss"]),
                    "val_dir_acc": float(val_metrics["dir_acc"]),
                    "val_reg_acc": float(val_metrics["reg_acc"]),
                    "val_num_samples": int(val_metrics["num_samples"]),
                }
            )

            val_primary_loss = val_metrics["dir_loss"] + val_metrics["reg_loss"]
            if val_primary_loss < best_val_primary_loss:
                best_val_primary_loss = val_primary_loss
                best_val_loss = float(val_metrics["total_loss"])
                best_epoch = epoch
                _save_checkpoint(
                    path=best_ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    fold_name=fold_name,
                    variant=args.variant,
                    epoch=epoch,
                    metrics={"train": train_metrics, "val": val_metrics},
                )

            if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
                _save_checkpoint(
                    path=fold_output / f"epoch_{epoch:03d}.pt",
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    fold_name=fold_name,
                    variant=args.variant,
                    epoch=epoch,
                    metrics={"train": train_metrics, "val": val_metrics},
                )

        if best_epoch < 0:
            raise RuntimeError("No best checkpoint was created.")

        pd.DataFrame(epoch_rows).to_csv(fold_output / "epoch_metrics.csv", index=False)

        _save_checkpoint(
            path=fold_output / "last_model.pt",
            model=model,
            optimizer=optimizer,
            config=config,
            fold_name=fold_name,
            variant=args.variant,
            epoch=args.epochs,
            metrics={"train": train_metrics, "val": val_metrics},
        )

        best_payload = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best_payload["state_dict"], strict=True)

        test_metrics = _run_epoch(
            model=model,
            dataloader=test_loader,
            device=device,
            variant=args.variant,
            d_feat=config.d_feat,
            optimizer=None,
            max_batches=args.max_eval_batches,
        )
        val_pred_rows, val_event_metrics = _export_split_predictions(
            model=model, split_name="val", dataloader=val_loader, fold_dir=fold_dir, output_dir=fold_output, config=config, variant=args.variant, device=device
        )
        test_pred_rows, test_event_metrics = _export_split_predictions(
            model=model, split_name="test", dataloader=test_loader, fold_dir=fold_dir, output_dir=fold_output, config=config, variant=args.variant, device=device
        )

        latent_counts = _export_fold_latents(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            fold_dir=fold_dir,
            output_dir=fold_output,
            config=config,
            variant=args.variant,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )

        fold_summary = {
            "fold": fold_name,
            "variant": args.variant,
            "best_epoch": int(best_epoch),
            "best_val_total_loss": float(best_val_loss),
            "test_total_loss": float(test_metrics["total_loss"]),
            "test_dir_loss": float(test_metrics["dir_loss"]),
            "test_reg_loss": float(test_metrics["reg_loss"]),
            "test_dir_acc": float(test_metrics["dir_acc"]),
            "test_reg_acc": float(test_metrics["reg_acc"]),
            "test_num_samples": int(test_metrics["num_samples"]),
            "dir_label_mode": config.dir_label_mode,
            "dir_vol_k": float(config.dir_vol_k),
            "dir_vol_col": config.dir_vol_col,
            "dir_ignore_index": int(config.dir_ignore_index),
            "val_pred_rows": int(val_pred_rows),
            "test_pred_rows": int(test_pred_rows),
            **{f"val_{k}": v for k, v in val_event_metrics.items()},
            **{f"test_{k}": v for k, v in test_event_metrics.items()},
            **latent_counts,
        }

        with open(fold_output / "fold_summary.json", "w", encoding="utf-8") as handle:
            json.dump(fold_summary, handle, indent=2)

        fold_rows.append(fold_summary)

    metrics_df = pd.DataFrame(fold_rows)
    metrics_df.to_csv(output_root / "fold_metrics.csv", index=False)
    return metrics_df


def parse_args(
    argv: Sequence[str] | None = None,
    default_variant: str = "gating",
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train MarketTransformer on walk-forward folds and export aligned "
            "latent vectors (.npy) for train/val/test/all splits."
        )
    )

    # Data / output
    parser.add_argument(
        "--fold-root",
        type=str,
        default="data/training",
        help="Directory containing fold subdirs (e.g. fold1/fold2/...) with labeled CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/transformer_runs",
        help="Directory where checkpoints, metrics, and latent arrays are saved.",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default=None,
        help="Optional comma-separated fold names (e.g. fold1,fold3).",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Optional cap on number of folds to run after fold selection.",
    )

    # Training
    parser.add_argument("--variant", choices=VARIANT_CHOICES, default=default_variant)
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs per fold.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device string (auto/cpu/cuda/mps).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="If >0, save extra checkpoint every N epochs.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Optional limit on train batches per epoch (useful for smoke tests).",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
        help="Optional limit on val/test batches (useful for smoke tests).",
    )

    # Model / labels
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--d-feat", type=int, default=16,
                        help="Price/tech feature channels (10 scaled + 1 RSI + 1 BB + 4 regime_prob).")
    parser.add_argument("--d-sent", type=int, default=3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--d-z", type=int, default=16)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--gate-beta", type=float, default=1.0)
    parser.add_argument(
        "--gate-mode",
        choices=[mode.value for mode in GateMode],
        default=GateMode.CROSS_ATTN.value,
    )
    parser.add_argument(
        "--readout-mode",
        choices=[mode.value for mode in ReadoutMode],
        default=ReadoutMode.ATTN_POOL.value,
    )
    parser.add_argument("--n-dir-classes", type=int, default=2,
                        help="Direction head output classes: 2=binary Up/Down (default), "
                             "3=Bear/Neutral/Bull.")
    parser.add_argument("--n-reg-classes", type=int, default=4)
    parser.add_argument("--lambda-dir", type=float, default=2.0)
    parser.add_argument("--lambda-reg", type=float, default=0.3)
    parser.add_argument("--dir-label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--dir-head-hidden",
        type=int,
        default=32,
        help="Hidden dim for two-layer direction MLP head; 0 = single linear.",
    )
    parser.add_argument(
        "--dir-n-forward",
        type=int,
        default=5,
        help="Direction label horizon in trading days (default 5). "
             "5-day forward returns have stronger momentum signal than 1-day.",
    )
    parser.add_argument(
        "--dir-label-mode",
        type=str,
        default="sign",
        choices=["sign", "vol_threshold", "quantile_3class"],
    )
    parser.add_argument("--dir-vol-k", type=float, default=0.50)
    parser.add_argument("--dir-vol-col", type=str, default="rolling_vol_20")
    parser.add_argument("--dir-ignore-index", type=int, default=-1)
    parser.add_argument(
        "--use-task-specific-heads",
        action="store_true",
        default=True,
        help="Direction head reads from h_pooled (bypasses the 64→16 bottleneck projection).",
    )
    parser.add_argument(
        "--no-task-specific-heads",
        dest="use_task_specific_heads",
        action="store_false",
        help="Revert direction head to read from z (16-dim bottleneck); ablation only.",
    )
    parser.add_argument(
        "--no-shuffle-train",
        dest="shuffle_train",
        action="store_false",
        help="Disable shuffling of training windows (default: shuffled).",
    )
    parser.set_defaults(shuffle_train=True)
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Max gradient norm for clip_grad_norm_ (0 = disabled).",
    )
    parser.add_argument(
        "--warmup-frac",
        type=float,
        default=0.05,
        help="Fraction of total epochs used for linear LR warm-up (0 = no warmup).",
    )

    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    default_variant: str = "gating",
) -> None:
    args = parse_args(argv=argv, default_variant=default_variant)
    metrics_df = run_training(args)
    print("\nTransformer training complete. Fold metrics:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
