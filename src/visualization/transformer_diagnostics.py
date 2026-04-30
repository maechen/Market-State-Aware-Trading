"""
transformer_diagnostics.py — Training effectiveness visualizations for MarketTransformer.

Produces 7 publication-ready figures from a completed transformer run directory:

  01_learning_curves.png        — Train/val total loss per fold with best-epoch marker
  02_component_losses.png       — Per-component (dir/reg/ret) train vs val loss per fold
  03_crossfold_test_metrics.png — Cross-fold bar chart of test accuracy and loss
  04_direction_accuracy.png     — Val direction accuracy vs epoch per fold (vs random baseline)
  05_regime_accuracy.png        — Val regime accuracy vs epoch per fold
  06_latent_pca.png             — PCA of test-split latent vectors across all folds
  07_val_vs_test_loss.png       — Best val loss vs test loss per fold (overfitting diagnostic)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# ── Aesthetic constants ────────────────────────────────────────────────────────

FOLD_COLORS = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A",
    "#F4A261", "#264653", "#8338EC", "#FB5607",
]
COMPONENT_COLORS = {"dir": "#E63946", "reg": "#2A9D8F", "ret": "#E9C46A"}
SPLIT_ALPHA = {"train": 0.45, "val": 1.0}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.dpi": 130,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

RANDOM_CHANCE_DIR = 0.5  # binary Up/Down (2-class default; n_dir_classes=2)


# ── Data loading helpers ───────────────────────────────────────────────────────

def _load_epoch_metrics(fold_dir: Path) -> pd.DataFrame:
    return pd.read_csv(fold_dir / "epoch_metrics.csv")


def _load_fold_summary(fold_dir: Path) -> dict:
    with open(fold_dir / "fold_summary.json") as f:
        return json.load(f)


def _load_fold_metrics(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(run_dir / "fold_metrics.csv")


def _load_run_config(run_dir: Path) -> dict:
    with open(run_dir / "run_config.json") as f:
        return json.load(f)


def _discover_folds(run_dir: Path) -> list[Path]:
    """Return sorted fold sub-directories that contain epoch_metrics.csv."""
    return sorted(
        [p for p in run_dir.iterdir() if p.is_dir() and (p / "epoch_metrics.csv").exists()],
        key=lambda p: p.name,
    )


# ── Figure 1: Per-fold learning curves ────────────────────────────────────────

def plot_learning_curves(run_dir: Path, out_dir: Path) -> Path:
    """Train vs val total loss across epochs for each fold."""
    folds = _discover_folds(run_dir)
    n_folds = len(folds)
    n_cols = 4
    n_rows = (n_folds + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    axes = np.array(axes).flatten()
    fig.suptitle("Learning Curves — Train vs Val Total Loss (per Fold)", fontsize=14, y=1.01)

    for ax_idx, fold_dir in enumerate(folds):
        ax = axes[ax_idx]
        df = _load_epoch_metrics(fold_dir)
        summary = _load_fold_summary(fold_dir)
        best_epoch = summary["best_epoch"]
        color = FOLD_COLORS[ax_idx % len(FOLD_COLORS)]

        epochs = df["epoch"].values
        ax.plot(epochs, df["train_total_loss"], color=color, alpha=0.45, lw=1.5, label="Train")
        ax.plot(epochs, df["val_total_loss"], color=color, lw=2.0, label="Val")
        ax.axvline(best_epoch, color=color, lw=1.2, linestyle=":", alpha=0.8, label=f"Best (ep {best_epoch})")

        ax.set_title(
            f"{fold_dir.name}  |  best val {summary['best_val_total_loss']:.3f}",
            fontsize=9, fontweight="bold",
        )
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Total Loss", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper right")

    for ax in axes[n_folds:]:
        ax.set_visible(False)

    fig.tight_layout()
    out_path = out_dir / "01_learning_curves.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ── Figure 2: Per-fold component losses ───────────────────────────────────────

def plot_component_losses(run_dir: Path, out_dir: Path) -> Path:
    """Dir / reg / ret train and val loss curves per fold."""
    folds = _discover_folds(run_dir)
    n_folds = len(folds)
    n_cols = 4
    n_rows = (n_folds + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    axes = np.array(axes).flatten()
    fig.suptitle("Component Losses per Fold  (solid = val, faded = train)", fontsize=14, y=1.01)

    legend_patches = [
        mpatches.Patch(color=COMPONENT_COLORS["dir"], label="Direction"),
        mpatches.Patch(color=COMPONENT_COLORS["reg"], label="Regime"),
        mpatches.Patch(color=COMPONENT_COLORS["ret"], label="Return"),
    ]

    for ax_idx, fold_dir in enumerate(folds):
        ax = axes[ax_idx]
        df = _load_epoch_metrics(fold_dir)
        summary = _load_fold_summary(fold_dir)
        epochs = df["epoch"].values

        for comp in ("dir", "reg", "ret"):
            c = COMPONENT_COLORS[comp]
            ax.plot(epochs, df[f"train_{comp}_loss"], color=c, alpha=0.35, lw=1.2)
            ax.plot(epochs, df[f"val_{comp}_loss"], color=c, lw=1.8)

        ax.axvline(summary["best_epoch"], color="gray", lw=1.0, linestyle=":", alpha=0.7)
        ax.set_title(fold_dir.name, fontsize=9, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Loss", fontsize=8)
        ax.tick_params(labelsize=7)

    for ax in axes[n_folds:]:
        ax.set_visible(False)

    fig.legend(handles=legend_patches, loc="lower right", fontsize=9, ncol=3,
               bbox_to_anchor=(1.0, 0.0))
    fig.tight_layout()
    out_path = out_dir / "02_component_losses.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ── Figure 3: Cross-fold test metrics bar chart ────────────────────────────────

def plot_crossfold_test_metrics(run_dir: Path, out_dir: Path) -> Path:
    """Grouped bar chart: test dir_acc / reg_acc / total_loss per fold."""
    fm = _load_fold_metrics(run_dir)
    folds = fm["fold"].tolist()
    x = np.arange(len(folds))
    width = 0.25

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Cross-Fold Test Performance", fontsize=14)

    # Accuracy panel
    dir_acc = fm["test_dir_acc"].values * 100
    reg_acc = fm["test_reg_acc"].values * 100
    ax_acc.bar(x - width / 2, dir_acc, width, color="#E63946", label="Direction Acc", zorder=3)
    ax_acc.bar(x + width / 2, reg_acc, width, color="#2A9D8F", label="Regime Acc", zorder=3)
    ax_acc.axhline(RANDOM_CHANCE_DIR * 100, color="#E63946", lw=1.5,
                   linestyle="--", alpha=0.6, label="Dir random chance (33.3%)")
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(folds, rotation=30, ha="right", fontsize=9)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=10)
    ax_acc.set_title("Test Accuracy per Fold", fontsize=11)
    ax_acc.set_ylim(0, 105)
    ax_acc.legend(fontsize=8)
    for i, (da, ra) in enumerate(zip(dir_acc, reg_acc)):
        ax_acc.text(i - width / 2, da + 1, f"{da:.0f}", ha="center", fontsize=7)
        ax_acc.text(i + width / 2, ra + 1, f"{ra:.0f}", ha="center", fontsize=7)

    # Loss panel
    total_loss = fm["test_total_loss"].values
    dir_loss = fm["test_dir_loss"].values
    reg_loss = fm["test_reg_loss"].values
    ret_loss = fm["test_ret_loss"].values
    w3 = 0.2
    ax_loss.bar(x - 1.5 * w3, total_loss, w3, color="#264653", label="Total", zorder=3)
    ax_loss.bar(x - 0.5 * w3, dir_loss, w3, color="#E63946", label="Direction", zorder=3)
    ax_loss.bar(x + 0.5 * w3, reg_loss, w3, color="#2A9D8F", label="Regime", zorder=3)
    ax_loss.bar(x + 1.5 * w3, ret_loss, w3, color="#E9C46A", label="Return", zorder=3)
    ax_loss.set_xticks(x)
    ax_loss.set_xticklabels(folds, rotation=30, ha="right", fontsize=9)
    ax_loss.set_ylabel("Loss", fontsize=10)
    ax_loss.set_title("Test Loss Components per Fold", fontsize=11)
    ax_loss.legend(fontsize=8)

    fig.tight_layout()
    out_path = out_dir / "03_crossfold_test_metrics.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ── Figure 4: Direction accuracy over epochs ──────────────────────────────────

def plot_direction_accuracy(run_dir: Path, out_dir: Path) -> Path:
    """Val direction accuracy per fold vs epoch with random-chance baseline."""
    folds = _discover_folds(run_dir)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(RANDOM_CHANCE_DIR * 100, color="black", lw=1.5, linestyle="--",
               alpha=0.5, label="Random chance (33.3%)", zorder=2)

    for ax_idx, fold_dir in enumerate(folds):
        df = _load_epoch_metrics(fold_dir)
        summary = _load_fold_summary(fold_dir)
        color = FOLD_COLORS[ax_idx % len(FOLD_COLORS)]
        epochs = df["epoch"].values
        val_acc = df["val_dir_acc"].values * 100
        best_acc = val_acc[summary["best_epoch"] - 1]
        ax.plot(epochs, val_acc, color=color, lw=1.8, label=f"{fold_dir.name} (best {best_acc:.1f}%)")
        ax.scatter([summary["best_epoch"]], [best_acc], color=color, s=50, zorder=5)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Val Direction Accuracy (%)", fontsize=11)
    ax.set_title("Direction Head: Validation Accuracy vs Epoch", fontsize=13)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    out_path = out_dir / "04_direction_accuracy.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ── Figure 5: Regime accuracy over epochs ────────────────────────────────────

def plot_regime_accuracy(run_dir: Path, out_dir: Path) -> Path:
    """Val regime accuracy per fold vs epoch."""
    folds = _discover_folds(run_dir)
    n_reg_classes = 4
    random_chance_reg = 1 / n_reg_classes

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(random_chance_reg * 100, color="black", lw=1.5, linestyle="--",
               alpha=0.5, label=f"Random chance ({random_chance_reg * 100:.0f}%)", zorder=2)

    for ax_idx, fold_dir in enumerate(folds):
        df = _load_epoch_metrics(fold_dir)
        summary = _load_fold_summary(fold_dir)
        color = FOLD_COLORS[ax_idx % len(FOLD_COLORS)]
        epochs = df["epoch"].values
        val_acc = df["val_reg_acc"].values * 100
        best_acc = val_acc[summary["best_epoch"] - 1]
        ax.plot(epochs, val_acc, color=color, lw=1.8, label=f"{fold_dir.name} (best {best_acc:.1f}%)")
        ax.scatter([summary["best_epoch"]], [best_acc], color=color, s=50, zorder=5)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Val Regime Accuracy (%)", fontsize=11)
    ax.set_title("Regime Head: Validation Accuracy vs Epoch", fontsize=13)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    out_path = out_dir / "05_regime_accuracy.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ── Figure 6: Latent-space PCA ────────────────────────────────────────────────

def plot_latent_pca(run_dir: Path, out_dir: Path) -> Path:
    """PCA of test-split latent vectors, coloured by fold."""
    folds = _discover_folds(run_dir)

    all_z: list[np.ndarray] = []
    fold_ids: list[int] = []

    for fold_idx, fold_dir in enumerate(folds):
        z_path = fold_dir / "latents_test.npy"
        if not z_path.exists():
            continue
        z = np.load(z_path)
        all_z.append(z)
        fold_ids.extend([fold_idx] * len(z))

    if not all_z:
        raise FileNotFoundError("No latents_test.npy files found under run_dir.")

    Z = np.concatenate(all_z, axis=0)
    fold_ids = np.array(fold_ids)

    pca = PCA(n_components=2, random_state=42)
    Z2 = pca.fit_transform(Z)
    var_exp = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(9, 7))
    for fold_idx, fold_dir in enumerate(folds):
        mask = fold_ids == fold_idx
        ax.scatter(
            Z2[mask, 0], Z2[mask, 1],
            color=FOLD_COLORS[fold_idx % len(FOLD_COLORS)],
            s=14, alpha=0.65, label=fold_dir.name, zorder=3,
        )

    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% var)", fontsize=11)
    ax.set_title("Latent Space PCA — Test-Split Representations by Fold", fontsize=13)
    ax.legend(fontsize=9, ncol=2, markerscale=1.8, loc="best")
    fig.tight_layout()
    out_path = out_dir / "06_latent_pca.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ── Figure 7: Best-val vs test loss (overfitting diagnostic) ──────────────────

def plot_val_vs_test_loss(run_dir: Path, out_dir: Path) -> Path:
    """
    Bar chart comparing best_val_total_loss vs test_total_loss per fold.
    A large gap (test >> val) signals overfitting or distribution shift.
    """
    fm = _load_fold_metrics(run_dir)
    folds = fm["fold"].tolist()
    x = np.arange(len(folds))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5))
    bars_val = ax.bar(x - width / 2, fm["best_val_total_loss"], width,
                      color="#457B9D", label="Best Val Loss", zorder=3)
    bars_test = ax.bar(x + width / 2, fm["test_total_loss"], width,
                       color="#E63946", label="Test Loss", zorder=3)

    for bar in bars_val:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)
    for bar in bars_test:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(folds, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Total Loss", fontsize=11)
    ax.set_title("Best Validation Loss vs Test Loss per Fold\n(gap = overfitting / regime shift)", fontsize=12)
    ax.legend(fontsize=10)
    fig.tight_layout()
    out_path = out_dir / "07_val_vs_test_loss.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_all(run_dir: str | Path, out_subdir: str = "plots") -> list[Path]:
    """
    Generate all 7 diagnostic figures for a completed transformer run.

    Args:
        run_dir    : Path to the variant output dir (e.g. data/transformer_runs/gating)
        out_subdir : Sub-directory name for plots, created inside run_dir.

    Returns:
        List of saved figure paths.
    """
    run_dir = Path(run_dir)
    out_dir = run_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    tasks = [
        ("01 learning curves",        plot_learning_curves),
        ("02 component losses",        plot_component_losses),
        ("03 cross-fold test metrics", plot_crossfold_test_metrics),
        ("04 direction accuracy",      plot_direction_accuracy),
        ("05 regime accuracy",         plot_regime_accuracy),
        ("06 val vs test loss",        plot_val_vs_test_loss),
    ]

    for name, fn in tasks:
        print(f"  Generating {name}...", flush=True)
        path = fn(run_dir, out_dir)
        saved.append(path)
        print(f"    → saved {path.relative_to(run_dir.parent.parent)}", flush=True)

    return saved
