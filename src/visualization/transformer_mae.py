"""
MAE diagnostics for transformer test predictions.

Reads per-fold `test_predictions.csv` artifacts and computes:
    - Direction MAE on class indices |y_dir_true - y_dir_pred|
    - Regime MAE on class indices |y_reg_true - y_reg_pred|

Outputs aggregate CSV summaries and a bar plot across variants.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "figure.dpi": 130,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def _discover_fold_prediction_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("fold*/test_predictions.csv"))


def _compute_fold_mae(pred_df: pd.DataFrame) -> dict[str, float]:
    dir_mae = float((pred_df["y_dir_true"] - pred_df["y_dir_pred"]).abs().mean())
    reg_mae = float((pred_df["y_reg_true"] - pred_df["y_reg_pred"]).abs().mean())
    return {
        "dir_mae": dir_mae,
        "reg_mae": reg_mae,
    }


def build_mae_tables(
    runs_root: str | Path,
    variants: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build fold-level and variant-level MAE tables from test_predictions.csv files.
    """
    runs_root = Path(runs_root)
    fold_rows: list[dict[str, float | str | int]] = []

    for variant in variants:
        run_dir = runs_root / variant
        pred_files = _discover_fold_prediction_files(run_dir)
        if not pred_files:
            raise FileNotFoundError(
                f"No fold*/test_predictions.csv found under: {run_dir}. "
                "Re-run transformer training after enabling prediction export."
            )

        for pred_path in pred_files:
            fold_name = pred_path.parent.name
            pred_df = pd.read_csv(pred_path)
            required_cols = {
                "y_dir_true",
                "y_dir_pred",
                "y_reg_true",
                "y_reg_pred",
            }
            missing = sorted(required_cols - set(pred_df.columns))
            if missing:
                raise KeyError(f"{pred_path} missing required columns: {missing}")

            maes = _compute_fold_mae(pred_df)
            fold_rows.append(
                {
                    "variant": variant,
                    "fold": fold_name,
                    "num_samples": int(len(pred_df)),
                    **maes,
                }
            )

    fold_df = pd.DataFrame(fold_rows).sort_values(["variant", "fold"]).reset_index(drop=True)
    summary_df = (
        fold_df.groupby("variant", as_index=False)[["dir_mae", "reg_mae"]]
        .mean()
        .sort_values("variant")
        .reset_index(drop=True)
    )
    return fold_df, summary_df


def plot_variant_mae(summary_df: pd.DataFrame, output_path: Path) -> Path:
    variants = summary_df["variant"].tolist()
    x = np.arange(len(variants))
    w = 0.3

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, summary_df["dir_mae"], width=w, color="#E63946", label="Direction MAE")
    ax.bar(x + w / 2, summary_df["reg_mae"], width=w, color="#2A9D8F", label="Regime MAE")

    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontsize=10)
    ax.set_ylabel("Mean Absolute Error", fontsize=11)
    ax.set_title("Transformer Test MAE by Variant", fontsize=13)
    ax.legend(fontsize=9)

    for idx, row in summary_df.iterrows():
        ax.text(x[idx] - w / 2, row["dir_mae"] + 0.005, f"{row['dir_mae']:.3f}", ha="center", fontsize=8)
        ax.text(x[idx] + w / 2, row["reg_mae"] + 0.005, f"{row['reg_mae']:.3f}", ha="center", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def generate_mae_artifacts(
    runs_root: str | Path,
    variants: list[str],
    out_dir: str | Path,
) -> dict[str, Path]:
    """
    Generate fold-level/summary MAE CSVs and variant comparison plot.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_df, summary_df = build_mae_tables(runs_root=runs_root, variants=variants)
    fold_csv = out_dir / "transformer_mae_by_fold.csv"
    summary_csv = out_dir / "transformer_mae_summary.csv"
    plot_png = out_dir / "transformer_mae_by_variant.png"

    fold_df.to_csv(fold_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    plot_variant_mae(summary_df, plot_png)

    return {
        "fold_csv": fold_csv,
        "summary_csv": summary_csv,
        "plot_png": plot_png,
    }
