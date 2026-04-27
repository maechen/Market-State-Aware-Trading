"""Variant wrapper: run transformer training with default gating behavior."""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path
from typing import Sequence

THIS_DIR = Path(__file__).resolve().parent
sys.path = [path_entry for path_entry in sys.path if Path(path_entry or ".").resolve() != THIS_DIR]

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.models.transformer.run import main

WINDOW_SIZE_SWEEP: tuple[int, ...] = (20, 30, 40, 50, 60)
DEFAULT_SINGLE_OUTPUT_DIR = "data/transformer_npy/gating"
DEFAULT_SWEEP_OUTPUT_DIR = "data/transformer_npy/gating_sweep"


def _has_arg(argv: Sequence[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in argv)


def _get_arg_value(argv: Sequence[str], flag: str) -> str | None:
    for idx, arg in enumerate(argv):
        if arg == flag and idx + 1 < len(argv):
            return argv[idx + 1]
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
    return None


def _with_default_arg(argv: list[str], flag: str, value: str) -> list[str]:
    if _has_arg(argv, flag):
        return list(argv)
    return [*argv, flag, value]


def _mean_best_val_loss(output_dir: Path) -> float:
    """
    Aggregate fold-level best validation losses for model selection.

    We define "best window size" as the one with the lowest mean
    `best_val_total_loss` across all folds.
    """
    metrics_path = output_dir / "fold_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing fold metrics file: {metrics_path}")

    losses: list[float] = []
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw = row.get("best_val_total_loss")
            if raw is None or raw == "":
                continue
            losses.append(float(raw))

    if not losses:
        raise ValueError(
            f"No best_val_total_loss values found in: {metrics_path}"
        )
    return sum(losses) / len(losses)


if __name__ == "__main__":
    base_argv = [*sys.argv[1:]]
    base_argv = _with_default_arg(base_argv, "--variant", "gating")
    base_argv = _with_default_arg(base_argv, "--gate-mode", "cross_attn")
    base_argv = _with_default_arg(base_argv, "--n-heads", "4")

    if _has_arg(base_argv, "--window-size"):
        base_argv = _with_default_arg(base_argv, "--output-dir", DEFAULT_SINGLE_OUTPUT_DIR)
        print("[Launcher] Starting gating single-run training...", flush=True)
        main(argv=base_argv, default_variant="gating")
    else:
        print(
            "[Launcher] Starting gating window-size sweep: "
            f"{', '.join(str(w) for w in WINDOW_SIZE_SWEEP)}",
            flush=True,
        )
        output_root = _get_arg_value(base_argv, "--output-dir") or DEFAULT_SWEEP_OUTPUT_DIR
        run_argv = [
            arg
            for arg in base_argv
            if not (arg == "--output-dir" or arg.startswith("--output-dir="))
        ]
        output_root_path = Path(output_root)
        output_root_path.mkdir(parents=True, exist_ok=True)

        sweep_scores: list[tuple[int, float]] = []

        for window_size in WINDOW_SIZE_SWEEP:
            run_output = output_root_path / f"w{window_size}"
            print(
                f"[Launcher] Running window_size={window_size} -> {run_output}",
                flush=True,
            )
            main(
                argv=[
                    *run_argv,
                    "--window-size",
                    str(window_size),
                    "--output-dir",
                    str(run_output),
                ],
                default_variant="gating",
            )
            score = _mean_best_val_loss(run_output)
            sweep_scores.append((window_size, score))
            print(
                f"[Sweep] window_size={window_size} "
                f"mean_best_val_total_loss={score:.6f}"
            )

        # Primary criterion: lowest mean best validation loss.
        # Tie-breaker: smaller window size for a simpler model.
        best_window, best_score = min(sweep_scores, key=lambda item: (item[1], item[0]))

        print(
            f"[Sweep] Selected best window_size={best_window} "
            f"(mean_best_val_total_loss={best_score:.6f})."
        )

        for window_size, _score in sweep_scores:
            if window_size == best_window:
                continue
            drop_dir = output_root_path / f"w{window_size}"
            if drop_dir.exists():
                shutil.rmtree(drop_dir)
                print(f"[Sweep] Removed non-best output: {drop_dir}")

        print("[Launcher] Sweep complete.", flush=True)
