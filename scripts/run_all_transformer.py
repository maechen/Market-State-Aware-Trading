"""
Master orchestration script: train all transformer variants, then plot diagnostics and MAE.

Runs the following pipeline in order:
  1. Train gating variant       → data/transformer_npy/gating/
  2. Train no_gating variant    → data/transformer_npy/no_gating/
  3. Train no_sentiment variant → data/transformer_npy/no_sentiment/
  4. Plot diagnostic figures for each variant (learning curves, component losses, …)
  5. Plot cross-variant MAE summary (direction, regime, return MAE per fold)

Usage (from repo root):
    python scripts/run_all_transformer.py
    python scripts/run_all_transformer.py --epochs 100 --window-size 20
    python scripts/run_all_transformer.py --skip-training        # only re-plot
    python scripts/run_all_transformer.py --variants gating,no_gating  # subset
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
SRC_ROOT = REPO_ROOT / "src"

# Ensure the scripts/ directory is NOT on sys.path so that plain `import run`
# cannot accidentally shadow anything, while the repo root IS available.
sys.path = [p for p in sys.path if Path(p or ".").resolve() != THIS_DIR]
for _p in (str(REPO_ROOT), str(SRC_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.models.transformer.run import main as transformer_main
from src.visualization.transformer_diagnostics import generate_all as diagnostics_generate_all
from src.visualization.transformer_mae import generate_mae_artifacts


# ---------------------------------------------------------------------------
# Variant definitions (mirrors run_gating / run_no_gating / run_no_sentiment)
# ---------------------------------------------------------------------------

VARIANTS: dict[str, dict] = {
    "gating": {
        "default_variant": "gating",
        "output_dir": "data/transformer_npy/gating",
        "fixed_args": [
            "--variant", "gating",
            "--gate-mode", "cross_attn",
            "--n-heads", "4",
        ],
    },
    "no_gating": {
        "default_variant": "no_gating",
        "output_dir": "data/transformer_npy/no_gating",
        "fixed_args": [
            "--variant", "no_gating",
            "--gate-mode", "master",
            "--n-heads", "4",
        ],
    },
    "no_sentiment": {
        "default_variant": "no_sentiment",
        "output_dir": "data/transformer_npy/no_sentiment",
        "fixed_args": [
            "--variant", "no_sentiment",
            "--gate-mode", "cross_attn",
            "--n-heads", "4",
        ],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_flag(argv: list[str], flag: str) -> bool:
    return any(a == flag or a.startswith(f"{flag}=") for a in argv)


def _inject_default(argv: list[str], flag: str, value: str) -> list[str]:
    """Add flag/value only when the flag is not already present."""
    if _has_flag(argv, flag):
        return argv
    return [*argv, flag, value]


def _elapsed(start: float) -> str:
    secs = time.monotonic() - start
    mins, secs = divmod(int(secs), 60)
    return f"{mins}m {secs:02d}s" if mins else f"{secs}s"


def _banner(msg: str) -> None:
    width = min(80, len(msg) + 4)
    bar = "=" * width
    print(f"\n{bar}\n  {msg}\n{bar}", flush=True)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def run_training_variant(
    variant_name: str,
    variant_cfg: dict,
    extra_argv: list[str],
) -> None:
    """Build the argv list for one variant and call the transformer main()."""
    argv = list(variant_cfg["fixed_args"])
    argv = _inject_default(argv, "--output-dir", variant_cfg["output_dir"])

    # Default training budget: 100 epochs, 20-day window.
    # User-supplied values in extra_argv take precedence (checked by _has_flag).
    argv = _inject_default(argv, "--epochs", "100")
    argv = _inject_default(argv, "--window-size", "20")

    # Forward any extra CLI args the user passed (epochs, window-size, folds, …).
    # User-supplied flags take precedence over the defaults injected above.
    for arg in extra_argv:
        # Avoid duplicating fixed variant-specific flags.
        flag = arg.split("=")[0]
        if not _has_flag(argv, flag):
            argv.append(arg)

    print(f"  argv: {' '.join(argv)}", flush=True)
    transformer_main(argv=argv, default_variant=variant_cfg["default_variant"])


# ---------------------------------------------------------------------------
# Diagnostics step
# ---------------------------------------------------------------------------

def run_diagnostics(variant_name: str, output_dir: Path) -> None:
    """Generate learning-curve / component-loss / fold diagnostic plots."""
    if not output_dir.exists():
        print(
            f"  [WARN] Output directory not found, skipping diagnostics: {output_dir}",
            flush=True,
        )
        return

    out_subdir = "plots"
    print(f"  Saving figures to: {output_dir / out_subdir}", flush=True)
    saved = diagnostics_generate_all(output_dir, out_subdir=out_subdir)
    print(f"  {len(saved)} figure(s) saved.", flush=True)


# ---------------------------------------------------------------------------
# MAE step
# ---------------------------------------------------------------------------

def run_mae(runs_root: Path, variants: list[str], out_dir: Path) -> None:
    """Compute and plot MAE across variants from test_predictions.csv files."""
    print(f"  runs_root : {runs_root}", flush=True)
    print(f"  variants  : {variants}", flush=True)
    print(f"  out_dir   : {out_dir}", flush=True)

    outputs = generate_mae_artifacts(
        runs_root=runs_root,
        variants=variants,
        out_dir=out_dir,
    )
    print(f"  {len(outputs)} artifact(s) saved:", flush=True)
    for key, path in outputs.items():
        print(f"    {key}: {path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train all transformer variants and generate diagnostic + MAE plots. "
            "Any unknown flags are forwarded verbatim to the transformer training scripts "
            "(e.g. --epochs, --window-size, --folds)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="gating,no_gating,no_sentiment",
        help="Comma-separated subset of variants to run (default: all three).",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        default=False,
        help="Skip transformer training and only regenerate plots from existing outputs.",
    )
    parser.add_argument(
        "--skip-diagnostics",
        action="store_true",
        default=False,
        help="Skip per-variant diagnostic plots.",
    )
    parser.add_argument(
        "--skip-mae",
        action="store_true",
        default=False,
        help="Skip cross-variant MAE plot.",
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default="data/transformer_npy",
        help="Root directory for transformer output (variant sub-dirs live here).",
    )
    parser.add_argument(
        "--mae-out-dir",
        type=str,
        default="data/transformer_npy/mae_plots",
        help="Directory where MAE CSV summaries and plot are saved.",
    )
    return parser.parse_known_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args, extra_argv = parse_args()

    selected_variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown_variants = [v for v in selected_variants if v not in VARIANTS]
    if unknown_variants:
        print(
            f"[ERROR] Unknown variant(s): {unknown_variants}. "
            f"Valid choices: {list(VARIANTS.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    runs_root = Path(args.runs_root)
    if not runs_root.is_absolute():
        runs_root = REPO_ROOT / runs_root

    mae_out_dir = Path(args.mae_out_dir)
    if not mae_out_dir.is_absolute():
        mae_out_dir = REPO_ROOT / mae_out_dir

    total_start = time.monotonic()

    _banner(f"Transformer pipeline — variants: {', '.join(selected_variants)}")

    # ------------------------------------------------------------------ #
    # Step 1: Training                                                     #
    # ------------------------------------------------------------------ #
    if args.skip_training:
        print("\n[Step 1/3] Training SKIPPED (--skip-training set).", flush=True)
    else:
        print(f"\n[Step 1/3] Training {len(selected_variants)} variant(s)…", flush=True)
        for idx, variant_name in enumerate(selected_variants, 1):
            variant_cfg = VARIANTS[variant_name]
            variant_start = time.monotonic()
            _banner(
                f"Training [{idx}/{len(selected_variants)}]: {variant_name}"
            )
            print(
                f"  output_dir: {variant_cfg['output_dir']}",
                flush=True,
            )
            run_training_variant(variant_name, variant_cfg, extra_argv)
            print(
                f"\n[Training] {variant_name} done in {_elapsed(variant_start)}.",
                flush=True,
            )

    # ------------------------------------------------------------------ #
    # Step 2: Per-variant diagnostic plots                                 #
    # ------------------------------------------------------------------ #
    if args.skip_diagnostics:
        print("\n[Step 2/3] Diagnostics SKIPPED (--skip-diagnostics set).", flush=True)
    else:
        print(
            f"\n[Step 2/3] Generating diagnostic plots for {len(selected_variants)} variant(s)…",
            flush=True,
        )
        for idx, variant_name in enumerate(selected_variants, 1):
            variant_cfg = VARIANTS[variant_name]
            output_dir = runs_root / variant_name
            diag_start = time.monotonic()
            _banner(
                f"Diagnostics [{idx}/{len(selected_variants)}]: {variant_name}"
            )
            run_diagnostics(variant_name, output_dir)
            print(
                f"[Diagnostics] {variant_name} done in {_elapsed(diag_start)}.",
                flush=True,
            )

    # ------------------------------------------------------------------ #
    # Step 3: Cross-variant MAE                                            #
    # ------------------------------------------------------------------ #
    if args.skip_mae:
        print("\n[Step 3/3] MAE plots SKIPPED (--skip-mae set).", flush=True)
    else:
        _banner("MAE plots [Step 3/3]")
        mae_start = time.monotonic()
        try:
            run_mae(
                runs_root=runs_root,
                variants=selected_variants,
                out_dir=mae_out_dir,
            )
        except FileNotFoundError as exc:
            print(
                f"  [WARN] MAE skipped — missing prediction files: {exc}",
                flush=True,
            )
            print(
                "  Re-run without --skip-training to generate test_predictions.csv first.",
                flush=True,
            )
        print(f"[MAE] Done in {_elapsed(mae_start)}.", flush=True)

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #
    _banner(f"Pipeline complete — total time: {_elapsed(total_start)}")
    print(
        "Outputs written to:\n"
        f"  Training + latents : {runs_root}/<variant>/\n"
        f"  Diagnostic plots   : {runs_root}/<variant>/plots/\n"
        f"  MAE plots          : {mae_out_dir}/",
        flush=True,
    )


if __name__ == "__main__":
    main()
