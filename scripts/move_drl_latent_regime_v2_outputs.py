"""
Move latent-regime v2 baseline outputs into their own folder.

Files under data/baselines/drl_latent_regime whose names start with
``rl_only`` are moved to data/baselines/drl_latent_regime_v2 with that prefix
renamed to ``drl_latent_regime_v2``. ``run_config`` files are moved too, but
their names are kept unchanged.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / "data" / "baselines" / "drl_latent_regime"
TARGET_DIR = PROJECT_ROOT / "data" / "baselines" / "drl_latent_regime_v2"
OLD_PREFIX = "rl_only"
NEW_PREFIX = "drl_latent_regime_v2"


def _target_path(source_path: Path) -> Path | None:
    if source_path.name.startswith(OLD_PREFIX):
        relative_parent = source_path.parent.relative_to(SOURCE_DIR)
        new_name = NEW_PREFIX + source_path.name[len(OLD_PREFIX) :]
        return TARGET_DIR / relative_parent / new_name

    if source_path.name.startswith("run_config"):
        relative_path = source_path.relative_to(SOURCE_DIR)
        return TARGET_DIR / relative_path

    return None


def _iter_moves() -> list[tuple[Path, Path]]:
    moves: list[tuple[Path, Path]] = []
    for source_path in SOURCE_DIR.rglob("*"):
        if not source_path.is_file():
            continue

        target_path = _target_path(source_path)
        if target_path is not None:
            moves.append((source_path, target_path))

    return sorted(moves, key=lambda move: str(move[0]))


def move_outputs(dry_run: bool = False) -> int:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source directory does not exist: {SOURCE_DIR}")

    moves = _iter_moves()
    if not moves:
        print("No matching files found.")
        return 0

    collisions = [target for _, target in moves if target.exists()]
    if collisions:
        collision_list = "\n".join(f"  {path}" for path in collisions)
        raise FileExistsError(f"Refusing to overwrite existing files:\n{collision_list}")

    for source_path, target_path in moves:
        print(f"{source_path.relative_to(PROJECT_ROOT)} -> {target_path.relative_to(PROJECT_ROOT)}")
        if dry_run:
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_path), str(target_path))

    action = "Would move" if dry_run else "Moved"
    print(f"{action} {len(moves)} file(s).")
    return len(moves)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Move rl_only outputs from data/baselines/drl_latent_regime to "
            "data/baselines/drl_latent_regime_v2 and rename the prefix."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the files that would be moved without changing anything.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    move_outputs(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
