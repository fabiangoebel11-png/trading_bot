"""Remove known V2/V3-transition model directories after an explicit review.

Run without arguments for a dry run. Deletion requires ``--apply`` and the
explicit ``--confirm DELETE`` token. Active V3 checkpoint directories are
protected by the allowlist below.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_MODEL_DIRS = (
    Path("data/models/tcn_crypto_1h"),
    Path("models/checkpoints"),
)
LEGACY_MODEL_DIRS = (
    Path("data/models/model1a_crypto"),
    Path("data/models/model1b_equity"),
    Path("data/models/tmp_wf4_prod_smoke"),
)


def _checked_directory(relative_path: Path, *, protect_active: bool = True) -> Path:
    candidate = PROJECT_ROOT / relative_path
    if candidate.is_symlink():
        raise RuntimeError(f"Refusing to follow symlink: {relative_path}")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise RuntimeError(f"Refusing path outside project root: {relative_path}")
    if protect_active:
        for active_path in ACTIVE_MODEL_DIRS:
            active = (PROJECT_ROOT / active_path).resolve()
            if resolved == active or active.is_relative_to(resolved):
                raise RuntimeError(f"Refusing to remove active V3 model path: {relative_path}")
    if candidate.exists() and not candidate.is_dir():
        raise RuntimeError(f"Expected a directory, found another file type: {relative_path}")
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="delete the listed legacy directories")
    parser.add_argument("--confirm", help="must be exactly DELETE when using --apply")
    args = parser.parse_args()

    try:
        targets = [(relative, _checked_directory(relative)) for relative in LEGACY_MODEL_DIRS]
        active = [(relative, _checked_directory(relative, protect_active=False)) for relative in ACTIVE_MODEL_DIRS]
    except RuntimeError as exc:
        parser.error(str(exc))

    print("Active V3 model directories (preserved):")
    for relative, path in active:
        print(f"  {'present' if path.is_dir() else 'missing':7} {relative}")

    existing = [(relative, path) for relative, path in targets if path.is_dir()]
    print("Legacy directories:")
    for relative, path in targets:
        status = "would remove" if path.is_dir() and not args.apply else "remove" if path.is_dir() else "absent"
        print(f"  {status:11} {relative}")

    if not args.apply:
        print("Dry run only. No files were removed. Review the list, then pass --apply --confirm DELETE to proceed.")
        return 0
    if args.confirm != "DELETE":
        parser.error("deletion requires --confirm DELETE")

    for relative, path in existing:
        shutil.rmtree(path)
        print(f"removed       {relative}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())