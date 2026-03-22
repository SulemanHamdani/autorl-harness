"""Local path bootstrap for running the scaffold from the repo root."""

from __future__ import annotations

from pathlib import Path
import sys


def bootstrap_autorl_paths() -> None:
    """Add scaffold directories to ``sys.path`` for local script execution."""

    repo_root = Path(__file__).resolve().parent
    scaffold_root = repo_root / "autorl"
    src_path = scaffold_root / "src"

    for path in (scaffold_root, src_path):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
