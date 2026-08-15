"""YAML configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from rsna_knee.utils.paths import project_root


def load_config(name: str = "default") -> dict[str, Any]:
    """
    Load a YAML config from ``configs/{name}.yaml``.

    Search order:
      1. ``{project_root}/configs/{name}.yaml`` (repo or Kaggle vendor root)
      2. ``/kaggle/working/rsna_knee_vendor/configs/{name}.yaml``
    """
    candidates = [
        project_root() / "configs" / f"{name}.yaml",
        Path("/kaggle/working/rsna_knee_vendor/configs") / f"{name}.yaml",
    ]
    for path in candidates:
        if path.is_file():
            with path.open(encoding="utf-8") as f:
                return yaml.safe_load(f)

    searched = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Config '{name}' not found.\nSearched:\n  - {searched}\n"
        "On Kaggle, re-push with scripts/sync_kaggle_train.py so configs/ are vendored."
    )
