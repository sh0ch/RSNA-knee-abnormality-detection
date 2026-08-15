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
      1. ``{project_root}/configs/{name}.yaml`` (local repo or Dataset root)
      2. ``/kaggle/input/rsna-knee-code/configs/{name}.yaml``
    """
    candidates = [
        project_root() / "configs" / f"{name}.yaml",
        Path("/kaggle/input/rsna-knee-code/configs") / f"{name}.yaml",
    ]
    for path in candidates:
        if path.is_file():
            with path.open(encoding="utf-8") as f:
                return yaml.safe_load(f)

    searched = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Config '{name}' not found.\nSearched:\n  - {searched}\n"
        "On Kaggle, attach Dataset rsna-knee-code "
        "(python scripts/publish_code_dataset.py) so configs/ are under "
        "/kaggle/input/rsna-knee-code/configs/."
    )
