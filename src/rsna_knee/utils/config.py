"""YAML configuration loading."""

from __future__ import annotations

from typing import Any

import yaml

from rsna_knee.utils.paths import project_root


def load_config(name: str = "default") -> dict[str, Any]:
    """Load a YAML config from configs/{name}.yaml."""
    path = project_root() / "configs" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
