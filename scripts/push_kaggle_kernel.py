#!/usr/bin/env python3
"""
Thin submit helper: copy a source notebook into kaggle/{eda|train}/ and push.

Does not inject bootstrap cells — setup lives in the source notebooks themselves.
Does not publish the code Dataset (run publish_code_dataset.py separately).

Usage:
    python scripts/push_kaggle_kernel.py eda
    python scripts/push_kaggle_kernel.py train
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from rsna_knee.utils.paths import project_root

TARGETS: dict[str, str] = {
    "eda": "kaggle_eda.yaml",
    "train": "kaggle_train.yaml",
}


def load_cfg(target: str) -> dict[str, Any]:
    cfg_path = project_root() / "configs" / TARGETS[target]
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def strip_notebook_outputs(nb: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(nb)
    for cell in out["cells"]:
        cell["outputs"] = []
        cell["execution_count"] = None
        cell.pop("id", None)
    return out


def copy_notebook(cfg: dict[str, Any]) -> Path:
    root = project_root()
    src = root / cfg["source_notebook"]
    kernel_dir = root / cfg["kernel_dir"]
    code_file = cfg.get("code_file")
    if not code_file:
        meta_path = kernel_dir / "kernel-metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        code_file = meta["code_file"]
    dst = kernel_dir / code_file
    kernel_dir.mkdir(parents=True, exist_ok=True)

    nb = json.loads(src.read_text(encoding="utf-8"))
    clean = strip_notebook_outputs(nb)
    dst.write_text(json.dumps(clean, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Copied {src} -> {dst} ({len(clean['cells'])} cells, outputs stripped)")
    return dst


def push_kernel(cfg: dict[str, Any]) -> None:
    kernel_dir = project_root() / cfg["kernel_dir"]
    meta = kernel_dir / "kernel-metadata.json"
    if not meta.is_file():
        raise FileNotFoundError(f"Missing {meta}")
    cmd = [sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(kernel_dir)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=sorted(TARGETS), help="Which kernel to push")
    parser.add_argument(
        "--copy-only",
        action="store_true",
        help="Copy notebook into kaggle/ dir without pushing",
    )
    args = parser.parse_args()
    cfg = load_cfg(args.target)
    copy_notebook(cfg)
    if not args.copy_only:
        push_kernel(cfg)


if __name__ == "__main__":
    main()
