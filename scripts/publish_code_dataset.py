#!/usr/bin/env python3
"""
Stage and publish the offline code Dataset (src/ + configs/) to Kaggle.

Dataset: simonhochwebde/rsna-knee-code (override via configs/kaggle_train.yaml).

Usage:
    python scripts/publish_code_dataset.py           # stage + create/version
    python scripts/publish_code_dataset.py --stage   # stage only (no API call)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from rsna_knee.utils.paths import project_root

CODE_DATASET_DIR = "kaggle/datasets/rsna-knee-code"
DEFAULT_SLUG = "simonhochwebde/rsna-knee-code"


def load_code_dataset_slug() -> str:
    """Prefer kaggle_train.yaml; fall back to kaggle_eda.yaml / default."""
    root = project_root()
    for name in ("kaggle_train.yaml", "kaggle_eda.yaml"):
        path = root / "configs" / name
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as f:
            cfg: dict[str, Any] = yaml.safe_load(f) or {}
        slug = cfg.get("code_dataset")
        if slug:
            return str(slug)
    return DEFAULT_SLUG


def mount_name(slug: str) -> str:
    return slug.split("/", 1)[-1]


def stage_code_dataset(slug: str | None = None) -> Path:
    """Copy src/rsna_knee + configs into the staging directory."""
    root = project_root()
    slug = slug or load_code_dataset_slug()
    stage = root / CODE_DATASET_DIR
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    shutil.copytree(root / "src" / "rsna_knee", stage / "src" / "rsna_knee")
    shutil.copytree(root / "configs", stage / "configs")

    meta = {
        "id": slug,
        "title": mount_name(slug),
        "licenses": [{"name": "CC0-1.0"}],
    }
    (stage / "dataset-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )
    n_py = len(list((stage / "src").rglob("*.py")))
    n_cfg = len(list((stage / "configs").glob("*.yaml")))
    print(f"Staged code dataset -> {stage} ({n_py} py, {n_cfg} yaml)")
    return stage


def push_code_dataset(slug: str | None = None) -> None:
    """Create or version the offline code Dataset on Kaggle."""
    slug = slug or load_code_dataset_slug()
    stage = project_root() / CODE_DATASET_DIR
    if not (stage / "dataset-metadata.json").is_file():
        stage_code_dataset(slug)

    def _create() -> None:
        create_cmd = [
            sys.executable,
            "-m",
            "kaggle",
            "datasets",
            "create",
            "-p",
            str(stage),
            "--dir-mode",
            "zip",
        ]
        print("Running:", " ".join(create_cmd))
        subprocess.run(create_cmd, check=True)

    status_cmd = [sys.executable, "-m", "kaggle", "datasets", "status", slug]
    status = subprocess.run(status_cmd, check=False, capture_output=True, text=True)
    status_text = (status.stdout or "") + (status.stderr or "")
    dataset_missing = status.returncode != 0 or any(
        s in status_text.lower()
        for s in ("403", "404", "not found", "forbidden", "does not exist")
    )
    if dataset_missing:
        print(f"Dataset {slug} not available yet — creating.")
        try:
            _create()
            return
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Failed to create code dataset. Common causes:\n"
                "  - Kaggle account needs phone verification (Settings -> Phone)\n"
                "  - Create the dataset once in the UI, then re-run\n"
                f"  - Confirm slug matches your user: {slug}"
            ) from exc

    version_cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "version",
        "-p",
        str(stage),
        "-m",
        f"sync {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        "--dir-mode",
        "zip",
    ]
    print("Running:", " ".join(version_cmd))
    result = subprocess.run(version_cmd, check=False, capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout or "Dataset versioned.")
        return

    combined = (result.stdout or "") + (result.stderr or "")
    print(combined)
    raise RuntimeError(f"Failed to version code dataset {slug} (exit {result.returncode})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        action="store_true",
        help="Only stage files under kaggle/datasets/rsna-knee-code (no Kaggle API)",
    )
    args = parser.parse_args()
    slug = load_code_dataset_slug()
    stage_code_dataset(slug)
    if not args.stage:
        push_code_dataset(slug)


if __name__ == "__main__":
    main()
