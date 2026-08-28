#!/usr/bin/env python3
"""
Publish exported ConvNeXt-Tiny weights as Kaggle Dataset
simonhochwebde/rsna-knee-pretrained.

Run after: python scripts/export_pretrained_weights.py

Usage:
    python scripts/publish_pretrained_weights.py
    python scripts/publish_pretrained_weights.py --stage
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from rsna_knee.models.weights import CONVNEXT_TINY_FILENAME, pretrained_weights_dir
from rsna_knee.utils.paths import project_root

DEFAULT_SLUG = "simonhochwebde/rsna-knee-pretrained"


def load_pretrained_dataset_slug() -> str:
    path = project_root() / "configs" / "kaggle_phase2_train.yaml"
    if path.is_file():
        with path.open(encoding="utf-8") as f:
            cfg: dict[str, Any] = yaml.safe_load(f) or {}
        slug = cfg.get("pretrained_dataset")
        if slug:
            return str(slug)
    return DEFAULT_SLUG


def _stage_metadata(stage: Path, slug: str) -> None:
    meta = {
        "id": slug,
        "title": slug.split("/", 1)[-1],
        "licenses": [{"name": "other"}],
    }
    (stage / "dataset-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )


def stage_pretrained_dataset(slug: str | None = None) -> Path:
    slug = slug or load_pretrained_dataset_slug()
    stage = pretrained_weights_dir()
    weights = stage / CONVNEXT_TINY_FILENAME
    if not weights.is_file():
        raise SystemExit(
            f"No pretrained weights at {weights}. "
            "Run: python scripts/export_pretrained_weights.py"
        )
    _stage_metadata(stage, slug)
    size_mb = weights.stat().st_size / (1024 * 1024)
    print(f"Staged pretrained dataset -> {stage} ({weights.name}, {size_mb:.1f} MB)")
    return stage


def push_pretrained_dataset(slug: str | None = None) -> None:
    slug = slug or load_pretrained_dataset_slug()
    stage = stage_pretrained_dataset(slug)

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
        print(f"Dataset {slug} not available yet — creating (~109 MB upload).")
        try:
            _create()
            return
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Failed to create pretrained dataset. Common causes:\n"
                "  - Kaggle account needs phone verification\n"
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
        f"convnext_tiny_imagenet {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        "--dir-mode",
        "zip",
    ]
    print("Running:", " ".join(version_cmd))
    result = subprocess.run(version_cmd, check=False, capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout or "Dataset versioned.")
        return
    print((result.stdout or "") + (result.stderr or ""))
    raise RuntimeError(f"Failed to version pretrained dataset {slug} (exit {result.returncode})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        action="store_true",
        help="Write dataset-metadata.json only (no Kaggle API)",
    )
    args = parser.parse_args()
    slug = load_pretrained_dataset_slug()
    if args.stage:
        stage_pretrained_dataset(slug)
        return
    push_pretrained_dataset(slug)
    print(
        f"Published {CONVNEXT_TINY_FILENAME} -> {slug}. "
        "Confirm it is attached on the Phase 2 kernel, then restart Jupyter Server."
    )


if __name__ == "__main__":
    main()
