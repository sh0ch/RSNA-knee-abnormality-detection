#!/usr/bin/env python3
"""
Publish Qwen2.5-1.5B-Instruct as Kaggle Dataset simonhochwebde/rsna-knee-llm.

Run after: python scripts/export_llm_weights.py

Usage:
    python scripts/publish_llm_dataset.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from rsna_knee.utils.paths import project_root

DEFAULT_SLUG = "simonhochwebde/rsna-knee-llm"
LLM_DIR = "data/llm/qwen2.5-1.5b-instruct"


def _stage_metadata(stage: Path, slug: str) -> None:
    meta = {
        "id": slug,
        "title": slug.split("/", 1)[-1],
        "licenses": [{"name": "Apache 2.0"}],
    }
    (stage / "dataset-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )


def push_llm_dataset(slug: str = DEFAULT_SLUG) -> None:
    stage = project_root() / LLM_DIR
    if not (stage / "config.json").is_file():
        raise SystemExit(
            f"No LLM weights at {stage}. Run: python scripts/export_llm_weights.py"
        )
    _stage_metadata(stage, slug)

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
        print(f"Dataset {slug} not available yet — creating (~3 GB upload).")
        try:
            _create()
            return
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Failed to create LLM dataset. Common causes:\n"
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
        f"qwen2.5-1.5b-instruct {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        "--dir-mode",
        "zip",
    ]
    print("Running:", " ".join(version_cmd))
    result = subprocess.run(version_cmd, check=False, capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout or "Dataset versioned.")
        return
    print((result.stdout or "") + (result.stderr or ""))
    raise RuntimeError(f"Failed to version LLM dataset {slug} (exit {result.returncode})")


def main() -> None:
    push_llm_dataset()
    print("Attach Dataset rsna-knee-llm on the Phase 2 kernel, then restart Jupyter Server.")


if __name__ == "__main__":
    main()
