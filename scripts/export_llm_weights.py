#!/usr/bin/env python3
"""
Download Qwen2.5-1.5B-Instruct for offline local use (Apache-2.0).

On Kaggle, prefer Add Input → Models → Qwen/Qwen2.5-1.5B-Instruct instead of
uploading this directory as a Dataset.

Usage:
    python scripts/export_llm_weights.py
    python scripts/export_llm_weights.py --publish
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from rsna_knee.reports.llm import DEFAULT_LLM_MODEL_ID
from rsna_knee.utils.paths import project_root

LLM_DIRNAME = "qwen2.5-1.5b-instruct"


def llm_weights_dir() -> Path:
    return project_root() / "data" / "llm" / LLM_DIRNAME


def export_qwen_instruct(output: Path, model_id: str = DEFAULT_LLM_MODEL_ID) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from exc

    output.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {model_id} -> {output}")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(output),
        ignore_patterns=["*.md", ".gitattributes"],
    )
    cfg = output / "config.json"
    if not cfg.is_file():
        raise FileNotFoundError(f"Download finished but {cfg} is missing")
    size_mb = sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) / (1024 * 1024)
    print(f"Wrote {output} ({size_mb:.0f} MB)")
    print("License: Qwen2.5-1.5B-Instruct — Apache-2.0 (Alibaba). Document in PROJECT_LOG.")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_LLM_MODEL_ID)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory for model files (default: data/llm/qwen2.5-1.5b-instruct)",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="After download, publish Dataset simonhochwebde/rsna-knee-llm",
    )
    args = parser.parse_args()
    out = args.output or llm_weights_dir()
    if not out.is_absolute():
        out = project_root() / out
    export_qwen_instruct(out, model_id=args.model_id)
    print("Next: python scripts/publish_llm_dataset.py")
    print("Then attach Dataset rsna-knee-llm on the Phase 2 kernel and restart Jupyter Server.")
    if args.publish:
        cmd = [sys.executable, str(project_root() / "scripts" / "publish_llm_dataset.py")]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
