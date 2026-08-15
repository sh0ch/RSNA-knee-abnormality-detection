#!/usr/bin/env python3
"""
Sync the Phase 1 train notebook between the repo and Kaggle (offline-submittable).

Source of truth: notebooks/03_phase1_image_baseline.ipynb
Kaggle copy:      kaggle/train/train.ipynb  (generated — vendors src/rsna_knee)

Unlike the EDA sync, this does NOT git-clone or pip-install (internet is OFF
for competition submission). Package code is written into /kaggle/working.

Usage:
    python scripts/sync_kaggle_train.py              # regenerate only
    python scripts/sync_kaggle_train.py --push       # regenerate + kaggle kernels push
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from rsna_knee.utils.paths import project_root

GENERATED_BANNER = (
    "<!-- AUTO-GENERATED from notebooks/03_phase1_image_baseline.ipynb — do not edit. "
    "Run: python scripts/sync_kaggle_train.py --push -->\n"
)


def load_config() -> dict[str, Any]:
    cfg_path = project_root() / "configs" / "kaggle_train.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def strip_notebook_outputs(nb: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(nb)
    for cell in out["cells"]:
        cell["outputs"] = []
        cell["execution_count"] = None
        cell.pop("id", None)
    return out


def _collect_package_files() -> list[tuple[str, str]]:
    """Return (relative posix path under src/, file text) for all package modules."""
    src_root = project_root() / "src" / "rsna_knee"
    files: list[tuple[str, str]] = []
    for path in sorted(src_root.rglob("*.py")):
        rel = path.relative_to(project_root() / "src").as_posix()
        files.append((rel, path.read_text(encoding="utf-8")))
    if not files:
        raise RuntimeError(f"No Python files found under {src_root}")
    return files


def _vendor_cell_source(files: list[tuple[str, str]]) -> str:
    """Build a notebook cell that materializes src/rsna_knee under /kaggle/working."""
    # Embed as a JSON object of path -> source for compact, safe round-trip.
    payload = json.dumps({rel: text for rel, text in files}, ensure_ascii=False)
    # Escape for embedding inside a Python triple-quoted string via json already.
    return textwrap.dedent(
        f'''\
        # Offline vendor: write package sources (no git / no pip / no internet)
        import json
        import os
        import sys
        from pathlib import Path

        ON_KAGGLE = (
            os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
            or "KAGGLE_URL_BASE" in os.environ
            or os.path.isdir("/kaggle/working")
        )

        _VENDORED = json.loads({payload!r})

        if ON_KAGGLE:
            VENDOR_ROOT = Path("/kaggle/working/rsna_knee_vendor")
            src_dir = VENDOR_ROOT / "src"
            for rel, text in _VENDORED.items():
                path = src_dir / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            sys.path.insert(0, str(src_dir))
            print(f"Vendored {{len(_VENDORED)}} modules -> {{src_dir}}")
        else:
            print("Local run — skipping vendor write (use repo src/ on PYTHONPATH).")
        '''
    )


def inject_offline_bootstrap(nb: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Prepend offline vendor cells; ensure setup finds local or vendored package."""
    out = strip_notebook_outputs(nb)
    files = _collect_package_files()

    bootstrap_md = {
        "cell_type": "markdown",
        "metadata": {"tags": ["kaggle-bootstrap"]},
        "source": [
            GENERATED_BANNER,
            "## Kaggle offline bootstrap\n",
            "\n",
            "Vendors `src/rsna_knee` into `/kaggle/working` — **no internet**, no git, no pip.\n",
            "\n",
            f"Phase 1 trains **from scratch** (no pretrained Dataset required).\n",
            "\n",
            "Sync: `python scripts/sync_kaggle_train.py --push`\n",
        ],
    }

    bootstrap_code = {
        "cell_type": "code",
        "metadata": {"tags": ["kaggle-bootstrap"]},
        "source": [line + "\n" for line in _vendor_cell_source(files).split("\n")],
        "outputs": [],
        "execution_count": None,
    }

    # Patch setup cell repo-root resolution if present (same pattern as EDA).
    for cell in out["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell.get("source", []))
        if "default_data_root" not in src:
            continue
        local_block = """# Resolve repo root when running locally from notebooks/
for candidate in (Path.cwd(), Path.cwd().parent):
    if (candidate / "src" / "rsna_knee").is_dir():
        REPO_ROOT = candidate
        break
else:
    REPO_ROOT = Path.cwd().parent"""
        unified_block = """# Kaggle: vendored package above; local: search from notebooks/
VENDOR_SRC = Path("/kaggle/working/rsna_knee_vendor/src")
if VENDOR_SRC.is_dir():
    REPO_ROOT = VENDOR_SRC.parent
    if str(VENDOR_SRC) not in sys.path:
        sys.path.insert(0, str(VENDOR_SRC))
else:
    for candidate in (Path.cwd(), Path.cwd().parent):
        if (candidate / "src" / "rsna_knee").is_dir():
            REPO_ROOT = candidate
            break
    else:
        REPO_ROOT = Path.cwd().parent"""
        if local_block in src:
            cell["source"] = src.replace(local_block, unified_block).splitlines(keepends=True)
        break

    out["cells"] = [bootstrap_md, bootstrap_code] + out["cells"]
    return out


def write_kernel_metadata(cfg: dict[str, Any]) -> Path:
    """Ensure kaggle/train/kernel-metadata.json matches offline submit settings."""
    kernel_dir = project_root() / cfg["kernel_dir"]
    kernel_dir.mkdir(parents=True, exist_ok=True)
    meta_path = kernel_dir / "kernel-metadata.json"
    pretrained = cfg.get("pretrained_dataset")
    # Kaggle dataset_sources want "username/slug" form (optional for from-scratch Phase 1).
    dataset_sources = [pretrained] if pretrained else []
    meta = {
        "id": cfg["kernel_id"],
        "title": "RSNA Knee Phase 1 Image Baseline",
        "code_file": "train.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": dataset_sources,
        "kernel_sources": [],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": [],
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {meta_path}")
    return meta_path


def copy_to_kaggle(cfg: dict[str, Any]) -> Path:
    root = project_root()
    src = root / cfg["source_notebook"]
    dst = root / cfg["output_notebook"]
    dst.parent.mkdir(parents=True, exist_ok=True)

    nb = json.loads(src.read_text(encoding="utf-8"))
    kaggle_nb = inject_offline_bootstrap(nb, cfg)
    dst.write_text(json.dumps(kaggle_nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {dst} ({len(kaggle_nb['cells'])} cells) from {src}")
    write_kernel_metadata(cfg)
    return dst


def push_to_kaggle(cfg: dict[str, Any]) -> None:
    kernel_dir = project_root() / cfg["kernel_dir"]
    cmd = [sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(kernel_dir)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push", action="store_true", help="Push kaggle/train to Kaggle after copy")
    args = parser.parse_args()
    cfg = load_config()
    copy_to_kaggle(cfg)
    if args.push:
        push_to_kaggle(cfg)


if __name__ == "__main__":
    main()
