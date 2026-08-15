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
    python scripts/sync_kaggle_train.py --import-run PATH  # import downloaded run + summary
    python scripts/sync_kaggle_train.py --logs       # fetch last commit run log
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import textwrap
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from rsna_knee.utils.paths import project_root

GENERATED_BANNER = (
    "<!-- AUTO-GENERATED from notebooks/03_phase1_image_baseline.ipynb — do not edit. "
    "Run: python scripts/sync_kaggle_train.py --push -->\n"
)

RUNS_DIR = "kaggle/train/runs"


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


def _collect_vendor_files() -> list[tuple[str, str]]:
    """
    Files to embed in the Kaggle notebook, paths relative to repo root.

    Includes ``src/rsna_knee/**/*.py`` and ``configs/*.yaml`` so ``load_config``
    works offline under ``/kaggle/working/rsna_knee_vendor``.
    """
    root = project_root()
    files: list[tuple[str, str]] = []

    src_root = root / "src" / "rsna_knee"
    for path in sorted(src_root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        files.append((rel, path.read_text(encoding="utf-8")))

    configs_dir = root / "configs"
    for path in sorted(configs_dir.glob("*.yaml")):
        rel = path.relative_to(root).as_posix()
        files.append((rel, path.read_text(encoding="utf-8")))

    if not any(rel.startswith("src/rsna_knee/") for rel, _ in files):
        raise RuntimeError(f"No package Python files found under {src_root}")
    if not any(rel.startswith("configs/") for rel, _ in files):
        raise RuntimeError(f"No YAML configs found under {configs_dir}")
    return files


def _vendor_cell_source(files: list[tuple[str, str]]) -> str:
    """Build a notebook cell that materializes package + configs under /kaggle/working."""
    payload = json.dumps({rel: text for rel, text in files}, ensure_ascii=False)
    return textwrap.dedent(
        f'''\
        # Offline vendor: write package + configs (no git / no pip / no internet)
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
            for rel, text in _VENDORED.items():
                path = VENDOR_ROOT / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            src_dir = VENDOR_ROOT / "src"
            sys.path.insert(0, str(src_dir))
            n_py = sum(1 for r in _VENDORED if r.endswith(".py"))
            n_cfg = sum(1 for r in _VENDORED if r.startswith("configs/"))
            print(f"Vendored {{n_py}} modules + {{n_cfg}} configs -> {{VENDOR_ROOT}}")
        else:
            print("Local run — skipping vendor write (use repo src/ on PYTHONPATH).")
        '''
    )


def inject_offline_bootstrap(nb: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Prepend offline vendor cells; ensure setup finds local or vendored package."""
    out = strip_notebook_outputs(nb)
    files = _collect_vendor_files()

    bootstrap_md = {
        "cell_type": "markdown",
        "metadata": {"tags": ["kaggle-bootstrap"]},
        "source": [
            GENERATED_BANNER,
            "## Kaggle offline bootstrap\n",
            "\n",
            "Vendors `src/rsna_knee` + `configs/` into `/kaggle/working` — "
            "**no internet**, no git, no pip.\n",
            "\n",
            "Phase 1 trains **from scratch** (no pretrained Dataset required).\n",
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


def pull_logs(cfg: dict[str, Any]) -> Path:
    from kaggle.api.kaggle_api_extended import KaggleApi

    runs_dir = project_root() / RUNS_DIR
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_path = runs_dir / "last_commit.log"

    api = KaggleApi()
    api.authenticate()
    logs = api.kernels_logs(cfg["kernel_id"])
    out_path.write_text(logs, encoding="utf-8")
    print(f"Wrote commit log to {out_path} ({len(logs):,} chars)")
    return out_path


def _output_text(output: dict[str, Any]) -> str:
    otype = output.get("output_type")
    if otype == "stream":
        text = output.get("text", "")
        return text if isinstance(text, str) else "".join(text)
    if otype in ("execute_result", "display_data"):
        data = output.get("data", {})
        parts: list[str] = []
        if "text/plain" in data:
            tp = data["text/plain"]
            parts.append(tp if isinstance(tp, str) else "".join(tp))
        if "text/html" in data:
            parts.append("[html table/display omitted]")
        if "image/png" in data:
            parts.append("[figure]")
        return "\n".join(parts)
    if otype == "error":
        return f"ERROR: {output.get('ename')}: {output.get('evalue')}"
    return ""


def summarize_notebook(nb: dict[str, Any]) -> str:
    lines: list[str] = [
        f"Extracted: {datetime.now(UTC).isoformat()}",
        f"Cells: {len(nb['cells'])}",
        "",
    ]
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src_lines = "".join(cell.get("source", [])).strip().splitlines()
        header = src_lines[0][:100] if src_lines else "(empty)"
        lines.append(f"=== Cell {i}: {header} ===")
        outputs = cell.get("outputs", [])
        if not outputs:
            lines.append("(no outputs)")
            lines.append("")
            continue
        for output in outputs:
            text = _output_text(output).strip()
            if text:
                lines.append(text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def import_run(path: Path) -> tuple[Path, Path]:
    if not path.is_file():
        raise FileNotFoundError(f"Notebook not found: {path}")

    runs_dir = project_root() / RUNS_DIR
    runs_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    archived = runs_dir / f"run_{stamp}.ipynb"
    latest = runs_dir / "latest.ipynb"
    summary_path = runs_dir / "last_run_summary.txt"

    shutil.copy2(path, archived)
    shutil.copy2(path, latest)

    nb = json.loads(path.read_text(encoding="utf-8"))
    summary = summarize_notebook(nb)
    summary_path.write_text(summary, encoding="utf-8")

    print(f"Archived run  : {archived}")
    print(f"Latest run    : {latest}")
    print(f"Run summary   : {summary_path}")
    return latest, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push", action="store_true", help="Push kaggle/train to Kaggle after copy")
    parser.add_argument(
        "--logs",
        action="store_true",
        help="Fetch stdout/stderr from the last Save & Run All commit",
    )
    parser.add_argument(
        "--import-run",
        metavar="PATH",
        help="Copy a downloaded executed notebook into kaggle/train/runs/ and write a summary",
    )
    args = parser.parse_args()
    cfg = load_config()

    if args.import_run:
        import_run(Path(args.import_run))
        return

    if args.logs:
        pull_logs(cfg)
        return

    copy_to_kaggle(cfg)
    if args.push:
        push_to_kaggle(cfg)


if __name__ == "__main__":
    main()
