#!/usr/bin/env python3
"""
Sync the Phase 1 train notebook between the repo and Kaggle (offline-submittable).

Source of truth: notebooks/03_phase1_image_baseline.ipynb
Kaggle copy:      kaggle/train/train.ipynb

Offline package delivery: stage ``src/`` + ``configs/`` into a Kaggle Dataset
(``code_dataset``) and attach it to the kernel. The notebook bootstrap only
adds that mount to ``sys.path`` — no embedded source blobs.

Usage:
    python scripts/sync_kaggle_train.py              # stage dataset + regenerate notebook
    python scripts/sync_kaggle_train.py --push       # stage, version dataset, push kernel
    python scripts/sync_kaggle_train.py --import-run PATH
    python scripts/sync_kaggle_train.py --logs
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
CODE_DATASET_DIR = "kaggle/datasets/rsna-knee-code"


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


def _code_dataset_slug(cfg: dict[str, Any]) -> str:
    """Return owner/slug for the offline code Dataset."""
    return str(cfg.get("code_dataset") or "simonhochwebde/rsna-knee-code")


def _code_dataset_mount_name(cfg: dict[str, Any]) -> str:
    """Folder name under /kaggle/input (slug without owner)."""
    return _code_dataset_slug(cfg).split("/", 1)[-1]


def stage_code_dataset(cfg: dict[str, Any]) -> Path:
    """Copy src/ + configs/ into the Kaggle Dataset staging directory."""
    root = project_root()
    stage = root / CODE_DATASET_DIR
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    shutil.copytree(root / "src" / "rsna_knee", stage / "src" / "rsna_knee")
    shutil.copytree(root / "configs", stage / "configs")

    slug = _code_dataset_slug(cfg)
    meta = {
        "id": slug,
        "title": _code_dataset_mount_name(cfg),
        "licenses": [{"name": "MIT"}],
    }
    (stage / "dataset-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )
    n_py = len(list((stage / "src").rglob("*.py")))
    n_cfg = len(list((stage / "configs").glob("*.yaml")))
    print(f"Staged code dataset -> {stage} ({n_py} py, {n_cfg} yaml)")
    return stage


def push_code_dataset(cfg: dict[str, Any]) -> None:
    """Create or version the offline code Dataset on Kaggle."""
    stage = project_root() / CODE_DATASET_DIR
    if not (stage / "dataset-metadata.json").is_file():
        stage_code_dataset(cfg)

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
    # First time: create instead of version.
    if "404" in combined or "does not exist" in combined.lower() or "not found" in combined.lower():
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
        return

    raise RuntimeError(f"Failed to push code dataset (exit {result.returncode})")


def _bootstrap_cell_source(cfg: dict[str, Any]) -> str:
    """Thin bootstrap: point sys.path at the attached code Dataset (no embedded sources)."""
    mount = _code_dataset_mount_name(cfg)
    slug = _code_dataset_slug(cfg)
    return textwrap.dedent(
        f'''\
        # Offline bootstrap: use attached code Dataset (no git / no pip / no embedded sources)
        import os
        import sys
        from pathlib import Path

        ON_KAGGLE = (
            os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
            or "KAGGLE_URL_BASE" in os.environ
            or os.path.isdir("/kaggle/working")
        )

        CODE_DATASET = "{slug}"
        CODE_MOUNT = Path("/kaggle/input/{mount}")

        if ON_KAGGLE:
            if not CODE_MOUNT.is_dir():
                raise FileNotFoundError(
                    f"Code dataset not mounted at {{CODE_MOUNT}}. "
                    f"Attach '{{CODE_DATASET}}' in notebook Input "
                    "(and re-run scripts/sync_kaggle_train.py --push after src/ changes)."
                )
            src_dir = CODE_MOUNT / "src"
            if not (src_dir / "rsna_knee").is_dir():
                raise FileNotFoundError(f"Expected package at {{src_dir / 'rsna_knee'}}")
            sys.path.insert(0, str(src_dir))
            print(f"Using code dataset: {{CODE_MOUNT}}")
        else:
            print("Local run — skipping Kaggle code-dataset bootstrap.")
        '''
    )


def inject_offline_bootstrap(nb: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Prepend thin offline bootstrap; patch setup cell for code-dataset root."""
    out = strip_notebook_outputs(nb)
    mount = _code_dataset_mount_name(cfg)
    slug = _code_dataset_slug(cfg)

    bootstrap_md = {
        "cell_type": "markdown",
        "metadata": {"tags": ["kaggle-bootstrap"]},
        "source": [
            GENERATED_BANNER,
            "## Kaggle offline bootstrap\n",
            "\n",
            f"Uses attached Dataset `{slug}` (`src/` + `configs/`). "
            "**No internet**, no git, no pip, no embedded source blobs.\n",
            "\n",
            "Phase 1 trains **from scratch**.\n",
            "\n",
            "Sync: `python scripts/sync_kaggle_train.py --push`\n",
        ],
    }

    bootstrap_code = {
        "cell_type": "code",
        "metadata": {"tags": ["kaggle-bootstrap"]},
        "source": [line + "\n" for line in _bootstrap_cell_source(cfg).split("\n")],
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
        # Also replace older vendor-based unified block if present.
        vendor_block = """# Kaggle: vendored package above; local: search from notebooks/
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
        unified_block = f"""# Kaggle: attached code Dataset; local: search from notebooks/
CODE_MOUNT = Path("/kaggle/input/{mount}")
if CODE_MOUNT.is_dir() and (CODE_MOUNT / "src" / "rsna_knee").is_dir():
    REPO_ROOT = CODE_MOUNT
    src_dir = CODE_MOUNT / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
else:
    for candidate in (Path.cwd(), Path.cwd().parent):
        if (candidate / "src" / "rsna_knee").is_dir():
            REPO_ROOT = candidate
            break
    else:
        REPO_ROOT = Path.cwd().parent"""
        if local_block in src:
            cell["source"] = src.replace(local_block, unified_block).splitlines(keepends=True)
        elif vendor_block in src:
            cell["source"] = src.replace(vendor_block, unified_block).splitlines(keepends=True)
        elif unified_block not in src and "CODE_MOUNT" not in src:
            # Source already has a prior CODE_MOUNT block from an earlier rebuild — leave it
            # unless it points at a different mount; rebuild from build_phase1 keeps local_block.
            pass
        break

    out["cells"] = [bootstrap_md, bootstrap_code] + out["cells"]
    return out


def write_kernel_metadata(cfg: dict[str, Any]) -> Path:
    """Ensure kaggle/train/kernel-metadata.json matches offline submit settings."""
    kernel_dir = project_root() / cfg["kernel_dir"]
    kernel_dir.mkdir(parents=True, exist_ok=True)
    meta_path = kernel_dir / "kernel-metadata.json"

    dataset_sources: list[str] = [_code_dataset_slug(cfg)]
    pretrained = cfg.get("pretrained_dataset")
    if pretrained:
        dataset_sources.append(str(pretrained))

    meta = {
        "id": cfg["kernel_id"],
        # Title slug must match the id suffix (Kaggle 409 if they diverge).
        "title": cfg.get("kernel_title")
        or cfg["kernel_id"].split("/", 1)[-1].replace("-", " ").title(),
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
    stage_code_dataset(cfg)

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
    push_code_dataset(cfg)
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
    parser.add_argument("--push", action="store_true", help="Push code dataset + kaggle/train kernel")
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
