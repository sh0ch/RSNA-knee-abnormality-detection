#!/usr/bin/env python3
"""
Sync the EDA notebook between the repo and Kaggle.

Source of truth: notebooks/02_eda_phase0.ipynb
Kaggle copy:      kaggle/eda/eda_phase0.ipynb  (generated, do not edit by hand)

Usage:
    python scripts/sync_kaggle_eda.py              # copy only
    python scripts/sync_kaggle_eda.py --push       # copy + kaggle kernels push
    python scripts/sync_kaggle_eda.py --pull       # download source from Kaggle (no outputs)
    python scripts/sync_kaggle_eda.py --logs       # fetch last commit run log
    python scripts/sync_kaggle_eda.py --import-run PATH  # import downloaded run + summary

See docs/KAGGLE_NOTEBOOK_SYNC.md for the full workflow.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from rsna_knee.utils.paths import project_root

GENERATED_BANNER = (
    "<!-- AUTO-GENERATED from notebooks/02_eda_phase0.ipynb — do not edit on disk. "
    "Run: python scripts/sync_kaggle_eda.py --push -->\n"
)

RUNS_DIR = "kaggle/eda/runs"
LATEST_RUN_NOTEBOOK = f"{RUNS_DIR}/latest.ipynb"
LAST_RUN_SUMMARY = f"{RUNS_DIR}/last_run_summary.txt"
LAST_COMMIT_LOG = f"{RUNS_DIR}/last_commit.log"


def load_config() -> dict[str, Any]:
    cfg_path = project_root() / "configs" / "kaggle_eda.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def strip_notebook_outputs(nb: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(nb)
    for cell in out["cells"]:
        cell["outputs"] = []
        cell["execution_count"] = None
        cell.pop("id", None)
    return out


def inject_kaggle_bootstrap(nb: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Prepend Kaggle bootstrap cells; patch setup cell for cloned repo path."""
    out = strip_notebook_outputs(nb)

    bootstrap_md = {
        "cell_type": "markdown",
        "metadata": {"tags": ["kaggle-bootstrap"]},
        "source": [
            GENERATED_BANNER,
            "## Kaggle bootstrap\n",
            "\n",
            "Installs this repo on Kaggle. Skipped when running locally.\n",
            "\n",
            f"Repo: `{cfg['repo_url']}`\n",
            "\n",
            "Sync from your machine: `python scripts/sync_kaggle_eda.py --push`\n",
        ],
    }

    bootstrap_code = {
        "cell_type": "code",
        "metadata": {"tags": ["kaggle-bootstrap"]},
        "source": [
            "import os\n",
            "import subprocess\n",
            "import sys\n",
            "\n",
            f'REPO_URL = "{cfg["repo_url"]}"\n',
            f'BRANCH = "{cfg["branch"]}"\n',
            'WORK_DIR = "/kaggle/working/rsna_knee_repo"\n',
            "\n",
            "ON_KAGGLE = (\n",
            '    os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None\n',
            '    or "KAGGLE_URL_BASE" in os.environ\n',
            '    or os.path.isdir("/kaggle/working")\n',
            ")\n",
            "\n",
            "if ON_KAGGLE:\n",
            "    if not os.path.exists(WORK_DIR):\n",
            "        subprocess.run(\n",
            '            ["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, WORK_DIR],\n',
            "            check=True,\n",
            "        )\n",
            "    else:\n",
            "        subprocess.run(\n",
            '            ["git", "-C", WORK_DIR, "pull", "--ff-only", "origin", BRANCH],\n',
            "            check=False,\n",
            "        )\n",
            '    subprocess.run(["pip", "install", "-q", "-e", f"{WORK_DIR}[dev]"], check=True)\n',
            '    sys.path.insert(0, f"{WORK_DIR}/src")\n',
            "else:\n",
            '    print("Local run — skipping Kaggle bootstrap.")\n',
        ],
    }

    setup_idx = next(
        i
        for i, cell in enumerate(out["cells"])
        if cell["cell_type"] == "code" and "default_data_root" in "".join(cell.get("source", []))
    )
    setup = out["cells"][setup_idx]
    setup_src = "".join(setup["source"])

    local_block = """# Resolve repo root when running locally from notebooks/
for candidate in (Path.cwd(), Path.cwd().parent):
    if (candidate / "src" / "rsna_knee").is_dir():
        REPO_ROOT = candidate
        break
else:
    REPO_ROOT = Path.cwd().parent"""

    unified_block = """# Kaggle: repo cloned above; local: search from notebooks/
WORK_DIR = Path("/kaggle/working/rsna_knee_repo")
if WORK_DIR.is_dir():
    REPO_ROOT = WORK_DIR
else:
    for candidate in (Path.cwd(), Path.cwd().parent):
        if (candidate / "src" / "rsna_knee").is_dir():
            REPO_ROOT = candidate
            break
    else:
        REPO_ROOT = Path.cwd().parent"""

    if local_block in setup_src:
        setup["source"] = setup_src.replace(local_block, unified_block).splitlines(keepends=True)
    elif unified_block not in setup_src:
        raise ValueError("Setup cell in source notebook does not match expected repo-root block")

    out["cells"] = [bootstrap_md, bootstrap_code] + out["cells"]
    return out


def copy_to_kaggle(cfg: dict[str, Any]) -> Path:
    root = project_root()
    src = root / cfg["source_notebook"]
    dst = root / cfg["output_notebook"]
    dst.parent.mkdir(parents=True, exist_ok=True)

    nb = json.loads(src.read_text(encoding="utf-8"))
    kaggle_nb = inject_kaggle_bootstrap(nb, cfg)
    dst.write_text(json.dumps(kaggle_nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {dst} ({len(kaggle_nb['cells'])} cells) from {src}")
    return dst


def push_to_kaggle(cfg: dict[str, Any]) -> None:
    kernel_dir = project_root() / cfg["kernel_dir"]
    cmd = [sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(kernel_dir)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def pull_from_kaggle(cfg: dict[str, Any]) -> Path:
    kernel_dir = project_root() / cfg["kernel_dir"]
    kernel_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "kernels",
        "pull",
        cfg["kernel_id"],
        "-p",
        str(kernel_dir),
        "-m",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    pulled = kernel_dir / "eda_phase0.ipynb"
    if not pulled.is_file():
        candidates = list(kernel_dir.glob("*.ipynb"))
        if len(candidates) == 1:
            pulled = candidates[0]
    print(f"Pulled notebook to {pulled}")
    print(
        "Source only (no outputs). For executed results use --import-run after downloading "
        "from Kaggle, or --logs for commit stdout."
    )
    return pulled


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
    parser.add_argument("--push", action="store_true", help="Push kaggle/eda to Kaggle after copy")
    parser.add_argument(
        "--pull",
        action="store_true",
        help="Pull notebook source from Kaggle (no outputs)",
    )
    parser.add_argument(
        "--logs",
        action="store_true",
        help="Fetch stdout/stderr from the last Save & Run All commit",
    )
    parser.add_argument(
        "--import-run",
        metavar="PATH",
        help="Copy a downloaded executed notebook into kaggle/eda/runs/ and write a summary",
    )
    args = parser.parse_args()
    cfg = load_config()

    if args.import_run:
        import_run(Path(args.import_run))
        return

    if args.logs:
        pull_logs(cfg)
        return

    if args.pull:
        pull_from_kaggle(cfg)
        return

    copy_to_kaggle(cfg)
    if args.push:
        push_to_kaggle(cfg)


if __name__ == "__main__":
    main()
