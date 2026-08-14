#!/usr/bin/env python3
"""
Sync the EDA notebook between the repo and Kaggle.

Source of truth: notebooks/02_eda_phase0.ipynb
Kaggle copy:      kaggle/eda/eda_phase0.ipynb  (generated, do not edit by hand)

Usage:
    python scripts/sync_kaggle_eda.py              # copy only
    python scripts/sync_kaggle_eda.py --push       # copy + kaggle kernels push
    python scripts/sync_kaggle_eda.py --pull       # download from Kaggle (review before merging)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from rsna_knee.utils.paths import project_root

GENERATED_BANNER = (
    "<!-- AUTO-GENERATED from notebooks/02_eda_phase0.ipynb — do not edit on disk. "
    "Run: python scripts/sync_kaggle_eda.py --push -->\n"
)


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
            'if os.environ.get("KAGGLE_KERNEL_RUN") == "True":\n',
            "    if not os.path.exists(WORK_DIR):\n",
            "        subprocess.run(\n",
            '            ["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, WORK_DIR],\n',
            "            check=True,\n",
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
        # Kaggle may name file from slug
        candidates = list(kernel_dir.glob("*.ipynb"))
        if len(candidates) == 1:
            pulled = candidates[0]
    print(f"Pulled notebook to {pulled}")
    print(
        "Review diff against notebooks/02_eda_phase0.ipynb and merge analysis cells manually "
        "(ignore bootstrap cells when copying back)."
    )
    return pulled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push", action="store_true", help="Push kaggle/eda to Kaggle after copy")
    parser.add_argument(
        "--pull",
        action="store_true",
        help="Pull notebook from Kaggle (for manual merge into source)",
    )
    args = parser.parse_args()
    cfg = load_config()

    if args.pull:
        pull_from_kaggle(cfg)
        return

    copy_to_kaggle(cfg)
    if args.push:
        push_to_kaggle(cfg)


if __name__ == "__main__":
    main()
