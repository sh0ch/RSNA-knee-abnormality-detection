#!/usr/bin/env python3
"""Build kaggle/eda/eda_phase0.ipynb from notebooks/02_eda_phase0.ipynb."""

from __future__ import annotations

import json
from pathlib import Path

from rsna_knee.utils.paths import project_root


def main() -> None:
    root = project_root()
    src = root / "notebooks" / "02_eda_phase0.ipynb"
    dst = root / "kaggle" / "eda" / "eda_phase0.ipynb"
    dst.parent.mkdir(parents=True, exist_ok=True)

    nb = json.loads(src.read_text(encoding="utf-8"))

    for cell in nb["cells"]:
        cell["outputs"] = []
        cell["execution_count"] = None
        cell.pop("id", None)

    bootstrap_md = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# RSNA Knee — EDA Phase 0 (Kaggle)\n",
            "\n",
            "Exploratory analysis on the full competition dataset.\n",
            "\n",
            "**Before running:** replace `YOUR_USERNAME` in the clone URL below with your GitHub username.\n",
            "\n",
            "Push from your machine: `kaggle kernels push -p kaggle/eda`\n",
        ],
    }

    bootstrap_code = {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "import os\n",
            "import subprocess\n",
            "import sys\n",
            "\n",
            'REPO_URL = "https://github.com/YOUR_USERNAME/RSNA_knee_abnormality_detection.git"\n',
            'BRANCH = "main"\n',
            'WORK_DIR = "/kaggle/working/rsna_knee_repo"\n',
            "\n",
            "if not os.path.exists(WORK_DIR):\n",
            "    subprocess.run(\n",
            '        ["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, WORK_DIR],\n',
            "        check=True,\n",
            "    )\n",
            "\n",
            'subprocess.run(["pip", "install", "-q", "-e", f"{WORK_DIR}[dev]"], check=True)\n',
            'sys.path.insert(0, f"{WORK_DIR}/src")\n',
        ],
    }

    setup = nb["cells"][2]
    setup_src = "".join(setup["source"])
    old = """# Resolve repo root when running locally from notebooks/
for candidate in (Path.cwd(), Path.cwd().parent):
    if (candidate / "src" / "rsna_knee").is_dir():
        REPO_ROOT = candidate
        break
else:
    REPO_ROOT = Path.cwd().parent"""
    new = """# Kaggle: repo cloned in the cell above; local: search from notebooks/
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
    if old not in setup_src:
        raise ValueError("Setup cell pattern not found — update build script")
    setup["source"] = setup_src.replace(old, new).splitlines(keepends=True)

    nb["cells"][0]["source"][0] = "# Phase 0 — Exploratory Data Analysis\n"
    nb["cells"] = [bootstrap_md, bootstrap_code] + nb["cells"]

    dst.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {dst} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
