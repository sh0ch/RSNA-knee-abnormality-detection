#!/usr/bin/env python3
"""Insert 5a upload-part cells into phase2 notebook (idempotent)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "04_phase2_pseudo_labels.ipynb"


def as_lines(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").strip("\n") + "\n"
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + ([lines[-1] + "\n"] if lines[-1] else [])


def src(cell: dict) -> str:
    return "".join(cell.get("source") or [])


PACK_MD = """## 5a. Pack volume cache (local only)

Decodes DICOMs into `/kaggle/working/volume_cache` (~14 GiB). **Does not upload.**

If `d16_h256_w256_s3` already has 4407 `.npy` files, skip this cell and run the three **Upload part** cells below.

Internet ON for uploads only (parts 1-3).
"""

PACK_CODE = """import gc

from rsna_knee.data import (
    materialize_writable_volume_cache,
    prepare_disk_volume_cache,
)

if "llm" in dir():
    unload = getattr(llm, "unload", None)
    if callable(unload):
        unload()
    else:
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("Released LLM weights")

CACHE_DIR = Path("/kaggle/working/volume_cache") if ON_KAGGLE else REPO_ROOT / "volume_cache"
volume_shape = tuple(cfg["data"]["volume_shape"])
max_series = int(cfg["data"]["max_series"])
_pseudo = pseudo_path if "pseudo_path" in dir() else cfg["data"].get("pseudo_labels_path")

cache_ds = prepare_disk_volume_cache(
    DATA_ROOT,
    CACHE_DIR,
    labeled_only=bool(cfg["data"].get("labeled_only", False)),
    pseudo_labels_path=_pseudo,
    min_confidence=float(cfg["data"].get("min_confidence", 0.7)),
    volume_shape=volume_shape,
    max_series=max_series,
    max_workers=8,
)
n_copied = materialize_writable_volume_cache(cache_ds)
n_npy = len(list(cache_ds.cache_dir.glob("*.npy")))
print(f"Cache ready: {cache_ds.cache_dir}  npy={n_npy}  copied_from_mounts={n_copied}")
print("Next: run Upload part 1, then 2, then 3 (internet ON).")
"""

UPLOAD_BLOCKS = [
    (
        """## 5a. Upload part 1 / 3 (~4.3 GiB)

Internet **ON**. Run this cell alone; wait until it finishes before part 2.

Creates Dataset `simonhochwebde/rsna-knee-volume-cache-p1`.
""",
        """import sys
for _m in list(sys.modules):
    if _m == "rsna_knee" or _m.startswith("rsna_knee."):
        del sys.modules[_m]

from rsna_knee.data.volume_cache import find_local_volume_cache_dir, push_volume_cache_part

CACHE_DIR = find_local_volume_cache_dir() or (
    Path("/kaggle/working/volume_cache") if "CACHE_DIR" not in dir() else CACHE_DIR
)
print("cache:", CACHE_DIR)
push_volume_cache_part(CACHE_DIR, 1, n_parts=3)
print("Part 1/3 done. Run Upload part 2.")
""",
    ),
    (
        """## 5a. Upload part 2 / 3""",
        """import sys
for _m in list(sys.modules):
    if _m == "rsna_knee" or _m.startswith("rsna_knee."):
        del sys.modules[_m]

from rsna_knee.data.volume_cache import find_local_volume_cache_dir, push_volume_cache_part

CACHE_DIR = find_local_volume_cache_dir() or (
    Path("/kaggle/working/volume_cache") if "CACHE_DIR" not in dir() else CACHE_DIR
)
print("cache:", CACHE_DIR)
push_volume_cache_part(CACHE_DIR, 2, n_parts=3)
print("Part 2/3 done. Run Upload part 3.")
""",
    ),
    (
        """## 5a. Upload part 3 / 3

After success: attach `rsna-knee-volume-cache-p1`, `-p2`, `-p3` on the kernel, restart Jupyter, then run **5b Train**.
""",
        """import sys
for _m in list(sys.modules):
    if _m == "rsna_knee" or _m.startswith("rsna_knee."):
        del sys.modules[_m]

from rsna_knee.data.volume_cache import find_local_volume_cache_dir, push_volume_cache_part

CACHE_DIR = find_local_volume_cache_dir() or (
    Path("/kaggle/working/volume_cache") if "CACHE_DIR" not in dir() else CACHE_DIR
)
print("cache:", CACHE_DIR)
push_volume_cache_part(CACHE_DIR, 3, n_parts=3)
print("Part 3/3 done. Attach all three Datasets and restart Jupyter.")
""",
    ),
]

TRAIN_MD = """## 5b. Train on pseudo-labels (OOF on 58 labeled)

Reads packed volumes from `/kaggle/working/volume_cache` and/or Datasets `rsna-knee-volume-cache-p1` … `-p3`. Does not re-decode DICOMs. Prefer GPU (T4), internet OFF.
"""


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # Update pack section
    for cell in cells:
        if cell.get("cell_type") == "markdown" and src(cell).startswith("## 5a. Pack"):
            cell["source"] = as_lines(PACK_MD)
        if cell.get("cell_type") == "code" and "prepare_disk_volume_cache" in src(cell):
            cell["source"] = as_lines(PACK_CODE)
            cell["outputs"] = []
            cell["execution_count"] = None
        if cell.get("cell_type") == "markdown" and src(cell).startswith("## 5b. Train"):
            cell["source"] = as_lines(TRAIN_MD)

    has_upload = any(
        c.get("cell_type") == "markdown" and src(c).startswith("## 5a. Upload part 1")
        for c in cells
    )
    if not has_upload:
        insert_at = next(
            i
            for i, c in enumerate(cells)
            if c.get("cell_type") == "markdown" and src(c).startswith("## 5b. Train")
        )
        new_cells: list[dict] = []
        for md, code in UPLOAD_BLOCKS:
            new_cells.append({"cell_type": "markdown", "metadata": {}, "source": as_lines(md)})
            new_cells.append(
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": as_lines(code),
                    "outputs": [],
                    "execution_count": None,
                }
            )
        cells[insert_at:insert_at] = new_cells

    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Updated {NB} ({len(cells)} cells, upload_cells={has_upload or True})")


if __name__ == "__main__":
    main()
