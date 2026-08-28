"""Find and publish hybrid pseudo-label CSVs (not committed to git)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from rsna_knee.utils.paths import is_kaggle_kernel, project_root

PSEUDO_FILENAMES: tuple[str, ...] = ("pseudo_labels.csv", "pseudo_labels.parquet")
DEFAULT_DATASET_SLUG = "simonhochwebde/rsna-knee-pseudo-labels"
_KAGGLE_MOUNTS: tuple[str, ...] = ("rsna-knee-pseudo-labels",)


def resolve_pseudo_labels_path(preferred: Path | str | None = None) -> Path | None:
    """
    Locate an existing pseudo-label artifact.

    Order: explicit path → ``/kaggle/working/outputs/`` → attached Dataset
    ``rsna-knee-pseudo-labels`` → local ``outputs/``.
    """
    candidates: list[Path] = []
    if preferred is not None:
        candidates.append(Path(preferred))
    if is_kaggle_kernel():
        work = Path("/kaggle/working/outputs")
        candidates.extend(work / name for name in PSEUDO_FILENAMES)
        root = Path("/kaggle/input")
        for mount_name in _KAGGLE_MOUNTS:
            mount = root / mount_name
            candidates.extend(mount / name for name in PSEUDO_FILENAMES)
            if mount.is_dir():
                try:
                    children = list(mount.iterdir())
                except OSError:
                    children = []
                for child in children:
                    if child.is_file() and child.name in PSEUDO_FILENAMES:
                        candidates.append(child)
                    elif child.is_dir():
                        candidates.extend(child / name for name in PSEUDO_FILENAMES)
    else:
        out = project_root() / "outputs"
        candidates.extend(out / name for name in PSEUDO_FILENAMES)

    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path
    return None


def ensure_working_copy(src: Path, dest: Path) -> Path:
    """Copy ``src`` to ``dest`` when they are different files."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = Path(src)
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    return dest


def push_pseudo_labels_dataset(
    csv_path: Path | str,
    *,
    slug: str = DEFAULT_DATASET_SLUG,
) -> Path:
    """
    Upload ``pseudo_labels.csv`` as a private Kaggle Dataset.

    Needs the Kaggle API (interactive kernel with internet, or local ``kaggle`` CLI).
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"No pseudo-label file at {csv_path}")

    if is_kaggle_kernel():
        stage = Path("/kaggle/working/_pseudo_labels_dataset")
    else:
        stage = project_root() / "kaggle" / "datasets" / "rsna-knee-pseudo-labels"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    shutil.copy2(csv_path, stage / "pseudo_labels.csv")
    meta = {
        "id": slug,
        "title": slug.split("/", 1)[-1],
        "licenses": [{"name": "CC0-1.0"}],
        "isPrivate": True,
    }
    (stage / "dataset-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    def _create() -> None:
        cmd = [
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
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    status = subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "status", slug],
        check=False,
        capture_output=True,
        text=True,
    )
    status_text = (status.stdout or "") + (status.stderr or "")
    missing = status.returncode != 0 or any(
        s in status_text.lower()
        for s in ("403", "404", "not found", "forbidden", "does not exist")
    )
    if missing:
        _create()
        return stage

    note = f"hybrid pseudo-labels {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}"
    cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "version",
        "-p",
        str(stage),
        "-m",
        note,
        "--dir-mode",
        "zip",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return stage
