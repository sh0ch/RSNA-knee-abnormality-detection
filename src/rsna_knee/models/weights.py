"""Offline pretrained weight resolution (no network at submit time)."""

from __future__ import annotations

import os
from pathlib import Path

from rsna_knee.utils.paths import is_kaggle_kernel, project_root

CONVNEXT_TINY_FILENAME = "convnext_tiny_imagenet.pth"

# Relative paths under /kaggle/input or the local repo.
_KAGGLE_WEIGHT_CANDIDATES: list[str] = [
    "rsna-knee-pretrained/convnext_tiny_imagenet.pth",
    "rsna-knee-pretrained/weights/convnext_tiny_imagenet.pth",
    "simonhochwebde/rsna-knee-pretrained/convnext_tiny_imagenet.pth",
]


def pretrained_weights_dir() -> Path:
    """Local cache directory for exported ImageNet weights (gitignored)."""
    return project_root() / "data" / "pretrained"


def default_convnext_tiny_path() -> Path:
    return pretrained_weights_dir() / CONVNEXT_TINY_FILENAME


def resolve_pretrained_weights(
    filename: str = CONVNEXT_TINY_FILENAME,
    *,
    explicit: Path | str | None = None,
    allow_missing: bool = False,
) -> Path | None:
    """
    Locate a pretrained ``.pth`` file without downloading.

    Search order:
      1. ``explicit`` path / ``RSNA_PRETRAINED_WEIGHTS`` env
      2. Kaggle dataset mounts under ``/kaggle/input``
      3. Local ``data/pretrained/{filename}``

    On Kaggle, missing weights raise ``FileNotFoundError`` unless ``allow_missing``.
    Locally, returns ``None`` when missing and ``allow_missing`` is True (tests / smoke).
    """
    candidates: list[Path] = []

    if explicit is not None:
        candidates.append(Path(explicit))

    env = os.environ.get("RSNA_PRETRAINED_WEIGHTS")
    if env:
        candidates.append(Path(env))

    if is_kaggle_kernel():
        input_root = Path("/kaggle/input")
        for rel in _KAGGLE_WEIGHT_CANDIDATES:
            candidates.append(input_root / rel)
        # Any dataset folder that contains the filename.
        if input_root.is_dir():
            for path in input_root.rglob(filename):
                candidates.append(path)

    candidates.append(pretrained_weights_dir() / filename)

    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve() if path.exists() else path
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.is_file():
            return path

    if allow_missing and not is_kaggle_kernel():
        return None

    searched = "\n  - ".join(str(p) for p in candidates[:12])
    raise FileNotFoundError(
        f"Pretrained weights '{filename}' not found (offline mode — no download).\n"
        f"Searched:\n  - {searched}\n"
        "Export locally with: python scripts/export_pretrained_weights.py\n"
        "Then upload the .pth as a Kaggle Dataset and attach it to the kernel."
    )
