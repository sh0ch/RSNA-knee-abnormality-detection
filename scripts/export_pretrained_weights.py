#!/usr/bin/env python3
"""
Export pretrained ConvNeXt-Tiny weights for offline Kaggle training.

Does not require local torch: ImageNet weights are fetched from the official
torchvision CDN (same file as ConvNeXt_Tiny_Weights.IMAGENET1K_V1).

Variants:
  imagenet    — torchvision ImageNet (default)
  radimagenet — copy user-provided RadImageNet checkpoint into data/pretrained/

Usage:
    python scripts/export_pretrained_weights.py
    python scripts/export_pretrained_weights.py --variant imagenet
    python scripts/export_pretrained_weights.py --variant radimagenet --source /path/to/rad.pth
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

from rsna_knee.models.weights import (
    PRETRAINED_WEIGHT_REGISTRY,
    pretrained_weights_dir,
)
from rsna_knee.utils.paths import project_root

# torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
# https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py
_IMAGENET_URL = "https://download.pytorch.org/models/convnext_tiny-983f1562.pth"
_EXPECTED_MIN_BYTES = 80 * 1024 * 1024  # official file is ~109 MB


def export_convnext_tiny_imagenet(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading ImageNet ConvNeXt-Tiny from {_IMAGENET_URL}")

    def _progress(block: int, block_size: int, total: int) -> None:
        if total <= 0:
            return
        done = min(block * block_size, total)
        pct = 100.0 * done / total
        print(f"\r  {done / (1024 * 1024):.1f} / {total / (1024 * 1024):.1f} MB ({pct:.0f}%)", end="")

    tmp = output.with_suffix(output.suffix + ".part")
    try:
        urllib.request.urlretrieve(_IMAGENET_URL, tmp, reporthook=_progress)
        print()
        size = tmp.stat().st_size
        if size < _EXPECTED_MIN_BYTES:
            raise RuntimeError(
                f"Download too small ({size} bytes); expected ~109 MB. Check the URL / network."
            )
        tmp.replace(output)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"Wrote {output} ({size_mb:.1f} MB)")
    print(
        "License note: torchvision ConvNeXt-Tiny ImageNet weights — BSD-3; "
        "document for competition pretrained-weight rules."
    )
    return output


def export_radimagenet_copy(source: Path, output: Path) -> Path:
    if not source.is_file():
        raise FileNotFoundError(f"RadImageNet source not found: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"Copied RadImageNet weights to {output} ({size_mb:.1f} MB)")
    print(
        "License note: verify RadImageNet license and competition rules before use. "
        "Document source in docs/PROJECT_LOG.md."
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=sorted(PRETRAINED_WEIGHT_REGISTRY.keys()),
        default="imagenet",
        help="Pretrained variant to export",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Source .pth for radimagenet variant (required for radimagenet)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .pth path (default: data/pretrained/<variant>.pth)",
    )
    args = parser.parse_args()

    filename = PRETRAINED_WEIGHT_REGISTRY[args.variant]
    out = args.output or (pretrained_weights_dir() / filename)
    if not out.is_absolute():
        out = project_root() / out

    if args.variant == "imagenet":
        export_convnext_tiny_imagenet(out)
    else:
        if args.source is None:
            raise SystemExit("--source is required for radimagenet variant")
        export_radimagenet_copy(args.source, out)

    print(f"Next: python scripts/publish_pretrained_weights.py  ({filename})")


if __name__ == "__main__":
    main()
