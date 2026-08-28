#!/usr/bin/env python3
"""
Export pretrained ConvNeXt-Tiny weights for offline Kaggle training.

Variants:
  imagenet   — torchvision ImageNet (default)
  radimagenet — copy user-provided RadImageNet checkpoint into data/pretrained/

Usage:
    python scripts/export_pretrained_weights.py
    python scripts/export_pretrained_weights.py --variant imagenet
    python scripts/export_pretrained_weights.py --variant radimagenet --source /path/to/rad.pth
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from rsna_knee.models.weights import (
    CONVNEXT_TINY_FILENAME,
    PRETRAINED_WEIGHT_REGISTRY,
    RAD_IMAGENET_CONVNEXT_TINY_FILENAME,
    pretrained_weights_dir,
)
from rsna_knee.utils.paths import project_root


def export_convnext_tiny_imagenet(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading ImageNet ConvNeXt-Tiny weights via torchvision…")
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    state = model.state_dict()
    torch.save(state, output)
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"Wrote {output} ({size_mb:.1f} MB, {len(state)} tensors)")
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

    print(
        "Upload to Kaggle Dataset 'rsna-knee-pretrained' and attach to train kernel "
        f"as {filename} (internet OFF)."
    )


if __name__ == "__main__":
    main()
