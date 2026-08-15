#!/usr/bin/env python3
"""
Export ImageNet ConvNeXt-Tiny weights for offline Kaggle training.

Requires internet once (local only). Upload the resulting .pth as a Kaggle
Dataset (e.g. rsna-knee-pretrained) and attach it to the Phase 1 train kernel.

Usage:
    python scripts/export_pretrained_weights.py
    python scripts/export_pretrained_weights.py --output path/to/file.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from rsna_knee.models.weights import CONVNEXT_TINY_FILENAME, pretrained_weights_dir
from rsna_knee.utils.paths import project_root


def export_convnext_tiny(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading ImageNet ConvNeXt-Tiny weights via torchvision…")
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    state = model.state_dict()
    torch.save(state, output)
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"Wrote {output} ({size_mb:.1f} MB, {len(state)} tensors)")
    print(
        "Upload this file to a Kaggle Dataset named e.g. 'rsna-knee-pretrained', "
        "then attach that dataset to the Phase 1 train kernel (internet OFF)."
    )
    print(
        "License note: torchvision ConvNeXt-Tiny ImageNet weights — BSD-3 / "
        "see torchvision docs; document for competition pretrained-weight rules."
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output .pth path (default: data/pretrained/{CONVNEXT_TINY_FILENAME})",
    )
    args = parser.parse_args()
    out = args.output or (pretrained_weights_dir() / CONVNEXT_TINY_FILENAME)
    if not out.is_absolute():
        out = project_root() / out
    export_convnext_tiny(out)


if __name__ == "__main__":
    main()
