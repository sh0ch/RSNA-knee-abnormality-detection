#!/usr/bin/env python3
"""
Upload the packed MRI volume cache as Dataset simonhochwebde/rsna-knee-volume-cache.

Run on the Kaggle Jupyter session after cache warming finishes (internet ON):

    python scripts/publish_volume_cache.py

Then attach that Dataset on the Phase 2 kernel and restart Jupyter. Training
will read the npy files from /kaggle/input and skip DICOM decode.

Usage:
    python scripts/publish_volume_cache.py
    python scripts/publish_volume_cache.py /tmp/rsna_volume_cache/d16_h256_w256_s3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rsna_knee.data.volume_cache import (
    DEFAULT_VOLUME_CACHE_SLUG,
    VOLUME_CACHE_PARTS,
    find_local_volume_cache_dir,
    push_volume_cache_dataset,
    push_volume_cache_part,
    push_volume_cache_parts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cache_dir",
        nargs="?",
        default=None,
        help="Folder of {StudyInstanceUID}.npy files (auto-detect if omitted)",
    )
    parser.add_argument("--slug", default=DEFAULT_VOLUME_CACHE_SLUG)
    parser.add_argument(
        "--parts",
        type=int,
        default=1,
        help=f"Split into N Datasets named {{slug}}-p1 … -pN (default {VOLUME_CACHE_PARTS} on Kaggle to avoid timeouts)",
    )
    parser.add_argument(
        "--part",
        type=int,
        default=None,
        help="Upload only this part (1-based). Implies --parts if unset.",
    )
    args = parser.parse_args()
    path = Path(args.cache_dir) if args.cache_dir else find_local_volume_cache_dir()
    if path is None or not path.is_dir():
        raise SystemExit(
            "No volume cache folder found. Pass the d16_h256_w256_s3 directory, "
            "or finish cache warming first."
        )
    n_parts = int(args.parts)
    if args.part is not None:
        n_parts = n_parts if n_parts > 1 else VOLUME_CACHE_PARTS
        push_volume_cache_part(path, int(args.part), n_parts=n_parts)
        print(f"Published part {args.part}/{n_parts} from {path}")
        return
    if n_parts > 1:
        push_volume_cache_parts(path, n_parts=n_parts)
        print(f"Published {n_parts} parts from {path}")
        return
    push_volume_cache_dataset(path, slug=args.slug)
    print(f"Published {path} -> {args.slug}")
    print("Attach Dataset simonhochwebde/rsna-knee-volume-cache on the kernel, then restart Jupyter.")


if __name__ == "__main__":
    sys.exit(main())
