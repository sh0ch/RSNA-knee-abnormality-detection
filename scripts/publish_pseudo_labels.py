#!/usr/bin/env python3
"""
Upload local outputs/pseudo_labels.csv as Dataset simonhochwebde/rsna-knee-pseudo-labels.

Run after: python scripts/pull_pseudo_labels.py

Usage:
    python scripts/publish_pseudo_labels.py
    python scripts/publish_pseudo_labels.py path/to/pseudo_labels.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rsna_knee.reports.persist import DEFAULT_DATASET_SLUG, push_pseudo_labels_dataset
from rsna_knee.utils.paths import project_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv",
        nargs="?",
        default=str(project_root() / "outputs" / "pseudo_labels.csv"),
        help="Local CSV (default: outputs/pseudo_labels.csv)",
    )
    parser.add_argument("--slug", default=DEFAULT_DATASET_SLUG)
    args = parser.parse_args()
    path = Path(args.csv)
    if not path.is_file():
        raise SystemExit(
            f"Missing {path}. Download first: python scripts/pull_pseudo_labels.py"
        )
    push_pseudo_labels_dataset(path, slug=args.slug)
    print(f"Published {path} -> {args.slug}")
    print("Attach that Dataset on the Phase 2 kernel, then restart Jupyter.")


if __name__ == "__main__":
    sys.exit(main())
