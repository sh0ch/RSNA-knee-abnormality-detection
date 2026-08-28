#!/usr/bin/env python3
"""
Download pseudo_labels.csv from a live Kaggle Jupyter Server to local outputs/.

Usage:
    python scripts/pull_pseudo_labels.py
    python scripts/pull_pseudo_labels.py "https://..../?token=..."

Uses the same Jupyter URL as sync_code_to_jupyter.py (.env KAGGLE_JUPYTER_URL).
Later upload: python scripts/publish_pseudo_labels.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rsna_knee.utils.paths import project_root

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from sync_code_to_jupyter import (  # noqa: E402
    JupyterContents,
    _jupyter_url_from_env,
    _load_dotenv,
    _parse_server,
)

REMOTE_CANDIDATES = (
    "outputs/pseudo_labels.csv",
    "outputs/pseudo_labels.parquet",
)


def pull(url: str, dest: Path | None = None) -> Path:
    base, token = _parse_server(url)
    client = JupyterContents(base, token)
    dest_dir = dest.parent if dest is not None else project_root() / "outputs"
    last_error: Exception | None = None
    for remote in REMOTE_CANDIDATES:
        suffix = Path(remote).suffix
        out = dest if dest is not None else dest_dir / Path(remote).name
        try:
            client.download_file(remote, out)
        except RuntimeError as exc:
            last_error = exc
            continue
        size = out.stat().st_size
        print(f"Downloaded {remote} -> {out} ({size / 1e6:.2f} MB)")
        if suffix == ".csv" and size < 100:
            raise SystemExit(f"{out} looks empty")
        return out
    raise SystemExit(
        "Could not find outputs/pseudo_labels.csv on the Jupyter Server.\n"
        f"Last error: {last_error}"
    )


def main() -> None:
    _load_dotenv(project_root() / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "url",
        nargs="?",
        default=_jupyter_url_from_env(),
        help="Jupyter URL, or KAGGLE_JUPYTER_URL in .env",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Local destination (default: outputs/pseudo_labels.csv)",
    )
    args = parser.parse_args()
    if not args.url:
        raise SystemExit(
            "Pass the Jupyter URL or set KAGGLE_JUPYTER_NOTEBOOKS_URL / "
            "KAGGLE_JUPYTER_URL in .env."
        )
    pull(args.url, args.output)


if __name__ == "__main__":
    sys.exit(main())
