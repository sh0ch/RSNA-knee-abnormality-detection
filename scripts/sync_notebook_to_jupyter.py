#!/usr/bin/env python3
"""Push local phase2 notebook to the live Kaggle Jupyter session."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sync_code_to_jupyter import (  # noqa: E402
    JupyterContents,
    _jupyter_url_from_env,
    _load_dotenv,
    _parse_server,
)

NOTEBOOK = ROOT / "notebooks" / "04_phase2_pseudo_labels.ipynb"
REMOTE_NAMES = (
    "__notebook_source__.ipynb",
    "04_phase2_pseudo_labels.ipynb",
    "notebooks/04_phase2_pseudo_labels.ipynb",
)


def main() -> None:
    _load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", nargs="?", default=_jupyter_url_from_env())
    args = parser.parse_args()
    if not args.url:
        raise SystemExit("Set KAGGLE_JUPYTER_NOTEBOOKS_URL in .env or pass URL")

    text = NOTEBOOK.read_text(encoding="utf-8")
    # Validate JSON
    json.loads(text)

    base, token = _parse_server(args.url)
    client = JupyterContents(base, token)
    client._request("GET", "/api/contents/?content=0")

    last_err: Exception | None = None
    for remote in REMOTE_NAMES:
        try:
            client.put_text(remote, text)
            print(f"Pushed {NOTEBOOK.name} -> {remote}")
            print("Reload the notebook tab in Cursor (close/reopen or revert file).")
            return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise SystemExit(f"Could not push notebook: {last_err}")


if __name__ == "__main__":
    main()
