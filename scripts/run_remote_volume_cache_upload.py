#!/usr/bin/env python3
"""
Sync latest src/ to Kaggle Jupyter, then upload volume cache parts 1–4.

Run from repo root (internet ON on the Kaggle kernel):

    python scripts/run_remote_volume_cache_upload.py
    python scripts/run_remote_volume_cache_upload.py --part 2   # resume
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from execute_on_jupyter import execute_code  # noqa: E402

from rsna_knee.data.volume_cache import VOLUME_CACHE_PARTS  # noqa: E402
from sync_code_to_jupyter import (  # noqa: E402
    JupyterContents,
    _jupyter_url_from_env,
    _load_dotenv,
    _parse_server,
    sync,
)


def _remote_has_push_part(url: str) -> bool:
    """Verify on the kernel filesystem (Jupyter API may list .py as notebooks)."""
    code = """
from pathlib import Path
p = Path("/kaggle/working/src/rsna_knee/data/volume_cache.py")
print("OK" if p.is_file() and "def push_volume_cache_part(" in p.read_text(encoding="utf-8") else "MISSING")
"""
    stdout, _ = execute_code(url, code, timeout_s=300.0)
    return "OK" in stdout


_UPLOAD_CODE = """
import sys
for name in list(sys.modules):
    if name == "rsna_knee" or name.startswith("rsna_knee."):
        del sys.modules[name]
if "/kaggle/working/src" not in sys.path:
    sys.path.insert(0, "/kaggle/working/src")

from pathlib import Path
from rsna_knee.data.volume_cache import (
    cached_npy_stems,
    push_volume_cache_part,
)

CACHE_DIR = Path("{cache_dir}")
if not CACHE_DIR.is_dir():
    raise SystemExit(f"Missing cache dir {{CACHE_DIR}}")
n = len(cached_npy_stems([CACHE_DIR]))
print(f"cache_dir={{CACHE_DIR}}  npy_count={{n}}", flush=True)

PART = {part}
N_PARTS = {n_parts}
print(f"=== uploading part {{PART}}/{{N_PARTS}} ===", flush=True)
push_volume_cache_part(CACHE_DIR, PART, n_parts=N_PARTS)
print(f"=== part {{PART}}/{{N_PARTS}} done ===", flush=True)
"""


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    _load_dotenv(root / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        default="/kaggle/working/volume_cache/d16_h256_w256_s3",
        help="Packed .npy folder on the Kaggle kernel",
    )
    parser.add_argument("--skip-verify", action="store_true", help="Skip post-sync file check")
    parser.add_argument("--skip-sync", action="store_true", help="Skip code sync")
    parser.add_argument(
        "--part",
        type=int,
        default=None,
        help=f"Upload only this part (1–{VOLUME_CACHE_PARTS})",
    )
    parser.add_argument("url", nargs="?", default=_jupyter_url_from_env())
    parser.add_argument("--timeout", type=float, default=14400.0)
    args = parser.parse_args()
    if not args.url:
        raise SystemExit("Set KAGGLE_JUPYTER_NOTEBOOKS_URL in .env")

    print("Syncing src/ + configs/ to Kaggle …")
    if not args.skip_sync:
        sync(args.url)
    else:
        print("(skipped sync)")

    if not args.skip_verify:
        print("Checking remote volume_cache.py …")
        if not _remote_has_push_part(args.url):
            raise SystemExit(
                "Remote volume_cache.py still missing push_volume_cache_part after sync."
            )
        print("Remote volume_cache.py OK (push_volume_cache_part present).")

    parts = [args.part] if args.part is not None else list(range(1, VOLUME_CACHE_PARTS + 1))
    for part in parts:
        if part < 1 or part > VOLUME_CACHE_PARTS:
            raise SystemExit(f"--part must be 1..{VOLUME_CACHE_PARTS}")
        print(
            f"\n{'=' * 60}\nUploading part {part}/{VOLUME_CACHE_PARTS} on Kaggle kernel …\n{'=' * 60}"
        )
        code = _UPLOAD_CODE.format(
            part=part,
            n_parts=VOLUME_CACHE_PARTS,
            cache_dir=args.cache_dir,
        )
        stdout, stderr = execute_code(args.url, code, timeout_s=args.timeout)
        if "=== part" not in stdout and "done" not in stdout.lower():
            combined = stdout + stderr
            if re.search(r"(Error|Traceback|Exception)", combined):
                raise SystemExit(f"Part {part} may have failed — check output above.")
    slugs = ", ".join(f"-p{i}" for i in range(1, VOLUME_CACHE_PARTS + 1))
    print(
        f"\nAll requested parts finished. Attach rsna-knee-volume-cache {slugs} on the kernel and restart Jupyter."
    )


if __name__ == "__main__":
    main()
