#!/usr/bin/env python3
"""
Push local src/ + configs/ onto a running Kaggle Jupyter Server.

Writes /kaggle/working/src and /kaggle/working/configs so the kernel always
imports this checkout — no Dataset version, no restart.

Usage:
    python scripts/sync_code_to_jupyter.py
    python scripts/sync_code_to_jupyter.py "https://..../?token=..."

URL comes from Kaggle: Run → Kaggle Jupyter Server → VS Code Compatible URL
or the Notebooks proxy URL. Store it in .env as KAGGLE_JUPYTER_URL or
KAGGLE_JUPYTER_NOTEBOOKS_URL.
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from rsna_knee.utils.paths import project_root


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _jupyter_url_from_env() -> str:
    for key in (
        "KAGGLE_JUPYTER_URL",
        "KAGGLE_JUPYTER_NOTEBOOKS_URL",
        "KAGGLE_JUPYTER_SERVER_URL",
    ):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return ""


def _parse_server(url: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(url.strip())
    token = urllib.parse.parse_qs(parsed.query).get("token", [""])[0]
    origin = f"{parsed.scheme}://{parsed.netloc}"
    prefix = parsed.path.rstrip("/")
    for suffix in ("/lab/tree", "/lab", "/tree", "/notebooks"):
        if prefix.endswith(suffix):
            prefix = prefix[: -len(suffix)]
            break
    if prefix == "/":
        prefix = ""
    # Kaggle notebooks proxy: auth is in the path (.../k/<id>/<jwe>/proxy).
    return f"{origin}{prefix}", token


def _contents_path(path: str) -> str:
    return "/api/contents/" + urllib.parse.quote(path, safe="/")


class JupyterContents:
    def __init__(self, base: str, token: str) -> None:
        self.base = base.rstrip("/")
        self.token = token
        self._ctx = ssl.create_default_context()

    def _url(self, api_path: str) -> str:
        url = f"{self.base}{api_path}"
        if self.token:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}token={urllib.parse.quote(self.token)}"
        return url

    def _request(self, method: str, api_path: str, payload: dict | None = None) -> None:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "User-Agent": "rsna-knee-sync"}
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        req = urllib.request.Request(
            self._url(api_path),
            data=data,
            method=method,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, context=self._ctx, timeout=60) as resp:
                resp.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"{method} {api_path} -> HTTP {exc.code}\n{body}\n"
                "Is the VS Code Compatible URL still valid (session not expired)?"
            ) from exc

    def mkdir(self, path: str) -> None:
        try:
            self._request(
                "PUT",
                _contents_path(path),
                {"type": "directory"},
            )
        except RuntimeError as exc:
            if "HTTP 400" not in str(exc) and "HTTP 409" not in str(exc):
                raise

    def put_text(self, path: str, text: str) -> None:
        name = path.rsplit("/", 1)[-1]
        self._request(
            "PUT",
            _contents_path(path),
            {
                "type": "file",
                "format": "text",
                "name": name,
                "path": path,
                "content": text,
            },
        )


def _ensure_parents(client: JupyterContents, rel: str, seen: set[str]) -> None:
    parts = Path(rel).parts[:-1]
    cur = ""
    for part in parts:
        cur = f"{cur}/{part}" if cur else part
        if cur in seen:
            continue
        client.mkdir(cur.replace("\\", "/"))
        seen.add(cur)


def sync(url: str) -> None:
    root = project_root()
    base, token = _parse_server(url)
    if not token and "/proxy" not in urllib.parse.urlparse(base).path:
        raise SystemExit(
            "Jupyter URL needs either ?token= (VS Code URL) or a Kaggle "
            ".../proxy path (Notebooks URL)."
        )
    client = JupyterContents(base, token)
    # Probe: empty directory list of root.
    client._request("GET", "/api/contents/?content=0")

    seen: set[str] = set()
    files: list[Path] = []
    files.extend(sorted((root / "src" / "rsna_knee").rglob("*.py")))
    files.extend(sorted((root / "configs").glob("*.yaml")))
    n = 0
    for path in files:
        rel = path.relative_to(root).as_posix()
        _ensure_parents(client, rel, seen)
        client.put_text(rel, path.read_text(encoding="utf-8"))
        n += 1
        print(f"  {rel}")
    print(f"Uploaded {n} files -> /kaggle/working/src + configs")
    print("Re-run the notebook setup cell (restart kernel if rsna_knee was already imported).")


def main() -> None:
    _load_dotenv(project_root() / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "url",
        nargs="?",
        default=_jupyter_url_from_env(),
        help="Jupyter URL, or KAGGLE_JUPYTER_URL / KAGGLE_JUPYTER_NOTEBOOKS_URL in .env",
    )
    args = parser.parse_args()
    if not args.url:
        raise SystemExit(
            "Pass the Jupyter URL or set KAGGLE_JUPYTER_NOTEBOOKS_URL / "
            "KAGGLE_JUPYTER_URL in .env."
        )
    sync(args.url)


if __name__ == "__main__":
    sys.exit(main())
