#!/usr/bin/env python3
"""Execute Python on a live Kaggle Jupyter kernel via the shell WebSocket."""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from websocket import WebSocketTimeoutException, create_connection

from sync_code_to_jupyter import (  # noqa: E402
    JupyterContents,
    _jupyter_url_from_env,
    _load_dotenv,
    _parse_server,
)


def _safe_write(stream, text: str) -> None:
    """Write kernel output on Windows consoles that lack full Unicode."""
    try:
        stream.write(text)
    except UnicodeEncodeError:
        stream.write(
            text.encode(stream.encoding or "utf-8", errors="replace").decode(
                stream.encoding or "utf-8", errors="replace"
            )
        )
    stream.flush()


def _kernel_id(client: JupyterContents) -> str:
    sessions = client.get_json("/api/sessions")
    if not sessions:
        raise RuntimeError("No active Jupyter sessions — open the notebook on Kaggle first.")
    for sess in sessions:
        kid = sess.get("kernel", {}).get("id")
        if kid:
            return kid
    raise RuntimeError(f"No kernel id in sessions: {sessions!r}")


def _ws_url(base: str, kernel_id: str, token: str) -> str:
    parsed = urllib.parse.urlparse(base)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    path = f"{parsed.path.rstrip('/')}/api/kernels/{kernel_id}/channels"
    url = urllib.parse.urlunparse((scheme, parsed.netloc, path, "", "", ""))
    if token:
        url = f"{url}?token={urllib.parse.quote(token)}"
    return url


def execute_code(
    url: str,
    code: str,
    *,
    timeout_s: float = 7200.0,
    poll_s: float = 0.25,
) -> tuple[str, str]:
    """Run ``code`` on the kernel; return (stdout, stderr) text."""
    base, token = _parse_server(url)
    client = JupyterContents(base, token)
    kid = _kernel_id(client)
    ws_url = _ws_url(base, kid, token)
    ctx = ssl.create_default_context()
    ws = create_connection(ws_url, sslopt={"context": ctx}, timeout=60)
    ws.settimeout(poll_s)

    msg_id = uuid.uuid4().hex
    req = {
        "header": {
            "msg_id": msg_id,
            "msg_type": "execute_request",
            "username": "",
            "session": "",
            "date": "",
            "version": "5.3",
        },
        "parent_header": {},
        "metadata": {},
        "content": {
            "code": code,
            "silent": False,
            "store_history": True,
            "user_expressions": {},
            "allow_stdin": False,
            "stop_on_error": True,
        },
        "channel": "shell",
        "buffers": [],
    }
    ws.send(json.dumps(req))

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    started = time.monotonic()
    done = False
    while not done:
        if time.monotonic() - started > timeout_s:
            ws.close()
            raise TimeoutError(f"Kernel execution exceeded {timeout_s:.0f}s")
        try:
            raw = ws.recv()
        except WebSocketTimeoutException:
            continue
        if not raw:
            continue
        msg = json.loads(raw)
        parent = msg.get("parent_header") or {}
        if parent.get("msg_id") != msg_id:
            continue
        msg_type = msg.get("msg_type")
        content = msg.get("content") or {}
        if msg_type == "stream":
            text = content.get("text", "")
            if content.get("name") == "stderr":
                stderr_parts.append(text)
                _safe_write(sys.stderr, text)
            else:
                stdout_parts.append(text)
                _safe_write(sys.stdout, text)
        elif msg_type == "error":
            stderr_parts.append("\n".join(content.get("traceback", [])))
            ws.close()
            raise RuntimeError("\n".join(content.get("traceback", [])))
        elif msg_type == "execute_reply":
            status = content.get("status")
            if status == "error":
                ws.close()
                raise RuntimeError(f"execute_reply error: {content}")
            if status == "ok":
                done = True
    ws.close()
    return "".join(stdout_parts), "".join(stderr_parts)


def main() -> None:
    _load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("code", nargs="?", help="Python source to run")
    parser.add_argument("-f", "--file", type=Path, help="Read code from file")
    parser.add_argument("url", nargs="?", default=_jupyter_url_from_env())
    parser.add_argument("--timeout", type=float, default=7200.0)
    args = parser.parse_args()
    if args.file:
        code = args.file.read_text(encoding="utf-8")
    elif args.code:
        code = args.code
    else:
        code = sys.stdin.read()
    if not code.strip():
        raise SystemExit("No code to execute")
    if not args.url:
        raise SystemExit("Set KAGGLE_JUPYTER_NOTEBOOKS_URL in .env or pass URL")
    execute_code(args.url, code, timeout_s=args.timeout)


if __name__ == "__main__":
    main()
