"""Shared HTTP helpers for m5-infer CLI subcommands.

Thin wrapper around urllib so the CLI has no extra runtime deps beyond what the
engine itself already pulls in. Uses :func:`app.core.config.get_settings` to
discover the server URL; the ``$M5_INFER_URL`` env var overrides it (useful
when pointing a local CLI at a remote server).
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Iterator

# ANSI colors — reused across CLI subcommands
GREEN = "\033[0;32m"
CYAN = "\033[0;36m"
YELLOW = "\033[0;33m"
RED = "\033[0;31m"
GRAY = "\033[0;90m"
BOLD = "\033[1m"
DIM = "\033[2m"
NC = "\033[0m"


def server_url() -> str:
    """Resolve the base URL of the local m5-infer server."""
    override = os.getenv("M5_INFER_URL")
    if override:
        return override.rstrip("/")
    try:
        from app.core.config import get_settings
        s = get_settings()
        host = s.server.host
        # 0.0.0.0 bind means any-interface; clients should hit 127.0.0.1.
        if host in ("0.0.0.0", "::"):
            host = "127.0.0.1"
        return f"http://{host}:{s.server.port}"
    except Exception:
        return "http://127.0.0.1:11436"


def api_get(path: str, timeout: float = 5.0) -> dict[str, Any] | None:
    """GET a JSON endpoint. Returns None on any failure."""
    url = f"{server_url()}{path}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def api_post(path: str, data: dict[str, Any], timeout: float = 300.0) -> dict[str, Any]:
    """POST JSON to an endpoint. Returns the parsed response or ``{"error": ...}``."""
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{server_url()}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")
            return {"error": body, "status_code": e.code}
        except Exception:
            return {"error": str(e), "status_code": e.code}
    except Exception as e:
        return {"error": str(e)}


def api_post_stream(
    path: str,
    data: dict[str, Any],
    timeout: float = 3600.0,
) -> Iterator[dict[str, Any]]:
    """POST and iterate SSE / newline-delimited JSON events from the response.

    Yields each parsed JSON chunk. Lines starting with ``data:`` are
    unwrapped; the ``[DONE]`` sentinel ends the stream. Any non-JSON line
    is skipped silently.
    """
    body = json.dumps(data).encode()
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    req = urllib.request.Request(f"{server_url()}{path}", data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    break
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except urllib.error.HTTPError as e:
        try:
            yield {"error": e.read().decode("utf-8", errors="replace"), "status_code": e.code}
        except Exception:
            yield {"error": str(e), "status_code": e.code}
    except Exception as e:
        yield {"error": str(e)}


def die_if_server_down() -> None:
    """Print a friendly hint and exit if the server is unreachable."""
    if api_get("/health", timeout=2.0) is None:
        print(
            f"{YELLOW}m5-infer server not reachable at {server_url()}.{NC}\n"
            f"  Start it with: {BOLD}m5-infer start{NC}",
            file=sys.stderr,
        )
        sys.exit(2)
