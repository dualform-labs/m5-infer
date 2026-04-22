"""`m5-infer pull <model>` — download + load a HuggingFace model through the server.

Until v1.1.2 Phase 2 adds SSE streaming to ``/v1/models/pull``, this uses the
synchronous endpoint and prints a best-effort heartbeat. Once Phase 2.1 lands
this module switches to the SSE consumer (a ``stream=True`` flag is already
accepted; the server side just ignores it for now).
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request

from app.cli._http import (
    BOLD,
    CYAN,
    DIM,
    GRAY,
    GREEN,
    NC,
    RED,
    YELLOW,
    api_get,
    api_post_stream,
    die_if_server_down,
    server_url,
)


def _supports_stream(pull_path: str) -> bool:
    """Probe whether the server understands the SSE query flag (v1.1.2+)."""
    # Cheap heuristic: v1.1.1 and earlier will happily accept the flag and
    # return the usual blocking JSON. v1.1.2+ returns ``text/event-stream``.
    # We run a HEAD against /openapi.json to look up the endpoint description,
    # but since we don't have a reliable marker yet, just always TRY streaming
    # and fall back to non-stream consumption if we don't see chunk-encoded
    # progress events within a grace window.
    return True  # keep simple; api_post_stream tolerates both shapes


def _hf_cache_size(repo_id: str) -> int | None:
    """Return the current on-disk size of the HF cache for this repo, or None."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        from pathlib import Path
    except ImportError:
        return None
    folder = Path(HF_HUB_CACHE) / ("models--" + repo_id.replace("/", "--"))
    if not folder.is_dir():
        return 0
    total = 0
    for item in folder.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            pass
    return total


def _fmt_size(nbytes: int) -> str:
    if nbytes >= 1 << 30:
        return f"{nbytes / (1 << 30):.2f} GB"
    if nbytes >= 1 << 20:
        return f"{nbytes / (1 << 20):.0f} MB"
    return f"{nbytes / 1024:.0f} KB"


def _heartbeat(repo_id: str, stop_flag: dict) -> None:
    """Background thread that prints on-disk-size growth every ~2 s."""
    t0 = time.perf_counter()
    last_size = _hf_cache_size(repo_id) or 0
    last_t = t0
    while not stop_flag["stop"]:
        time.sleep(2.0)
        cur = _hf_cache_size(repo_id) or 0
        now = time.perf_counter()
        delta = cur - last_size
        dt = now - last_t
        mbps = (delta / (1 << 20)) / dt if dt > 0 else 0
        elapsed = int(now - t0)
        sys.stdout.write(
            f"\r  {CYAN}downloading{NC}  {_fmt_size(cur):>10s}  "
            f"{GRAY}({mbps:5.1f} MB/s · elapsed {elapsed}s){NC}"
            + " " * 6
        )
        sys.stdout.flush()
        last_size = cur
        last_t = now
    sys.stdout.write("\r" + " " * 70 + "\r")
    sys.stdout.flush()


def run(args) -> int:
    die_if_server_down()
    repo = args.model
    if "/" not in repo:
        print(f"{YELLOW}Invalid model id '{repo}'. Expected format: org/name{NC}")
        return 1

    print(f"{BOLD}m5-infer pull{NC}  {repo}")
    print(f"  {GRAY}server: {server_url()}{NC}")

    # Start heartbeat thread for blocking-mode progress visibility
    stop_flag = {"stop": False}
    hb = threading.Thread(target=_heartbeat, args=(repo, stop_flag), daemon=True)
    hb.start()

    saw_stream_event = False
    final: dict | None = None
    try:
        for event in api_post_stream("/v1/models/pull", {"model": repo}):
            if "error" in event:
                stop_flag["stop"] = True
                hb.join(timeout=1.0)
                code = event.get("status_code", "?")
                print(f"  {RED}✗ pull failed ({code}):{NC} {event['error']}")
                return 1
            if event.get("status"):
                final = event
                saw_stream_event = True
                continue
            # Per-file progress event (Phase 2.1 format)
            if "file" in event and "percent" in event:
                saw_stream_event = True
                stop_flag["stop"] = True
                hb.join(timeout=0.5)
                pct = event["percent"]
                speed = event.get("speed_mbps", 0)
                eta = event.get("eta_s", 0)
                sys.stdout.write(
                    f"\r  {CYAN}{event['file'][:40]:<40s}{NC}  "
                    f"{pct:5.1f}%  {speed:5.1f} MB/s  "
                    f"{GRAY}ETA {int(eta)}s{NC}    "
                )
                sys.stdout.flush()
    except KeyboardInterrupt:
        stop_flag["stop"] = True
        hb.join(timeout=1.0)
        print(f"\n  {YELLOW}aborted (download may continue on server){NC}")
        return 130

    stop_flag["stop"] = True
    hb.join(timeout=1.0)

    if saw_stream_event:
        print()
    total = _hf_cache_size(repo) or 0
    if final and final.get("status") in ("success", "loaded", "already_loaded"):
        print(f"  {GREEN}✓ {final['status']}{NC}  {_fmt_size(total)} on disk")
        return 0
    if final:
        print(f"  {CYAN}server returned:{NC} {json.dumps(final)[:200]}")
        return 0
    print(f"  {GREEN}✓ complete{NC}  {_fmt_size(total)} on disk")
    return 0
