"""`m5-infer pull <model>` — download + load a HuggingFace model through the server.

Consumes the NDJSON streaming response from ``/v1/models/pull`` (v1.1.2+).
For backward compatibility with older servers that ignore the stream flag and
return a single JSON blob, a background polling thread reports on-disk cache
growth so the user still sees progress.
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
    failed = False
    try:
        for event in api_post_stream("/v1/models/pull", {"model": repo, "stream": True}):
            # Transport-level failure (HTTP error from api_post_stream)
            if "error" in event and "phase" not in event:
                stop_flag["stop"] = True
                hb.join(timeout=1.0)
                code = event.get("status_code", "?")
                print(f"  {RED}✗ pull failed ({code}):{NC} {event['error']}")
                return 1
            phase = event.get("phase", "")
            saw_stream_event = True
            if phase == "error":
                stop_flag["stop"] = True
                hb.join(timeout=1.0)
                print(
                    f"  {RED}✗ {event.get('error_code','error')}:{NC} "
                    f"{event.get('error','(no detail)')}"
                )
                failed = True
                break
            if phase == "downloading":
                # Server reports per-second size / speed. Suppress our own
                # heartbeat and render the server's number.
                stop_flag["stop"] = True
                size = event.get("bytes_dl", 0)
                mbps = event.get("mbps", 0)
                elapsed = event.get("elapsed_s", 0)
                sys.stdout.write(
                    f"\r  {CYAN}downloading{NC}  {_fmt_size(size):>10s}  "
                    f"{GRAY}({mbps:5.1f} MB/s · elapsed {elapsed}s){NC}"
                    + " " * 8
                )
                sys.stdout.flush()
            elif phase == "load_start":
                stop_flag["stop"] = True
                sys.stdout.write(f"\r  {CYAN}loading model into MLX...{NC}" + " " * 30)
                sys.stdout.flush()
            elif phase == "success":
                final = event
    except KeyboardInterrupt:
        stop_flag["stop"] = True
        hb.join(timeout=1.0)
        print(f"\n  {YELLOW}aborted (download may continue on server){NC}")
        return 130

    stop_flag["stop"] = True
    hb.join(timeout=1.0)

    if failed:
        return 1
    if saw_stream_event:
        print()
    total = _hf_cache_size(repo) or 0
    if final and final.get("status") in ("loaded", "already_loaded"):
        print(f"  {GREEN}✓ {final['status']}{NC}  {_fmt_size(total)} on disk")
        return 0
    if final:
        # Backward-compat with old servers (single JSON blob, no "phase")
        print(f"  {CYAN}server returned:{NC} {json.dumps(final)[:200]}")
        return 0
    print(f"  {GREEN}✓ complete{NC}  {_fmt_size(total)} on disk")
    return 0
