"""`m5-infer status` and `m5-infer stop` subcommand implementations."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from typing import Any

from app.cli._http import (
    BOLD,
    CYAN,
    DIM,
    GREEN,
    GRAY,
    NC,
    RED,
    YELLOW,
    api_get,
    server_url,
)


def _fmt_memory(mem: dict[str, Any] | None) -> str:
    if not mem:
        return f"{GRAY}(unknown){NC}"
    active = mem.get("active_gb", 0.0)
    peak = mem.get("peak_gb", 0.0)
    return f"{active:.2f} GB active / {peak:.2f} GB peak"


def _fmt_cache(mtab: dict[str, Any] | None) -> str:
    if not mtab or not mtab.get("lookups"):
        return f"{GRAY}(no activity yet){NC}"
    hits = mtab.get("exact_hits", 0) + mtab.get("partial_hits", 0)
    lookups = mtab.get("lookups", 0)
    rate = (hits / lookups * 100) if lookups else 0
    return f"{hits}/{lookups} ({rate:.1f}%)"


def run_status(_args=None) -> int:
    """Pretty-print the engine status via /health."""
    health = api_get("/health", timeout=3.0)
    if health is None:
        print(
            f"{YELLOW}m5-infer server not reachable at {server_url()}.{NC}\n"
            f"  Start it with: {BOLD}m5-infer start{NC}"
        )
        return 1

    ready = health.get("status") == "ready"
    status_color = GREEN if ready else YELLOW
    print(f"{BOLD}m5-infer status{NC}  {status_color}{health.get('status')}{NC}")
    print(f"  {GRAY}url:{NC}       {server_url()}")
    model = health.get("model") or "(none loaded)"
    print(f"  {GRAY}model:{NC}     {model}")
    print(f"  {GRAY}memory:{NC}    {_fmt_memory(health.get('memory'))}")
    print(f"  {GRAY}cache:{NC}     {_fmt_cache(health.get('mtab'))}")

    # Optional v1.1.2+ fields
    version = health.get("version")
    if version:
        print(f"  {GRAY}version:{NC}   {version}")
    uptime = health.get("uptime_s")
    if uptime:
        mins, secs = divmod(int(uptime), 60)
        hours, mins = divmod(mins, 60)
        days, hours = divmod(hours, 24)
        parts = []
        if days:  parts.append(f"{days}d")
        if hours: parts.append(f"{hours}h")
        if mins:  parts.append(f"{mins}m")
        parts.append(f"{secs}s")
        print(f"  {GRAY}uptime:{NC}    {' '.join(parts)}")
    served = health.get("requests_served")
    if served is not None:
        print(f"  {GRAY}requests:{NC}  {served}")

    return 0 if ready else 1


def _find_pids_on_port(port: int) -> list[int]:
    """Return PIDs listening on ``port`` (macOS / Linux via lsof)."""
    try:
        result = subprocess.run(
            ["lsof", "-t", "-i", f":{port}", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    return [int(pid) for pid in result.stdout.split() if pid.isdigit()]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def run_stop(_args=None) -> int:
    """Locate and stop the server process on the configured port.

    Sends SIGTERM, waits up to ``grace`` seconds for graceful shutdown, then
    falls back to SIGKILL. Returns 0 only when every target PID is confirmed
    gone.
    """
    import time as _time

    url = server_url()
    try:
        port = int(url.rsplit(":", 1)[1])
    except (ValueError, IndexError):
        port = 11436

    pids = _find_pids_on_port(port)
    if not pids:
        print(f"{GRAY}No m5-infer server found on port {port}.{NC}")
        return 0

    grace_s = 5.0
    killed: list[int] = []

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except PermissionError:
            print(f"{RED}Permission denied to stop PID {pid}.{NC}", file=sys.stderr)
            return 1

        # Poll for up to grace_s
        deadline = _time.monotonic() + grace_s
        while _pid_alive(pid) and _time.monotonic() < deadline:
            _time.sleep(0.2)

        if _pid_alive(pid):
            # Graceful shutdown timed out — escalate
            try:
                os.kill(pid, signal.SIGKILL)
                print(f"{GREEN}SIGKILL PID {pid}{NC} {DIM}(SIGTERM timed out after {grace_s:.0f}s){NC}")
            except ProcessLookupError:
                pass
            killed.append(pid)
        else:
            print(f"{GREEN}Stopped PID {pid}{NC} {DIM}(port {port}){NC}")
            killed.append(pid)

    # Final port check
    remaining = _find_pids_on_port(port)
    if remaining:
        print(
            f"{RED}Port {port} still bound after stop attempt: {remaining}{NC}",
            file=sys.stderr,
        )
        return 1
    return 0
