"""`m5-infer cache list|clear` — inspect and purge engine caches.

Caches we manage:

* **CTRSP** (Cross-Turn Recurrent State Persist) — full GDN + FA state
  snapshots keyed by token-bytes hash. Disk: ``$DATA/state/ctrsp/*.npz``.
* **OIRC** (Opt-In Response Cache) — deterministic full-response replay.
  In-memory only (dies with the server); listed for completeness.
* **TPC** (Tokenizer Path Cache) — raw-bytes → token-list LRU. In-memory.

The server stores runtime state in the XDG data dir; the CLI reads it
directly from disk so this command works even with the server stopped.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from app.cli._http import BOLD, CYAN, DIM, GRAY, GREEN, NC, YELLOW, api_get


def _fmt_size(nbytes: int) -> str:
    if nbytes >= 1 << 30:
        return f"{nbytes / (1 << 30):.2f} GB"
    if nbytes >= 1 << 20:
        return f"{nbytes / (1 << 20):.0f} MB"
    if nbytes >= 1 << 10:
        return f"{nbytes / 1024:.0f} KB"
    return f"{nbytes} B"


def _dir_size(path: Path) -> int:
    if not path.is_dir():
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            pass
    return total


def _ctrsp_dir() -> Path:
    from app.core.paths import ctrsp_dir
    return ctrsp_dir()


def run_list(_args) -> int:
    """Print current cache usage."""
    ctrsp = _ctrsp_dir()
    ctrsp_size = _dir_size(ctrsp)
    ctrsp_count = len(list(ctrsp.glob("*.npz"))) if ctrsp.is_dir() else 0

    print(f"{BOLD}m5-infer caches:{NC}")
    print()
    print(f"  {CYAN}L2 CTRSP{NC}  {GRAY}(disk, persistent){NC}")
    print(f"    path:    {ctrsp}")
    print(f"    entries: {ctrsp_count}")
    print(f"    size:    {_fmt_size(ctrsp_size)}")
    print()

    # OIRC + TPC are in-memory; query the server for rough stats.
    print(f"  {CYAN}L1 OIRC{NC}  {GRAY}(in-memory, opt-in){NC}")
    health = api_get("/health", timeout=2.0)
    if health and health.get("oirc"):
        oirc = health["oirc"]
        print(f"    entries: {oirc.get('size', 0)} / {oirc.get('max_entries', '?')}")
        print(f"    hits:    {oirc.get('hits', 0)}")
    else:
        print(f"    {DIM}(server not reachable, or OIRC idle){NC}")
    print()

    print(f"  {CYAN}L3 TPC{NC}   {GRAY}(in-memory, tokenizer lookups){NC}")
    if health and health.get("mtab"):
        mtab = health["mtab"]
        print(f"    lookups: {mtab.get('lookups', 0)}")
        print(f"    hits:    {mtab.get('exact_hits', 0) + mtab.get('partial_hits', 0)}")
    else:
        print(f"    {DIM}(server not reachable, or TPC idle){NC}")
    return 0


def run_clear(args) -> int:
    """Purge one or more caches."""
    targets = []
    if args.all:
        targets = ["ctrsp"]  # OIRC / TPC are in-memory only; cleared on restart
    else:
        if args.ctrsp: targets.append("ctrsp")
        if args.oirc:  targets.append("oirc")
        if args.tpc:   targets.append("tpc")

    if not targets:
        print(f"{YELLOW}Specify what to clear: --ctrsp, --oirc, --tpc, or --all{NC}")
        return 1

    cleared_any = False
    if "ctrsp" in targets:
        path = _ctrsp_dir()
        if path.is_dir():
            size = _dir_size(path)
            shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
            print(f"  {GREEN}✓{NC} CTRSP cleared ({_fmt_size(size)} freed)")
            cleared_any = True
        else:
            print(f"  {DIM}CTRSP already empty{NC}")

    if "oirc" in targets or "tpc" in targets:
        # These are in-memory; purging requires hitting the server
        names = [n for n in ("oirc", "tpc") if n in targets]
        print(
            f"  {YELLOW}⚠{NC}  {'/'.join(names).upper()} is in-memory only. "
            f"Restart the server to clear, or call /v1/cache/clear if the "
            f"endpoint exists (future v1.1.3)."
        )

    return 0 if cleared_any or ("oirc" in targets or "tpc" in targets) else 1
