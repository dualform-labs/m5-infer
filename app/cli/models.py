"""`m5-infer models` — list loaded / cached models.

Shows:

* Which model the running server has loaded (via ``/v1/models``).
* What's sitting in the HuggingFace hub cache (local filesystem scan; does
  not require the server to be running).
"""

from __future__ import annotations

from pathlib import Path

from app.cli._http import BOLD, CYAN, DIM, GRAY, GREEN, NC, YELLOW, api_get


def _hf_cache_dir() -> Path:
    """Return the HuggingFace hub cache root."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        return Path(HF_HUB_CACHE)
    except Exception:
        return Path.home() / ".cache" / "huggingface" / "hub"


def _cache_entries() -> list[tuple[str, int]]:
    """Scan the HF cache and return ``[(repo_id, bytes)]`` pairs."""
    root = _hf_cache_dir()
    if not root.is_dir():
        return []
    entries: list[tuple[str, int]] = []
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith("models--"):
            continue
        repo_id = child.name.replace("models--", "").replace("--", "/", 1)
        # Sum size across all snapshots (HF uses hardlinks into blobs/, so we
        # sum the blobs/ directory to get an accurate on-disk size)
        blobs = child / "blobs"
        total = 0
        if blobs.is_dir():
            for blob in blobs.iterdir():
                try:
                    total += blob.stat().st_size
                except OSError:
                    pass
        entries.append((repo_id, total))
    return sorted(entries, key=lambda x: -x[1])


def _fmt_size(nbytes: int) -> str:
    if nbytes >= 1 << 30:
        return f"{nbytes / (1 << 30):.1f} GB"
    if nbytes >= 1 << 20:
        return f"{nbytes / (1 << 20):.0f} MB"
    return f"{nbytes / 1024:.0f} KB"


def run(args) -> int:
    """Dispatch ``m5-infer models --loaded|--cached|(both)``."""
    show_loaded = args.loaded or not args.cached
    show_cached = args.cached or not args.loaded

    if show_loaded:
        models = api_get("/v1/models", timeout=3.0)
        print(f"{BOLD}Loaded on server:{NC}")
        if models is None:
            print(f"  {YELLOW}(server not reachable){NC}")
        else:
            data = models.get("data") or []
            if not data:
                print(f"  {DIM}(no model loaded){NC}")
            for item in data:
                mid = item.get("id", "?")
                print(f"  {GREEN}▸{NC} {mid}")
        print()

    if show_cached:
        entries = _cache_entries()
        cache_root = _hf_cache_dir()
        print(f"{BOLD}In HF cache:{NC} {GRAY}{cache_root}{NC}")
        if not entries:
            print(f"  {DIM}(empty){NC}")
        else:
            # Highlight anything containing "mlx" as actually usable here.
            for repo, size in entries:
                marker = f"{CYAN}▸{NC}" if "mlx" in repo.lower() else f"{DIM}▸{NC}"
                print(f"  {marker} {repo:<60s} {_fmt_size(size):>10s}")
        total = sum(size for _, size in entries)
        print(f"  {GRAY}total: {_fmt_size(total)}{NC}")

    return 0
