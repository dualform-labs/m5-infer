"""Runtime path resolution for m5-infer.

Decides where the engine reads configuration and writes state/logs, in a way
that works for both source installs (editable clone) and PyPI installs
(site-packages, no cwd-relative `configs/` or `state/`).

Resolution rules
----------------

Data dir (state, logs, ctrsp snapshots, metrics.db):
1. ``$M5_INFER_DATA_DIR`` (explicit override).
2. If the current working directory looks like the source tree (``pyproject.toml``
   AND ``app/`` both present), use the cwd — preserves the historical
   ``./state/``, ``./logs/`` layout for developers.
3. ``$XDG_DATA_HOME/m5-infer`` if set, else ``~/.local/share/m5-infer``.

Config dir (engine.toml, models.toml):
1. ``$M5_INFER_CONFIG`` (file or directory).
2. ``./configs/<name>`` (cwd).
3. ``$XDG_CONFIG_HOME/m5-infer/<name>`` if set, else ``~/.config/m5-infer/<name>``.
4. Walk parents of this file until a ``pyproject.toml`` + ``configs/`` is found
   (editable install / in-tree execution).
5. ``app/_defaults/<name>`` bundled inside the installed package.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Data dir (state / logs / snapshots)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def data_root() -> Path:
    """Return the base directory for runtime state (state/, logs/, …)."""
    override = os.getenv("M5_INFER_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()

    cwd = Path.cwd()
    if (cwd / "pyproject.toml").is_file() and (cwd / "app").is_dir():
        return cwd

    xdg = os.getenv("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return base / "m5-infer"


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def state_dir() -> Path:
    """Directory for persistent state (CTRSP snapshots, SSD n-gram tables, …)."""
    return _ensure(data_root() / "state")


def logs_dir() -> Path:
    """Directory for server and benchmark logs."""
    return _ensure(data_root() / "logs")


def ctrsp_dir() -> Path:
    """Directory for CTRSP snapshot files."""
    return _ensure(state_dir() / "ctrsp")


def metrics_db_path() -> Path:
    """Default path for the SQLite metrics store."""
    return state_dir() / "metrics.db"


# ---------------------------------------------------------------------------
# Config dir (engine.toml / models.toml)
# ---------------------------------------------------------------------------


def _config_search_order(name: str) -> Iterable[Path]:
    """Yield candidate paths for a config file, in priority order.

    Only external (user-editable) paths are yielded. The package-bundled
    default is loaded separately because it lives inside ``importlib.resources``
    and may not be a real filesystem path on every platform.
    """
    override = os.getenv("M5_INFER_CONFIG")
    if override:
        p = Path(override).expanduser()
        if p.is_dir():
            yield p / name
        else:
            yield p

    yield Path.cwd() / "configs" / name

    xdg_config = os.getenv("XDG_CONFIG_HOME")
    xdg_base = Path(xdg_config).expanduser() if xdg_config else Path.home() / ".config"
    yield xdg_base / "m5-infer" / name

    # Editable install / in-tree execution: walk up from this file
    current = Path(__file__).resolve().parent
    for ancestor in (current, *current.parents):
        if (ancestor / "pyproject.toml").is_file() and (ancestor / "configs").is_dir():
            yield ancestor / "configs" / name
            break


def load_bundled_config(name: str) -> str | None:
    """Read the package-bundled default config text, or ``None`` if missing."""
    try:
        from importlib.resources import files
    except ImportError:  # pragma: no cover
        return None
    try:
        bundled = files("app").joinpath("_defaults", name)
        if bundled.is_file():
            return bundled.read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, AttributeError):
        return None
    return None


def find_config(name: str) -> tuple[Path | None, str | None]:
    """Locate a config file.

    Returns ``(path, text)`` where exactly one of them is populated. The path
    form is returned when an external (user) file is found; the text form is
    returned when the bundled default inside the installed package is used.

    Raises ``FileNotFoundError`` if neither an external file nor a bundled
    default exists.
    """
    for cand in _config_search_order(name):
        if cand.is_file():
            return cand, None

    text = load_bundled_config(name)
    if text is not None:
        return None, text

    searched = [str(p) for p in _config_search_order(name)]
    raise FileNotFoundError(
        f"Config file '{name}' not found. Searched:\n"
        + "\n".join(f"  - {p}" for p in searched)
        + "\n  - bundled app/_defaults/" + name
        + "\nRun `m5-infer init` to create one in ./configs/."
    )
