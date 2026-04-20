"""Sub Model Controller — manages dynamic loading/unloading of heavy and speed sub models."""

from __future__ import annotations
import time
import mlx.core as mx
from app.backend.mlx_text_backend import MLXTextBackend
from app.engine.memory_guard import MemoryGuard
from app.engine.mmrs_registry import get_mmrs_registry
from app.core.model_registry import get_registry
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

class SubModelController:
    """Manages sub model lifecycle: load, use, unload."""

    def __init__(self, memory_guard: MemoryGuard):
        self._memory_guard = memory_guard
        self._heavy_backend: MLXTextBackend | None = None
        self._speed_backend: MLXTextBackend | None = None
        self._heavy_loaded_at: float = 0
        self._speed_loaded_at: float = 0

    def load_heavy_sub(self) -> MLXTextBackend:
        """Load the heavy sub model (e.g., Gemma 4). Raises if insufficient memory."""
        registry = get_registry()
        profile = registry.get_heavy_sub()
        if profile is None:
            raise RuntimeError("No heavy sub model configured")

        if not self._memory_guard.can_load_sub_model(profile.estimated_model_ram_gb):
            raise MemoryError(
                f"Insufficient memory for {profile.name} "
                f"(needs {profile.estimated_model_ram_gb} GB)"
            )

        if self._heavy_backend and self._heavy_backend.is_loaded():
            return self._heavy_backend

        # MMRS: O(1) pointer swap if already resident.
        mmrs = get_mmrs_registry()
        if mmrs.is_active():
            resident = mmrs.get_resident("heavy")
            if resident is not None:
                logger.info("MMRS heavy hit — routing is a pointer swap (0 ms)")
                self._heavy_backend = resident  # type: ignore[assignment]
                self._heavy_loaded_at = time.time()
                return resident  # type: ignore[return-value]

        logger.info("Loading heavy sub model: %s", profile.name)
        backend = MLXTextBackend()
        backend.load_model(profile.path)
        self._heavy_backend = backend
        self._heavy_loaded_at = time.time()

        mem = backend.get_memory_usage()
        logger.info(
            "Heavy sub loaded: %s (%.2f GB active)",
            profile.name, mem["active_gb"],
        )
        if mmrs.is_active():
            mmrs.register("heavy", profile.path, backend, mem["active_gb"])
        return backend

    def load_speed_sub(self) -> MLXTextBackend:
        """Load the speed (draft) sub model for speculative decoding."""
        registry = get_registry()
        profile = registry.get_speed_sub()
        if profile is None:
            raise RuntimeError("No speed sub model configured")

        if not profile.path:
            raise RuntimeError("Speed sub model path not configured")

        if self._speed_backend and self._speed_backend.is_loaded():
            return self._speed_backend

        logger.info("Loading speed sub model: %s", profile.name)
        backend = MLXTextBackend()
        backend.load_model(profile.path)
        self._speed_backend = backend
        self._speed_loaded_at = time.time()
        return backend

    def unload_heavy_sub(self) -> None:
        if self._heavy_backend:
            logger.info("Unloading heavy sub model")
            self._heavy_backend.unload_model()
            self._heavy_backend = None

    def unload_speed_sub(self) -> None:
        if self._speed_backend:
            logger.info("Unloading speed sub model")
            self._speed_backend.unload_model()
            self._speed_backend = None

    def unload_all(self) -> None:
        self.unload_heavy_sub()
        self.unload_speed_sub()

    def get_heavy_backend(self) -> MLXTextBackend | None:
        return self._heavy_backend if self._heavy_backend and self._heavy_backend.is_loaded() else None

    def get_speed_backend(self) -> MLXTextBackend | None:
        return self._speed_backend if self._speed_backend and self._speed_backend.is_loaded() else None

    def check_warm_ttl(self) -> None:
        """Unload sub models that have been idle beyond warm_ttl."""
        settings = get_settings()
        from app.core.config import get_sub_model_config
        sub_cfg = get_sub_model_config()
        ttl = sub_cfg.warm_ttl_sec if sub_cfg else 180
        now = time.time()

        if self._heavy_backend and self._heavy_loaded_at > 0:
            if now - self._heavy_loaded_at > ttl:
                self.unload_heavy_sub()

        if self._speed_backend and self._speed_loaded_at > 0:
            if now - self._speed_loaded_at > ttl:
                self.unload_speed_sub()

    def get_stats(self) -> dict:
        return {
            "heavy_loaded": self._heavy_backend is not None and self._heavy_backend.is_loaded(),
            "speed_loaded": self._speed_backend is not None and self._speed_backend.is_loaded(),
        }
