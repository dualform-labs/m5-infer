"""Main resident model manager for the M5 MLX Inference Engine.

Manages the lifecycle of the main model (e.g. Qwen3.5-9B):
loading on server startup, health-check introspection, model
switching, and clean unload on shutdown.

Supports loading any MLX-compatible model from HuggingFace by repo ID.
"""

from __future__ import annotations

from app.backend.mlx_text_backend import MLXTextBackend
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.model_registry import get_registry

logger = get_logger(__name__)


class MainModelManager:
    """Manages the main resident model lifecycle."""

    def __init__(self) -> None:
        self._backend = MLXTextBackend()
        self._is_ready = False
        self._current_model_id: str = ""
        self._loaded_models_history: list[str] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        """Load the main model and warm it up.

        Called once during server startup.  Retrieves the main model
        profile from the registry and delegates to :class:`MLXTextBackend`.
        """
        registry = get_registry()
        profile = registry.get_main()

        logger.info(
            "Loading main model: %s from %s",
            profile.name,
            profile.path,
        )

        try:
            self._backend.load_model(profile.path)
        except Exception:
            logger.exception("Failed to load main model: %s", profile.name)
            raise

        self._is_ready = True
        self._current_model_id = profile.path
        self._loaded_models_history.append(profile.path)

        mem = self._backend.get_memory_usage()
        logger.info(
            "Main model loaded. Active memory: %.2f GB, Peak: %.2f GB",
            mem["active_gb"],
            mem["peak_gb"],
        )

    async def load_model(self, model_id: str) -> dict:
        """Load a model by HuggingFace repo ID or local path.

        If a different model is already loaded, unloads it first.
        mlx_lm.load automatically downloads from HuggingFace if needed.

        Args:
            model_id: HuggingFace repo ID (e.g. 'mlx-community/Llama-3.2-3B-Instruct-4bit')
                       or local path to model directory.

        Returns:
            dict with model info and memory usage.
        """
        if self._is_ready and self._current_model_id == model_id:
            logger.info("Model %s already loaded", model_id)
            return {
                "status": "already_loaded",
                "model": model_id,
                "memory": self._backend.get_memory_usage(),
            }

        # Unload current model if one is loaded
        if self._is_ready:
            logger.info("Unloading current model %s to load %s", self._current_model_id, model_id)
            self._backend.unload_model()
            self._is_ready = False

        logger.info("Loading model: %s (will download from HuggingFace if needed)", model_id)
        try:
            self._backend.load_model(model_id)
        except Exception as e:
            logger.exception("Failed to load model: %s", model_id)
            raise RuntimeError(f"Failed to load model {model_id}: {e}") from e

        self._is_ready = True
        self._current_model_id = model_id
        if model_id not in self._loaded_models_history:
            self._loaded_models_history.append(model_id)

        mem = self._backend.get_memory_usage()
        logger.info(
            "Model loaded: %s (%.2f GB active, %.2f GB peak)",
            model_id, mem["active_gb"], mem["peak_gb"],
        )
        return {
            "status": "loaded",
            "model": model_id,
            "memory": mem,
        }

    def get_current_model_id(self) -> str:
        """Return the currently loaded model's HF repo ID or path."""
        return self._current_model_id

    def get_loaded_history(self) -> list[str]:
        """Return list of all models that have been loaded in this session."""
        return list(self._loaded_models_history)

    async def shutdown(self) -> None:
        """Unload the main model and release memory.

        Called during server shutdown.  Safe to call even if the model
        was never loaded (or was already unloaded).
        """
        if not self._is_ready:
            logger.debug("shutdown() called but main model was not loaded")
            return

        logger.info("Unloading main model")
        try:
            self._backend.unload_model()
        except Exception:
            logger.exception("Error during main model unload")
        finally:
            self._is_ready = False

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """Return True when the main model is loaded and ready to serve."""
        return self._is_ready

    def get_backend(self) -> MLXTextBackend:
        """Return the underlying :class:`MLXTextBackend`.

        Raises:
            RuntimeError: If the model has not been loaded yet.
        """
        if not self._is_ready:
            raise RuntimeError("Main model not ready")
        return self._backend

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    async def health_check(self) -> dict:
        """Return a health-status dict suitable for the ``/health`` endpoint."""
        if not self._is_ready:
            return {"status": "not_ready", "model": None}

        mem = self._backend.get_memory_usage()
        return {
            "status": "ready",
            "model": self._backend.get_model_name(),
            "memory": mem,
        }
