"""RDMS Draft model loader.

Loads a small draft model at startup and keeps it resident in memory.
The draft is used as the "speculator" in RDMS speculative decoding —
it predicts the next k tokens which the main model then verifies.

Memory cost (4-bit quantized):
  Qwen2.5-0.5B  ~0.4 GB
  Qwen2.5-1.5B  ~0.8 GB
  Qwen2.5-3B    ~1.7 GB
  Qwen2.5-7B    ~4.0 GB
"""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)


class DraftModelLoader:
    """Singleton loader for the resident draft model.

    Use:
        loader = DraftModelLoader()
        loader.load("mlx-community/Qwen2.5-1.5B-Instruct-4bit")
        if loader.is_loaded():
            model, tokenizer = loader.get()
    """

    _instance: "DraftModelLoader | None" = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not getattr(self, "_initialized", False):
            self._model = None
            self._tokenizer = None
            self._path = ""
            self._initialized = True

    def load(self, model_path: str) -> bool:
        """Load draft model from HuggingFace path.

        Returns True on success, False on failure (logs error).
        Idempotent: if already loaded with same path, no-op.
        """
        if not model_path:
            return False
        if self._model is not None and self._path == model_path:
            return True

        try:
            from mlx_lm import load as mlx_load
            logger.info("RDMS: loading draft model %s", model_path)
            model, tokenizer = mlx_load(model_path)
            self._model = model
            self._tokenizer = tokenizer
            self._path = model_path
            logger.info("RDMS: draft model loaded successfully (%s)", model_path)
            return True
        except Exception:
            logger.exception("RDMS: failed to load draft model %s", model_path)
            self._model = None
            self._tokenizer = None
            self._path = ""
            return False

    def unload(self) -> None:
        """Release draft model from memory."""
        self._model = None
        self._tokenizer = None
        self._path = ""
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass
        logger.info("RDMS: draft model unloaded")

    def is_loaded(self) -> bool:
        return self._model is not None

    def get(self) -> tuple[Any, Any]:
        """Return (model, tokenizer) tuple. Raises if not loaded."""
        if self._model is None:
            raise RuntimeError("Draft model not loaded — call load() first")
        return self._model, self._tokenizer

    def get_path(self) -> str:
        return self._path

    def stats(self) -> dict:
        return {
            "loaded": self.is_loaded(),
            "path": self._path,
        }


# Convenience accessor (singleton)
def get_draft_loader() -> DraftModelLoader:
    return DraftModelLoader()
