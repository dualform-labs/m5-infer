"""MLX Vision Backend — handles image+text inference via multimodal models."""

from __future__ import annotations
from app.backend.mlx_text_backend import MLXTextBackend
from app.core.logging import get_logger

logger = get_logger(__name__)

class MLXVisionBackend(MLXTextBackend):
    """Extends MLXTextBackend with vision (image+text) capabilities.

    Uses the same mlx_lm infrastructure but adds image preprocessing
    and vision-specific prompt construction.

    Full implementation in Phase 7 (Multimodal Speed Path).
    """

    def __init__(self):
        super().__init__()
        self._supports_vision = False

    def load_model(self, path: str, **kwargs) -> None:
        super().load_model(path, **kwargs)
        # Check if loaded model supports vision
        if self._model is not None and hasattr(self._model, 'vision_model'):
            self._supports_vision = True
            logger.info("Vision model detected")

    def supports_vision(self) -> bool:
        return self._supports_vision
