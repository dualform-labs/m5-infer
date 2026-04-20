"""Image Preprocessor — resolution policy and optimization for vision input."""

from __future__ import annotations
from dataclasses import dataclass
from app.core.logging import get_logger

logger = get_logger(__name__)

@dataclass
class ImageConfig:
    """Configuration for image preprocessing."""
    max_resolution: int = 1024
    target_resolution: int = 768
    jpeg_quality: int = 85
    fast_mode: bool = False

class ImagePreprocessor:
    """Preprocesses images for vision model input.

    Full implementation in Phase 7 (Multimodal Speed Path).
    Placeholder for now.
    """

    def __init__(self, config: ImageConfig | None = None):
        self._config = config or ImageConfig()

    def preprocess(self, image_path: str) -> dict:
        """Preprocess an image for model input. Returns model-ready format."""
        # Phase 7: implement actual preprocessing
        logger.info("Image preprocessing: %s (stub)", image_path)
        return {"path": image_path, "processed": False}

    def estimate_vision_tokens(self, image_path: str) -> int:
        """Estimate how many tokens an image will consume."""
        # Rough estimate: 256 tokens for a standard image
        return 256
