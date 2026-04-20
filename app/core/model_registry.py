"""Model registry for M5 MLX Inference Engine.

Loads ModelProfile instances from models.toml and provides
lookup by role or name. This module does NOT load actual ML models;
it only holds metadata (ModelProfile data objects).
"""

from __future__ import annotations

from app.core.config import get_model_configs
from app.planner.plan_types import ModelProfile, ModelRole


class ModelRegistry:
    """Holds ModelProfile instances built from models.toml."""

    def __init__(self) -> None:
        raw_configs = get_model_configs()
        self._profiles: dict[str, ModelProfile] = {}

        for _key, cfg in raw_configs.items():
            profile = ModelProfile(
                name=cfg.name,
                path=cfg.path,
                modality=cfg.modality,
                role=ModelRole(cfg.role),
                load_strategy=cfg.load_strategy,
                supports_vision=cfg.supports_vision,
                estimated_model_ram_gb=cfg.estimated_model_ram_gb,
                max_context_tokens_hard=cfg.max_context_tokens_hard,
                max_output_tokens_hard=cfg.max_output_tokens_hard,
            )
            self._profiles[profile.name] = profile

        # Validate: at least one main model must exist.
        if not any(p.role == ModelRole.MAIN for p in self._profiles.values()):
            raise ValueError(
                "No model with role='main' found in models.toml. "
                "At least one main model must be configured."
            )

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def get_main(self) -> ModelProfile:
        """Return the model with role='main'.

        Guaranteed to exist (constructor validates this).
        """
        for p in self._profiles.values():
            if p.role == ModelRole.MAIN:
                return p
        # Unreachable after __init__ validation, but keeps mypy happy.
        raise ValueError("No main model configured")  # pragma: no cover

    def get_heavy_sub(self) -> ModelProfile | None:
        """Return the model with role='sub_heavy', or None."""
        for p in self._profiles.values():
            if p.role == ModelRole.SUB_HEAVY:
                return p
        return None

    def get_speed_sub(self) -> ModelProfile | None:
        """Return the model with role='sub_speed', or None."""
        for p in self._profiles.values():
            if p.role == ModelRole.SUB_SPEED:
                return p
        return None

    def get_by_name(self, name: str) -> ModelProfile | None:
        """Return a profile by its unique name, or None."""
        return self._profiles.get(name)

    def list_all(self) -> list[ModelProfile]:
        """Return all registered profiles."""
        return list(self._profiles.values())

    def is_vision_capable(self, name: str) -> bool:
        """Check if the named model supports vision input."""
        profile = self._profiles.get(name)
        if profile is None:
            return False
        return profile.supports_vision


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------

_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    """Return the singleton ModelRegistry, creating it on first call."""
    global _registry  # noqa: PLW0603
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
