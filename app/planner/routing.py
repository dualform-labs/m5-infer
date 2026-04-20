from __future__ import annotations
from app.planner.plan_types import RoutingDecision, ModelRole
from app.core.model_registry import get_registry
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

def route_request(
    has_image: bool = False,
    force_heavy: bool = False,
    model_hint: str | None = None,
) -> RoutingDecision:
    """Decide which model to use for a request."""
    settings = get_settings()
    registry = get_registry()

    # If specific model requested
    if model_hint:
        profile = registry.get_by_name(model_hint)
        if profile:
            return RoutingDecision(
                selected_model=profile.name,
                use_sub_model=profile.role != ModelRole.MAIN,
                requires_sub_load=profile.role != ModelRole.MAIN,
                reason=f"User requested model: {model_hint}",
                unload_after_run=profile.role != ModelRole.MAIN,
                use_vision_path=has_image,
            )

    # Image input -> heavy sub
    if has_image and settings.planner.route_image_input_to_sub:
        heavy = registry.get_heavy_sub()
        if heavy and heavy.supports_vision:
            return RoutingDecision(
                selected_model=heavy.name,
                use_sub_model=True,
                requires_sub_load=True,
                reason="Image input requires vision model",
                unload_after_run=True,
                use_vision_path=True,
            )

    # Heavy reasoning -> heavy sub
    if force_heavy and settings.planner.route_heavy_reasoning_to_sub:
        heavy = registry.get_heavy_sub()
        if heavy:
            return RoutingDecision(
                selected_model=heavy.name,
                use_sub_model=True,
                requires_sub_load=True,
                reason="Heavy reasoning requested",
                unload_after_run=True,
                use_vision_path=False,
            )

    # Default -> main model
    main = registry.get_main()
    return RoutingDecision(
        selected_model=main.name,
        use_sub_model=False,
        requires_sub_load=False,
        reason="Default routing to main model",
        unload_after_run=False,
        use_vision_path=False,
    )
