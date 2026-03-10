"""Concentration model registry."""

import logging
from collections.abc import Callable

from osipy.common.exceptions import DataValidationError

logger = logging.getLogger(__name__)

_CONCENTRATION_MODEL_REGISTRY: dict[str, Callable] = {}


def register_concentration_model(name: str):
    """Decorator to register a concentration conversion model."""

    def decorator(func: Callable) -> Callable:
        if name in _CONCENTRATION_MODEL_REGISTRY:
            logger.warning("Overwriting concentration model '%s'", name)
        _CONCENTRATION_MODEL_REGISTRY[name] = func
        return func

    return decorator


def get_concentration_model(name: str) -> Callable:
    """Get a concentration conversion model by name."""
    if name not in _CONCENTRATION_MODEL_REGISTRY:
        valid = ", ".join(sorted(_CONCENTRATION_MODEL_REGISTRY.keys()))
        raise DataValidationError(
            f"Unknown concentration model: {name}. Valid: {valid}",
        )
    return _CONCENTRATION_MODEL_REGISTRY[name]


def list_concentration_models() -> list[str]:
    """List registered concentration conversion models."""
    return sorted(_CONCENTRATION_MODEL_REGISTRY.keys())
