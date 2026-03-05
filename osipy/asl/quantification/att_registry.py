"""ATT model registry."""

import logging

from osipy.common.exceptions import DataValidationError

logger = logging.getLogger(__name__)

ATT_MODEL_REGISTRY: dict[str, type] = {}


def register_att_model(name: str):
    """Decorator to register an ATT estimation model.

    Parameters
    ----------
    name : str
        Registry key for the model (e.g. ``'buxton'``).

    Returns
    -------
    Callable
        Class decorator.
    """

    def decorator(cls):
        if name in ATT_MODEL_REGISTRY:
            logger.warning(
                "Overwriting '%s' (%s) with %s",
                name,
                ATT_MODEL_REGISTRY[name].__name__,
                cls.__name__,
            )
        ATT_MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_att_model(name: str):
    """Get an ATT model instance by name.

    Parameters
    ----------
    name : str
        Model name (e.g. ``'buxton'``).

    Returns
    -------
    BaseASLModel
        Model instance.

    Raises
    ------
    DataValidationError
        If model name is not recognized.
    """
    if name not in ATT_MODEL_REGISTRY:
        valid = ", ".join(sorted(ATT_MODEL_REGISTRY.keys()))
        raise DataValidationError(f"Unknown ATT model: {name}. Valid: {valid}")
    return ATT_MODEL_REGISTRY[name]()


def list_att_models() -> list[str]:
    """Return names of all registered ATT models.

    Returns
    -------
    list[str]
        Sorted list of registered model names.
    """
    return sorted(ATT_MODEL_REGISTRY.keys())
