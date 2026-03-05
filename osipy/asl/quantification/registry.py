"""ASL quantification model registry."""

import logging

from osipy.common.exceptions import DataValidationError

logger = logging.getLogger(__name__)

ASL_QUANTIFICATION_REGISTRY: dict[str, type] = {}


def register_quantification_model(name: str):
    """Decorator to register an ASL quantification model.

    Parameters
    ----------
    name : str
        Registry key for the model (e.g. ``'pcasl_single_pld'``).

    Returns
    -------
    Callable
        Class decorator.
    """

    def decorator(cls):
        if name in ASL_QUANTIFICATION_REGISTRY:
            logger.warning(
                "Overwriting '%s' (%s) with %s",
                name,
                ASL_QUANTIFICATION_REGISTRY[name].__name__,
                cls.__name__,
            )
        ASL_QUANTIFICATION_REGISTRY[name] = cls
        return cls

    return decorator


def get_quantification_model(name: str):
    """Get an ASL quantification model instance by name.

    Parameters
    ----------
    name : str
        Model name (e.g. ``'pcasl_single_pld'``).

    Returns
    -------
    BaseASLModel
        Model instance.

    Raises
    ------
    DataValidationError
        If model name is not recognized.
    """
    if name not in ASL_QUANTIFICATION_REGISTRY:
        valid = ", ".join(sorted(ASL_QUANTIFICATION_REGISTRY.keys()))
        raise DataValidationError(
            f"Unknown quantification model: {name}. Valid: {valid}"
        )
    return ASL_QUANTIFICATION_REGISTRY[name]()


def list_quantification_models() -> list[str]:
    """Return names of all registered ASL quantification models.

    Returns
    -------
    list[str]
        Sorted list of registered model names.
    """
    return sorted(ASL_QUANTIFICATION_REGISTRY.keys())
