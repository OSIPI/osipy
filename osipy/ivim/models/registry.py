"""IVIM model registry for dynamic lookup and extension.

This module provides the registry pattern for IVIM signal models,
enabling runtime model registration and retrieval by name.
"""

import logging
from typing import Any

from osipy.common.exceptions import DataValidationError
from osipy.ivim.models.biexponential import IVIMModel

logger = logging.getLogger(__name__)

# IVIM model registry for dynamic lookup
IVIM_MODEL_REGISTRY: dict[str, type[IVIMModel]] = {}


def register_ivim_model(name: str):
    """Decorator to register an IVIM signal model.

    Registers an ``IVIMModel`` subclass in ``IVIM_MODEL_REGISTRY`` so it
    can be looked up by name via ``get_ivim_model()``.

    Parameters
    ----------
    name : str
        Registry key for the model (e.g. ``'triexponential'``).

    Returns
    -------
    Callable
        Class decorator.

    Examples
    --------
    >>> from osipy.ivim.models import register_ivim_model, IVIMModel
    >>> @register_ivim_model("triexponential")
    ... class TriexponentialModel(IVIMModel):
    ...     # implement abstract methods ...
    ...     pass
    """

    def decorator(cls: type[IVIMModel]) -> type[IVIMModel]:
        if name in IVIM_MODEL_REGISTRY:
            logger.warning(
                "Overwriting existing IVIM model '%s' (%s) with %s",
                name,
                IVIM_MODEL_REGISTRY[name].__name__,
                cls.__name__,
            )
        IVIM_MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_ivim_model(name: str, **kwargs: Any) -> IVIMModel:
    """Get an IVIM model instance by name.

    Parameters
    ----------
    name : str
        Model name: ``'biexponential'``, ``'simplified'``, or any name
        added via ``register_ivim_model()``.
    **kwargs
        Additional keyword arguments passed to the model constructor.

    Returns
    -------
    IVIMModel
        Model instance.

    Raises
    ------
    DataValidationError
        If model name is not recognized.
    """
    if name not in IVIM_MODEL_REGISTRY:
        valid = ", ".join(sorted(IVIM_MODEL_REGISTRY.keys()))
        msg = f"Unknown IVIM model: {name}. Valid models: {valid}"
        raise DataValidationError(msg)
    return IVIM_MODEL_REGISTRY[name](**kwargs)
