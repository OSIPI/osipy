"""Fitter registry for osipy.

Standard registry for BaseFitter subclasses. Follows the same
pattern as all other osipy registries (DSC deconvolution, DCE models, etc.).
"""

import logging

from osipy.common.exceptions import DataValidationError

logger = logging.getLogger(__name__)

FITTER_REGISTRY: dict[str, type] = {}


def register_fitter(name: str):
    """Register a fitter class.

    Parameters
    ----------
    name : str
        Registry key for the fitter (e.g., 'lm').
    """

    def decorator(cls):
        if name in FITTER_REGISTRY:
            logger.warning(
                "Overwriting '%s' (%s) with %s",
                name,
                FITTER_REGISTRY[name].__name__,
                cls.__name__,
            )
        FITTER_REGISTRY[name] = cls
        return cls

    return decorator


def get_fitter(name: str):
    """Get a fitter instance by name.

    Parameters
    ----------
    name : str
        Registry key for the fitter.

    Returns
    -------
    BaseFitter
        A new fitter instance.

    Raises
    ------
    DataValidationError
        If the fitter name is not recognized.
    """
    if name not in FITTER_REGISTRY:
        valid = ", ".join(sorted(FITTER_REGISTRY.keys()))
        raise DataValidationError(f"Unknown fitter: {name}. Valid: {valid}")
    return FITTER_REGISTRY[name]()


def list_fitters() -> list[str]:
    """List registered canonical fitter names.

    Returns
    -------
    list[str]
        Sorted list of registered fitter names (excludes aliases).
    """
    return sorted(FITTER_REGISTRY.keys())
