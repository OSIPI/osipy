"""Model fitting infrastructure for osipy.

This module provides the abstract base class for fitters and
concrete implementations for Levenberg-Marquardt (least squares)
and Bayesian (MAP) fitting.

Registry-driven extensibility: use ``register_fitter`` / ``get_fitter`` /
``list_fitters`` to register, lookup, and enumerate fitters by name.
"""

from osipy.common.fitting.base import BaseFitter
from osipy.common.fitting.batch import create_empty_maps, create_parameter_maps
from osipy.common.fitting.bayesian import BayesianFitter
from osipy.common.fitting.least_squares import LevenbergMarquardtFitter
from osipy.common.fitting.registry import (
    get_fitter,
    list_fitters,
    register_fitter,
    register_fitter_alias,
)
from osipy.common.fitting.result import FittingResult

__all__ = [
    "BaseFitter",
    "BayesianFitter",
    "FittingResult",
    "LevenbergMarquardtFitter",
    "create_empty_maps",
    "create_parameter_maps",
    "get_fitter",
    "list_fitters",
    "register_fitter",
    "register_fitter_alias",
]
