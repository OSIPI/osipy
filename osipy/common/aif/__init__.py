"""Arterial Input Function handling for osipy.

This module provides classes and functions for working with AIFs,
including population-based models and automatic detection.
"""

from osipy.common.aif.base import ArterialInputFunction, BaseAIF
from osipy.common.aif.base_detector import BaseAIFDetector
from osipy.common.aif.delay import shift_aif
from osipy.common.aif.detection import (
    AIFDetectionParams,
    AIFDetectionResult,
    detect_aif,
)
from osipy.common.aif.detection_registry import (
    get_aif_detector,
    list_aif_detectors,
    register_aif_detector,
)
from osipy.common.aif.hematocrit import DEFAULT_HEMATOCRIT, correct_hematocrit
from osipy.common.aif.population import (
    AIF_REGISTRY,
    FritzHansenAIF,
    FritzHansenAIFParams,
    GeorgiouAIF,
    GeorgiouAIFParams,
    McGrathAIF,
    McGrathAIFParams,
    ParkerAIF,
    ParkerAIFParams,
    PopulationAIFType,
    WeinmannAIF,
    WeinmannAIFParams,
    get_population_aif,
    list_aifs,
    register_aif,
)

__all__ = [
    "AIF_REGISTRY",
    "DEFAULT_HEMATOCRIT",
    "AIFDetectionParams",
    "AIFDetectionResult",
    "ArterialInputFunction",
    "BaseAIF",
    "BaseAIFDetector",
    "FritzHansenAIF",
    "FritzHansenAIFParams",
    "GeorgiouAIF",
    "GeorgiouAIFParams",
    "McGrathAIF",
    "McGrathAIFParams",
    "ParkerAIF",
    "ParkerAIFParams",
    "PopulationAIFType",
    "WeinmannAIF",
    "WeinmannAIFParams",
    "correct_hematocrit",
    "detect_aif",
    "get_aif_detector",
    "get_population_aif",
    "list_aif_detectors",
    "list_aifs",
    "register_aif",
    "register_aif_detector",
    "shift_aif",
]
