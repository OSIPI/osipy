"""Base classes for all osipy components and signal models.

This module provides ``BaseComponent``, the lightweight shared base for
ALL registered osipy components (models, fitters, correctors, detectors),
and ``BaseSignalModel``, the abstract base for parametric forward models.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class BaseComponent(ABC):
    """Shared interface for all registered osipy components.

    Every registered class in osipy — signal models, fitters, leakage
    correctors, arrival detectors, calibration methods, AIF detectors,
    and AIF models — inherits from this base to guarantee a minimal
    introspection interface.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return human-readable component name."""
        ...

    @property
    @abstractmethod
    def reference(self) -> str:
        """Return primary literature citation for this component.

        Returns
        -------
        str
            Citation string.
        """
        ...


class BaseSignalModel(BaseComponent):
    """Abstract base class for all parametric signal models.

    All modality-specific model base classes (BasePerfusionModel, IVIMModel,
    BaseASLModel) inherit from this class to ensure a consistent interface
    for registry-driven dispatch and fitting.
    """

    @property
    @abstractmethod
    def parameters(self) -> list[str]:
        """Return list of parameter names."""
        ...

    @property
    @abstractmethod
    def parameter_units(self) -> dict[str, str]:
        """Return mapping of parameter names to their units.

        Returns
        -------
        dict[str, str]
            Parameter units (e.g., {'Ktrans': '1/min', 've': 'fraction'}).
        """
        ...

    @abstractmethod
    def get_bounds(self) -> dict[str, tuple[float, float]]:
        """Return parameter bounds as {name: (lower, upper)}."""
        ...

    def params_to_array(self, params: Any) -> "NDArray[np.floating[Any]]":
        """Convert parameter dataclass or dict to array.

        Parameters
        ----------
        params : Any
            Parameter dataclass or dict.

        Returns
        -------
        NDArray[np.floating]
            Parameter values in standard order.
        """
        if isinstance(params, dict):
            return np.array([params[p] for p in self.parameters])
        values = []
        for name in self.parameters:
            if hasattr(params, name):
                values.append(getattr(params, name))
            elif hasattr(params, name.lower()):
                values.append(getattr(params, name.lower()))
            else:
                msg = f"Parameter '{name}' not found in {type(params).__name__}"
                raise AttributeError(msg)
        return np.array(values)

    def array_to_params(self, values: "NDArray[np.floating[Any]]") -> dict[str, float]:
        """Convert array to parameter dictionary.

        Parameters
        ----------
        values : NDArray[np.floating]
            Parameter values in standard order.

        Returns
        -------
        dict[str, float]
            Parameter dictionary.
        """
        return dict(zip(self.parameters, values, strict=True))

    def bounds_to_arrays(
        self,
    ) -> tuple["NDArray[np.floating[Any]]", "NDArray[np.floating[Any]]"]:
        """Convert bounds dict to lower/upper arrays.

        Returns
        -------
        tuple[NDArray, NDArray]
            (lower_bounds, upper_bounds) arrays.
        """
        bounds = self.get_bounds()
        lower = np.array([bounds[p][0] for p in self.parameters])
        upper = np.array([bounds[p][1] for p in self.parameters])
        return lower, upper
