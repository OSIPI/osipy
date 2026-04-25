"""Population-based arterial input functions.

This module implements standard population AIFs for DCE-MRI analysis,
providing consistent reference functions when individual AIF measurement
is not possible.

OSIPI CAPLEX model codes:
- Parker AIF: M.IC2.001
- Georgiou AIF: M.IC2.002

GPU/CPU agnostic using the xp array module pattern.

References
----------
.. [1] Parker GJM et al. (2006). Experimentally-derived functional form for a
   population-averaged high-temporal-resolution arterial input function for
   dynamic contrast-enhanced MRI. Magn Reson Med 56(5):993-1000.
.. [2] Georgiou L et al. (2014). A functional form for a representative
   individual arterial input function. Magn Reson Med 73(3):1241-1249.
.. [3] Fritz-Hansen T et al. (1996). Capillary transfer constant of Gd-DTPA
   in the myocardium at rest and during vasodilation assessed by MRI.
   Magn Reson Med 35(2):139-144.
.. [4] Weinmann HJ et al. (1984). Pharmacokinetics of GdDTPA/dimeglumine after
   intravenous injection into healthy volunteers. Physiological Chemistry
   and Physics and Medical NMR 16(2):167-172.
.. [5] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
.. [6] Dickie BR et al. MRM 2024. doi:10.1002/mrm.29840
"""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from osipy.common.aif.base import ArterialInputFunction, BaseAIF
from osipy.common.backend.array_module import get_array_module
from osipy.common.exceptions import AIFError
from osipy.common.types import AIFType

if TYPE_CHECKING:
    from numpy.typing import NDArray


class PopulationAIFType(Enum):
    """Enumeration of available population AIF models."""

    PARKER = "parker"
    GEORGIOU = "georgiou"
    FRITZ_HANSEN = "fritz_hansen"
    WEINMANN = "weinmann"
    MCGRATH = "mcgrath"


@dataclass
class ParkerAIFParams:
    """Parameters for the Parker population AIF.

    The Parker AIF is described by:
        Cb(t) = A1/(σ1√(2π)) * exp(-(t-T1)²/(2σ1²))
              + A2/(σ2√(2π)) * exp(-(t-T2)²/(2σ2²))
              + α * exp(-β*t) / (1 + exp(-s(t-τ)))

    Default values are from Parker et al. (2006).
    """

    # First Gaussian (bolus peak)
    a1: float = 0.809  # mmol·min
    t1: float = 0.17046  # min
    sigma1: float = 0.0563  # min

    # Second Gaussian (recirculation)
    a2: float = 0.330  # mmol·min
    t2: float = 0.365  # min
    sigma2: float = 0.132  # min

    # Exponential tail (clearance)
    alpha: float = 1.050  # mmol
    beta: float = 0.1685  # min⁻¹
    s: float = 38.078  # min⁻¹
    tau: float = 0.483  # min


class ParkerAIF(BaseAIF):
    """Parker population AIF (OSIPI M.IC2.001).

    The Parker AIF is a widely-used population-averaged arterial input
    function derived from measurements in a cohort of cancer patients.
    It consists of two Gaussian components (bolus arrival and recirculation)
    and a modified exponential tail.

    **Derivation context (Parker 2006, PMID 17036301):** the canonical
    parameters were fit from 23 abdominal cancer patients at 1.5T
    receiving gadodiamide, across 113 visits. Use with caution outside
    that regime — 3T, brain / head-and-neck studies, or different
    contrast agents (e.g. gadobutrol, protein-binding agents) may
    require a different population AIF (consider ``GeorgiouAIF``,
    OSIPI M.IC2.002) or a subject-specific measured AIF.

    GPU/CPU agnostic - operates on same device as input time array.

    References
    ----------
    .. [1] Parker GJM et al. (2006). Magn Reson Med 56(5):993-1000.
           doi:10.1002/mrm.21066, PMID 17036301.
    .. [2] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
    """

    def __init__(self, params: ParkerAIFParams | None = None) -> None:
        """Initialize Parker AIF.

        Parameters
        ----------
        params : ParkerAIFParams, optional
            Model parameters. Uses published defaults if not provided.
        """
        self.params = params or ParkerAIFParams()

    @property
    def name(self) -> str:
        """Return AIF model name."""
        return "Parker Population AIF"

    @property
    def reference(self) -> str:
        """Return literature citation."""
        return "Parker GJM et al. (2006). Magn Reson Med 56(5):993-1000."

    def get_parameters(self) -> dict[str, float]:
        """Return current AIF model parameters.

        Returns
        -------
        dict[str, float]
            Parameter names and values.
        """
        p = self.params
        return {
            "a1": p.a1,
            "t1": p.t1,
            "sigma1": p.sigma1,
            "a2": p.a2,
            "t2": p.t2,
            "sigma2": p.sigma2,
            "alpha": p.alpha,
            "beta": p.beta,
            "s": p.s,
            "tau": p.tau,
        }

    def get_concentration(
        self, t: "NDArray[np.floating[Any]]"
    ) -> "NDArray[np.floating[Any]]":
        """Get AIF concentration values at specified time points.

        Parameters
        ----------
        t : NDArray[np.floating]
            Time points in seconds.

        Returns
        -------
        NDArray[np.floating]
            AIF concentration values (mM).
        """
        return self(t).concentration

    def __call__(self, t: "NDArray[np.floating[Any]]") -> ArterialInputFunction:
        """Generate AIF at specified time points.

        GPU/CPU agnostic - operates on same device as input.

        Parameters
        ----------
        t : NDArray[np.floating]
            Time points in seconds.

        Returns
        -------
        ArterialInputFunction
            Generated AIF with concentration values.
        """
        xp = get_array_module(t)

        # Convert time to minutes (Parker AIF uses minutes)
        t_min = t / 60.0

        p = self.params

        # First Gaussian (bolus peak)
        gaussian1 = (
            p.a1
            / (p.sigma1 * xp.sqrt(2 * np.pi))
            * xp.exp(-((t_min - p.t1) ** 2) / (2 * p.sigma1**2))
        )

        # Second Gaussian (recirculation)
        gaussian2 = (
            p.a2
            / (p.sigma2 * xp.sqrt(2 * np.pi))
            * xp.exp(-((t_min - p.t2) ** 2) / (2 * p.sigma2**2))
        )

        # Modified exponential (sigmoid modulated clearance)
        sigmoid = 1.0 / (1.0 + xp.exp(-p.s * (t_min - p.tau)))
        exponential = p.alpha * xp.exp(-p.beta * t_min) * sigmoid

        # Total blood plasma concentration
        cb = gaussian1 + gaussian2 + exponential

        return ArterialInputFunction(
            time=t,
            concentration=cb,
            aif_type=AIFType.POPULATION,
            population_model="Parker",
        )


@dataclass
class GeorgiouAIFParams:
    """Parameters for the Georgiou population AIF.

    The Georgiou AIF extends the Parker model with improved individual
    variability handling.

    Default values are from Georgiou et al. (2014).
    """

    # First Gaussian
    a1: float = 0.37  # mmol·min
    m1: float = 0.11  # min
    sigma1: float = 0.055  # min

    # Second Gaussian
    a2: float = 0.33  # mmol·min
    m2: float = 0.28  # min
    sigma2: float = 0.095  # min

    # Exponential decay
    alpha: float = 5.0  # mmol
    beta: float = 0.05  # min⁻¹


class GeorgiouAIF(BaseAIF):
    """Georgiou population AIF (OSIPI M.IC2.002).

    The Georgiou AIF provides a simplified functional form for
    individual AIF representation, useful for population-based
    corrections.

    GPU/CPU agnostic - operates on same device as input time array.

    References
    ----------
    .. [1] Georgiou L et al. (2014). Magn Reson Med 73(3):1241-1249.
    .. [2] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
    """

    def __init__(self, params: GeorgiouAIFParams | None = None) -> None:
        """Initialize Georgiou AIF.

        Parameters
        ----------
        params : GeorgiouAIFParams, optional
            Model parameters. Uses published defaults if not provided.
        """
        self.params = params or GeorgiouAIFParams()

    @property
    def name(self) -> str:
        """Return AIF model name."""
        return "Georgiou Population AIF"

    @property
    def reference(self) -> str:
        """Return literature citation."""
        return "Georgiou L et al. (2014). Magn Reson Med 73(3):1241-1249."

    def get_parameters(self) -> dict[str, float]:
        """Return current AIF model parameters."""
        p = self.params
        return {
            "a1": p.a1,
            "m1": p.m1,
            "sigma1": p.sigma1,
            "a2": p.a2,
            "m2": p.m2,
            "sigma2": p.sigma2,
            "alpha": p.alpha,
            "beta": p.beta,
        }

    def get_concentration(
        self, t: "NDArray[np.floating[Any]]"
    ) -> "NDArray[np.floating[Any]]":
        """Get AIF concentration values at specified time points.

        Parameters
        ----------
        t : NDArray[np.floating]
            Time points in seconds.

        Returns
        -------
        NDArray[np.floating]
            AIF concentration values (mM).
        """
        return self(t).concentration

    def __call__(self, t: "NDArray[np.floating[Any]]") -> ArterialInputFunction:
        """Generate AIF at specified time points.

        GPU/CPU agnostic - operates on same device as input.

        Parameters
        ----------
        t : NDArray[np.floating]
            Time points in seconds.

        Returns
        -------
        ArterialInputFunction
            Generated AIF with concentration values.
        """
        xp = get_array_module(t)

        # Convert to minutes
        t_min = t / 60.0

        p = self.params

        # First Gaussian
        gaussian1 = (
            p.a1
            / (p.sigma1 * xp.sqrt(2 * np.pi))
            * xp.exp(-((t_min - p.m1) ** 2) / (2 * p.sigma1**2))
        )

        # Second Gaussian
        gaussian2 = (
            p.a2
            / (p.sigma2 * xp.sqrt(2 * np.pi))
            * xp.exp(-((t_min - p.m2) ** 2) / (2 * p.sigma2**2))
        )

        # Exponential decay (only for t > 0)
        exponential = xp.where(t_min > 0, p.alpha * xp.exp(-p.beta * t_min), 0.0)

        cb = gaussian1 + gaussian2 + exponential

        return ArterialInputFunction(
            time=t,
            concentration=cb,
            aif_type=AIFType.POPULATION,
            population_model="Georgiou",
        )


@dataclass
class FritzHansenAIFParams:
    """Parameters for the Fritz-Hansen population AIF.

    The Fritz-Hansen AIF uses a bi-exponential model commonly used
    for cardiac perfusion studies.

    Default values are from Fritz-Hansen et al. (1996).
    """

    # First exponential (fast component)
    a1: float = 3.99  # mmol
    m1: float = 0.144  # min⁻¹

    # Second exponential (slow component)
    a2: float = 4.78  # mmol
    m2: float = 0.0111  # min⁻¹


class FritzHansenAIF(BaseAIF):
    """Fritz-Hansen population AIF.

    The Fritz-Hansen AIF is a bi-exponential model originally developed
    for cardiac MRI studies. It provides a simpler representation
    suitable for myocardial perfusion analysis.

    GPU/CPU agnostic - operates on same device as input time array.

    References
    ----------
    Fritz-Hansen T et al. (1996). Magn Reson Med 35(2):139-144.
    """

    def __init__(self, params: FritzHansenAIFParams | None = None) -> None:
        """Initialize Fritz-Hansen AIF.

        Parameters
        ----------
        params : FritzHansenAIFParams, optional
            Model parameters. Uses published defaults if not provided.
        """
        self.params = params or FritzHansenAIFParams()

    @property
    def name(self) -> str:
        """Return AIF model name."""
        return "Fritz-Hansen Population AIF"

    @property
    def reference(self) -> str:
        """Return literature citation."""
        return "Fritz-Hansen T et al. (1996). Magn Reson Med 35(2):139-144."

    def get_parameters(self) -> dict[str, float]:
        """Return current AIF model parameters."""
        p = self.params
        return {
            "a1": p.a1,
            "m1": p.m1,
            "a2": p.a2,
            "m2": p.m2,
        }

    def get_concentration(
        self, t: "NDArray[np.floating[Any]]"
    ) -> "NDArray[np.floating[Any]]":
        """Get AIF concentration values at specified time points.

        Parameters
        ----------
        t : NDArray[np.floating]
            Time points in seconds.

        Returns
        -------
        NDArray[np.floating]
            AIF concentration values (mM).
        """
        return self(t).concentration

    def __call__(self, t: "NDArray[np.floating[Any]]") -> ArterialInputFunction:
        """Generate AIF at specified time points.

        GPU/CPU agnostic - operates on same device as input.

        Parameters
        ----------
        t : NDArray[np.floating]
            Time points in seconds.

        Returns
        -------
        ArterialInputFunction
            Generated AIF with concentration values.
        """
        xp = get_array_module(t)

        # Convert to minutes
        t_min = t / 60.0

        p = self.params

        # Bi-exponential decay
        cb = p.a1 * xp.exp(-p.m1 * t_min) + p.a2 * xp.exp(-p.m2 * t_min)

        # Ensure non-negative values
        cb = xp.maximum(cb, 0.0)

        return ArterialInputFunction(
            time=t,
            concentration=cb,
            aif_type=AIFType.POPULATION,
            population_model="Fritz-Hansen",
        )


@dataclass
class WeinmannAIFParams:
    """Parameters for the Weinmann population AIF.

    The Weinmann AIF uses a bi-exponential model for Gd-DTPA clearance.

    Default values are from Weinmann et al. (1984).
    """

    # First exponential (fast component - distribution phase)
    a1: float = 3.99  # mmol·kg/L
    m1: float = 0.144  # min⁻¹

    # Second exponential (slow component - elimination phase)
    a2: float = 4.78  # mmol·kg/L
    m2: float = 0.0111  # min⁻¹

    # Dose factor (mmol/kg)
    dose: float = 0.1  # typical clinical dose


class WeinmannAIF(BaseAIF):
    """Weinmann population AIF.

    The Weinmann AIF is a bi-exponential model describing the
    pharmacokinetics of Gd-DTPA after intravenous injection.
    It is one of the earliest population AIFs and is still widely
    used as a reference.

    GPU/CPU agnostic - operates on same device as input time array.

    References
    ----------
    Weinmann HJ et al. (1984). Physiological Chemistry and Physics
    and Medical NMR 16(2):167-172.
    """

    def __init__(self, params: WeinmannAIFParams | None = None) -> None:
        """Initialize Weinmann AIF.

        Parameters
        ----------
        params : WeinmannAIFParams, optional
            Model parameters. Uses published defaults if not provided.
        """
        self.params = params or WeinmannAIFParams()

    @property
    def name(self) -> str:
        """Return AIF model name."""
        return "Weinmann Population AIF"

    @property
    def reference(self) -> str:
        """Return literature citation."""
        return "Weinmann HJ et al. (1984). Physiol Chem Phys Med NMR 16(2):167-172."

    def get_parameters(self) -> dict[str, float]:
        """Return current AIF model parameters."""
        p = self.params
        return {
            "a1": p.a1,
            "m1": p.m1,
            "a2": p.a2,
            "m2": p.m2,
            "dose": p.dose,
        }

    def get_concentration(
        self, t: "NDArray[np.floating[Any]]"
    ) -> "NDArray[np.floating[Any]]":
        """Get AIF concentration values at specified time points.

        Parameters
        ----------
        t : NDArray[np.floating]
            Time points in seconds.

        Returns
        -------
        NDArray[np.floating]
            AIF concentration values (mM).
        """
        return self(t).concentration

    def __call__(self, t: "NDArray[np.floating[Any]]") -> ArterialInputFunction:
        """Generate AIF at specified time points.

        GPU/CPU agnostic - operates on same device as input.

        Parameters
        ----------
        t : NDArray[np.floating]
            Time points in seconds.

        Returns
        -------
        ArterialInputFunction
            Generated AIF with concentration values.
        """
        xp = get_array_module(t)

        # Convert to minutes
        t_min = t / 60.0

        p = self.params

        # Bi-exponential decay scaled by dose
        cb = p.dose * (p.a1 * xp.exp(-p.m1 * t_min) + p.a2 * xp.exp(-p.m2 * t_min))

        # Ensure non-negative values
        cb = xp.maximum(cb, 0.0)

        return ArterialInputFunction(
            time=t,
            concentration=cb,
            aif_type=AIFType.POPULATION,
            population_model="Weinmann",
        )


@dataclass
class McGrathAIFParams:
    """Parameters for the McGrath preclinical population AIF.

    The McGrath AIF is modeled as a gamma-variate plus exponential washout:
        Cp(t) = A * (t/tau)^alpha * exp(-(t - tau)/tau) * H(t)
              + B * exp(-beta * t) * H(t)

    Default values are from McGrath DM et al. (2009), Table 1, Model B.
    Designed for small animal (rodent) DCE-MRI studies.
    """

    # Gamma-variate component
    a_gv: float = 4.0  # amplitude (mM)
    alpha: float = 2.0  # shape parameter (dimensionless)
    tau: float = 0.15  # peak time (min)

    # Exponential washout component
    a_exp: float = 0.5  # exponential amplitude (mM)
    beta: float = 0.1  # washout rate (min^-1)


class McGrathAIF(BaseAIF):
    """McGrath preclinical population AIF.

    A population AIF for small animal (rodent) DCE-MRI studies,
    modeled as a gamma-variate plus exponential washout.

    GPU/CPU agnostic - operates on same device as input time array.

    References
    ----------
    .. [1] McGrath DM et al. (2009). Magn Reson Med 61(5):1173-1184.
    """

    def __init__(self, params: McGrathAIFParams | None = None) -> None:
        """Initialize McGrath AIF.

        Parameters
        ----------
        params : McGrathAIFParams, optional
            Model parameters. Uses published defaults if not provided.
        """
        self.params = params or McGrathAIFParams()

    @property
    def name(self) -> str:
        """Return AIF model name."""
        return "McGrath Preclinical AIF"

    @property
    def reference(self) -> str:
        """Return literature citation."""
        return "McGrath DM et al. (2009). Magn Reson Med 61(5):1173-1184."

    def get_parameters(self) -> dict[str, float]:
        """Return current AIF model parameters.

        Returns
        -------
        dict[str, float]
            Parameter names and values.
        """
        p = self.params
        return {
            "a_gv": p.a_gv,
            "alpha": p.alpha,
            "tau": p.tau,
            "a_exp": p.a_exp,
            "beta": p.beta,
        }

    def get_concentration(
        self, t: "NDArray[np.floating[Any]]"
    ) -> "NDArray[np.floating[Any]]":
        """Get AIF concentration values at specified time points.

        Parameters
        ----------
        t : NDArray[np.floating]
            Time points in seconds.

        Returns
        -------
        NDArray[np.floating]
            AIF concentration values (mM).
        """
        return self(t).concentration

    def __call__(self, t: "NDArray[np.floating[Any]]") -> ArterialInputFunction:
        """Generate AIF at specified time points.

        GPU/CPU agnostic - operates on same device as input.

        Parameters
        ----------
        t : NDArray[np.floating]
            Time points in seconds.

        Returns
        -------
        ArterialInputFunction
            Generated AIF with concentration values.
        """
        xp = get_array_module(t)

        # Convert to minutes (McGrath AIF uses minutes internally)
        t_min = t / 60.0

        p = self.params

        # Gamma-variate: A * (t/tau)^alpha * exp(-(t - tau)/tau) for t > 0
        # Avoid division by zero and negative values
        safe_t = xp.maximum(t_min, 0.0)

        gamma_variate = xp.where(
            t_min > 0,
            p.a_gv * (safe_t / p.tau) ** p.alpha * xp.exp(-(safe_t - p.tau) / p.tau),
            0.0,
        )

        # Exponential washout
        exponential = xp.where(
            t_min > 0,
            p.a_exp * xp.exp(-p.beta * safe_t),
            0.0,
        )

        cb = gamma_variate + exponential
        cb = xp.maximum(cb, 0.0)

        return ArterialInputFunction(
            time=t,
            concentration=cb,
            aif_type=AIFType.POPULATION,
            population_model="McGrath",
        )


import logging

_logger = logging.getLogger(__name__)

# Module-level AIF registry for dynamic lookup and extension
AIF_REGISTRY: dict[str, type[BaseAIF]] = {
    "parker": ParkerAIF,
    "georgiou": GeorgiouAIF,
    "fritz_hansen": FritzHansenAIF,
    "fritz-hansen": FritzHansenAIF,
    "weinmann": WeinmannAIF,
    "mcgrath": McGrathAIF,
}


def register_aif(name: str):
    """Decorator to register a population AIF model.

    Registers a ``BaseAIF`` subclass in ``AIF_REGISTRY`` so it can be
    looked up by name via ``get_population_aif()``.

    Parameters
    ----------
    name : str
        Registry key for the AIF (e.g. ``'my_aif'``).

    Returns
    -------
    Callable
        Class decorator.

    Examples
    --------
    >>> from osipy.common.aif.population import register_aif
    >>> from osipy.common.aif.base import BaseAIF
    >>> @register_aif("custom")
    ... class CustomAIF(BaseAIF):
    ...     # implement abstract methods ...
    ...     pass
    """

    def decorator(cls: type[BaseAIF]) -> type[BaseAIF]:
        if name in AIF_REGISTRY:
            _logger.warning(
                "Overwriting existing AIF '%s' (%s) with %s",
                name,
                AIF_REGISTRY[name].__name__,
                cls.__name__,
            )
        AIF_REGISTRY[name] = cls
        return cls

    return decorator


# Factory function for population AIFs
def get_population_aif(aif_type: str | PopulationAIFType) -> BaseAIF:
    """Get a population AIF model instance.

    Parameters
    ----------
    aif_type : str or PopulationAIFType
        Type of population AIF: 'parker', 'georgiou', 'fritz_hansen', 'weinmann',
        or any name added via ``register_aif()``.

    Returns
    -------
    BaseAIF
        AIF model instance.

    Raises
    ------
    AIFError
        If AIF type is not recognized.
    """
    if isinstance(aif_type, PopulationAIFType):
        aif_type = aif_type.value

    if isinstance(aif_type, str):
        aif_type = aif_type.lower()

    if aif_type not in AIF_REGISTRY:
        valid = sorted({k for k in AIF_REGISTRY if "-" not in k})
        msg = f"Unknown AIF type: {aif_type}. Valid types: {valid}"
        raise AIFError(msg)

    return AIF_REGISTRY[aif_type]()


def list_aifs() -> list[str]:
    """Return names of all registered AIF models.

    Returns only canonical names, filtering out aliases (e.g. hyphenated
    variants like ``'fritz-hansen'``).

    Returns
    -------
    list[str]
        Sorted list of registered AIF names.
    """
    return sorted(k for k in AIF_REGISTRY if "-" not in k)


def parker_aif_curve(
    t: "NDArray[np.floating[Any]]",
    params: ParkerAIFParams | None = None,
) -> "NDArray[np.floating[Any]]":
    """Compute Parker AIF concentration values directly.

    Convenience function that returns just the concentration array
    without wrapping in an ArterialInputFunction object.

    GPU/CPU agnostic - operates on same device as input.

    Parameters
    ----------
    t : NDArray[np.floating]
        Time points in seconds.
    params : ParkerAIFParams, optional
        Model parameters. Uses published defaults if not provided.

    Returns
    -------
    NDArray[np.floating]
        AIF concentration values (mM).
    """
    aif = ParkerAIF(params)
    return aif.get_concentration(t)
