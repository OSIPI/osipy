"""Core type definitions for osipy.

This module defines enumerations and base dataclasses used across all
perfusion modality modules. Acquisition parameter names follow MRI
conventions and OSIPI lexicon definitions where applicable.

References
----------
.. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
.. [2] OSIPI ASL Lexicon, https://osipi.github.io/ASL-Lexicon/
.. [3] Dickie BR et al. MRM 2024. doi:10.1002/mrm.30101
.. [4] Suzuki Y et al. MRM 2024;91(4):1411-1421.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Modality(Enum):
    """Perfusion imaging modality.

    Attributes
    ----------
    DCE : str
        Dynamic Contrast-Enhanced MRI.
    DSC : str
        Dynamic Susceptibility Contrast MRI.
    ASL : str
        Arterial Spin Labeling.
    IVIM : str
        Intravoxel Incoherent Motion.
    """

    DCE = "dce"
    DSC = "dsc"
    ASL = "asl"
    IVIM = "ivim"


class LabelingType(Enum):
    """ASL labeling scheme type.

    Attributes
    ----------
    PCASL : str
        Pseudo-continuous Arterial Spin Labeling.
    PASL_FAIR : str
        Pulsed ASL with Flow-sensitive Alternating Inversion Recovery.
    PASL_EPISTAR : str
        Pulsed ASL with Echo Planar Imaging and Signal Targeting with
        Alternating Radiofrequency.
    PASL_PICORE : str
        Pulsed ASL with Proximal Inversion with Control for Off-Resonance Effects.
    VSASL : str
        Velocity-Selective Arterial Spin Labeling.

    References
    ----------
    Alsop DC et al. (2015). Recommended implementation of ASL. MRM.
    """

    PCASL = "pcasl"
    PASL_FAIR = "pasl_fair"
    PASL_EPISTAR = "pasl_epistar"
    PASL_PICORE = "pasl_picore"
    VSASL = "vsasl"


class FittingMethod(Enum):
    """Model fitting algorithm.

    Attributes
    ----------
    LM : str
        Levenberg-Marquardt non-linear least squares optimization.
    BAYESIAN : str
        Bayesian inference with uncertainty estimation.
    NEURAL_NETWORK : str
        Neural network-based fitting (future).
    """

    LM = "lm"
    BAYESIAN = "bayesian"
    NEURAL_NETWORK = "neural_network"


class AIFType(Enum):
    """Arterial Input Function type.

    Attributes
    ----------
    MEASURED : str
        AIF extracted from image data.
    POPULATION : str
        Published population-based AIF model.
    AUTOMATIC : str
        Automatically detected from image data.
    """

    MEASURED = "measured"
    POPULATION = "population"
    AUTOMATIC = "automatic"


@dataclass
class AcquisitionParams:
    """Base acquisition parameters common to all modalities.

    Attributes
    ----------
    tr : float | None
        Repetition time in milliseconds.
    te : float | None
        Echo time in milliseconds.
    flip_angle : float | None
        Flip angle in degrees.
    field_strength : float | None
        Magnetic field strength in Tesla.
    """

    tr: float | None = None
    te: float | None = None
    flip_angle: float | None = None
    field_strength: float | None = None


@dataclass
class DCEAcquisitionParams(AcquisitionParams):
    """DCE-MRI specific acquisition parameters.

    Attributes
    ----------
    flip_angles : list[float]
        Flip angles for variable flip angle T1 mapping in degrees.
    baseline_frames : int
        Number of pre-contrast frames for baseline calculation.
    temporal_resolution : float
        Time between dynamic acquisitions in seconds.
    relaxivity : float
        Contrast agent relaxivity in mM⁻¹s⁻¹. Default is 4.5 for
        gadolinium-based agents at 3T.
    t1_assumed : float | None
        Assumed pre-contrast T1 value in milliseconds when T1 mapping
        data is unavailable. If None, a measured T1 map is required for
        signal-to-concentration conversion. Typical values at 3T:
        - Breast tissue: ~1400 ms
        - Brain white matter: ~800 ms
        - Brain gray matter: ~1200 ms
        - Blood: ~1600 ms

    References
    ----------
    Tofts PS (1997). Modeling tracer kinetics in DCE-MRI. JMRI.
    """

    flip_angles: list[float] = field(default_factory=list)
    baseline_frames: int = 5
    temporal_resolution: float = 1.0
    relaxivity: float = 4.5
    t1_assumed: float | None = None


@dataclass
class DSCAcquisitionParams(AcquisitionParams):
    """DSC-MRI specific acquisition parameters.

    Attributes
    ----------
    baseline_frames : int
        Number of pre-bolus frames for baseline calculation.

    Notes
    -----
    The `te` parameter from the base class is required for ΔR2* calculation.

    References
    ----------
    Ostergaard L et al. (1996). High resolution CBF measurement. MRM.
    """

    baseline_frames: int = 10


@dataclass
class ASLAcquisitionParams(AcquisitionParams):
    """ASL specific acquisition parameters.

    Attributes
    ----------
    labeling_type : LabelingType
        ASL labeling scheme.
    pld : float | list[float]
        Post-labeling delay(s) in milliseconds.
    labeling_duration : float
        Labeling duration in milliseconds.
    background_suppression : bool
        Whether background suppression was applied.
    bs_efficiency : float
        Background suppression efficiency factor (0 to 1).
    m0_scale : float | None
        M0 calibration scaling value.

    References
    ----------
    Alsop DC et al. (2015). Recommended implementation of ASL. MRM.
    """

    labeling_type: LabelingType = LabelingType.PCASL
    pld: float | list[float] = 1800.0
    labeling_duration: float = 1800.0
    background_suppression: bool = False
    bs_efficiency: float = 1.0
    m0_scale: float | None = None


def _get_default_ivim_b_values() -> list[int]:
    """Get default IVIM b-values from the canonical defaults module.

    Uses a lazy import to avoid circular dependency (defaults.py imports Modality).
    """
    from osipy.common.io.metadata.defaults import DEFAULT_IVIM_PARAMS

    return list(DEFAULT_IVIM_PARAMS["b_values"])


@dataclass
class IVIMAcquisitionParams(AcquisitionParams):
    """IVIM specific acquisition parameters.

    Attributes
    ----------
    b_values : NDArray[np.floating]
        Array of b-values in s/mm².

    References
    ----------
    Le Bihan D et al. (1988). IVIM MR imaging. Radiology.
    """

    b_values: "NDArray[np.floating[Any]]" = field(
        default_factory=lambda: np.array(_get_default_ivim_b_values())
    )


# Type alias for any acquisition params
AnyAcquisitionParams = (
    AcquisitionParams
    | DCEAcquisitionParams
    | DSCAcquisitionParams
    | ASLAcquisitionParams
    | IVIMAcquisitionParams
)
