"""ASL labeling scheme definitions and parameters.

This module defines the parameters for different ASL labeling schemes
per the OSIPI ASL Lexicon terminology:

- PASL (Pulsed ASL) -- labeling efficiency (alpha), inversion time (TI) in ms
- CASL (Continuous ASL) -- labeling duration (tau/LD) in ms, PLD in ms
- pCASL (pseudo-Continuous ASL) -- labeling duration (tau/LD) in ms, PLD in ms

References
----------
.. [1] OSIPI ASL Lexicon, https://osipi.github.io/ASL-Lexicon/
.. [2] Alsop DC et al. (2015). Recommended implementation of arterial
   spin-labeled perfusion MRI for clinical applications.
   Magn Reson Med 73(1):102-116.
.. [3] Dai W et al. (2008). Continuous flow-driven inversion for arterial
   spin labeling using pulsed radio frequency and gradient fields.
   Magn Reson Med 60(6):1488-1497.
"""

from dataclasses import dataclass
from enum import Enum


class LabelingScheme(Enum):
    """Enumeration of ASL labeling schemes."""

    PASL = "pasl"
    CASL = "casl"
    PCASL = "pcasl"


@dataclass
class PASLParams:
    """Parameters for Pulsed ASL (PASL).

    Attributes
    ----------
    ti : float
        Inversion time (TI) in milliseconds.
        Time from labeling pulse to image acquisition.
    ti1 : float | None
        First TI for QUIPSS II or Q2TIPS (ms).
        Time at which saturation pulse is applied.
    bolus_duration : float
        Bolus duration (τ) in milliseconds.
        Effective labeling duration.
    labeling_efficiency : float
        Labeling efficiency (α), typically 0.95-0.98 for PASL.
    inversion_flip_angle : float
        Flip angle of inversion pulse in degrees.
    """

    ti: float = 1800.0  # ms
    ti1: float | None = 700.0  # ms (for QUIPSS)
    bolus_duration: float = 700.0  # ms
    labeling_efficiency: float = 0.98
    inversion_flip_angle: float = 180.0


@dataclass
class CASLParams:
    """Parameters for Continuous ASL (CASL).

    Attributes
    ----------
    label_duration : float
        Duration of labeling RF pulse (τ) in milliseconds.
    pld : float
        Post-labeling delay (PLD) in milliseconds.
    labeling_efficiency : float
        Labeling efficiency (α), typically 0.68-0.73 for CASL.
    rf_power : float
        RF power during labeling in μT.
    gradient_amplitude : float
        Labeling gradient amplitude in mT/m.
    """

    label_duration: float = 1800.0  # ms
    pld: float = 1800.0  # ms
    labeling_efficiency: float = 0.71
    rf_power: float = 3.0  # μT
    gradient_amplitude: float = 10.0  # mT/m


@dataclass
class PCASLParams:
    """Parameters for pseudo-Continuous ASL (pCASL).

    pCASL is the recommended labeling scheme per ISMRM consensus.

    Attributes
    ----------
    label_duration : float
        Duration of labeling train (τ) in milliseconds.
    pld : float
        Post-labeling delay in milliseconds.
    labeling_efficiency : float
        Labeling efficiency (α), typically 0.85-0.95 for pCASL.
    average_b1 : float
        Average B1 during labeling in μT.
    average_gradient : float
        Average gradient during labeling in mT/m.
    pulse_duration : float
        Duration of each RF pulse in microseconds.
    pulse_gap : float
        Gap between RF pulses in microseconds.
    num_pulses : int | None
        Number of RF pulses in labeling train.
    """

    label_duration: float = 1800.0  # ms
    pld: float = 1800.0  # ms
    labeling_efficiency: float = 0.85
    average_b1: float = 1.5  # μT
    average_gradient: float = 0.5  # mT/m
    pulse_duration: float = 500.0  # μs
    pulse_gap: float = 500.0  # μs
    num_pulses: int | None = None


def compute_labeling_efficiency(
    scheme: LabelingScheme,
    params: PASLParams | CASLParams | PCASLParams | None = None,
    measured_efficiency: float | None = None,
) -> float:
    """Compute or return labeling efficiency.

    Parameters
    ----------
    scheme : LabelingScheme
        Labeling scheme type.
    params : PASLParams | CASLParams | PCASLParams | None
        Labeling parameters.
    measured_efficiency : float | None
        If provided, use this measured value instead of default.

    Returns
    -------
    float
        Labeling efficiency (0 to 1).

    Notes
    -----
    Default efficiencies per ISMRM consensus (Alsop et al., 2015):
    - PASL: 0.98
    - CASL: 0.71
    - pCASL: 0.85
    """
    if measured_efficiency is not None:
        return measured_efficiency

    if params is not None:
        return params.labeling_efficiency

    # Default values from consensus paper
    defaults = {
        LabelingScheme.PASL: 0.98,
        LabelingScheme.CASL: 0.71,
        LabelingScheme.PCASL: 0.85,
    }

    return defaults.get(scheme, 0.85)
