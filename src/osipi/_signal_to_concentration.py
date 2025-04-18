import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._electromagnetic_property import R1_to_C_linear_relaxivity


def S_to_C_via_R1_SPGR(
    S: ArrayLike,
    S_baseline: np.floating,
    R10: np.floating,
    TR: np.floating,
    a: np.floating,
    r1: np.floating,
) -> NDArray[np.floating]:
    """
    Signal to concentration via
    electromagnetic property (SPGR, FXL, analytical, linear R1 relaxivity)

    Converts S -> R1 -> C

    Args:
        S (ArrayLike): Vector of magnitude signals in a.u. [OSIPI code Q.MS1.001]
        S_baseline (np.floating): Pre-contrast magnitude signal in a.u. [OSIPI code Q.MS1.001]
        R10 (np.floating): Native longitudinal relaxation rate in units of /s. [OSIPI code Q.EL1.002]
        TR (np.floating): Repetition time in units of s. [OSIPI code Q.MS1.006]
        a (np.floating): Prescribed flip angle in units of deg. [OSIPI code Q.MS1.007]
        r1 (np.floating): Longitudinal relaxivity in units of /s/mM. [OSIPI code Q.EL1.015]

    Returns:
         NDArray[np.floating]:
            Vector of indicator total (across all compartments) indicator
            concentration in units of mM. [OSIPI code Q.IC1.001]

    References:
        - Lexicon URL: https://osipi.github.io/OSIPI_CAPLEX/perfusionProcesses/
        - Lexicon code: P.SC2.002
        - OSIPI name: ConvertSToCViaEP
        - Signal to electromagnetic property conversion method:
          - model-based [OSIPI code P.SE1.001]
            - Inversion method: analytical inversion [OSIPI code G.MI1.001]
            - Forward model: Spoiled gradient recalled echo model [OSIPI code M.SM2.002]
        - Electromagnetic property inverse model:
          - model-based [OSIPI code P.EC1.001]
            - Inversion method: analytical inversion [OSIPI code G.MI1.001]
            - Forward model:
                longitudinal relaxation rate, linear with relaxivity model [OSIPI code M.EL1.003]
    """
    R1 = S_to_R1_SPGR(S, S_baseline, R10, TR, a)  # S -> R1
    return R1_to_C_linear_relaxivity(R1, R10, r1)  # R1 -> C


def S_to_R1_SPGR(
    S: ArrayLike,
    S_baseline: np.floating,
    R10: np.floating,
    TR: np.floating,
    a: np.floating,
) -> NDArray[np.floating]:
    """
    Signal to electromagnetic property conversion (analytical, SPGR, FXL)

    Converts Signal to R1

    Args:
        S (ArrayLike): Vector of magnitude signals in a.u. [OSIPI code Q.MS1.001]
        S_baseline (np.floating): Pre-contrast magnitude signal in a.u. [OSIPI code Q.MS1.001]
        R10 (np.floating): Native longitudinal relaxation rate in units of /s. [OSIPI code Q.EL1.002]
        TR (np.floating): Repetition time in units of s. [OSIPI code Q.MS1.006]
        a (np.floating): Prescribed flip angle in units of deg. [OSIPI code Q.MS1.007]

    Returns:
        NDArray[np.floating]: Vector of R1 in units of /s. [OSIPI code Q.EL1.001]

    References:
        - Lexicon URL: https://osipi.github.io/OSIPI_CAPLEX/perfusionProcesses/#
        - Lexicon code: P.SE1.001
        - OSIPI name: model-based
          - Inversion method: analytical inversion [OSIPI code G.MI1.001]
          - Forward model: Spoiled gradient recalled echo model [OSIPI code M.SM2.002]
        - Adapted from contribution of LEK_UoEdinburgh_UK
    """
    # Convert input to numpy array with appropriate dtype
    S_arr = np.asarray(S)
    # Ensure floating-point dtype
    if not np.issubdtype(S_arr.dtype, np.floating):
        S_arr = S_arr.astype(np.float64)

    # Check S is a 1D array of floats
    if not (S_arr.ndim == 1):
        raise TypeError("S must be a 1D array-like object of floating point values")

    a_rad = a * np.pi / 180
    # Estimate fully T1-relaxed signal S0 in units of a.u. [OSIPI code Q.MS1.010], then R1
    exp_TR_R10 = np.exp(-TR * R10)
    sin_a = np.sin(a_rad)
    cos_a = np.cos(a_rad)
    S0 = S_baseline * (1 - cos_a * exp_TR_R10) / (sin_a * (1 - exp_TR_R10))
    return np.log(((S0 * sin_a) - S_arr) / (S0 * sin_a - (S_arr * cos_a))) * (-1 / TR)  # R1
