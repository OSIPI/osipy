import numpy as np
from numpy.typing import ArrayLike, NDArray


def signal_linear(R1: ArrayLike, k: np.floating) -> NDArray[np.floating]:
    """Linear model for relationship between R1 and magnitude signal
    Args:
        R1 (ArrayLike): longitudinal relaxation rate in units of /s. [OSIPI code Q.EL1.001]
        k (np.floating): proportionality constant in a.u. S [OSIPI code Q.GE1.009]
    Returns:
        NDArray[np.floating]: magnitude signal in a.u. [OSIPI code Q.MS1.001]
    References:
        - Lexicon url: https://osipi.github.io/OSIPI_CAPLEX/perfusionModels/#LinModel_SM2
        - Lexicon code: M.SM2.001
        - OSIPI name: Linear model
        - Adapted from equation given in the Lexicon
    """
    # Convert input to numpy array with appropriate dtype
    R1_arr = np.asarray(R1)
    # Ensure floating-point dtype
    if not np.issubdtype(R1_arr.dtype, np.floating):
        R1_arr = R1_arr.astype(np.float64)

    # calculate signal
    return k * R1_arr  # S


def signal_SPGR(
    R1: ArrayLike,
    S0: ArrayLike,
    TR: np.floating,
    a: np.floating,
) -> NDArray[np.floating]:
    """Steady-state signal for SPGR sequence.
    Args:
        R1 (ArrayLike): longitudinal relaxation rate in units of /s. [OSIPI code Q.EL1.001]
        S0 (ArrayLike): fully T1-relaxed signal in a.u. [OSIPI code Q.MS1.010]
        TR (np.floating): repetition time in units of s. [OSIPI code Q.MS1.006]
        a (np.floating): prescribed flip angle in units of deg. [OSIPI code Q.MS1.007]
    Returns:
        NDArray[np.floating]: magnitude signal in a.u. [OSIPI code Q.MS1.001]
    References:
        - Lexicon url: https://osipi.github.io/OSIPI_CAPLEX/perfusionModels/#SPGR%20model
        - Lexicon code: M.SM2.002
        - OSIPI name: Spoiled gradient recalled echo model
        - Adapted from equation given in the Lexicon and contribution from MJT_UoEdinburgh_UK
    """
    # Convert inputs to numpy arrays with appropriate dtype
    R1_arr = np.asarray(R1)
    S0_arr = np.asarray(S0)

    # Ensure floating-point dtype
    if not np.issubdtype(R1_arr.dtype, np.floating):
        R1_arr = R1_arr.astype(np.float64)
    if not np.issubdtype(S0_arr.dtype, np.floating):
        S0_arr = S0_arr.astype(np.float64)

    # calculate signal
    a_rad = a * np.pi / 180
    exp_TR_R1 = np.exp(-TR * R1_arr)
    return S0_arr * (((1.0 - exp_TR_R1) * np.sin(a_rad)) / (1.0 - exp_TR_R1 * np.cos(a_rad)))  # S
