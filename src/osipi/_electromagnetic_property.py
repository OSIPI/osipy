import numpy as np
from numpy.typing import ArrayLike, NDArray


def R1_to_C_linear_relaxivity(
    R1: ArrayLike, R10: np.floating, r1: np.floating
) -> NDArray[np.floating]:
    """
    Electromagnetic property inverse model:
    - longitudinal relaxation rate, linear with relaxivity

    Converts R1 to tissue concentration

    Args:
        R1 (ArrayLike):
            Vector of longitudinal relaxation rate in units of /s. [OSIPI code Q.EL1.001]
        R10 (np.floating):
            Native longitudinal relaxation rate in units of /s. [OSIPI code Q.EL1.002]
        r1 (np.floating):
            Longitudinal relaxivity in units of /s/mM. [OSIPI code Q.EL1.015]

    Returns:
        NDArray[np.floating]:
            Vector of indicator concentration in units of mM. [OSIPI code Q.IC1.001]

    References:
        - Lexicon URL: https://osipi.github.io/OSIPI_CAPLEX/perfusionProcesses/#
        - Lexicon code: P.EC1.001
        - OSIPI name: model-based
          - Inversion method: analytical inversion [OSIPI code G.MI1.001]
          - Forward model:
            longitudinal relaxation rate, linear with relaxivity model [OSIPI code M.EL1.003]
        - Adapted from equation given in lexicon
    """
    # Convert input to numpy array with appropriate dtype
    R1_arr = np.asarray(R1)
    # Ensure floating-point dtype
    if not np.issubdtype(R1_arr.dtype, np.floating):
        R1_arr = R1_arr.astype(np.float64)

    # Check R1 is a 1D array
    if not (R1_arr.ndim == 1):
        raise TypeError("R1 must be a 1D array-like object of floating point values")
    elif not (r1 >= 0):
        raise ValueError("r1 must be positive")
    return (R1_arr - R10) / r1  # C
