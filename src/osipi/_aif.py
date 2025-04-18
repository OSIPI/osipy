import numpy as np
from numpy.typing import ArrayLike, NDArray


def aif_parker(
    t: ArrayLike, BAT: np.floating = 0.0, Hct: np.floating = 0.0
) -> NDArray[np.floating]:
    """AIF model as defined by Parker et al (2005)

    Args:
        t (ArrayLike): array of time points in units of sec. [OSIPI code Q.GE1.004]
        BAT (np.floating, optional):
            Time in seconds before the bolus arrives. Defaults to 0. [OSIPI code Q.BA1.001]
        Hct (np.floating, optional):
            Hematocrit. Defaults to 0.0. [OSIPI code Q.PH1.012]

    Returns:
        NDArray[np.floating]: Concentrations in mM for each time point in t.

    See Also:
        `aif_georgiou`
        `aif_weinmann`

    References:
        - Lexicon url:
            https://osipi.github.io/OSIPI_CAPLEX/perfusionModels/#arterial-input-function-models
        - Lexicon code: M.IC2.001
        - OSIPI name: Parker AIF model
        - Adapted from contribution by: MB_QBI_UoManchester_UK

    Example:
        Create an array of time points covering 6 min in steps of 1 sec,
        calculate the Parker AIF at these time points and plot the results.

        Import packages:

        >>> import matplotlib.pyplot as plt
        >>> import osipi

        Calculate AIF and plot

        >>> t = np.arange(0, 6 * 60, 1)
        >>> ca = osipi.aif_parker(t)
        >>> plt.plot(t, ca)
        >>> plt.show()

    """
    # Convert input to numpy array with appropriate dtype
    t_arr = np.asarray(t)
    # Ensure floating-point dtype
    if not np.issubdtype(t_arr.dtype, np.floating):
        t_arr = t_arr.astype(np.float64)

    # Convert from OSIPI units (sec) to units used internally (mins)
    t_min = t_arr / 60
    bat_min = BAT / 60

    t_offset = t_min - bat_min

    # A1/(SD1*sqrt(2*PI)) * exp(-(t_offset-m1)^2/(2*var1))
    # A1 = 0.833, SD1 = 0.055, m1 = 0.171
    gaussian1 = 5.73258 * np.exp(
        -1.0 * (t_offset - 0.17046) * (t_offset - 0.17046) / (2.0 * 0.0563 * 0.0563)
    )

    # A2/(SD2*sqrt(2*PI)) * exp(-(t_offset-m2)^2/(2*var2))
    # A2 = 0.336, SD2 = 0.134, m2 = 0.364
    gaussian2 = 0.997356 * np.exp(
        -1.0 * (t_offset - 0.365) * (t_offset - 0.365) / (2.0 * 0.132 * 0.132)
    )
    # alpha*exp(-beta*t_offset) / (1+exp(-s(t_offset-tau)))
    # alpha = 1.064, beta = 0.166, s = 37.772, tau = 0.482
    sigmoid = 1.050 * np.exp(-0.1685 * t_offset) / (1.0 + np.exp(-38.078 * (t_offset - 0.483)))

    pop_aif = (gaussian1 + gaussian2 + sigmoid) / (1.0 - Hct)

    return pop_aif


def aif_georgiou(t: ArrayLike, BAT: np.floating = 0.0) -> NDArray[np.floating]:
    """AIF model as defined by Georgiou et al.

    Note:
        This function is not yet implemented.
        If you are implementing it yourself please consider submitting a code contribution to OSIPI,
        so nobody ever has to write this function again!

    Args:
        t (ArrayLike): array of time points in units of sec. [OSIPI code Q.GE1.004]
        BAT (np.floating, optional):
            Time in seconds before the bolus arrives. Defaults to 0sec. [OSIPI code Q.BA1.001]

    Returns:
        NDArray[np.floating]: Concentrations in mM for each time point in t.

    See Also:
        `aif_parker`
        `aif_weinmann`

    References:
        - Lexicon url:
            https://osipi.github.io/OSIPI_CAPLEX/perfusionModels/#arterial-input-function-models
        - Lexicon code: M.IC2.002
        - OSIPI name: Georgiou AIF model
        - Adapted from contribution by: TBC

    Example:

        Create an array of time points covering 6min in steps of 1sec,
        calculate the Georgiou AIF at these time points and plot the results.

        Import packages:

        >>> import matplotlib.pyplot as plt
        >>> import osipi

        Calculate AIF and plot

        >>> t = np.arange(0, 6 * 60, 0.1)
        >>> ca = osipi.aif_georgiou(t)
        >>> plt.plot(t, ca)
        >>> plt.show()

    """

    msg = "This function is not yet implemented \n"
    msg += (
        "If you implement it yourself, please consider submitting it as an OSIPI code contribution"
    )
    raise NotImplementedError(msg)


def aif_weinmann(t: ArrayLike, BAT: np.floating = 0.0) -> NDArray[np.floating]:
    """AIF model as defined by Weinmann et al.

    Note:
        This function is not yet implemented.
        If you are implementing it yourself please consider submitting a code contribution to OSIPI,
        so nobody ever has to write this function again!

    Args:
        t (ArrayLike): array of time points in units of sec. [OSIPI code Q.GE1.004]
        BAT (np.floating, optional):
            Time in seconds before the bolus arrives. Defaults to 0sec. [OSIPI code Q.BA1.001]

    Returns:
        NDArray[np.floating]: Concentrations in mM for each time point in t.

    See Also:
        `aif_parker`
        `aif_georgiou`

    References:
        - Lexicon url:
            https://osipi.github.io/OSIPI_CAPLEX/perfusionModels/#arterial-input-function-models
        - Lexicon code: M.IC2.003
        - OSIPI name: Weinmann AIF model
        - Adapted from contribution by: TBC

    Example:

        Create an array of time points covering 6min in steps of 1sec,
        calculate the Weinmann AIF at these time points and plot the results.

        Import packages:

        >>> import matplotlib.pyplot as plt
        >>> import osipi

        Calculate AIF and plot

        >>> t = np.arange(0, 6 * 60, 0.1)
        >>> ca = osipi.aif_weinmann(t)
        >>> plt.plot(t, ca)

    """

    msg = "This function is not yet implemented \n"
    msg += (
        "If you implement it yourself, please consider submitting it as an OSIPI code contribution"
    )
    raise NotImplementedError(msg)
