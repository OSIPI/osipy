import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import interp1d

from ._convolution import exp_conv


def tofts(
    t: ArrayLike,
    ca: ArrayLike,
    Ktrans: np.floating,
    ve: np.floating,
    Ta: np.floating = 30.0,
    discretization_method: str = "conv",
) -> NDArray[np.floating]:
    """Tofts model as defined by Tofts and Kermode (1991)

    Args:
        t (ArrayLike): array of time points in units of sec. [OSIPI code Q.GE1.004]
        ca (ArrayLike):
            Arterial concentrations in mM for each time point in t. [OSIPI code Q.IC1.001]
        Ktrans (np.floating):
            Volume transfer constant in units of 1/min. [OSIPI code Q.PH1.008]
        ve (np.floating):
            Relative volume fraction of the extracellular
            extravascular compartment (e). [OSIPI code Q.PH1.001.[e]]
        Ta (np.floating, optional):
            Arterial delay time,
            i.e., difference in onset time between tissue curve and AIF in units of sec. Defaults to 30 seconds. [OSIPI code Q.PH1.007]
        discretization_method (str, optional): Defines the discretization method. Options include

            – 'conv': Numerical convolution (default) [OSIPI code G.DI1.001]

            – 'exp': Exponential convolution [OSIPI code G.DI1.006]


    Returns:
        NDArray[np.floating]: Tissue concentrations in mM for each time point in t.

    See Also:
        `extended_tofts`

    References:
        - Lexicon url:
            https://osipi.github.io/OSIPI_CAPLEX/perfusionModels/#indicator-kinetic-models
        - Lexicon code: M.IC1.004
        - OSIPI name: Tofts Model
        - Adapted from contributions by: LEK_UoEdinburgh_UK, ST_USyd_AUS, MJT_UoEdinburgh_UK

    Example:

        Create an array of time points covering 6 min in steps of 1 sec,
        calculate the Parker AIF at these time points, calculate tissue concentrations
        using the Tofts model and plot the results.

        Import packages:

        >>> import matplotlib.pyplot as plt
        >>> import osipi
        >>> import numpy

        Calculate AIF:

        >>> t = np.arange(0, 6 * 60, 1)
        >>> ca = osipi.aif_parker(t)

        Calculate tissue concentrations and plot:

        >>> Ktrans = 0.6  # in units of 1/min
        >>> ve = 0.2  # takes values from 0 to 1
        >>> ct = osipi.tofts(t, ca, Ktrans, ve)
        >>> plt.plot(t, ca, "r", t, ct, "b")

    """
    # Convert inputs to numpy arrays with appropriate dtype
    t_arr = np.asarray(t)
    ca_arr = np.asarray(ca)

    # Ensure floating-point dtype
    if not np.issubdtype(t_arr.dtype, np.floating):
        t_arr = t_arr.astype(np.float64)
    if not np.issubdtype(ca_arr.dtype, np.floating):
        ca_arr = ca_arr.astype(np.float64)

    if not np.allclose(np.diff(t_arr), np.diff(t_arr)[0]):
        warnings.warn(
            ("Non-uniform time spacing detected. Time array may be" " resampled."),
            stacklevel=2,
        )

    if Ktrans <= 0 or ve <= 0:
        ct = 0 * ca_arr

    else:
        # Convert units
        Ktrans = Ktrans / 60  # from 1/min to 1/sec

        if discretization_method == "exp":  # Use exponential convolution
            # Shift the AIF by the arterial delay time (if not zero)
            if Ta != 0:
                f = interp1d(
                    t_arr,
                    ca_arr,
                    kind="linear",
                    bounds_error=False,
                    fill_value=0,
                )
                ca_arr = (t_arr > Ta) * f(t_arr - Ta)

            Tc = ve / Ktrans
            ct = ve * exp_conv(Tc, t_arr, ca_arr)

        else:  # Use convolution by default
            # Calculate the impulse response function
            kep = Ktrans / ve
            imp = Ktrans * np.exp(-1 * kep * t_arr)

            # Shift the AIF by the arterial delay time (if not zero)
            if Ta != 0:
                f = interp1d(
                    t_arr,
                    ca_arr,
                    kind="linear",
                    bounds_error=False,
                    fill_value=0,
                )
                ca_arr = (t_arr > Ta) * f(t_arr - Ta)

            # Check if time data grid is uniformly spaced
            if np.allclose(np.diff(t_arr), np.diff(t_arr)[0]):
                # Convolve impulse response with AIF
                convolution = np.convolve(ca_arr, imp)

                # Discard unwanted points and make sure time spacing
                # is correct
                ct = convolution[0 : len(t_arr)] * t_arr[1]
            else:
                # Resample at the smallest spacing
                dt = np.min(np.diff(t_arr))
                t_resampled = np.linspace(t_arr[0], t_arr[-1], int((t_arr[-1] - t_arr[0]) / dt))
                ca_func = interp1d(
                    t_arr,
                    ca_arr,
                    kind="quadratic",
                    bounds_error=False,
                    fill_value=0,
                )
                imp_func = interp1d(
                    t_arr,
                    imp,
                    kind="quadratic",
                    bounds_error=False,
                    fill_value=0,
                )
                ca_resampled = ca_func(t_resampled)
                imp_resampled = imp_func(t_resampled)
                # Convolve impulse response with AIF
                convolution = np.convolve(ca_resampled, imp_resampled)

                # Discard unwanted points and make sure time spacing is correct
                ct_resampled = convolution[0 : len(t_resampled)] * t_resampled[1]

                # Restore time grid spacing
                ct_func = interp1d(
                    t_resampled,
                    ct_resampled,
                    kind="quadratic",
                    bounds_error=False,
                    fill_value=0,
                )
                ct = ct_func(t_arr)

    return ct


def extended_tofts(
    t: ArrayLike,
    ca: ArrayLike,
    Ktrans: np.floating,
    ve: np.floating,
    vp: np.floating,
    Ta: np.floating = 30.0,
    discretization_method: str = "conv",
) -> NDArray[np.floating]:
    """Extended tofts model as defined by Tofts (1997)

    Args:
        t (ArrayLike):
            array of time points in units of sec. [OSIPI code Q.GE1.004]
        ca (ArrayLike):
        vp (np.floating):
            Relative volyme fraction of the plasma compartment (p). [OSIPI code Q.PH1.001.[p]]
        Ta (np.floating, optional):
            Arterial delay time, i.e., difference in onset time
            between tissue curve and AIF in units of sec.
            Defaults to 30 seconds. [OSIPI code Q.PH1.007]
        discretization_method (str, optional):
            Defines the discretization method. Options include

            – 'conv': Numerical convolution (default) [OSIPI code G.DI1.001]

            – 'exp': Exponential convolution [OSIPI code G.DI1.006]


    Returns:
        NDArray[np.floating]: Tissue concentrations in mM for each time point in t.

    See Also:
        `tofts`

    References:
        - Lexicon url: https://osipi.github.io/OSIPI_CAPLEX/perfusionModels/#indicator-kinetic-models
        - Lexicon code: M.IC1.005
        - OSIPI name: Extended Tofts Model
        - Adapted from contributions by: LEK_UoEdinburgh_UK, ST_USyd_AUS, MJT_UoEdinburgh_UK

    Example:

        Create an array of time points covering 6 min in steps of 1 sec,
        calculate the Parker AIF at these time points, calculate tissue concentrations
        using the Extended Tofts model and plot the results.

        Import packages:

        >>> import matplotlib.pyplot as plt
        >>> import osipi

        Calculate AIF

        >>> t = np.arange(0, 6 * 60, 0.1)
        >>> ca = osipi.aif_parker(t)

        Calculate tissue concentrations and plot

        >>> Ktrans = 0.6  # in units of 1/min
        >>> ve = 0.2  # takes values from 0 to 1
        >>> vp = 0.3  # takes values from 0 to 1
        >>> ct = osipi.extended_tofts(t, ca, Ktrans, ve, vp)
        >>> plt.plot(t, ca, "r", t, ct, "b")

    """
    # Convert inputs to numpy arrays with appropriate dtype
    t_arr = np.asarray(t)
    ca_arr = np.asarray(ca)

    # Ensure floating-point dtype
    if not np.issubdtype(t_arr.dtype, np.floating):
        t_arr = t_arr.astype(np.float64)
    if not np.issubdtype(ca_arr.dtype, np.floating):
        ca_arr = ca_arr.astype(np.float64)

    if not np.allclose(np.diff(t_arr), np.diff(t_arr)[0]):
        warnings.warn(
            ("Non-uniform time spacing detected. Time array may be" " resampled."),
            stacklevel=2,
        )

    if Ktrans <= 0 or ve <= 0:
        ct = vp * ca_arr

    else:
        # Convert units
        Ktrans = Ktrans / 60  # from 1/min to 1/sec

        if discretization_method == "exp":  # Use exponential convolution
            # Shift the AIF by the arterial delay time (if not zero)
            if Ta != 0:
                f = interp1d(
                    t_arr,
                    ca_arr,
                    kind="linear",
                    bounds_error=False,
                    fill_value=0,
                )
                ca_arr = (t_arr > Ta) * f(t_arr - Ta)

            Tc = ve / Ktrans
            # expconv calculates convolution of ca and
            # (1/Tc)exp(-t/Tc), add vp*ca term for extended model
            ct = (vp * ca_arr) + ve * exp_conv(Tc, t_arr, ca_arr)

        else:  # Use convolution by default
            # Calculate the impulse response function
            kep = Ktrans / ve
            imp = Ktrans * np.exp(-1 * kep * t_arr)

            # Shift the AIF by the arterial delay time (if not zero)
            if Ta != 0:
                f = interp1d(
                    t_arr,
                    ca_arr,
                    kind="linear",
                    bounds_error=False,
                    fill_value=0,
                )
                ca_arr = (t_arr > Ta) * f(t_arr - Ta)

            # Check if time data grid is uniformly spaced
            if np.allclose(np.diff(t_arr), np.diff(t_arr)[0]):
                # Convolve impulse response with AIF
                convolution = np.convolve(ca_arr, imp)

                # Discard unwanted points, make sure time spacing is
                # correct and add vp*ca term for extended model
                ct = convolution[0 : len(t_arr)] * t_arr[1] + (vp * ca_arr)
            else:
                # Resample at the smallest spacing
                dt = np.min(np.diff(t_arr))
                t_resampled = np.linspace(t_arr[0], t_arr[-1], int((t_arr[-1] - t_arr[0]) / dt))
                ca_func = interp1d(
                    t_arr,
                    ca_arr,
                    kind="quadratic",
                    bounds_error=False,
                    fill_value=0,
                )
                imp_func = interp1d(
                    t_arr,
                    imp,
                    kind="quadratic",
                    bounds_error=False,
                    fill_value=0,
                )
                ca_resampled = ca_func(t_resampled)
                imp_resampled = imp_func(t_resampled)
                # Convolve impulse response with AIF
                convolution = np.convolve(ca_resampled, imp_resampled)

                # Discard unwanted points, make sure time spacing is
                # correct and add vp*ca term for extended model
                ct_resampled = convolution[0 : len(t_resampled)] * t_resampled[1] + (
                    vp * ca_resampled
                )

                # Restore time grid spacing
                ct_func = interp1d(
                    t_resampled,
                    ct_resampled,
                    kind="quadratic",
                    bounds_error=False,
                    fill_value=0,
                )
                ct = ct_func(t_arr)

    return ct
