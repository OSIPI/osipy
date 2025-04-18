import numpy as np
from numpy.typing import ArrayLike, NDArray


def exp_conv(T: np.floating, t: ArrayLike, a: ArrayLike) -> NDArray[np.floating]:
    """Exponential convolution operation of (1/T)exp(-t/T) with a.

    Args:
        T (np.floating): exponent in time units
        t (ArrayLike): array of time points
        a (ArrayLike): array to be convolved with time exponential

    Returns:
        NDArray[np.floating]: convolved array
    """
    # Convert inputs to numpy arrays
    t_arr = np.asarray(t)
    a_arr = np.asarray(a)

    # Ensure floating-point dtype while preserving original precision
    if not np.issubdtype(t_arr.dtype, np.floating):
        t_arr = t_arr.astype(np.float64)
    if not np.issubdtype(a_arr.dtype, np.floating):
        a_arr = a_arr.astype(np.float64)

    if T == 0:
        return a_arr

    n = len(t_arr)
    f = np.zeros((n,), dtype=t_arr.dtype)  # Use same dtype as t_arr

    x = (t_arr[1 : n - 1] - t_arr[0 : n - 2]) / T
    da = (a_arr[1 : n - 1] - a_arr[0 : n - 2]) / x

    E = np.exp(-x)
    E0 = 1 - E
    E1 = x - E0

    add = a_arr[0 : n - 2] * E0 + da * E1

    for i in range(0, n - 2):
        f[i + 1] = E[i] * f[i] + add[i]

    f[n - 1] = f[n - 2]
    return f
