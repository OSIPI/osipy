"""T1 mapping methods for DCE-MRI.

This module provides functions for computing T1 relaxation time maps
from various acquisition schemes, estimating native R1 (OSIPI: Q.EL1.001).

Available methods:
    - VFA (OSIPI: P.NR2.002): Variable Flip Angle
    - Look-Locker (OSIPI: P.NR2.004): Multi-delay inversion recovery

References
----------
.. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
"""

from osipy.common.dataset import PerfusionDataset
from osipy.dce.t1_mapping.look_locker import compute_t1_look_locker
from osipy.dce.t1_mapping.registry import (
    get_t1_method,
    list_t1_methods,
    register_t1_method,
)
from osipy.dce.t1_mapping.vfa import T1MappingResult, compute_t1_vfa


def compute_t1_map(
    dataset: PerfusionDataset,
    method: str = "vfa",
) -> T1MappingResult:
    """Compute T1 map from multi-flip-angle or Look-Locker data.

    Parameters
    ----------
    dataset : PerfusionDataset
        Dataset with multiple flip angles (VFA) or inversion times (Look-Locker).
    method : str, default="vfa"
        T1 mapping method: 'vfa' or 'look_locker'.

    Returns
    -------
    T1MappingResult
        Result containing T1 map, M0 map, and quality mask.

    Raises
    ------
    ValueError
        If unknown method or incompatible dataset.

    References
    ----------
    Deoni et al. (2003). Rapid combined T1 and T2 mapping. MRM.

    Examples
    --------
    >>> from osipy.dce.t1_mapping import compute_t1_map
    >>> result = compute_t1_map(dataset, method="vfa")
    >>> print(f"Mean T1: {result.t1_map.statistics()['mean']:.0f} ms")
    """
    t1_func = get_t1_method(method)
    return t1_func(dataset)


__all__ = [
    "T1MappingResult",
    "compute_t1_look_locker",
    "compute_t1_map",
    "compute_t1_vfa",
    "get_t1_method",
    "list_t1_methods",
    "register_t1_method",
]
