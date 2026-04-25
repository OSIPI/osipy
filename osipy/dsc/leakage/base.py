"""Base class for DSC leakage correction methods (OSIPI: P.LC1.001).

Leakage correction addresses contrast agent extravasation across the
blood-brain barrier, which contaminates the delta-R2* signal used to
compute perfusion parameters.

References
----------
.. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
.. [2] Dickie BR et al. MRM 2024. doi:10.1002/mrm.29840
"""

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from osipy.common.models.base import BaseComponent

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class BaseLeakageCorrector(BaseComponent):
    """Abstract base class for leakage correction methods (OSIPI: P.LC1.001).

    Subclasses implement specific correction algorithms to separate
    intravascular delta-R2* signal from extravascular leakage contributions,
    estimating leakage coefficients K1 (OSIPI: Q.LC1.001) and K2
    (OSIPI: Q.LC1.002).

    References
    ----------
    .. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
    """

    @abstractmethod
    def correct(
        self,
        delta_r2: "NDArray[np.floating[Any]]",
        aif: "NDArray[np.floating[Any]]",
        time: "NDArray[np.floating[Any]]",
        mask: "NDArray[np.bool_] | None" = None,
        **kwargs: Any,
    ) -> Any:
        """Perform leakage correction (OSIPI: P.LC1.001).

        Parameters
        ----------
        delta_r2 : NDArray[np.floating]
            Uncorrected delta-R2* data.
        aif : NDArray[np.floating]
            Arterial input function.
        time : NDArray[np.floating]
            Time points in seconds.
        mask : NDArray[np.bool_] | None
            Brain mask.
        **kwargs : Any
            Method-specific parameters.

        Returns
        -------
        LeakageCorrectionResult
            Corrected data and leakage coefficients.
        """
        ...
