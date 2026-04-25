"""Base class for DSC deconvolution methods.

Deconvolution recovers the tissue residue function R(t) from the observed
concentration-time curve C(t) and the arterial input function (AIF):

    C(t) = CBF * AIF(t) ** R(t)

where ** denotes convolution. The residue function describes the fraction of
tracer still present in the tissue at time t after an ideal bolus injection.

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


class BaseDeconvolver(BaseComponent):
    """Abstract base class for deconvolution methods.

    Subclasses implement specific deconvolution algorithms (sSVD, cSVD, oSVD)
    to recover the residue function and estimate perfusion parameters
    including CBF (OSIPI: Q.PH1.003), MTT (OSIPI: Q.PH1.006), and
    arterial delay Ta (OSIPI: Q.PH1.007).

    References
    ----------
    .. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
    """

    @abstractmethod
    def deconvolve(
        self,
        concentration: "NDArray[np.floating[Any]]",
        aif: "NDArray[np.floating[Any]]",
        time: "NDArray[np.floating[Any]]",
        mask: "NDArray[np.bool_] | None" = None,
        **kwargs: Any,
    ) -> Any:
        """Perform deconvolution to recover the residue function.

        Parameters
        ----------
        concentration : NDArray[np.floating]
            Tissue concentration curves.
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
        DeconvolutionResult
            Residue function and derived perfusion parameters.
        """
        ...
