"""Config models for DSC deconvolution methods.

Each SVD deconvolution method (sSVD, cSVD, oSVD) declares its tunable knobs as a
:class:`~osipy.common.config.MethodConfig`. These compose into a discriminated
union for the CLI config, and :func:`~osipy.common.config.construct_from_config`
builds the corresponding fitter directly from a validated config instance.

References
----------
.. [1] Ostergaard L et al. MRM 1996;36(5):715-725.
.. [2] Wu O et al. MRM 2003;50(1):164-174.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from osipy.common.config import MethodConfig
from osipy.dsc.deconvolution.svd_fitters import CSVDFitter, OSVDFitter, SSVDFitter


class SSVDConfig(MethodConfig):
    """Standard truncated-SVD deconvolution (single global threshold)."""

    method: Literal["sSVD"] = "sSVD"
    threshold: float = Field(
        0.2, gt=0.0, lt=1.0, description="SVD truncation threshold (fraction of max)"
    )


class CSVDConfig(MethodConfig):
    """Block-circulant SVD deconvolution (delay-insensitive)."""

    method: Literal["cSVD"] = "cSVD"
    threshold: float = Field(
        0.2, gt=0.0, lt=1.0, description="SVD truncation threshold (fraction of max)"
    )


class OSVDConfig(MethodConfig):
    """Oscillation-index SVD deconvolution (per-voxel adaptive threshold)."""

    method: Literal["oSVD"] = "oSVD"
    oscillation_index: float = Field(
        0.035, gt=0.0, description="target oscillation index (Wu et al. 2003)"
    )
    default_threshold: float = Field(
        0.2, gt=0.0, lt=1.0, description="fallback truncation threshold"
    )


#: name -> config model (source for the discriminated union)
DECONVOLVER_CONFIGS: dict[str, type[MethodConfig]] = {
    "sSVD": SSVDConfig,
    "cSVD": CSVDConfig,
    "oSVD": OSVDConfig,
}

#: name -> fitter class (the live implementation, constructed from config)
DECONVOLVER_REGISTRY = {
    "sSVD": SSVDFitter,
    "cSVD": CSVDFitter,
    "oSVD": OSVDFitter,
}

__all__ = [
    "DECONVOLVER_CONFIGS",
    "DECONVOLVER_REGISTRY",
    "CSVDConfig",
    "OSVDConfig",
    "SSVDConfig",
]
