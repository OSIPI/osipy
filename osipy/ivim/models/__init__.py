"""IVIM signal models.

This module defines IVIM signal decay models for DWI analysis,
implementing the bi-exponential and simplified IVIM models for
separating diffusion and perfusion contributions.

Models are registered via ``@register_ivim_model`` and can be
retrieved by name with ``get_ivim_model()``.

References
----------
.. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
.. [2] Le Bihan D et al. (1988). Radiology 168(2):497-505.
"""

from osipy.ivim.models.biexponential import (
    IVIMBiexponentialModel,
    IVIMModel,
    IVIMParams,
    IVIMSimplifiedModel,
)
from osipy.ivim.models.registry import (
    get_ivim_model,
    register_ivim_model,
)

# Register built-in models via the decorator (applied here to avoid
# circular imports between registry.py and biexponential.py).
register_ivim_model("biexponential")(IVIMBiexponentialModel)
register_ivim_model("simplified")(IVIMSimplifiedModel)

__all__ = [
    "IVIMBiexponentialModel",
    "IVIMModel",
    "IVIMParams",
    "IVIMSimplifiedModel",
    "get_ivim_model",
    "register_ivim_model",
]
