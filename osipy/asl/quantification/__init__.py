"""ASL CBF quantification module.

This module implements CBF quantification from ASL difference images
following the ISMRM consensus recommendations. Parameter naming and
units follow the OSIPI ASL Lexicon: CBF in mL/100g/min, ATT in ms,
M0 in a.u.

References
----------
.. [1] OSIPI ASL Lexicon, https://osipi.github.io/ASL-Lexicon/
"""

from osipy.asl.quantification.att_registry import (
    get_att_model,
    list_att_models,
    register_att_model,
)
from osipy.asl.quantification.base import BaseASLModel
from osipy.asl.quantification.cbf import (
    ASLQuantificationParams,
    ASLQuantificationResult,
    compute_control_label_difference,
    compute_pcasl_difference,
    list_difference_methods,
    quantify_cbf,
    register_difference_method,
)
from osipy.asl.quantification.multi_pld import (
    MultiPLDParams,
    MultiPLDResult,
    quantify_multi_pld,
)
from osipy.asl.quantification.registry import (
    get_quantification_model,
    list_quantification_models,
    register_quantification_model,
)

__all__ = [
    "ASLQuantificationParams",
    "ASLQuantificationResult",
    "BaseASLModel",
    "MultiPLDParams",
    "MultiPLDResult",
    "compute_control_label_difference",
    "compute_pcasl_difference",
    "get_att_model",
    "get_quantification_model",
    "list_att_models",
    "list_difference_methods",
    "list_quantification_models",
    "quantify_cbf",
    "quantify_multi_pld",
    "register_att_model",
    "register_difference_method",
    "register_quantification_model",
]
