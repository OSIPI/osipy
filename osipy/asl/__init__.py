"""ASL perfusion analysis module.

This module provides tools for Arterial Spin Labeling MRI analysis,
including CBF quantification for different labeling schemes (PASL, CASL,
pCASL). Parameter naming and units follow the OSIPI ASL Lexicon:

- CBF (cerebral blood flow): mL/100g/min
- ATT (arterial transit time): ms
- M0 (equilibrium magnetization): a.u.

Functions
---------
quantify_cbf
    Compute absolute CBF from ASL difference images.
apply_m0_calibration
    Apply M0 calibration to ASL data.
compute_labeling_efficiency
    Compute/apply labeling efficiency corrections.

References
----------
.. [1] OSIPI ASL Lexicon, https://osipi.github.io/ASL-Lexicon/
.. [2] Alsop DC et al. (2015). Recommended implementation of arterial
   spin-labeled perfusion MRI for clinical applications: A consensus of
   the ISMRM Perfusion Study Group and the European Consortium for ASL
   in Dementia. Magn Reson Med 73(1):102-116.
.. [3] Suzuki Y et al. (2024). ASL lexicon and reporting
   recommendations: A consensus report from the ISMRM Open Science
   Initiative for Perfusion Imaging (OSIPI). MRM 91(5):1743-1760.
"""

# Labeling schemes
# Calibration
from osipy.asl.calibration import (
    M0CalibrationParams,
    apply_m0_calibration,
    compute_m0_from_pd,
    get_m0_calibration,
    list_m0_calibrations,
    register_m0_calibration,
)
from osipy.asl.labeling import (
    CASLParams,
    LabelingScheme,
    PASLParams,
    PCASLParams,
    compute_labeling_efficiency,
)

# Quantification
from osipy.asl.quantification import (
    ASLQuantificationParams,
    ASLQuantificationResult,
    compute_control_label_difference,
    compute_pcasl_difference,
    get_att_model,
    get_quantification_model,
    list_att_models,
    list_difference_methods,
    list_quantification_models,
    quantify_cbf,
    register_att_model,
    register_difference_method,
    register_quantification_model,
)

__all__ = [
    "ASLQuantificationParams",
    "ASLQuantificationResult",
    "CASLParams",
    # Labeling
    "LabelingScheme",
    "M0CalibrationParams",
    "PASLParams",
    "PCASLParams",
    # Calibration
    "apply_m0_calibration",
    "compute_control_label_difference",
    "compute_labeling_efficiency",
    "compute_m0_from_pd",
    "compute_pcasl_difference",
    "get_att_model",
    "get_m0_calibration",
    "get_quantification_model",
    "list_att_models",
    "list_difference_methods",
    "list_m0_calibrations",
    "list_quantification_models",
    # Quantification
    "quantify_cbf",
    "register_att_model",
    "register_difference_method",
    "register_m0_calibration",
    # Quantification registries
    "register_quantification_model",
]
