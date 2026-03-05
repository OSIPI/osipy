"""Pydantic v2 models for YAML pipeline configuration.

Provides validation models for each modality (DCE, DSC, ASL, IVIM),
a top-level ``PipelineConfig`` model, ``load_config()`` for parsing
YAML files, and ``dump_defaults()`` for generating commented templates.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared configuration sections
# ---------------------------------------------------------------------------


class DataConfig(BaseModel):
    """Data loading configuration."""

    format: str = "auto"
    mask: str | None = None
    t1_map: str | None = None
    aif_file: str | None = None
    m0_data: str | None = None
    b_values: list[float] | None = None
    b_values_file: str | None = None
    subject: str | None = None
    session: str | None = None


class OutputConfig(BaseModel):
    """Output configuration."""

    format: str = "nifti"


class BackendConfig(BaseModel):
    """GPU/CPU backend configuration."""

    force_cpu: bool = False


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"


# ---------------------------------------------------------------------------
# Fitting configuration models
# ---------------------------------------------------------------------------


class DCEFittingConfig(BaseModel):
    """DCE model fitting configuration from YAML."""

    fitter: str = "lm"
    max_iterations: int = 100
    tolerance: float = 1e-6
    r2_threshold: float = 0.5
    bounds: dict[str, list[float]] | None = None
    initial_guess: dict[str, float] | None = None

    @field_validator("fitter")
    @classmethod
    def validate_fitter(cls, v: str) -> str:
        """Validate fitter name against registry."""
        from osipy.common.fitting.registry import FITTER_REGISTRY

        if v not in FITTER_REGISTRY:
            valid = sorted(FITTER_REGISTRY.keys())
            msg = f"Invalid fitter '{v}'. Valid: {valid}"
            raise ValueError(msg)
        return v

    @field_validator("bounds")
    @classmethod
    def validate_bounds(
        cls, v: dict[str, list[float]] | None
    ) -> dict[str, list[float]] | None:
        """Validate bounds are [lower, upper] pairs."""
        if v is None:
            return v
        for name, pair in v.items():
            if len(pair) != 2:
                msg = f"Bounds for '{name}' must be [lower, upper], got {pair}"
                raise ValueError(msg)
            if pair[0] > pair[1]:
                msg = f"Lower bound > upper bound for '{name}': {pair}"
                raise ValueError(msg)
        return v


class BayesianIVIMFittingConfig(BaseModel):
    """Bayesian IVIM fitting configuration from YAML."""

    prior_scale: float = 1.5
    noise_std: float | None = None
    compute_uncertainty: bool = True


class IVIMFittingConfig(BaseModel):
    """IVIM model fitting configuration from YAML."""

    max_iterations: int = 500
    tolerance: float = 1e-6
    bounds: dict[str, list[float]] | None = None
    initial_guess: dict[str, float] | None = None
    bayesian: BayesianIVIMFittingConfig = BayesianIVIMFittingConfig()

    @field_validator("bounds")
    @classmethod
    def validate_bounds(
        cls, v: dict[str, list[float]] | None
    ) -> dict[str, list[float]] | None:
        """Validate bounds are [lower, upper] pairs."""
        if v is None:
            return v
        for name, pair in v.items():
            if len(pair) != 2:
                msg = f"Bounds for '{name}' must be [lower, upper], got {pair}"
                raise ValueError(msg)
            if pair[0] > pair[1]:
                msg = f"Lower bound > upper bound for '{name}': {pair}"
                raise ValueError(msg)
        return v


# ---------------------------------------------------------------------------
# DCE modality
# ---------------------------------------------------------------------------


class DCEAcquisitionYAML(BaseModel):
    """DCE acquisition parameters from YAML."""

    tr: float | None = None
    flip_angles: list[float] | None = None
    baseline_frames: int = 5
    relaxivity: float = 4.5
    t1_assumed: float | None = None


class ROIConfig(BaseModel):
    """Region-of-interest configuration for limiting processing."""

    enabled: bool = False
    center: list[int] | None = None
    radius: int = 10


class DCEPipelineYAML(BaseModel):
    """DCE pipeline settings from YAML."""

    model: str = "extended_tofts"
    t1_mapping_method: str = "vfa"
    aif_source: str = "population"
    population_aif: str = "parker"
    save_intermediate: bool = False
    acquisition: DCEAcquisitionYAML = DCEAcquisitionYAML()
    roi: ROIConfig = ROIConfig()
    fitting: DCEFittingConfig = DCEFittingConfig()

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate DCE model name against registry."""
        from osipy.dce import list_models

        valid = list_models()
        if v not in valid:
            msg = f"Invalid DCE model '{v}'. Valid: {valid}"
            raise ValueError(msg)
        return v

    @field_validator("t1_mapping_method")
    @classmethod
    def validate_t1_method(cls, v: str) -> str:
        """Validate T1 mapping method."""
        valid = ["vfa", "look_locker"]
        if v not in valid:
            msg = f"Invalid T1 mapping method '{v}'. Valid: {valid}"
            raise ValueError(msg)
        return v

    @field_validator("aif_source")
    @classmethod
    def validate_aif_source(cls, v: str) -> str:
        """Validate AIF source."""
        valid = ["population", "detect", "manual"]
        if v not in valid:
            msg = f"Invalid AIF source '{v}'. Valid: {valid}"
            raise ValueError(msg)
        return v


# ---------------------------------------------------------------------------
# DSC modality
# ---------------------------------------------------------------------------


class DSCPipelineYAML(BaseModel):
    """DSC pipeline settings from YAML."""

    te: float = 30.0
    deconvolution_method: str = "oSVD"
    apply_leakage_correction: bool = True
    svd_threshold: float = 0.2
    baseline_frames: int = 10
    hematocrit_ratio: float = 0.73

    @field_validator("deconvolution_method")
    @classmethod
    def validate_deconv(cls, v: str) -> str:
        """Validate deconvolution method against registry."""
        from osipy.dsc import list_deconvolvers

        valid = list_deconvolvers()
        if v not in valid:
            msg = f"Invalid deconvolution method '{v}'. Valid: {valid}"
            raise ValueError(msg)
        return v


# ---------------------------------------------------------------------------
# ASL modality
# ---------------------------------------------------------------------------


class ASLPipelineYAML(BaseModel):
    """ASL pipeline settings from YAML."""

    labeling_scheme: str = "pcasl"
    pld: float = 1800.0
    label_duration: float = 1800.0
    t1_blood: float = 1650.0
    labeling_efficiency: float = 0.85
    m0_method: str = "single"
    t1_tissue: float = 1330.0
    partition_coefficient: float = 0.9
    difference_method: str = "pairwise"
    label_control_order: str = "label_first"

    @field_validator("labeling_scheme")
    @classmethod
    def validate_labeling(cls, v: str) -> str:
        """Validate ASL labeling scheme."""
        valid = ["pasl", "casl", "pcasl"]
        if v not in valid:
            msg = f"Invalid labeling scheme '{v}'. Valid: {valid}"
            raise ValueError(msg)
        return v

    @field_validator("m0_method")
    @classmethod
    def validate_m0(cls, v: str) -> str:
        """Validate M0 calibration method."""
        valid = ["single", "voxelwise", "reference_region"]
        if v not in valid:
            msg = f"Invalid M0 method '{v}'. Valid: {valid}"
            raise ValueError(msg)
        return v

    @field_validator("label_control_order")
    @classmethod
    def validate_order(cls, v: str) -> str:
        """Validate label/control ordering."""
        valid = ["label_first", "control_first"]
        if v not in valid:
            msg = f"Invalid label/control order '{v}'. Valid: {valid}"
            raise ValueError(msg)
        return v


# ---------------------------------------------------------------------------
# IVIM modality
# ---------------------------------------------------------------------------


class IVIMPipelineYAML(BaseModel):
    """IVIM pipeline settings from YAML."""

    fitting_method: str = "segmented"
    b_threshold: float = 200.0
    normalize_signal: bool = True
    fitting: IVIMFittingConfig = IVIMFittingConfig()

    @field_validator("fitting_method")
    @classmethod
    def validate_fitting(cls, v: str) -> str:
        """Validate IVIM fitting method."""
        valid = ["segmented", "full", "bayesian"]
        if v not in valid:
            msg = f"Invalid fitting method '{v}'. Valid: {valid}"
            raise ValueError(msg)
        return v


# ---------------------------------------------------------------------------
# Top-level configuration
# ---------------------------------------------------------------------------

_MODALITY_PIPELINE_MODELS: dict[str, type[BaseModel]] = {
    "dce": DCEPipelineYAML,
    "dsc": DSCPipelineYAML,
    "asl": ASLPipelineYAML,
    "ivim": IVIMPipelineYAML,
}


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration from YAML."""

    modality: str
    pipeline: dict[str, Any] = {}
    data: DataConfig = DataConfig()
    output: OutputConfig = OutputConfig()
    backend: BackendConfig = BackendConfig()
    logging: LoggingConfig = LoggingConfig()

    @field_validator("modality")
    @classmethod
    def validate_modality(cls, v: str) -> str:
        """Validate modality name."""
        valid = list(_MODALITY_PIPELINE_MODELS.keys())
        if v not in valid:
            msg = f"Invalid modality '{v}'. Valid: {valid}"
            raise ValueError(msg)
        return v

    def get_modality_config(self) -> BaseModel:
        """Get validated modality-specific pipeline config.

        Returns
        -------
        BaseModel
            Validated modality-specific pipeline configuration.
        """
        model_cls = _MODALITY_PIPELINE_MODELS[self.modality]
        return model_cls(**self.pipeline)


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> PipelineConfig:
    """Load and validate a YAML pipeline configuration file.

    Parameters
    ----------
    path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    PipelineConfig
        Validated pipeline configuration.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    pydantic.ValidationError
        If the config fails validation.
    """
    config_path = Path(path)
    if not config_path.exists():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)

    with config_path.open() as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        msg = f"Config file must contain a YAML mapping, got {type(raw).__name__}"
        raise ValueError(msg)

    config = PipelineConfig(**raw)
    # Eagerly validate the modality-specific pipeline config
    config.get_modality_config()
    return config


# ---------------------------------------------------------------------------
# dump_defaults
# ---------------------------------------------------------------------------

_DEFAULT_TEMPLATES: dict[str, str] = {
    "dce": """\
modality: dce
pipeline:
  model: extended_tofts            # tofts | extended_tofts | patlak | 2cxm | 2cum
  t1_mapping_method: vfa           # vfa | look_locker
  aif_source: population           # population | detect | manual
  population_aif: parker           # parker | georgiou | fritz_hansen | weinmann | mcgrath
  save_intermediate: false
  acquisition:
    baseline_frames: 5
    relaxivity: 4.5                # mM^-1 s^-1, contrast agent r1 relaxivity
    # --- overrides: auto-detected from DICOM when available ---
    # tr: 5.0                      # ms, repetition time of the dynamic acquisition
    # flip_angles: [2, 5, 10, 15]  # degrees, VFA flip angles for T1 mapping
    # t1_assumed: 1400.0           # ms, assumed T1 when no T1 map data exists
  roi:
    enabled: false                 # set true to process only an ROI for faster iteration
    # center: [128, 128, 8]        # voxel center [x, y, z] (default: volume center)
    # radius: 10                   # radius in voxels
  fitting:
    fitter: lm                     # lm | bayesian
    max_iterations: 100
    tolerance: 1.0e-6
    r2_threshold: 0.5             # minimum R^2 for a voxel fit to be considered valid
    # bounds:                      # override model defaults (omit to use model defaults)
    #   Ktrans: [0.0, 5.0]        # [lower, upper], 1/min
    #   ve: [0.001, 1.0]          # [lower, upper], mL/100mL
    #   vp: [0.0, 0.2]            # [lower, upper], mL/100mL
    # initial_guess:               # override data-driven initial estimates
    #   Ktrans: 0.1               # 1/min
    #   ve: 0.2                   # mL/100mL
    #   vp: 0.02                  # mL/100mL
data:
  format: auto                     # auto | nifti | dicom | bids
  # mask: brain_mask.nii.gz        # tissue mask, relative to data_path or absolute
  # t1_map: t1_map.nii.gz          # pre-computed T1 map (skips T1 mapping step)
  # aif_file: aif.txt              # custom AIF (requires aif_source: manual)
  # subject: "01"                  # BIDS subject ID (required when format: bids)
  # session: "01"                  # BIDS session ID
output:
  format: nifti                    # nifti
backend:
  force_cpu: false
logging:
  level: INFO                      # DEBUG | INFO | WARNING | ERROR
""",
    "dsc": """\
modality: dsc
pipeline:
  te: 30.0                         # ms, echo time
  deconvolution_method: oSVD       # oSVD | cSVD | sSVD
  apply_leakage_correction: true
  svd_threshold: 0.2               # truncation threshold for SVD
  baseline_frames: 10              # number of pre-bolus frames for baseline
  hematocrit_ratio: 0.73           # large-to-small vessel hematocrit ratio
data:
  format: auto                     # auto | nifti | dicom | bids
  # mask: brain_mask.nii.gz        # tissue mask, relative to data_path or absolute
  # subject: "01"                  # BIDS subject ID (required when format: bids)
  # session: "01"                  # BIDS session ID
output:
  format: nifti                    # nifti
backend:
  force_cpu: false
logging:
  level: INFO                      # DEBUG | INFO | WARNING | ERROR
""",
    "asl": """\
modality: asl
pipeline:
  labeling_scheme: pcasl           # pcasl | pasl | casl
  pld: 1800.0                     # ms, post-labeling delay
  label_duration: 1800.0          # ms, labeling duration
  t1_blood: 1650.0                # ms, longitudinal relaxation time of blood
  labeling_efficiency: 0.85       # labeling efficiency (0 to 1)
  m0_method: single               # single | voxelwise | reference_region
  t1_tissue: 1330.0               # ms, longitudinal relaxation time of tissue
  partition_coefficient: 0.9      # blood-brain partition coefficient (mL/g)
  difference_method: pairwise     # pairwise | surround | mean
  label_control_order: label_first  # label_first | control_first
data:
  format: auto                     # auto | nifti | dicom | bids
  # m0_data: m0.nii.gz             # M0 calibration image (M0=1.0 if omitted)
  # mask: brain_mask.nii.gz        # tissue mask, relative to data_path or absolute
  # subject: "01"                  # BIDS subject ID (required when format: bids)
  # session: "01"                  # BIDS session ID
output:
  format: nifti                    # nifti
backend:
  force_cpu: false
logging:
  level: INFO                      # DEBUG | INFO | WARNING | ERROR
""",
    "ivim": """\
modality: ivim
pipeline:
  fitting_method: segmented        # segmented | full | bayesian
  b_threshold: 200.0              # s/mm^2, threshold separating D and D* regimes
  normalize_signal: true          # normalize to S(b=0) before fitting
  fitting:
    max_iterations: 500
    tolerance: 1.0e-6
    # bounds:                      # override model defaults (omit to use model defaults)
    #   S0: [0.0, 1.0e+10]        # signal units
    #   D: [1.0e-4, 5.0e-3]       # mm^2/s
    #   D_star: [2.0e-3, 0.1]     # mm^2/s
    #   f: [0.0, 0.7]             # dimensionless
    # initial_guess:               # override data-driven initial estimates
    #   D: 1.0e-3                 # mm^2/s
    #   D_star: 0.01              # mm^2/s
    #   f: 0.1                    # dimensionless
    # bayesian:                    # only used when fitting_method: bayesian
    #   n_samples: 1000
    #   n_burn: 200
    #   n_thin: 2
    #   prior_d: [1.0e-3, 5.0e-4]       # [mean, std] mm^2/s
    #   prior_d_star: [0.015, 0.01]      # [mean, std] mm^2/s
    #   prior_f: [0.1, 0.1]             # [mean, std]
    #   proposal_scale: 0.1
data:
  format: auto                     # auto | nifti | dicom | bids
  # b_values: [0, 10, 20, 50, 100, 200, 400, 800]  # s/mm^2 (auto-detected from DICOM/BIDS)
  # b_values_file: bvals.txt       # alternative: load b-values from text file
  # mask: brain_mask.nii.gz        # tissue mask, relative to data_path or absolute
  # subject: "01"                  # BIDS subject ID (required when format: bids)
  # session: "01"                  # BIDS session ID
output:
  format: nifti                    # nifti
backend:
  force_cpu: false
logging:
  level: INFO                      # DEBUG | INFO | WARNING | ERROR
""",
}


def dump_defaults(modality: str) -> str:
    """Generate a commented YAML template for the given modality.

    Parameters
    ----------
    modality : str
        Modality name: 'dce', 'dsc', 'asl', or 'ivim'.

    Returns
    -------
    str
        Commented YAML template string.

    Raises
    ------
    ValueError
        If modality is not recognized.
    """
    modality = modality.lower()
    if modality not in _DEFAULT_TEMPLATES:
        valid = sorted(_DEFAULT_TEMPLATES.keys())
        msg = f"Unknown modality '{modality}'. Valid: {valid}"
        raise ValueError(msg)
    return _DEFAULT_TEMPLATES[modality]
