"""Pydantic v2 models for YAML pipeline configuration.

Provides validation models for each modality (DCE, DSC, ASL, IVIM),
a top-level ``PipelineConfig`` model, ``load_config()`` for parsing
YAML files, and ``dump_defaults()`` for generating commented templates.

The ``dump_defaults()`` function auto-generates YAML from Pydantic model
introspection, so every ``Field(description=...)`` annotation automatically
appears in the output.  This eliminates drift between the config schema
and the dumped templates.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared configuration sections
# ---------------------------------------------------------------------------


class DataConfig(BaseModel):
    """Data loading configuration."""

    format: str = Field(
        default="auto", description="data format (auto | nifti | dicom | bids)"
    )
    mask: str | None = Field(default=None, description="tissue mask path")
    t1_map: str | None = Field(
        default=None, description="pre-computed T1 map (DCE only)"
    )
    aif_file: str | None = Field(
        default=None, description="custom AIF file (requires aif_source: manual)"
    )
    m0_data: str | None = Field(
        default=None, description="M0 calibration image (ASL only)"
    )
    b_values: list[float] | None = Field(
        default=None, description="b-values in s/mm^2 (IVIM only)"
    )
    b_values_file: str | None = Field(
        default=None, description="path to b-values text file"
    )
    subject: str | None = Field(default=None, description="BIDS subject ID")
    session: str | None = Field(default=None, description="BIDS session ID")


class OutputConfig(BaseModel):
    """Output configuration."""

    format: str = Field(default="nifti", description="output format")


class BackendConfig(BaseModel):
    """GPU/CPU backend configuration."""

    force_cpu: bool = Field(
        default=False, description="force CPU even if GPU is available"
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(
        default="INFO", description="log level (DEBUG | INFO | WARNING | ERROR)"
    )


# ---------------------------------------------------------------------------
# Fitting configuration models
# ---------------------------------------------------------------------------


class DCEFittingConfig(BaseModel):
    """DCE model fitting configuration from YAML."""

    fitter: str = Field(default="lm", description="fitting method (lm | bayesian)")
    max_iterations: int = Field(default=100, description="maximum LM iterations")
    tolerance: float = Field(default=1e-6, description="convergence tolerance")
    r2_threshold: float = Field(
        default=0.5, description="minimum R^2 for a valid voxel fit"
    )
    bounds: dict[str, list[float]] | None = Field(
        default=None,
        description="per-parameter bounds override {name: [lower, upper]}",
    )
    initial_guess: dict[str, float] | None = Field(
        default=None,
        description="per-parameter initial guess override {name: value}",
    )

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

    prior_scale: float = Field(
        default=1.5, description="prior distribution scale factor"
    )
    noise_std: float | None = Field(
        default=None, description="noise standard deviation (null for auto-estimate)"
    )
    compute_uncertainty: bool = Field(
        default=True, description="compute parameter uncertainty estimates"
    )


class IVIMFittingConfig(BaseModel):
    """IVIM model fitting configuration from YAML."""

    max_iterations: int = Field(default=500, description="maximum iterations")
    tolerance: float = Field(default=1e-6, description="convergence tolerance")
    bounds: dict[str, list[float]] | None = Field(
        default=None,
        description="per-parameter bounds override {name: [lower, upper]}",
    )
    initial_guess: dict[str, float] | None = Field(
        default=None,
        description="per-parameter initial guess override {name: value}",
    )
    bayesian: BayesianIVIMFittingConfig = Field(
        default_factory=BayesianIVIMFittingConfig,
        description="Bayesian fitting options (used when fitting_method: bayesian)",
    )

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

    tr: float | None = Field(default=None, description="repetition time (ms)")
    flip_angles: list[float] | None = Field(
        default=None, description="VFA flip angles (degrees)"
    )
    baseline_frames: int = Field(
        default=5, description="number of pre-contrast baseline frames"
    )
    relaxivity: float = Field(
        default=4.5, description="contrast agent r1 relaxivity (mM^-1 s^-1)"
    )
    t1_assumed: float | None = Field(
        default=None,
        description="assumed T1 when no T1 map is available (ms)",
    )


class ROIConfig(BaseModel):
    """Region-of-interest configuration for limiting processing."""

    enabled: bool = Field(default=False, description="set true to process only an ROI")
    center: list[int] | None = Field(default=None, description="voxel center [x, y, z]")
    radius: int = Field(default=10, description="radius in voxels")


class DCEPipelineYAML(BaseModel):
    """DCE pipeline settings from YAML."""

    model: str = Field(
        default="extended_tofts",
        description="PK model (tofts | extended_tofts | patlak | 2cxm | 2cum)",
    )
    t1_mapping_method: str = Field(
        default="vfa", description="T1 mapping method (vfa | look_locker)"
    )
    aif_source: str = Field(
        default="population", description="AIF source (population | detect | manual)"
    )
    population_aif: str = Field(
        default="parker",
        description=(
            "population AIF model"
            " (parker | georgiou | fritz_hansen | weinmann | mcgrath)"
        ),
    )
    save_intermediate: bool = Field(
        default=False, description="save intermediate results"
    )
    acquisition: DCEAcquisitionYAML = Field(
        default_factory=DCEAcquisitionYAML,
        description="acquisition parameters",
    )
    roi: ROIConfig = Field(
        default_factory=ROIConfig,
        description="region-of-interest settings",
    )
    fitting: DCEFittingConfig = Field(
        default_factory=DCEFittingConfig,
        description="model fitting settings",
    )

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

    te: float = Field(default=30.0, description="echo time (ms)")
    deconvolution_method: str = Field(
        default="oSVD", description="deconvolution method (oSVD | cSVD | sSVD)"
    )
    apply_leakage_correction: bool = Field(
        default=True, description="apply BSW leakage correction"
    )
    svd_threshold: float = Field(default=0.2, description="SVD truncation threshold")
    baseline_frames: int = Field(
        default=10, description="number of pre-bolus baseline frames"
    )
    hematocrit_ratio: float = Field(
        default=0.73, description="large-to-small vessel hematocrit ratio"
    )

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

    labeling_scheme: str = Field(
        default="pcasl", description="labeling scheme (pcasl | pasl | casl)"
    )
    pld: float = Field(default=1800.0, description="post-labeling delay (ms)")
    label_duration: float = Field(default=1800.0, description="labeling duration (ms)")
    t1_blood: float = Field(default=1650.0, description="T1 of blood (ms)")
    labeling_efficiency: float = Field(
        default=0.85, description="labeling efficiency (0 to 1)"
    )
    m0_method: str = Field(
        default="single",
        description="M0 calibration method (single | voxelwise | reference_region)",
    )
    t1_tissue: float = Field(default=1330.0, description="T1 of tissue (ms)")
    partition_coefficient: float = Field(
        default=0.9, description="blood-brain partition coefficient (mL/g)"
    )
    difference_method: str = Field(
        default="pairwise",
        description="ASL difference method (pairwise | surround | mean)",
    )
    label_control_order: str = Field(
        default="label_first",
        description="label/control ordering (label_first | control_first)",
    )

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

    fitting_method: str = Field(
        default="segmented",
        description="fitting method (segmented | full | bayesian)",
    )
    b_threshold: float = Field(
        default=200.0,
        description="b-value threshold separating D and D* regimes (s/mm^2)",
    )
    normalize_signal: bool = Field(
        default=True, description="normalize signal to S(b=0)"
    )
    fitting: IVIMFittingConfig = Field(
        default_factory=IVIMFittingConfig,
        description="fitting options",
    )

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
# dump_defaults — auto-generated from Pydantic model introspection
# ---------------------------------------------------------------------------


def _is_model_class(annotation: Any) -> bool:
    """Return True if *annotation* is a Pydantic BaseModel subclass."""
    try:
        return isinstance(annotation, type) and issubclass(annotation, BaseModel)
    except TypeError:
        return False


def _format_yaml_value(value: Any) -> str:
    """Format a Python value as a YAML scalar string."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        s = repr(value)
        if "e" in s or "E" in s:
            # Normalise exponent: 1e-06 -> 1.0e-6
            parts = s.lower().split("e")
            if "." not in parts[0]:
                parts[0] += ".0"
            exp_val = int(parts[1])
            parts[1] = str(exp_val)
            s = "e".join(parts)
        elif "." not in s:
            s += ".0"
        return s
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        # Quote strings that YAML would misinterpret
        if value.lower() in ("true", "false", "null", "yes", "no", "on", "off"):
            return f'"{value}"'
        return value
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        items = ", ".join(_format_yaml_value(v) for v in value)
        return f"[{items}]"
    return str(value)


def _generate_section_yaml(
    model_cls: type[BaseModel],
    indent_level: int = 0,
) -> list[str]:
    """Generate commented YAML lines from a Pydantic model's fields.

    Active fields (non-None defaults) are emitted first, followed by
    commented-out optional fields (None defaults).  Nested Pydantic
    sub-models are recursed into and rendered as YAML sub-sections.
    ``Field(description=...)`` values appear as inline ``# comments``.
    """
    indent = "  " * indent_level
    active_lines: list[str] = []
    commented_lines: list[str] = []

    for field_name, field_info in model_cls.model_fields.items():
        # Resolve the effective default value
        default = field_info.default
        # Pydantic v2 stores mutable defaults via default_factory
        try:
            from pydantic_core import PydanticUndefinedType

            if isinstance(default, PydanticUndefinedType):
                if field_info.default_factory is not None:
                    default = field_info.default_factory()
                else:
                    default = None
        except ImportError:
            pass

        description = field_info.description
        annotation = field_info.annotation
        comment_suffix = f"  # {description}" if description else ""

        if _is_model_class(annotation):
            # Nested Pydantic model — render as a YAML sub-section
            section_lines = [f"{indent}{field_name}:{comment_suffix}"]
            section_lines.extend(_generate_section_yaml(annotation, indent_level + 1))
            active_lines.extend(section_lines)
        elif default is None:
            # Optional field — emit as a commented-out line
            commented_lines.append(f"{indent}# {field_name}: null{comment_suffix}")
        else:
            yaml_val = _format_yaml_value(default)
            active_lines.append(f"{indent}{field_name}: {yaml_val}{comment_suffix}")

    return active_lines + commented_lines


def dump_defaults(modality: str) -> str:
    """Generate a commented YAML template for the given modality.

    The template is auto-generated from Pydantic model field definitions,
    so it always reflects the current schema.  Fields with non-None
    defaults are emitted as active YAML; optional fields (default None)
    are shown as commented-out lines.  ``Field(description=...)`` values
    appear as inline comments.

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
    if modality not in _MODALITY_PIPELINE_MODELS:
        valid = sorted(_MODALITY_PIPELINE_MODELS.keys())
        msg = f"Unknown modality '{modality}'. Valid: {valid}"
        raise ValueError(msg)

    pipeline_cls = _MODALITY_PIPELINE_MODELS[modality]

    lines = [f"modality: {modality}"]

    # Pipeline section (modality-specific)
    lines.append("pipeline:")
    lines.extend(_generate_section_yaml(pipeline_cls, indent_level=1))

    # Shared sections
    for section_name, section_cls in [
        ("data", DataConfig),
        ("output", OutputConfig),
        ("backend", BackendConfig),
        ("logging", LoggingConfig),
    ]:
        lines.append(f"{section_name}:")
        lines.extend(_generate_section_yaml(section_cls, indent_level=1))

    return "\n".join(lines) + "\n"
