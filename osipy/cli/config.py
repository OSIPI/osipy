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
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared configuration sections
# ---------------------------------------------------------------------------


class DataConfig(BaseModel):
    """Data loading configuration."""

    format: str = Field(
        default="auto",
        description="auto | nifti | dicom | bids",
    )
    mask: str | None = Field(
        default=None,
        description="tissue mask, relative to data_path or absolute",
        examples=["brain_mask.nii.gz"],
    )
    t1_map: str | None = Field(
        default=None,
        description="pre-computed T1 map (skips T1 mapping step)",
        examples=["t1_map.nii.gz"],
    )
    aif_file: str | None = Field(
        default=None,
        description="custom AIF (requires aif_source: manual)",
        examples=["aif.txt"],
    )
    m0_data: str | None = Field(
        default=None,
        description="M0 calibration image (M0=1.0 if omitted)",
        examples=["m0.nii.gz"],
    )
    b_values: list[float] | None = Field(
        default=None,
        description="s/mm^2 (auto-detected from DICOM/BIDS)",
        examples=[[0, 10, 20, 50, 100, 200, 400, 800]],
    )
    b_values_file: str | None = Field(
        default=None,
        description="alternative: load b-values from text file",
        examples=["bvals.txt"],
    )
    subject: str | None = Field(
        default=None,
        description="BIDS subject ID (required when format: bids)",
        examples=["01"],
    )
    session: str | None = Field(
        default=None,
        description="BIDS session ID",
        examples=["01"],
    )


class OutputConfig(BaseModel):
    """Output configuration."""

    format: str = Field(default="nifti", description="nifti")


class BackendConfig(BaseModel):
    """GPU/CPU backend configuration."""

    force_cpu: bool = Field(
        default=False,
        description="force CPU execution even if GPU is available",
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO", description="DEBUG | INFO | WARNING | ERROR")


# ---------------------------------------------------------------------------
# Fitting configuration models
# ---------------------------------------------------------------------------


class DCEFittingConfig(BaseModel):
    """DCE model fitting configuration from YAML."""

    fitter: str = Field(default="lm", description="lm | bayesian")
    max_iterations: int = Field(default=100)
    tolerance: float = Field(default=1e-6)
    r2_threshold: float = Field(
        default=0.5,
        description="minimum R^2 for a voxel fit to be considered valid",
    )
    fit_delay: bool = Field(
        default=False,
        description=(
            "jointly fit an arterial delay parameter with the DCE model "
            "(adds one parameter per voxel)"
        ),
    )
    bounds: dict[str, list[float]] | None = Field(
        default=None,
        description="override model defaults (omit to use model defaults)",
        json_schema_extra={
            "yaml_example": (
                "Ktrans: [0.0, 5.0]        # [lower, upper], 1/min\n"
                "ve: [0.001, 1.0]          # [lower, upper], fraction [0, 1]\n"
                "vp: [0.0, 0.2]            # [lower, upper], fraction [0, 1]"
            )
        },
    )
    initial_guess: dict[str, float] | None = Field(
        default=None,
        description="override data-driven initial estimates",
        json_schema_extra={
            "yaml_example": (
                "Ktrans: 0.1               # 1/min\n"
                "ve: 0.2                   # fraction [0, 1]\n"
                "vp: 0.02                  # fraction [0, 1]"
            )
        },
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

    prior_scale: float = Field(default=1.5)
    noise_std: float | None = Field(default=None, examples=[0.01])
    compute_uncertainty: bool = Field(default=True)


class IVIMFittingConfig(BaseModel):
    """IVIM model fitting configuration from YAML."""

    max_iterations: int = Field(default=500)
    tolerance: float = Field(default=1e-6)
    bounds: dict[str, list[float]] | None = Field(
        default=None,
        description="override model defaults (omit to use model defaults)",
        json_schema_extra={
            "yaml_example": (
                "S0: [0.0, 1.0e+10]        # signal units\n"
                "D: [1.0e-4, 5.0e-3]       # mm^2/s\n"
                "D_star: [2.0e-3, 0.1]     # mm^2/s\n"
                "f: [0.0, 0.7]             # dimensionless"
            )
        },
    )
    initial_guess: dict[str, float] | None = Field(
        default=None,
        description="override data-driven initial estimates",
        json_schema_extra={
            "yaml_example": (
                "D: 1.0e-3                 # mm^2/s\n"
                "D_star: 0.01              # mm^2/s\n"
                "f: 0.1                    # dimensionless"
            )
        },
    )
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

    baseline_frames: int = Field(default=5)
    relaxivity: float = Field(
        default=4.5,
        description="mM^-1 s^-1, contrast agent r1 relaxivity",
    )
    # Overrides: auto-detected from DICOM when available.
    tr: float | None = Field(
        default=None,
        description="ms, repetition time of the dynamic acquisition",
        examples=[5.0],
    )
    flip_angles: list[float] | None = Field(
        default=None,
        description="degrees, VFA flip angles for T1 mapping",
        examples=[[2, 5, 10, 15]],
    )
    t1_assumed: float | None = Field(
        default=None,
        description="ms, assumed T1 when no T1 map data exists",
        examples=[1400.0],
    )


class DCEPipelineYAML(BaseModel):
    """DCE pipeline settings from YAML."""

    model: str = Field(
        default="extended_tofts",
        description="tofts | extended_tofts | patlak | 2cxm | 2cum",
    )
    t1_mapping_method: str = Field(default="vfa", description="vfa | look_locker")
    aif_source: str = Field(
        default="population", description="population | detect | manual"
    )
    population_aif: str = Field(
        default="parker",
        description="parker | georgiou | fritz_hansen | weinmann | mcgrath",
    )
    save_intermediate: bool = Field(default=False)
    acquisition: DCEAcquisitionYAML = DCEAcquisitionYAML()
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

    te: float = Field(default=30.0, description="ms, echo time")
    deconvolution_method: str = Field(default="oSVD", description="oSVD | cSVD | sSVD")
    apply_leakage_correction: bool = Field(default=True)
    svd_threshold: float = Field(
        default=0.2, description="truncation threshold for SVD"
    )
    baseline_frames: int = Field(
        default=10, description="number of pre-bolus frames for baseline"
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

    labeling_scheme: str = Field(default="pcasl", description="pcasl | pasl | casl")
    pld: float = Field(default=1800.0, description="ms, post-labeling delay")
    label_duration: float = Field(default=1800.0, description="ms, labeling duration")
    t1_blood: float = Field(
        default=1650.0, description="ms, longitudinal relaxation time of blood"
    )
    labeling_efficiency: float = Field(
        default=0.85, description="labeling efficiency (0 to 1)"
    )
    m0_method: str = Field(
        default="single", description="single | voxelwise | reference_region"
    )
    t1_tissue: float = Field(
        default=1330.0, description="ms, longitudinal relaxation time of tissue"
    )
    partition_coefficient: float = Field(
        default=0.9, description="blood-brain partition coefficient (mL/g)"
    )
    difference_method: str = Field(
        default="pairwise", description="pairwise | surround | mean"
    )
    label_control_order: str = Field(
        default="label_first", description="label_first | control_first"
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
        default="segmented", description="segmented | full | bayesian"
    )
    b_threshold: float = Field(
        default=200.0,
        description="s/mm^2, threshold separating D and D* regimes",
    )
    normalize_signal: bool = Field(
        default=True, description="normalize to S(b=0) before fitting"
    )
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

    # Read bytes so we can detect/strip a UTF-16 BOM or embedded null bytes
    # from configs written on Windows with older osipy versions (those
    # wrote files using the platform default encoding rather than UTF-8).
    raw_bytes = config_path.read_bytes()
    if raw_bytes.startswith((b"\xff\xfe", b"\xfe\xff")) or b"\x00" in raw_bytes[:1024]:
        msg = (
            f"Config file {config_path} is not valid UTF-8 "
            "(detected UTF-16 BOM or embedded null bytes). This is a known "
            "Windows issue with configs generated by older osipy versions. "
            "Regenerate the file with `osipy --help-me-pls` or re-encode it: "
            "`python -c 'import pathlib, sys; p=pathlib.Path(sys.argv[1]); "
            'p.write_text(p.read_text(encoding="utf-16"), encoding="utf-8")\' '
            f"{config_path}`"
        )
        raise ValueError(msg)

    raw = yaml.safe_load(raw_bytes.decode("utf-8"))

    if not isinstance(raw, dict):
        msg = f"Config file must contain a YAML mapping, got {type(raw).__name__}"
        raise ValueError(msg)

    config = PipelineConfig(**raw)
    # Eagerly validate the modality-specific pipeline config
    config.get_modality_config()
    return config


# ---------------------------------------------------------------------------
# dump_defaults — generate commented templates from pydantic model metadata
# ---------------------------------------------------------------------------


def _yaml_scalar(value: Any) -> str:
    """Render a Python scalar as a YAML-compatible inline string."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        # Quote strings that could be ambiguous YAML (e.g. "01").
        if value and (value.isdigit() or value.startswith("0")):
            return f'"{value}"'
        return value
    if isinstance(value, float):
        # YAML requires a decimal or exponent marker for floats.
        text = f"{value:.6g}"
        if "." not in text and "e" not in text and "E" not in text:
            text += ".0"
        return text
    if isinstance(value, list):
        return "[" + ", ".join(_yaml_scalar(x) for x in value) + "]"
    return str(value)


def _render_model_yaml(model: BaseModel, indent: int = 0) -> list[str]:
    """Recursively render a pydantic model instance as commented YAML lines.

    For each field on ``model``:

    * If the value is a nested ``BaseModel``, emit ``name:`` and recurse.
    * If the default value is not ``None``, emit ``name: value  # description``.
    * If the default is ``None``, emit a commented-out example line using
      the field's ``examples=[...]`` metadata, or — for complex dict fields —
      use the multi-line block in ``json_schema_extra["yaml_example"]``.

    Parameters
    ----------
    model : BaseModel
        Populated pydantic model (use defaults via ``ModelCls()``).
    indent : int
        Number of leading spaces (for nesting).

    Returns
    -------
    list[str]
        YAML lines (without trailing newlines).
    """
    pad = " " * indent
    lines: list[str] = []
    for name, field in type(model).model_fields.items():
        description = (field.description or "").strip()
        extra = field.json_schema_extra or {}
        value = getattr(model, name)

        # Nested pydantic model: recurse.
        if isinstance(value, BaseModel):
            lines.append(f"{pad}{name}:")
            lines.extend(_render_model_yaml(value, indent=indent + 2))
            continue

        # Commented-out optional field with a multi-line example block.
        if value is None and isinstance(extra, dict) and "yaml_example" in extra:
            header = f"{pad}# {name}:"
            if description:
                header += f"  # {description}"
            lines.append(header)
            for sub in str(extra["yaml_example"]).splitlines():
                lines.append(f"{pad}#   {sub}" if sub else f"{pad}#")
            continue

        # Commented-out optional field with a single example value.
        if value is None:
            example = None
            examples = getattr(field, "examples", None)
            if examples:
                example = examples[0]
            rendered_example = (
                _yaml_scalar(example) if example is not None else "<value>"
            )
            line = f"{pad}# {name}: {rendered_example}"
            if description:
                line += f"  # {description}"
            lines.append(line)
            continue

        # Normal field with a default value.
        line = f"{pad}{name}: {_yaml_scalar(value)}"
        if description:
            line += f"  # {description}"
        lines.append(line)

    return lines


def dump_defaults(modality: str) -> str:
    """Generate a commented YAML template for the given modality.

    The template is generated by introspecting the pydantic models used to
    validate each section — adding a field to a model (with
    ``Field(description=..., examples=[...])``) automatically surfaces it
    in the rendered output with a matching inline comment.

    Parameters
    ----------
    modality : str
        Modality name: ``'dce'``, ``'dsc'``, ``'asl'``, or ``'ivim'``.

    Returns
    -------
    str
        Commented YAML template string (ends with a trailing newline).

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

    lines: list[str] = []
    lines.append(f"modality: {modality}")
    lines.append("pipeline:")
    lines.extend(_render_model_yaml(pipeline_cls(), indent=2))
    lines.append("data:")
    lines.extend(_render_model_yaml(DataConfig(), indent=2))
    lines.append("output:")
    lines.extend(_render_model_yaml(OutputConfig(), indent=2))
    lines.append("backend:")
    lines.extend(_render_model_yaml(BackendConfig(), indent=2))
    lines.append("logging:")
    lines.extend(_render_model_yaml(LoggingConfig(), indent=2))

    return "\n".join(lines) + "\n"
