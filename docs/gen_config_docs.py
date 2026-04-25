"""Generate YAML configuration reference docs automatically.

This script is run by mkdocs-gen-files during the build process.
It introspects the Pydantic config models from ``osipy.cli.config``
and calls runtime registries to produce an always-accurate reference
page at ``reference/cli-config.md``.
"""

from __future__ import annotations

import inspect
import re
import types
import typing
from typing import Any, get_args, get_origin

import mkdocs_gen_files
from pydantic import BaseModel

from osipy.cli.config import (
    ASLPipelineYAML,
    BackendConfig,
    BayesianIVIMFittingConfig,
    DataConfig,
    DCEAcquisitionYAML,
    DCEFittingConfig,
    DCEPipelineYAML,
    DSCPipelineYAML,
    IVIMFittingConfig,
    IVIMPipelineYAML,
    LoggingConfig,
    OutputConfig,
    PipelineConfig,
)

# ---------------------------------------------------------------------------
# Valid-value mapping: (ClassName, field_name) -> callable returning list
# ---------------------------------------------------------------------------

VALID_VALUES: dict[tuple[str, str], Any] = {
    ("PipelineConfig", "modality"): lambda: ["dce", "dsc", "asl", "ivim"],
    ("DCEPipelineYAML", "model"): lambda: _safe_registry("osipy.dce", "list_models"),
    ("DCEPipelineYAML", "t1_mapping_method"): lambda: ["vfa", "look_locker"],
    ("DCEPipelineYAML", "aif_source"): lambda: [
        "population",
        "detect",
        "manual",
    ],
    ("DCEPipelineYAML", "population_aif"): lambda: _safe_registry(
        "osipy.common.aif", "list_aifs"
    ),
    ("DSCPipelineYAML", "deconvolution_method"): lambda: _safe_registry(
        "osipy.dsc", "list_deconvolvers"
    ),
    ("ASLPipelineYAML", "labeling_scheme"): lambda: [
        "pasl",
        "casl",
        "pcasl",
    ],
    ("ASLPipelineYAML", "m0_method"): lambda: [
        "single",
        "voxelwise",
        "reference_region",
    ],
    ("ASLPipelineYAML", "label_control_order"): lambda: [
        "label_first",
        "control_first",
    ],
    ("IVIMPipelineYAML", "fitting_method"): lambda: [
        "segmented",
        "full",
        "bayesian",
    ],
    ("DCEFittingConfig", "fitter"): lambda: _safe_registry(
        "osipy.common.fitting.registry", "list_fitters"
    ),
    ("DataConfig", "format"): lambda: ["auto", "nifti", "dicom", "bids"],
    ("OutputConfig", "format"): lambda: ["nifti"],
    ("LoggingConfig", "level"): lambda: ["DEBUG", "INFO", "WARNING"],
}


def _safe_registry(module: str, func: str) -> list[str]:
    """Import *module* and call *func*, returning [] on failure."""
    try:
        mod = __import__(module, fromlist=[func])
        return getattr(mod, func)()
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Anchor / heading helpers
# ---------------------------------------------------------------------------

# Maps model class name -> markdown anchor used in the generated page.
# Populated by _render_table as headings are emitted.
_MODEL_ANCHORS: dict[str, str] = {}


def _heading_to_anchor(heading: str) -> str:
    """Convert a markdown heading to the anchor slug MkDocs generates."""
    slug = heading.lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug.strip())
    return slug


# ---------------------------------------------------------------------------
# Type formatting helpers
# ---------------------------------------------------------------------------

_TYPE_NAMES: dict[type, str] = {
    str: "string",
    float: "number",
    int: "integer",
    bool: "boolean",
}


def _format_type(annotation: Any) -> str:
    """Render a Python type annotation as a readable string."""
    if annotation is inspect.Parameter.empty or annotation is None:
        return ""

    origin = get_origin(annotation)

    # Union (e.g. str | None) — handles both typing.Union and PEP 604 X | Y
    if origin is typing.Union or isinstance(annotation, types.UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        has_none = len(args) < len(get_args(annotation))
        if len(args) == 1 and has_none:
            return f"{_format_type(args[0])} or null"
        parts = " | ".join(_format_type(a) for a in args)
        return f"{parts} or null" if has_none else parts

    # list[X]
    if origin is list:
        inner = get_args(annotation)
        if inner:
            return f"list of {_format_type(inner[0])}s"
        return "list"

    # dict[K, V]
    if origin is dict:
        return "mapping"

    # Pydantic sub-model — link to its section anchor
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        anchor = _MODEL_ANCHORS.get(annotation.__name__, annotation.__name__.lower())
        return f"[{annotation.__name__}](#{anchor})"

    return _TYPE_NAMES.get(annotation, getattr(annotation, "__name__", str(annotation)))


def _format_default(default: Any) -> str:
    """Render a default value for display."""
    from pydantic_core import PydanticUndefined

    if default is ... or default is PydanticUndefined:
        return "**required**"
    if isinstance(default, BaseModel):
        cls_name = type(default).__name__
        anchor = _MODEL_ANCHORS.get(cls_name, cls_name.lower())
        return f"*(see [{cls_name}](#{anchor}))*"
    if default is None:
        return "null"
    if isinstance(default, bool):
        return str(default).lower()
    if isinstance(default, str):
        return f'`"{default}"`'
    return f"`{default}`"


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------


def _render_table(
    model: type[BaseModel],
    *,
    heading: str,
    heading_level: int = 3,
    yaml_prefix: str = "",
) -> list[str]:
    """Render a markdown table for *model*'s fields.

    Returns a list of markdown lines.
    """
    # Register the anchor so cross-references resolve correctly
    anchor = _heading_to_anchor(heading)
    _MODEL_ANCHORS[model.__name__] = anchor

    prefix = "#" * heading_level
    lines: list[str] = [f"{prefix} {heading}", ""]
    if model.__doc__:
        lines.append(model.__doc__.strip().split("\n")[0])
        lines.append("")

    lines.append("| Field | Type | Default | Valid values |")
    lines.append("|-------|------|---------|--------------|")

    for name, field_info in model.model_fields.items():
        annotation = field_info.annotation
        type_str = _format_type(annotation)
        default_str = _format_default(field_info.default)
        yaml_key = f"`{yaml_prefix}{name}`" if yaml_prefix else f"`{name}`"

        # Look up valid values
        key = (model.__name__, name)
        if key in VALID_VALUES:
            try:
                vals = VALID_VALUES[key]()
            except Exception:
                vals = []
            valid_str = ", ".join(f"`{v}`" for v in vals) if vals else ""
        else:
            valid_str = ""

        lines.append(f"| {yaml_key} | {type_str} | {default_str} | {valid_str} |")

    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Main document assembly
# ---------------------------------------------------------------------------

# Heading constants — used by _render_table to register anchors, and
# referenced by sub-model links in _format_type / _format_default.
# Shared-section models are rendered BEFORE pipeline models so that
# cross-references from PipelineConfig fields resolve correctly.

_SHARED_HEADINGS: list[tuple[type[BaseModel], str]] = [
    (DataConfig, "`data:` (DataConfig)"),
    (OutputConfig, "`output:` (OutputConfig)"),
    (BackendConfig, "`backend:` (BackendConfig)"),
    (LoggingConfig, "`logging:` (LoggingConfig)"),
]


def generate() -> str:
    """Build the full reference page as a markdown string."""
    _MODEL_ANCHORS.clear()

    # Pre-register anchors for all models so forward-references work.
    # These will be overwritten with the same values by _render_table.
    _pre_register = [
        (PipelineConfig, "PipelineConfig"),
        *_SHARED_HEADINGS,
        (DCEPipelineYAML, "`pipeline:` (DCEPipelineYAML)"),
        (DCEAcquisitionYAML, "`pipeline.acquisition:` (DCEAcquisitionYAML)"),
        (DCEFittingConfig, "`pipeline.fitting:` (DCEFittingConfig)"),
        (DSCPipelineYAML, "`pipeline:` (DSCPipelineYAML)"),
        (ASLPipelineYAML, "`pipeline:` (ASLPipelineYAML)"),
        (IVIMPipelineYAML, "`pipeline:` (IVIMPipelineYAML)"),
        (IVIMFittingConfig, "`pipeline.fitting:` (IVIMFittingConfig)"),
        (
            BayesianIVIMFittingConfig,
            "`pipeline.fitting.bayesian:` (BayesianIVIMFittingConfig)",
        ),
    ]
    for model_cls, heading in _pre_register:
        _MODEL_ANCHORS[model_cls.__name__] = _heading_to_anchor(heading)

    doc: list[str] = []

    doc.append("# YAML Configuration Reference")
    doc.append("")
    doc.append(
        "Auto-generated from the Pydantic config models in "
        "`osipy.cli.config` and runtime registries. "
        "This page is rebuilt on every documentation build, so it always "
        "reflects the current code."
    )
    doc.append("")

    # -- Top-level --------------------------------------------------------
    doc.append("## Top-level fields")
    doc.append("")
    doc.append("Every configuration file has these top-level keys:")
    doc.append("")
    doc.extend(_render_table(PipelineConfig, heading="PipelineConfig", heading_level=3))

    # -- Shared sections --------------------------------------------------
    doc.append("## Shared sections")
    doc.append("")
    doc.append("These sections are common to all modalities.")
    doc.append("")

    for model, heading in _SHARED_HEADINGS:
        doc.extend(_render_table(model, heading=heading, heading_level=3))

    # -- DCE --------------------------------------------------------------
    doc.append("## DCE Pipeline")
    doc.append("")
    doc.append("Set `modality: dce` and configure these under `pipeline:`.")
    doc.append("")
    doc.extend(
        _render_table(
            DCEPipelineYAML,
            heading="`pipeline:` (DCEPipelineYAML)",
            heading_level=3,
        )
    )
    doc.extend(
        _render_table(
            DCEAcquisitionYAML,
            heading="`pipeline.acquisition:` (DCEAcquisitionYAML)",
            heading_level=4,
        )
    )
    doc.extend(
        _render_table(
            DCEFittingConfig,
            heading="`pipeline.fitting:` (DCEFittingConfig)",
            heading_level=4,
        )
    )

    # -- DSC --------------------------------------------------------------
    doc.append("## DSC Pipeline")
    doc.append("")
    doc.append("Set `modality: dsc` and configure these under `pipeline:`.")
    doc.append("")
    doc.extend(
        _render_table(
            DSCPipelineYAML,
            heading="`pipeline:` (DSCPipelineYAML)",
            heading_level=3,
        )
    )

    # -- ASL --------------------------------------------------------------
    doc.append("## ASL Pipeline")
    doc.append("")
    doc.append("Set `modality: asl` and configure these under `pipeline:`.")
    doc.append("")
    doc.extend(
        _render_table(
            ASLPipelineYAML,
            heading="`pipeline:` (ASLPipelineYAML)",
            heading_level=3,
        )
    )

    # -- IVIM -------------------------------------------------------------
    doc.append("## IVIM Pipeline")
    doc.append("")
    doc.append("Set `modality: ivim` and configure these under `pipeline:`.")
    doc.append("")
    doc.extend(
        _render_table(
            IVIMPipelineYAML,
            heading="`pipeline:` (IVIMPipelineYAML)",
            heading_level=3,
        )
    )
    doc.extend(
        _render_table(
            IVIMFittingConfig,
            heading="`pipeline.fitting:` (IVIMFittingConfig)",
            heading_level=4,
        )
    )
    doc.extend(
        _render_table(
            BayesianIVIMFittingConfig,
            heading="`pipeline.fitting.bayesian:` (BayesianIVIMFittingConfig)",
            heading_level=5,
        )
    )

    return "\n".join(doc)


# ---------------------------------------------------------------------------
# Entry point — called by mkdocs-gen-files
# ---------------------------------------------------------------------------

content = generate()
with mkdocs_gen_files.open("reference/cli-config.md", "w") as f:
    f.write(content)
