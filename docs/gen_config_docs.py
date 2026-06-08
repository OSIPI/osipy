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
    DataConfig,
    DCEAcquisitionYAML,
    DCEFittingConfig,
    DCEPipelineYAML,
    DSCPipelineYAML,
    IVIMPipelineYAML,
    LoggingConfig,
    OutputConfig,
    PipelineConfig,
)

# ---------------------------------------------------------------------------
# Valid-value mapping: (ClassName, field_name) -> callable returning list
# ---------------------------------------------------------------------------

# Only genuinely-flat string fields need an explicit valid-value list. The
# component-selection blocks (model, t1_mapping_method, concentration,
# population_aif, deconvolution, m0, difference, quantification, fitting,
# IVIM model) are discriminated unions whose members are documented as nested
# sub-tables, so their valid values are self-describing via the discriminator.
VALID_VALUES: dict[tuple[str, str], Any] = {
    ("PipelineConfig", "modality"): lambda: ["dce", "dsc", "asl", "ivim"],
    ("DCEPipelineYAML", "aif_source"): lambda: [
        "population",
        "detect",
        "manual",
    ],
    ("ASLPipelineYAML", "labeling_scheme"): lambda: [
        "pasl",
        "casl",
        "pcasl",
    ],
    ("ASLPipelineYAML", "label_control_order"): lambda: [
        "label_first",
        "control_first",
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


def _union_members(annotation: Any) -> list[type[BaseModel]]:
    """Return the pydantic ``MethodConfig`` members of a (possibly single) union.

    Component-selection fields are typed as a discriminated union of
    ``MethodConfig`` subclasses (or a single such class). Returns the member
    classes in a stable, name-sorted order.
    """
    origin = get_origin(annotation)
    if origin is typing.Union or isinstance(annotation, types.UnionType):
        members = [a for a in get_args(annotation) if a is not type(None)]
    elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
        members = [annotation]
    else:
        return []
    members = [m for m in members if isinstance(m, type) and issubclass(m, BaseModel)]
    return sorted(members, key=lambda m: m.__name__)


def _render_component_union(
    annotation: Any,
    *,
    field_name: str,
    discriminator: str,
    heading_level: int,
) -> list[str]:
    """Render each member of a component-selection union as a sub-table.

    Each member is a :class:`MethodConfig` whose ``discriminator`` literal
    names the selectable option; selecting it surfaces exactly that member's
    fields.
    """
    lines: list[str] = []
    for member in _union_members(annotation):
        # The discriminator literal value identifies the selectable option.
        disc_field = member.model_fields.get(discriminator)
        choice = ""
        if disc_field is not None:
            choice_args = get_args(disc_field.annotation)
            if choice_args:
                choice = str(choice_args[0])
        title = (
            f"`pipeline.{field_name}` with `{discriminator}: {choice}`"
            if choice
            else f"`pipeline.{field_name}` ({member.__name__})"
        )
        lines.extend(_render_table(member, heading=title, heading_level=heading_level))
    return lines


def _render_modality_components(parent: type[BaseModel]) -> list[str]:
    """Render every component-selection sub-block for a modality pipeline model.

    Each selectable component is a discriminated union; its members are rendered
    as nested sub-tables (heading level 4) so the reference mirrors the nested
    YAML shape.
    """
    lines: list[str] = []
    for field_name, discriminator in _COMPONENT_FIELDS.get(parent, []):
        annotation = parent.model_fields[field_name].annotation
        lines.extend(
            _render_component_union(
                annotation,
                field_name=field_name,
                discriminator=discriminator,
                heading_level=4,
            )
        )
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

# Per-modality component-selection fields that are discriminated unions of
# ``MethodConfig`` members. Each entry maps the parent pipeline model to a list
# of ``(field_name, discriminator)`` pairs; each member is rendered as its own
# sub-table so selecting a method/mode/model surfaces exactly its parameters.
_COMPONENT_FIELDS: dict[type[BaseModel], list[tuple[str, str]]] = {
    DCEPipelineYAML: [
        ("model", "method"),
        ("t1_mapping_method", "method"),
        ("concentration", "method"),
        ("population_aif", "name"),
    ],
    DSCPipelineYAML: [("deconvolution", "method")],
    ASLPipelineYAML: [
        ("m0", "method"),
        ("difference", "method"),
        ("quantification", "mode"),
    ],
    IVIMPipelineYAML: [("fitting", "method"), ("model", "model")],
}


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
    ]
    for model_cls, heading in _pre_register:
        _MODEL_ANCHORS[model_cls.__name__] = _heading_to_anchor(heading)

    # Pre-register anchors for every component-union member so that the
    # ``A | B | C`` type links rendered in the parent table resolve to the
    # sub-tables emitted later.
    for parent, fields in _COMPONENT_FIELDS.items():
        for field_name, discriminator in fields:
            annotation = parent.model_fields[field_name].annotation
            for member in _union_members(annotation):
                disc_field = member.model_fields.get(discriminator)
                choice = ""
                if disc_field is not None:
                    choice_args = get_args(disc_field.annotation)
                    if choice_args:
                        choice = str(choice_args[0])
                title = (
                    f"`pipeline.{field_name}` with `{discriminator}: {choice}`"
                    if choice
                    else f"`pipeline.{field_name}` ({member.__name__})"
                )
                _MODEL_ANCHORS[member.__name__] = _heading_to_anchor(title)

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
    doc.extend(_render_modality_components(DCEPipelineYAML))

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
    doc.extend(_render_modality_components(DSCPipelineYAML))

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
    doc.extend(_render_modality_components(ASLPipelineYAML))

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
    doc.extend(_render_modality_components(IVIMPipelineYAML))

    return "\n".join(doc)


# ---------------------------------------------------------------------------
# Entry point — called by mkdocs-gen-files
# ---------------------------------------------------------------------------

content = generate()
with mkdocs_gen_files.open("reference/cli-config.md", "w") as f:
    f.write(content)
