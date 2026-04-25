"""Interactive configuration wizard for osipy.

Walks the user through an interactive Q&A to produce a validated YAML
config file.  Invoked via ``osipy --help-me-pls``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Prompt utilities
# ---------------------------------------------------------------------------

_MODALITY_DESCRIPTIONS: dict[str, str] = {
    "dce": "DCE-MRI  - Dynamic Contrast-Enhanced (pharmacokinetic modeling)",
    "dsc": "DSC-MRI  - Dynamic Susceptibility Contrast (deconvolution-based perfusion)",
    "asl": "ASL      - Arterial Spin Labeling (CBF quantification, no contrast agent)",
    "ivim": "IVIM     - Intravoxel Incoherent Motion (bi-exponential DWI fitting)",
}

_DATA_FORMAT_OPTIONS = ["auto", "nifti", "dicom", "bids"]

# Inline comments for generated YAML lines
_PIPELINE_COMMENTS: dict[str, str] = {
    "model": "pharmacokinetic model",
    "t1_mapping_method": "T1 mapping method",
    "aif_source": "AIF source",
    "population_aif": "population AIF model",
    "te": "ms, echo time",
    "deconvolution_method": "deconvolution method",
    "apply_leakage_correction": "correct for contrast agent leakage",
    "svd_threshold": "truncation threshold for SVD",
    "hematocrit_ratio": "large-to-small vessel hematocrit ratio",
    "labeling_scheme": "ASL labeling scheme",
    "pld": "ms, post-labeling delay",
    "label_duration": "ms, labeling duration",
    "t1_blood": "ms, longitudinal relaxation of blood",
    "labeling_efficiency": "labeling efficiency (0 to 1)",
    "m0_method": "M0 calibration method",
    "t1_tissue": "ms, longitudinal relaxation of tissue",
    "partition_coefficient": "blood-brain partition coefficient (mL/g)",
    "difference_method": "label-control subtraction method",
    "label_control_order": "label/control ordering",
    "fitting_method": "IVIM fitting method",
    "b_threshold": "s/mm^2, threshold separating D and D* regimes",
    "normalize_signal": "normalize to S(b=0) before fitting",
    "baseline_frames": "number of pre-bolus baseline frames",
    "relaxivity": "mM^-1 s^-1, contrast agent r1 relaxivity",
    "t1_assumed": "ms, assumed T1 when no T1 data is available",
}


def _check_tty() -> None:
    """Exit with a helpful message if stdin is not a terminal."""
    if not sys.stdin.isatty():
        print(
            "Error: --help-me-pls requires an interactive terminal.\n"
            "Use --dump-defaults MODALITY to generate a config non-interactively.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def _prompt_choice(
    prompt: str,
    options: list[str],
    descriptions: list[str] | None = None,
    default: str | None = None,
) -> str:
    """Show a numbered menu and return the selected option string.

    Accepts a number (1-based) or the option name. Enter selects the
    default when one is provided.
    """
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        marker = " (default)" if opt == default else ""
        desc = f" - {descriptions[i - 1]}" if descriptions else ""
        print(f"  [{i}] {opt}{desc}{marker}")

    while True:
        try:
            hint = f" [{default}]" if default else ""
            raw = input(f"  Choice{hint}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit(130) from None

        if raw == "" and default is not None:
            return default

        # Accept by number
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]

        # Accept by name (case-insensitive)
        lower = raw.lower()
        for opt in options:
            if opt.lower() == lower:
                return opt

        print(f"  Invalid choice. Enter 1-{len(options)} or a name from the list.")


def _prompt_value(
    prompt: str,
    default: Any = None,
    expected_type: type = str,
    required: bool = False,
) -> Any:
    """Prompt for a single value with type coercion.

    Returns *None* when the user presses Enter on an optional field.
    """
    while True:
        try:
            hint = f" [{default}]" if default is not None else ""
            raw = input(f"  {prompt}{hint}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit(130) from None

        if raw == "":
            if default is not None:
                return default
            if not required:
                return None
            print("  A value is required.")
            continue

        try:
            if expected_type is bool:
                if raw.lower() in ("true", "yes", "y", "1"):
                    return True
                if raw.lower() in ("false", "no", "n", "0"):
                    return False
                print("  Enter yes or no.")
                continue
            return expected_type(raw)
        except (ValueError, TypeError):
            print(f"  Could not parse as {expected_type.__name__}. Try again.")


def _prompt_yes_no(prompt: str, default: bool = True) -> bool:
    """Prompt for a yes/no answer."""
    hint = "Y/n" if default else "y/N"
    while True:
        try:
            raw = input(f"  {prompt} [{hint}]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit(130) from None

        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Enter y or n.")


def _prompt_path(
    prompt: str,
    must_exist: bool = False,
    default: str | None = None,
) -> str | None:
    """Prompt for a file path.  Returns *None* when skipped."""
    while True:
        try:
            hint = f" [{default}]" if default else " (Enter to skip)"
            raw = input(f"  {prompt}{hint}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit(130) from None

        if raw == "":
            return default

        if must_exist and not Path(raw).exists():
            print(f"  File not found: {raw}")
            continue

        return raw


# ---------------------------------------------------------------------------
# Per-modality collectors
# ---------------------------------------------------------------------------


def _collect_data_config(
    modality: str, pipeline_config: dict[str, Any]
) -> dict[str, Any]:
    """Collect the ``data:`` section (shared, with modality-aware fields).

    Parameters
    ----------
    modality : str
        Selected modality.
    pipeline_config : dict[str, Any]
        Already-collected pipeline settings.  Used to decide which
        optional data prompts to show (e.g. T1 map only when the user
        has T1 data, AIF file only when ``aif_source`` is ``"manual"``).
    """
    print("\n--- Data Configuration ---")
    cfg: dict[str, Any] = {}

    cfg["format"] = _prompt_choice(
        "Data format:",
        _DATA_FORMAT_OPTIONS,
        default="auto",
    )

    mask = _prompt_path("Tissue mask path")
    if mask:
        cfg["mask"] = mask

    if cfg["format"] == "bids":
        cfg["subject"] = _prompt_value("BIDS subject ID", required=True)
        session = _prompt_value("BIDS session ID")
        if session:
            cfg["session"] = session

    # --- Modality-specific data fields ---
    if modality == "dce":
        # Only ask for T1 map when the user has T1 data (no t1_assumed)
        acq = pipeline_config.get("acquisition", {})
        if "t1_assumed" not in acq:
            t1_map = _prompt_path(
                "Pre-computed T1 map path (skip if raw T1 data is in the data dir)"
            )
            if t1_map:
                cfg["t1_map"] = t1_map

        # Only ask for AIF file when the user chose manual AIF
        if pipeline_config.get("aif_source") == "manual":
            aif_file = _prompt_path("Custom AIF file path", must_exist=False)
            if aif_file:
                cfg["aif_file"] = aif_file

    elif modality == "asl":
        m0 = _prompt_path("M0 calibration image path")
        if m0:
            cfg["m0_data"] = m0
        else:
            print("  Note: Without M0 data, CBF will be in relative units (M0=1).")

    elif modality == "ivim":
        b_source = _prompt_choice(
            "B-values source:",
            ["auto", "inline", "file"],
            default="auto",
        )
        if b_source == "file":
            bfile = _prompt_path("B-values file path")
            if bfile:
                cfg["b_values_file"] = bfile
        elif b_source == "inline":
            raw = _prompt_value(
                "B-values (comma-separated, s/mm^2)",
                required=True,
            )
            cfg["b_values"] = [float(x) for x in str(raw).split(",")]

    return cfg


def _collect_dce_config() -> dict[str, Any]:
    """Collect DCE pipeline settings."""
    from osipy.common.aif.population import list_aifs
    from osipy.dce import list_models
    from osipy.dce.t1_mapping.registry import list_t1_methods

    print("\n--- DCE Pipeline Settings ---")
    cfg: dict[str, Any] = {}

    models = list_models()
    cfg["model"] = _prompt_choice(
        "Pharmacokinetic model:", models, default="extended_tofts"
    )

    # T1 data availability determines whether we do T1 mapping or use
    # an assumed value.  This also controls what the data section asks for.
    has_t1_data = _prompt_yes_no(
        "Do you have T1-weighted data (VFA flip angles or Look-Locker)?",
        default=True,
    )

    acquisition: dict[str, Any] = {}

    if has_t1_data:
        t1_methods = list_t1_methods()
        cfg["t1_mapping_method"] = _prompt_choice(
            "T1 mapping method:", t1_methods, default="vfa"
        )
    else:
        # No T1 data — use an assumed value and skip T1 mapping config
        acquisition["t1_assumed"] = _prompt_value(
            "Assumed T1 value (ms)", default=1400.0, expected_type=float
        )

    # AIF source
    aif_sources = ["population", "detect", "manual"]
    cfg["aif_source"] = _prompt_choice("AIF source:", aif_sources, default="population")

    if cfg["aif_source"] == "population":
        aifs = list_aifs()
        cfg["population_aif"] = _prompt_choice(
            "Population AIF:", aifs, default="parker"
        )

    # Acquisition parameters
    acquisition["baseline_frames"] = _prompt_value(
        "Number of baseline frames", default=5, expected_type=int
    )
    acquisition["relaxivity"] = _prompt_value(
        "Contrast agent relaxivity (mM^-1 s^-1)",
        default=4.5,
        expected_type=float,
    )

    cfg["acquisition"] = acquisition

    return cfg


def _collect_dsc_config() -> dict[str, Any]:
    """Collect DSC pipeline settings."""
    from osipy.dsc import list_deconvolvers

    print("\n--- DSC Pipeline Settings ---")
    cfg: dict[str, Any] = {}

    methods = list_deconvolvers()
    cfg["deconvolution_method"] = _prompt_choice(
        "Deconvolution method:", methods, default="oSVD"
    )

    cfg["te"] = _prompt_value("Echo time TE (ms)", default=30.0, expected_type=float)
    cfg["apply_leakage_correction"] = _prompt_yes_no(
        "Apply leakage correction?", default=True
    )
    cfg["svd_threshold"] = _prompt_value(
        "SVD truncation threshold", default=0.2, expected_type=float
    )
    cfg["baseline_frames"] = _prompt_value(
        "Number of baseline frames", default=10, expected_type=int
    )
    cfg["hematocrit_ratio"] = _prompt_value(
        "Hematocrit ratio", default=0.73, expected_type=float
    )

    return cfg


def _collect_asl_config() -> dict[str, Any]:
    """Collect ASL pipeline settings."""
    print("\n--- ASL Pipeline Settings ---")
    cfg: dict[str, Any] = {}

    labeling_schemes = ["pcasl", "pasl", "casl"]
    cfg["labeling_scheme"] = _prompt_choice(
        "Labeling scheme:", labeling_schemes, default="pcasl"
    )
    cfg["pld"] = _prompt_value(
        "Post-labeling delay PLD (ms)",
        default=1800.0,
        expected_type=float,
    )
    cfg["label_duration"] = _prompt_value(
        "Label duration (ms)", default=1800.0, expected_type=float
    )
    cfg["t1_blood"] = _prompt_value(
        "T1 of blood (ms)", default=1650.0, expected_type=float
    )
    cfg["labeling_efficiency"] = _prompt_value(
        "Labeling efficiency (0-1)", default=0.85, expected_type=float
    )

    m0_methods = ["single", "voxelwise", "reference_region"]
    cfg["m0_method"] = _prompt_choice(
        "M0 calibration method:", m0_methods, default="single"
    )

    cfg["t1_tissue"] = _prompt_value(
        "T1 of tissue (ms)", default=1330.0, expected_type=float
    )
    cfg["partition_coefficient"] = _prompt_value(
        "Partition coefficient (mL/g)", default=0.9, expected_type=float
    )

    diff_methods = ["pairwise", "surround", "mean"]
    cfg["difference_method"] = _prompt_choice(
        "Difference method:", diff_methods, default="pairwise"
    )

    orders = ["label_first", "control_first"]
    cfg["label_control_order"] = _prompt_choice(
        "Label/control order:", orders, default="label_first"
    )

    return cfg


def _collect_ivim_config() -> dict[str, Any]:
    """Collect IVIM pipeline settings."""
    print("\n--- IVIM Pipeline Settings ---")
    cfg: dict[str, Any] = {}

    fitting_methods = ["segmented", "full", "bayesian"]
    cfg["fitting_method"] = _prompt_choice(
        "Fitting method:", fitting_methods, default="segmented"
    )
    cfg["b_threshold"] = _prompt_value(
        "B-value threshold (s/mm^2)", default=200.0, expected_type=float
    )
    cfg["normalize_signal"] = _prompt_yes_no(
        "Normalize signal to S(b=0)?", default=True
    )

    if cfg["fitting_method"] == "bayesian":
        print(
            "  Note: Bayesian fitting uses default priors and MCMC settings.\n"
            "  Edit the generated YAML to customize (fitting.bayesian section)."
        )

    return cfg


# ---------------------------------------------------------------------------
# YAML generation
# ---------------------------------------------------------------------------


def _format_yaml_value(value: Any) -> str:
    """Format a Python value for YAML output."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        items = ", ".join(str(v) for v in value)
        return f"[{items}]"
    if isinstance(value, str):
        # Quote strings that could be misinterpreted by YAML
        if value in ("true", "false", "null", "yes", "no", "on", "off") or value == "":
            return f'"{value}"'
        return value
    if isinstance(value, float):
        # Use scientific notation for small values
        if 0 < abs(value) < 0.01:
            return f"{value:.1e}"
        return str(value)
    return str(value)


def _generate_yaml(
    modality: str,
    pipeline_config: dict[str, Any],
    data_config: dict[str, Any],
) -> str:
    """Build a YAML config string with inline comments."""
    lines: list[str] = []
    lines.append(f"modality: {modality}")
    lines.append("pipeline:")
    for key, val in pipeline_config.items():
        if isinstance(val, dict):
            # Nested sub-section (e.g. acquisition)
            lines.append(f"  {key}:")
            for sub_key, sub_val in val.items():
                comment = _PIPELINE_COMMENTS.get(sub_key, "")
                suffix = f"  # {comment}" if comment else ""
                formatted = _format_yaml_value(sub_val)
                lines.append(f"    {sub_key}: {formatted}{suffix}")
        else:
            comment = _PIPELINE_COMMENTS.get(key, "")
            suffix = f"  # {comment}" if comment else ""
            formatted = _format_yaml_value(val)
            lines.append(f"  {key}: {formatted}{suffix}")

    lines.append("data:")
    for key, val in data_config.items():
        formatted = _format_yaml_value(val)
        lines.append(f"  {key}: {formatted}")

    lines.append("output:")
    lines.append("  format: nifti")
    lines.append("backend:")
    lines.append("  force_cpu: false")
    lines.append("logging:")
    lines.append("  level: INFO")
    lines.append("")

    return "\n".join(lines)


def _validate_config(yaml_str: str) -> None:
    """Validate generated YAML by round-tripping through Pydantic models.

    Raises
    ------
    Exception
        Re-raises any validation error from ``PipelineConfig``.
    """
    import yaml

    from osipy.cli.config import PipelineConfig

    raw = yaml.safe_load(yaml_str)
    config = PipelineConfig(**raw)
    config.get_modality_config()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


_MODALITY_COLLECTORS: dict[str, Any] = {
    "dce": _collect_dce_config,
    "dsc": _collect_dsc_config,
    "asl": _collect_asl_config,
    "ivim": _collect_ivim_config,
}


def run_wizard() -> None:
    """Run the interactive configuration wizard.

    Walks the user through modality selection, data configuration, and
    modality-specific pipeline settings, then writes a validated YAML
    config file.
    """
    _check_tty()

    print("osipy Configuration Wizard")
    print("=" * 40)

    # Step 1: Select modality
    modalities = list(_MODALITY_DESCRIPTIONS.keys())
    descriptions = list(_MODALITY_DESCRIPTIONS.values())
    modality = _prompt_choice(
        "Select a modality:",
        modalities,
        descriptions=descriptions,
    )

    # Step 2: Modality-specific pipeline settings (collected first so
    # data prompts can adapt — e.g. T1 map vs assumed T1, AIF file)
    collector = _MODALITY_COLLECTORS[modality]
    pipeline_config = collector()

    # Step 3: Data configuration (informed by pipeline choices)
    data_config = _collect_data_config(modality, pipeline_config)

    # Step 4: Generate, validate, write
    yaml_str = _generate_yaml(modality, pipeline_config, data_config)

    print("\n--- Generated Configuration ---")
    print(yaml_str)

    # Validate
    try:
        _validate_config(yaml_str)
        print("Config validated successfully.")
    except Exception as exc:
        print(f"Warning: generated config did not validate: {exc}", file=sys.stderr)

    # Prompt for output path
    output_path = _prompt_path("Output file path", default="config.yaml")
    if output_path is None:
        output_path = "config.yaml"

    out = Path(output_path)
    if out.exists() and not _prompt_yes_no(
        f"{out} already exists. Overwrite?", default=False
    ):
        print("Aborted.")
        return

    out.write_text(yaml_str, encoding="utf-8")
    print(f"\nConfig written to {out}")
    print(f"Next step: osipy {out} /path/to/data")
