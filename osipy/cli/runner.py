"""Pipeline runner for YAML-configured execution.

Orchestrates data loading, pipeline execution, and result saving
based on validated YAML configuration.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from osipy.cli.config import PipelineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_pipeline(
    config: PipelineConfig,
    data_path: str | Path,
    output_dir: str | Path | None = None,
) -> None:
    """Run a modality pipeline from YAML configuration.

    Parameters
    ----------
    config : PipelineConfig
        Validated pipeline configuration.
    data_path : str or Path
        Path to input data (NIfTI file or directory).
    output_dir : str or Path or None
        Output directory. If None, defaults to ``{data_path}/osipy_output``
        when *data_path* is a directory, or ``{data_path_parent}/osipy_output``
        when *data_path* is a file.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        msg = f"Data path not found: {data_path}"
        raise FileNotFoundError(msg)

    # Determine output directory
    if output_dir is not None:
        out = Path(output_dir)
    elif data_path.is_dir():
        out = data_path / "osipy_output"
    else:
        out = data_path.parent / "osipy_output"
    out.mkdir(parents=True, exist_ok=True)

    # Configure backend
    if config.backend.force_cpu:
        from osipy.common.backend import GPUConfig, set_backend

        set_backend(GPUConfig(force_cpu=True))
        logger.info("Forced CPU execution")

    # Dispatch to modality handler
    logger.info("Running %s pipeline...", config.modality.upper())

    handlers: dict[str, Any] = {
        "dce": _run_dce,
        "dsc": _run_dsc,
        "asl": _run_asl,
        "ivim": _run_ivim,
    }

    handler = handlers[config.modality]
    t_start = time.perf_counter()
    handler(config, data_path, out)
    elapsed = time.perf_counter() - t_start

    # Save run metadata
    _save_metadata(config, data_path, out, elapsed_seconds=elapsed)
    logger.info("Results saved to %s", out)
    logger.info("Total pipeline time: %.1f s", elapsed)


# ---------------------------------------------------------------------------
# Path / mask helpers
# ---------------------------------------------------------------------------


def _resolve_path(relative: str, base_dir: Path) -> Path:
    """Resolve a data file path relative to *base_dir*."""
    p = Path(relative)
    if p.is_absolute():
        return p
    return base_dir / p


def _load_mask(mask_path: str | None, base_dir: Path) -> NDArray[np.bool_] | None:
    """Load mask file if specified."""
    if mask_path is None:
        return None
    import nibabel as nib

    path = _resolve_path(mask_path, base_dir)
    if not path.exists():
        logger.warning("Mask file not found: %s", path)
        return None
    img = nib.load(path)
    return np.asarray(img.dataobj, dtype=bool)


def _load_nifti_array(
    file_path: str, base_dir: Path
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]] | None:
    """Load a NIfTI file as (data, affine), returning None when absent."""
    import nibabel as nib

    path = _resolve_path(file_path, base_dir)
    if not path.exists():
        logger.warning("File not found: %s", path)
        return None
    img = nib.load(path)
    return np.asarray(img.dataobj, dtype=np.float64), np.asarray(
        img.affine, dtype=np.float64
    )


def _load_data(config: PipelineConfig, data_path: Path, modality: str) -> Any:
    """Load data using the appropriate loader for *data_path*.

    Dispatches directly to :func:`load_nifti`, :func:`load_bids`, or the
    new :func:`discover_dicom` + :func:`load_dicom_series` stack. Picks
    the best-matching series when discovery returns more than one
    candidate (prefers the first ``dynamic`` or ``dynamic_frame`` group).

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration (provides ``data.format``, ``data.subject``,
        ``data.session``).
    data_path : Path
        Path to input data (file or directory).
    modality : str
        Modality string (``"DCE"``, ``"DSC"``, ``"ASL"``, ``"IVIM"``).
    """
    from osipy.common.types import Modality

    fmt = config.data.format
    if fmt == "auto":
        fmt = _detect_format(data_path)

    modality_enum = (
        Modality(modality.lower()) if isinstance(modality, str) else modality
    )

    if fmt == "nifti":
        from osipy.common.io.nifti import load_nifti

        return load_nifti(data_path, modality=modality_enum)

    if fmt == "bids":
        from osipy.common.io.bids import load_bids

        if config.data.subject is None:
            msg = "subject is required for BIDS format"
            raise ValueError(msg)
        return load_bids(
            data_path,
            subject=config.data.subject,
            session=config.data.session,
            modality=modality_enum,
            interactive=False,
        )

    if fmt == "dicom":
        from osipy.common.io.discovery import discover_dicom, load_dicom_series

        series_list = discover_dicom(data_path)
        chosen = _select_dataset_series(series_list)
        return load_dicom_series(chosen, modality=modality_enum)

    msg = f"Unknown data format: {fmt}"
    raise ValueError(msg)


def _detect_format(path: Path) -> str:
    """Detect whether *path* points at NIfTI, DICOM, or BIDS data."""
    if path.is_file():
        if path.name.endswith((".nii", ".nii.gz")):
            return "nifti"
        return "dicom"
    if (path / "dataset_description.json").exists():
        return "bids"
    if any(p.name.endswith((".nii", ".nii.gz")) for p in path.iterdir() if p.is_file()):
        return "nifti"
    return "dicom"


def _select_dataset_series(series_list: list[Any]) -> Any:
    """Pick the series (or group) to load from a discovery result.

    Preference order:
        * largest ``dynamic_frame`` group (stacked as 4D multi-series)
        * first ``dynamic`` series
        * first ``t1_look_locker`` series
        * single imaged series if nothing better matches
    """
    from collections import defaultdict

    if not series_list:
        msg = "discover_dicom returned no series"
        raise ValueError(msg)

    groups: dict[str, list[Any]] = defaultdict(list)
    for s in series_list:
        if s.role_hint == "dynamic_frame" and s.group_key:
            groups[s.group_key].append(s)
    if groups:
        # Biggest cluster wins.
        best = max(groups.values(), key=len)
        return best

    for s in series_list:
        if s.role_hint == "dynamic":
            return s

    for s in series_list:
        if s.role_hint == "t1_look_locker":
            return s

    imaged = [s for s in series_list if s.role_hint != "unknown"]
    if imaged:
        return imaged[0]
    return series_list[0]


# ---------------------------------------------------------------------------
# Output statistics
# ---------------------------------------------------------------------------


def _log_parameter_stats(
    parameter_maps: dict[str, Any],
    quality_mask: NDArray[np.bool_] | None,
    elapsed: float,
) -> None:
    """Log summary statistics for computed parameter maps.

    Parameters
    ----------
    parameter_maps : dict[str, ParameterMap]
        Computed parameter maps keyed by name.
    quality_mask : NDArray[np.bool_] | None
        Overall quality mask.
    elapsed : float
        Elapsed time for the fitting/computation step in seconds.
    """
    from osipy.common.parameter_map import ParameterMap

    if quality_mask is not None:
        total = int(quality_mask.size)
        valid = int(np.sum(quality_mask))
        logger.info(
            "Quality: %d / %d voxels valid (%.1f%%)",
            valid,
            total,
            100.0 * valid / max(total, 1),
        )

    logger.info("Parameter statistics (valid voxels only):")
    for _name, pmap in parameter_maps.items():
        if not isinstance(pmap, ParameterMap):
            continue
        stats = pmap.statistics()
        logger.info(
            "  %-10s  mean=%.4g  std=%.4g  min=%.4g  max=%.4g  median=%.4g  [%s]",
            pmap.name,
            stats["mean"],
            stats["std"],
            stats["min"],
            stats["max"],
            stats["median"],
            pmap.units,
        )

    logger.info("Computation time: %.2f s", elapsed)


# ---------------------------------------------------------------------------
# Per-modality handlers
# ---------------------------------------------------------------------------


def _discover_dce_dicom(data_path: Path) -> tuple[list[Any], Any] | None:
    """Discover VFA + dynamic series for DCE pipelines.

    Runs :func:`discover_dicom`, then extracts:

    * the list of VFA single-flip series (role ``vfa``), sorted by FA;
    * the dynamic dataset — either a single ``dynamic`` series (all
      timepoints in one ``SeriesInstanceUID``) or a ``dynamic_frame``
      cluster (one 3D series per timepoint, accumulated into a list).

    Returns ``None`` when the discovery result does not contain both a
    VFA group and a dynamic series, so the caller can fall through to
    the generic loader (which handles NIfTI / single-series DICOM).
    """
    from collections import defaultdict

    from osipy.common.io.discovery import discover_dicom

    if not data_path.is_dir():
        return None

    series_list = discover_dicom(data_path)
    if not series_list:
        return None

    dce_series: Any | None = None
    for s in series_list:
        if s.role_hint == "dynamic":
            dce_series = s
            break
    if dce_series is None:
        groups: dict[str, list[Any]] = defaultdict(list)
        for s in series_list:
            if s.role_hint == "dynamic_frame" and s.group_key:
                groups[s.group_key].append(s)
        if groups:
            dce_series = max(groups.values(), key=len)

    if dce_series is None:
        return None

    # Discovery already restricts ``role_hint == "vfa"`` to coherent
    # same-shape clusters with distinct flip angles, so no further
    # filtering is needed here.
    vfa = sorted(
        (s for s in series_list if s.role_hint == "vfa"),
        key=lambda s: s.flip_angle if s.flip_angle is not None else 0.0,
    )

    if not vfa:
        return None

    vfa_desc = ", ".join(f"{s.flip_angle:.0f}°" for s in vfa)
    if isinstance(dce_series, list):
        dyn_label = f"{len(dce_series)} frames @ {dce_series[0].description!r}"
    else:
        dyn_label = (
            f"{dce_series.n_temporal_positions} TPIs @ {dce_series.description!r}"
        )
    logger.info("DCE discovery: VFA (%s) + dynamic (%s)", vfa_desc, dyn_label)
    return vfa, dce_series


def _run_dce(config: PipelineConfig, data_path: Path, output_dir: Path) -> None:
    from osipy.common.types import DCEAcquisitionParams
    from osipy.pipeline.dce_pipeline import DCEPipeline, DCEPipelineConfig

    mc = config.get_modality_config()  # DCEPipelineYAML
    base_dir = data_path if data_path.is_dir() else data_path.parent

    # Try VFA-aware DCE discovery first. Falls through for NIfTI inputs or
    # DICOM directories with no VFA companion.
    discovered = _discover_dce_dicom(data_path)

    if discovered is not None:
        _run_dce_from_dicom(config, mc, discovered, data_path, output_dir)
        return

    # Fall back to generic loader for NIfTI / BIDS / simple DICOM
    dataset = _load_data(config, data_path, "DCE")
    affine = dataset.affine
    mask = _load_mask(config.data.mask, base_dir)

    # Load optional pre-computed T1 map
    t1_map = None
    if config.data.t1_map is not None:
        loaded = _load_nifti_array(config.data.t1_map, base_dir)
        if loaded is not None:
            from osipy.common.parameter_map import ParameterMap

            t1_data, _ = loaded
            t1_map = ParameterMap(
                name="T1",
                symbol="T1",
                units="ms",
                values=t1_data,
                affine=affine,
            )

    # Build acquisition params.
    # Prefer DICOM-detected TR / flip angle over config values; the config
    # values act as fallbacks for when metadata is missing.
    acq = mc.acquisition  # type: ignore[attr-defined]
    detected = dataset.acquisition_params  # from DICOM / NIfTI headers
    detected_tr = getattr(detected, "tr", None) if detected else None
    tr_value = detected_tr if detected_tr is not None else acq.tr

    # For the dynamic flip angle, check DICOM metadata first.  The
    # config flip_angles list is intended for VFA T1 mapping, so we
    # should NOT use it as the dynamic FA.  Only fall back to
    # config flip_angles when nothing was detected.
    detected_fa = getattr(detected, "flip_angle", None) if detected else None
    flip_angles_value = (
        [detected_fa] if detected_fa is not None else (acq.flip_angles or [])
    )

    if detected_tr is not None and detected_tr != acq.tr:
        logger.info(
            "Using DICOM-detected TR=%.2f ms (config had %.2f ms)",
            detected_tr,
            acq.tr if acq.tr is not None else 0,
        )
    if detected_fa is not None:
        logger.info("Using DICOM-detected flip angle=%.1f°", detected_fa)

    acq_params = DCEAcquisitionParams(
        tr=tr_value,
        flip_angles=flip_angles_value,
        baseline_frames=acq.baseline_frames,
        relaxivity=acq.relaxivity,
        t1_assumed=acq.t1_assumed,
    )

    # Build pipeline config
    fitting = mc.fitting  # type: ignore[attr-defined]  # DCEFittingConfig
    bounds_override = (
        {k: tuple(v) for k, v in fitting.bounds.items()} if fitting.bounds else None
    )

    pipeline_cfg = DCEPipelineConfig(
        model=mc.model,  # type: ignore[attr-defined]
        t1_mapping_method=mc.t1_mapping_method,  # type: ignore[attr-defined]
        aif_source=mc.aif_source,  # type: ignore[attr-defined]
        population_aif=mc.population_aif,  # type: ignore[attr-defined]
        acquisition_params=acq_params,
        save_intermediate=mc.save_intermediate,  # type: ignore[attr-defined]
        fitter=fitting.fitter,
        bounds_override=bounds_override,
        initial_guess_override=fitting.initial_guess,
        max_iterations=fitting.max_iterations,
        tolerance=fitting.tolerance,
        r2_threshold=fitting.r2_threshold,
        fit_delay=fitting.fit_delay,
    )

    # Construct time vector
    time_array = dataset.time_points
    if time_array is None:
        tr_sec = (acq.tr / 1000.0) if acq.tr is not None else 1.0
        time_array = np.arange(dataset.data.shape[-1]) * tr_sec

    # Run
    pipeline = DCEPipeline(pipeline_cfg)
    t_fit = time.perf_counter()
    result = pipeline.run(
        dce_data=dataset,
        time=time_array,
        t1_map=t1_map,
        mask=mask,
    )
    elapsed_fit = time.perf_counter() - t_fit

    # Stats and save
    _log_parameter_stats(
        result.fit_result.parameter_maps,
        result.fit_result.quality_mask,
        elapsed_fit,
    )
    _save_results(
        result.fit_result.parameter_maps,
        result.fit_result.quality_mask,
        output_dir,
        affine,
    )


def _run_dce_from_dicom(
    config: PipelineConfig,
    mc: Any,
    discovered: tuple[list[Any], Any],
    data_path: Path,
    output_dir: Path,
) -> None:
    """Run DCE pipeline from discovered DICOM VFA + perfusion series.

    ``discovered`` is ``(vfa_series_sorted_by_flip_angle, dce_series)``
    where ``dce_series`` is either a single :class:`SeriesInfo` whose
    ``SeriesInstanceUID`` already contains every timepoint, or a list
    of per-timepoint :class:`SeriesInfo` objects — one 3D series per
    dynamic frame — which ``load_dicom_series`` stacks into 4D. The
    function loads both, then delegates fitting to :class:`DCEPipeline`
    so that the DICOM entry shares one code path with the NIfTI/YAML
    entry.
    """
    from osipy.common.io.discovery import load_dicom_series
    from osipy.common.types import DCEAcquisitionParams, Modality
    from osipy.pipeline.dce_pipeline import DCEPipeline, DCEPipelineConfig

    vfa_series, dce_series = discovered
    acq = mc.acquisition  # type: ignore[attr-defined]

    # ---- Step 1: Load VFA stack ----
    logger.info("[Step 1] Loading VFA data (%d flip angles)...", len(vfa_series))
    vfa_dataset = load_dicom_series(vfa_series, modality=Modality.DCE)
    vfa_flip_angles = vfa_dataset.acquisition_params.flip_angles
    vfa_tr = vfa_dataset.acquisition_params.tr
    if vfa_tr is None:
        msg = "VFA series has no RepetitionTime in its acquisition params"
        raise ValueError(msg)
    logger.info(
        "  VFA shape: %s, flip angles: %s, TR: %.1f ms",
        vfa_dataset.data.shape,
        vfa_flip_angles,
        vfa_tr,
    )

    # ---- Step 2: Load perfusion data ----
    logger.info("[Step 2] Loading perfusion data...")
    dce_dataset = load_dicom_series(dce_series, modality=Modality.DCE)
    time_seconds = dce_dataset.time_points
    dce_acq = dce_dataset.acquisition_params
    temporal_resolution = (
        float(np.mean(np.diff(time_seconds))) if len(time_seconds) >= 2 else 1.0
    )
    logger.info(
        "  DCE shape: %s, time: %.1f-%.1f s (dt=%.2f s)",
        dce_dataset.data.shape,
        time_seconds[0],
        time_seconds[-1],
        temporal_resolution,
    )

    affine = dce_dataset.affine
    mask = _load_mask(config.data.mask, data_path.parent)

    # ---- Build pipeline config and run ----
    dce_acq_params = DCEAcquisitionParams(
        tr=dce_acq.tr,
        te=dce_acq.te,
        flip_angles=[dce_acq.flip_angle] if dce_acq.flip_angle is not None else [],
        temporal_resolution=temporal_resolution,
        relaxivity=acq.relaxivity,
        field_strength=dce_acq.field_strength
        if dce_acq.field_strength is not None
        else 1.5,
        baseline_frames=acq.baseline_frames,
    )

    fitting = mc.fitting  # type: ignore[attr-defined]  # DCEFittingConfig
    dicom_bounds_override = (
        {k: tuple(v) for k, v in fitting.bounds.items()} if fitting.bounds else None
    )

    pipeline_cfg = DCEPipelineConfig(
        model=mc.model,  # type: ignore[attr-defined]
        t1_mapping_method="vfa",
        aif_source=mc.aif_source,  # type: ignore[attr-defined]
        population_aif=mc.population_aif,  # type: ignore[attr-defined]
        acquisition_params=dce_acq_params,
        save_intermediate=mc.save_intermediate,  # type: ignore[attr-defined]
        fitter=fitting.fitter,
        bounds_override=dicom_bounds_override,
        initial_guess_override=fitting.initial_guess,
        max_iterations=fitting.max_iterations,
        tolerance=fitting.tolerance,
        r2_threshold=fitting.r2_threshold,
        fit_delay=fitting.fit_delay,
    )

    logger.info("[Step 3-6] Running DCE pipeline (%s model)...", mc.model)  # type: ignore[attr-defined]
    pipeline = DCEPipeline(pipeline_cfg)
    t_fit = time.perf_counter()
    result = pipeline.run(
        dce_data=dce_dataset,
        time=time_seconds,
        t1_data=vfa_dataset,
        flip_angles=np.asarray(vfa_flip_angles, dtype=float),
        tr=vfa_tr,
        mask=mask,
    )
    elapsed_fit = time.perf_counter() - t_fit

    fitted = int(result.fit_result.quality_mask.sum())
    total = int(np.prod(dce_dataset.data.shape[:3]))
    logger.info(
        "  Fitted: %d/%d voxels (%.1f%%)", fitted, total, 100 * fitted / max(total, 1)
    )

    # ---- Step 7: Stats and save ----
    _log_parameter_stats(
        result.fit_result.parameter_maps,
        result.fit_result.quality_mask,
        elapsed_fit,
    )
    logger.info("Saving results...")
    _save_results(
        result.fit_result.parameter_maps,
        result.fit_result.quality_mask,
        output_dir,
        affine,
    )


def _run_dsc(config: PipelineConfig, data_path: Path, output_dir: Path) -> None:
    from osipy.pipeline.dsc_pipeline import DSCPipeline, DSCPipelineConfig

    mc = config.get_modality_config()  # DSCPipelineYAML
    base_dir = data_path if data_path.is_dir() else data_path.parent

    dataset = _load_data(config, data_path, "DSC")
    affine = dataset.affine
    mask = _load_mask(config.data.mask, base_dir)

    pipeline_cfg = DSCPipelineConfig(
        te=mc.te,  # type: ignore[attr-defined]
        deconvolution_method=mc.deconvolution_method,  # type: ignore[attr-defined]
        apply_leakage_correction=mc.apply_leakage_correction,  # type: ignore[attr-defined]
        svd_threshold=mc.svd_threshold,  # type: ignore[attr-defined]
    )

    time_array = dataset.time_points
    if time_array is None:
        time_array = np.arange(dataset.data.shape[-1], dtype=np.float64)

    pipeline = DSCPipeline(pipeline_cfg)
    t_fit = time.perf_counter()
    result = pipeline.run(dsc_signal=dataset, time=time_array, mask=mask)
    elapsed_fit = time.perf_counter() - t_fit

    # Collect perfusion maps
    maps: dict[str, Any] = {}
    pm = result.perfusion_maps
    maps["cbv"] = pm.cbv
    maps["cbf"] = pm.cbf
    maps["mtt"] = pm.mtt
    if pm.ttp is not None:
        maps["ttp"] = pm.ttp
    if pm.tmax is not None:
        maps["tmax"] = pm.tmax
    if pm.delay is not None:
        maps["delay"] = pm.delay

    _log_parameter_stats(maps, pm.quality_mask, elapsed_fit)
    _save_results(maps, pm.quality_mask, output_dir, affine)


def _run_asl(config: PipelineConfig, data_path: Path, output_dir: Path) -> None:
    from osipy.asl import LabelingScheme
    from osipy.pipeline.asl_pipeline import ASLPipeline, ASLPipelineConfig

    mc = config.get_modality_config()  # ASLPipelineYAML
    base_dir = data_path if data_path.is_dir() else data_path.parent

    dataset = _load_data(config, data_path, "ASL")
    affine = dataset.affine
    mask = _load_mask(config.data.mask, base_dir)

    # Map string to LabelingScheme enum
    scheme_map = {
        "pasl": LabelingScheme.PASL,
        "casl": LabelingScheme.CASL,
        "pcasl": LabelingScheme.PCASL,
    }
    labeling_scheme = scheme_map[mc.labeling_scheme]  # type: ignore[attr-defined]

    pipeline_cfg = ASLPipelineConfig(
        labeling_scheme=labeling_scheme,
        pld=mc.pld,  # type: ignore[attr-defined]
        label_duration=mc.label_duration,  # type: ignore[attr-defined]
        t1_blood=mc.t1_blood,  # type: ignore[attr-defined]
        labeling_efficiency=mc.labeling_efficiency,  # type: ignore[attr-defined]
        m0_method=mc.m0_method,  # type: ignore[attr-defined]
    )

    # Load M0 calibration data
    m0: NDArray[np.floating[Any]] | float
    if config.data.m0_data is not None:
        loaded = _load_nifti_array(config.data.m0_data, base_dir)
        if loaded is not None:
            m0 = loaded[0]
        else:
            m0 = 1.0
            logger.warning("M0 data not found, using M0=1.0")
    else:
        m0 = 1.0
        logger.warning("No M0 data specified, using M0=1.0")

    pipeline = ASLPipeline(pipeline_cfg)
    t_fit = time.perf_counter()
    result = pipeline.run_from_alternating(
        asl_data=dataset.data,
        m0_data=m0,
        label_control_order=mc.label_control_order,  # type: ignore[attr-defined]
        mask=mask,
    )
    elapsed_fit = time.perf_counter() - t_fit

    maps: dict[str, Any] = {"cbf": result.cbf_result.cbf_map}
    _log_parameter_stats(maps, result.cbf_result.quality_mask, elapsed_fit)
    _save_results(maps, result.cbf_result.quality_mask, output_dir, affine)


def _run_ivim(config: PipelineConfig, data_path: Path, output_dir: Path) -> None:
    from osipy.ivim import FittingMethod
    from osipy.pipeline.ivim_pipeline import IVIMPipeline, IVIMPipelineConfig

    mc = config.get_modality_config()  # IVIMPipelineYAML
    base_dir = data_path if data_path.is_dir() else data_path.parent

    dataset = _load_data(config, data_path, "IVIM")
    affine = dataset.affine
    mask = _load_mask(config.data.mask, base_dir)

    # Map string to FittingMethod enum
    method_map = {
        "segmented": FittingMethod.SEGMENTED,
        "full": FittingMethod.FULL,
        "bayesian": FittingMethod.BAYESIAN,
    }
    fitting_method = method_map[mc.fitting_method]  # type: ignore[attr-defined]

    fitting = mc.fitting  # type: ignore[attr-defined]  # IVIMFittingConfig
    bounds = (
        {k: tuple(v) for k, v in fitting.bounds.items()} if fitting.bounds else None
    )

    # Convert Bayesian config if using Bayesian method
    bayesian_params = None
    if fitting_method == FittingMethod.BAYESIAN:
        bc = fitting.bayesian
        bayesian_params = {
            "prior_scale": bc.prior_scale,
            "noise_std": bc.noise_std,
            "compute_uncertainty": bc.compute_uncertainty,
        }

    pipeline_cfg = IVIMPipelineConfig(
        fitting_method=fitting_method,
        b_threshold=mc.b_threshold,  # type: ignore[attr-defined]
        normalize_signal=mc.normalize_signal,  # type: ignore[attr-defined]
        bounds=bounds,
        initial_guess=fitting.initial_guess,
        max_iterations=fitting.max_iterations,
        tolerance=fitting.tolerance,
        bayesian_params=bayesian_params,
    )

    # Get b-values from config or file
    b_values: NDArray[np.floating[Any]] | None = None
    if config.data.b_values is not None:
        b_values = np.array(config.data.b_values, dtype=np.float64)
    elif config.data.b_values_file is not None:
        bval_path = _resolve_path(config.data.b_values_file, base_dir)
        if bval_path.exists():
            b_values = np.loadtxt(bval_path, dtype=np.float64).ravel()
        else:
            logger.warning("b-values file not found: %s", bval_path)

    if b_values is None and hasattr(dataset, "acquisition_params"):
        acq_params = dataset.acquisition_params
        if acq_params is not None and hasattr(acq_params, "b_values"):
            b_values = np.asarray(acq_params.b_values, dtype=np.float64)
            logger.info(
                "Using b-values from loaded metadata (%d values)", len(b_values)
            )

    if b_values is None:
        msg = "b-values must be provided via 'data.b_values' or 'data.b_values_file'"
        raise ValueError(msg)

    pipeline = IVIMPipeline(pipeline_cfg)
    t_fit = time.perf_counter()
    result = pipeline.run(dwi_data=dataset, b_values=b_values, mask=mask)
    elapsed_fit = time.perf_counter() - t_fit

    fr = result.fit_result
    maps: dict[str, Any] = {
        "d": fr.d_map,
        "d_star": fr.d_star_map,
        "f": fr.f_map,
        "s0": fr.s0_map,
    }
    _log_parameter_stats(maps, fr.quality_mask, elapsed_fit)
    _save_results(maps, fr.quality_mask, output_dir, affine)


# ---------------------------------------------------------------------------
# Result saving helpers
# ---------------------------------------------------------------------------


def _save_results(
    parameter_maps: dict[str, Any],
    quality_mask: NDArray[np.bool_] | None,
    output_dir: Path,
    affine: NDArray[np.floating[Any]] | None = None,
) -> None:
    """Save parameter maps and quality mask as NIfTI files."""
    from osipy.common.io.nifti import save_nifti

    for name, pmap in parameter_maps.items():
        filename = f"{name.lower().replace('*', '_star')}.nii.gz"
        filepath = output_dir / filename
        save_nifti(pmap, filepath, affine=affine)
        logger.info("  Saved %s -> %s", name, filepath)

    if quality_mask is not None:
        save_nifti(
            quality_mask.astype(np.uint8),
            output_dir / "quality_mask.nii.gz",
            affine=affine,
        )
        logger.info("  Saved quality_mask -> %s", output_dir / "quality_mask.nii.gz")


def _save_metadata(
    config: PipelineConfig,
    data_path: Path,
    output_dir: Path,
    *,
    elapsed_seconds: float | None = None,
) -> None:
    """Save run metadata as JSON."""
    from osipy._version import __version__

    metadata: dict[str, Any] = {
        "osipy_version": __version__,
        "modality": config.modality,
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "data_path": str(data_path),
        "config": config.model_dump(),
    }
    if elapsed_seconds is not None:
        metadata["elapsed_seconds"] = round(elapsed_seconds, 2)
    meta_path = output_dir / "osipy_run.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("  Saved metadata -> %s", meta_path)
