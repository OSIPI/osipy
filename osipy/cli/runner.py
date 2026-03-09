"""Pipeline runner for YAML-configured execution.

Orchestrates data loading, pipeline execution, and result saving
based on validated YAML configuration.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from osipy.common.backend.array_module import get_array_module, to_gpu

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
    """Load data using the universal loader, respecting config format.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration (provides ``data.format``, ``data.subject``,
        ``data.session``).
    data_path : Path
        Path to input data (file or directory).
    modality : str
        Modality string (``"DCE"``, ``"DSC"``, ``"ASL"``, ``"IVIM"``).

    Returns
    -------
    PerfusionDataset
        Loaded dataset.
    """
    from osipy.common.io.load import load_perfusion

    fmt = config.data.format  # "auto", "nifti", "dicom", "bids"
    return load_perfusion(
        path=data_path,
        modality=modality,
        format=fmt,
        subject=config.data.subject,
        session=config.data.session,
        interactive=False,
    )


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


def _discover_dce_dicom_series(
    data_path: Path,
) -> tuple[list[Path], Path] | None:
    """Discover VFA and perfusion series in a DCE DICOM directory tree.

    Handles multi-level layouts like:
    ``visit_dir / study_dir / series_dir / *.dcm``

    Parameters
    ----------
    data_path : Path
        Visit-level (or study-level) directory.

    Returns
    -------
    tuple[list[Path], Path] | None
        ``(vfa_dirs_sorted_by_flip_angle, perfusion_dir)`` or None
        if the layout is not a recognizable DCE DICOM structure.
    """
    import re

    vfa_dirs: list[Path] = []
    perfusion_dir: Path | None = None

    # Walk up to 3 levels to find series directories with DICOM files
    for series_dir in _iter_series_dirs(data_path):
        name_lower = series_dir.name.lower()
        if re.search(r"ax\s+\d+\s+flip", name_lower):
            vfa_dirs.append(series_dir)
        elif "perfusion" in name_lower:
            perfusion_dir = series_dir

    if not vfa_dirs or perfusion_dir is None:
        return None

    # Sort VFA dirs by flip angle extracted from directory name
    def _flip_angle_key(d: Path) -> int:
        m = re.search(r"ax\s+(\d+)\s+flip", d.name.lower())
        return int(m.group(1)) if m else 0

    vfa_dirs.sort(key=_flip_angle_key)
    return vfa_dirs, perfusion_dir


def _iter_series_dirs(root: Path) -> list[Path]:
    """Yield leaf directories that contain DICOM files.

    Searches up to 3 levels deep under *root*.
    """
    results: list[Path] = []

    def _has_dcm(d: Path) -> bool:
        return any(f.suffix.lower() == ".dcm" for f in d.iterdir() if f.is_file())

    for child in root.iterdir():
        if not child.is_dir():
            continue
        if _has_dcm(child):
            results.append(child)
            continue
        # One level deeper (study → series)
        for grandchild in child.iterdir():
            if not grandchild.is_dir():
                continue
            if _has_dcm(grandchild):
                results.append(grandchild)

    return results


def _load_dicom_volume(series_dir: Path) -> tuple[np.ndarray, Any]:
    """Load a single DICOM series as a 3D volume.

    Returns (volume, first_dcm) where volume has shape
    ``(cols, rows, slices)`` following NIfTI transposed convention.
    """
    import pydicom

    dcm_files = sorted(series_dir.glob("*.dcm"))
    if not dcm_files:
        msg = f"No DICOM files in {series_dir}"
        raise FileNotFoundError(msg)

    slices: list[tuple[float, Any]] = []
    for f in dcm_files:
        dcm = pydicom.dcmread(f)
        loc = float(getattr(dcm, "SliceLocation", 0))
        slices.append((loc, dcm))

    slices.sort(key=lambda x: x[0])
    first_dcm = slices[0][1]
    rows, cols = first_dcm.Rows, first_dcm.Columns
    volume = np.zeros((cols, rows, len(slices)), dtype=np.float32)

    for i, (_, dcm) in enumerate(slices):
        volume[:, :, i] = dcm.pixel_array.astype(np.float32).T

    return volume, first_dcm


def _load_perfusion_dicom(
    perfusion_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load 4D DCE perfusion data from a single DICOM series directory.

    Groups files by AcquisitionNumber (temporal position) and computes
    the time vector from AcquisitionTime DICOM tags.

    Returns ``(signal_4d, time_seconds, metadata)`` where signal_4d
    has shape ``(cols, rows, slices, timepoints)``.
    """
    import pydicom

    dcm_files = sorted(perfusion_dir.glob("*.dcm"))
    if not dcm_files:
        msg = f"No DICOM files in {perfusion_dir}"
        raise FileNotFoundError(msg)

    # Group files by temporal position
    temporal_data: dict[int, list[tuple[float, Any]]] = {}
    acquisition_times: dict[int, str] = {}

    for f in dcm_files:
        dcm = pydicom.dcmread(f)
        t_idx = int(dcm.AcquisitionNumber)
        loc = float(getattr(dcm, "SliceLocation", 0))

        if t_idx not in temporal_data:
            temporal_data[t_idx] = []
            if hasattr(dcm, "AcquisitionTime"):
                acquisition_times[t_idx] = str(dcm.AcquisitionTime)

        temporal_data[t_idx].append((loc, dcm))

    min_t = min(temporal_data)
    first_dcm = temporal_data[min_t][0][1]
    rows, cols = first_dcm.Rows, first_dcm.Columns
    n_slices = len(temporal_data[min_t])
    n_timepoints = len(temporal_data)

    signal_4d = np.zeros((cols, rows, n_slices, n_timepoints), dtype=np.float32)

    sorted_t = sorted(temporal_data)
    for t_out, t_idx in enumerate(sorted_t):
        slices = temporal_data[t_idx]
        slices.sort(key=lambda x: x[0])
        for z_idx, (_, dcm) in enumerate(slices):
            signal_4d[:, :, z_idx, t_out] = dcm.pixel_array.astype(np.float32).T

    # Build time vector from AcquisitionTime
    def _parse_dicom_time(t: str) -> float:
        h, m, s = int(t[:2]), int(t[2:4]), float(t[4:])
        return h * 3600 + m * 60 + s

    if acquisition_times:
        times_sec = [_parse_dicom_time(acquisition_times[t]) for t in sorted_t]
        time_seconds = np.array([t - times_sec[0] for t in times_sec])
        temporal_resolution = float(np.mean(np.diff(time_seconds)))
    else:
        tr_ms = float(first_dcm.RepetitionTime)
        temporal_resolution = tr_ms * n_slices / 1000.0
        time_seconds = np.arange(n_timepoints) * temporal_resolution

    from osipy.common.io.dicom import build_affine_from_dicom

    slice_thickness = float(getattr(first_dcm, "SliceThickness", 1.0))
    affine = build_affine_from_dicom(first_dcm, slice_thickness, transpose_slices=True)

    metadata = {
        "tr": float(first_dcm.RepetitionTime),
        "te": float(first_dcm.EchoTime),
        "flip_angle": float(first_dcm.FlipAngle),
        "field_strength": float(getattr(first_dcm, "MagneticFieldStrength", 1.5)),
        "temporal_resolution": temporal_resolution,
        "affine": affine,
    }
    return signal_4d, time_seconds, metadata


def _build_roi_mask(
    spatial_shape: tuple[int, ...],
    roi_cfg: Any,
) -> NDArray[np.bool_] | None:
    """Build a spherical ROI mask from pipeline ROI configuration.

    Parameters
    ----------
    spatial_shape : tuple
        (x, y, z) shape of the volume.
    roi_cfg : ROIConfig
        ROI configuration from YAML (has .enabled, .center, .radius).

    Returns
    -------
    NDArray[np.bool_] | None
        Boolean mask or None if ROI is disabled.
    """
    if not roi_cfg.enabled:
        return None

    radius = roi_cfg.radius
    if roi_cfg.center is not None:
        cx, cy, cz = roi_cfg.center
    else:
        # Default to volume center
        cx = spatial_shape[0] // 2
        cy = spatial_shape[1] // 2
        cz = spatial_shape[2] // 2

    logger.info("Using ROI: center=(%d, %d, %d), radius=%d", cx, cy, cz, radius)

    # Build coordinate grids within the bounding box
    x0 = max(cx - radius, 0)
    x1 = min(cx + radius + 1, spatial_shape[0])
    y0 = max(cy - radius, 0)
    y1 = min(cy + radius + 1, spatial_shape[1])
    z0 = max(cz - radius, 0)
    z1 = min(cz + radius + 1, spatial_shape[2])

    mask = np.zeros(spatial_shape, dtype=bool)
    zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
    mask[x0:x1, y0:y1, z0:z1] = dist_sq.transpose(2, 1, 0) <= radius**2
    return mask


def _run_dce(config: PipelineConfig, data_path: Path, output_dir: Path) -> None:
    from osipy.common.types import DCEAcquisitionParams
    from osipy.pipeline.dce_pipeline import DCEPipeline, DCEPipelineConfig

    mc = config.get_modality_config()  # DCEPipelineYAML
    base_dir = data_path if data_path.is_dir() else data_path.parent

    # Try DCE-specific DICOM discovery first (visit-level directory)
    discovered = _discover_dce_dicom_series(data_path)

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
    )

    # Construct time vector
    time_array = dataset.time_points
    if time_array is None:
        tr_sec = (acq.tr / 1000.0) if acq.tr is not None else 1.0
        time_array = np.arange(dataset.data.shape[-1]) * tr_sec

    # ROI mask
    roi_mask = _build_roi_mask(
        dataset.data.shape[:3],
        mc.roi,  # type: ignore[attr-defined]
    )
    if roi_mask is not None:
        mask = roi_mask if mask is None else (mask & roi_mask)

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
    discovered: tuple[list[Path], Path],
    data_path: Path,
    output_dir: Path,
) -> None:
    """Run DCE pipeline from discovered DICOM VFA + perfusion series."""
    from osipy.common.io.dicom import build_affine_from_dicom
    from osipy.common.parameter_map import ParameterMap
    from osipy.common.types import DCEAcquisitionParams
    from osipy.dce import compute_t1_vfa, signal_to_concentration
    from osipy.dce.fitting import fit_model

    vfa_dirs, perfusion_dir = discovered
    acq = mc.acquisition  # type: ignore[attr-defined]

    # ---- Step 1: Load VFA data ----
    logger.info("[Step 1] Loading VFA data (%d flip angles)...", len(vfa_dirs))
    vfa_volumes: list[np.ndarray] = []
    flip_angles: list[float] = []
    vfa_meta: dict[str, Any] | None = None

    for vfa_dir in vfa_dirs:
        vol, dcm = _load_dicom_volume(vfa_dir)
        vfa_volumes.append(vol)
        flip_angles.append(float(dcm.FlipAngle))
        if vfa_meta is None:
            slice_thickness = float(getattr(dcm, "SliceThickness", 1.0))
            vfa_meta = {
                "tr": float(dcm.RepetitionTime),
                "affine": build_affine_from_dicom(
                    dcm, slice_thickness, transpose_slices=True
                ),
            }

    vfa_volume = np.stack(vfa_volumes, axis=-1)
    logger.info(
        "  VFA shape: %s, flip angles: %s, TR: %.1f ms",
        vfa_volume.shape,
        flip_angles,
        vfa_meta["tr"],  # type: ignore[index]
    )

    # ---- Step 2: Load perfusion data ----
    logger.info("[Step 2] Loading perfusion data...")
    signal_4d, time_seconds, dce_meta = _load_perfusion_dicom(perfusion_dir)
    logger.info(
        "  DCE shape: %s, time: %.1f–%.1f s (dt=%.2f s)",
        signal_4d.shape,
        time_seconds[0],
        time_seconds[-1],
        dce_meta["temporal_resolution"],
    )

    affine = dce_meta["affine"]
    spatial_shape = signal_4d.shape[:3]

    # ---- Transfer perfusion signal to GPU for pipeline compute ----
    signal_4d = to_gpu(signal_4d)

    # ---- ROI mask ----
    roi_mask = _build_roi_mask(
        spatial_shape,
        mc.roi,  # type: ignore[attr-defined]
    )
    mask = _load_mask(config.data.mask, data_path.parent)
    if roi_mask is not None:
        mask = roi_mask if mask is None else (mask & roi_mask)

    # ---- Step 3: Compute T1 map ----
    logger.info("[Step 3] Computing T1 map (VFA)...")
    t1_result = compute_t1_vfa(
        signal=vfa_volume,
        flip_angles=flip_angles,
        tr=vfa_meta["tr"],  # type: ignore[index]
        method="linear",
    )

    t1_map = ParameterMap(
        name="T1",
        symbol="T1",
        units="ms",
        values=t1_result.t1_map.values,
        affine=vfa_meta["affine"],  # type: ignore[index]
        quality_mask=t1_result.quality_mask,
    )

    valid = (
        int(np.sum(t1_result.quality_mask)) if t1_result.quality_mask is not None else 0
    )
    total = int(np.prod(vfa_volume.shape[:3]))
    logger.info("  Valid voxels: %d/%d (%.1f%%)", valid, total, 100 * valid / total)

    # ---- Step 4: Signal to concentration ----
    logger.info("[Step 4] Converting signal to concentration...")
    relaxivity = acq.relaxivity
    baseline_frames = acq.baseline_frames

    dce_acq_params = DCEAcquisitionParams(
        tr=dce_meta["tr"],
        te=dce_meta["te"],
        flip_angles=[dce_meta["flip_angle"]],
        temporal_resolution=dce_meta["temporal_resolution"],
        relaxivity=relaxivity,
        field_strength=dce_meta["field_strength"],
        baseline_frames=baseline_frames,
    )

    concentration = signal_to_concentration(
        signal=signal_4d,
        t1_map=t1_map,
        acquisition_params=dce_acq_params,
    )
    logger.info("  Concentration shape: %s", concentration.shape)

    # ---- Step 5: Generate AIF ----
    aif_source = mc.aif_source  # type: ignore[attr-defined]
    logger.info("[Step 5] Generating AIF (source=%s)...", aif_source)

    if aif_source == "population":
        from osipy.common.aif import get_population_aif

        aif_model = get_population_aif(mc.population_aif)  # type: ignore[attr-defined]
        aif = aif_model(time_seconds)
        logger.info(
            "  AIF peak: %.2f mM at t=%.1f s",
            aif.concentration.max(),
            time_seconds[int(np.argmax(aif.concentration))],
        )
    else:
        from osipy.common.aif import detect_aif
        from osipy.common.dataset import PerfusionDataset
        from osipy.common.types import Modality

        conc_dataset = PerfusionDataset(
            data=concentration,
            affine=affine,
            modality=Modality.DCE,
            time_points=time_seconds,
        )
        aif_result = detect_aif(conc_dataset, roi_mask=mask)
        aif = aif_result.aif

    # ---- Step 6: Build fit mask and fit model ----
    logger.info("[Step 6] Fitting %s model...", mc.model)  # type: ignore[attr-defined]

    # Combine quality masks for fitting
    # concentration may be on GPU, so transfer all masks to same device
    xp = get_array_module(concentration)
    t1_quality = xp.asarray(
        t1_result.quality_mask
        if t1_result.quality_mask is not None
        else np.ones(spatial_shape, dtype=bool)
    )
    t1_bounds = xp.asarray(t1_map.values > 100) & xp.asarray(t1_map.values < 5000)
    has_enhancement = xp.any(concentration > 0.01, axis=-1)
    fit_mask = t1_quality & t1_bounds & has_enhancement
    if mask is not None:
        fit_mask = fit_mask & xp.asarray(mask)

    n_fit = int(fit_mask.sum())
    logger.info("  Fitting %d voxels...", n_fit)

    fitting = mc.fitting  # type: ignore[attr-defined]  # DCEFittingConfig
    dicom_bounds_override = (
        {k: tuple(v) for k, v in fitting.bounds.items()} if fitting.bounds else None
    )

    t_fit = time.perf_counter()
    fit_result = fit_model(
        model_name=mc.model,  # type: ignore[attr-defined]
        concentration=concentration,
        aif=aif,
        time=time_seconds,
        mask=fit_mask,
        fitter=fitting.fitter,
        bounds_override=dicom_bounds_override,
    )
    elapsed_fit = time.perf_counter() - t_fit

    fitted = int(fit_result.quality_mask.sum())
    logger.info(
        "  Fitted: %d/%d voxels (%.1f%%)",
        fitted,
        n_fit,
        100 * fitted / max(n_fit, 1),
    )

    # ---- Step 7: Stats and save ----
    _log_parameter_stats(
        fit_result.parameter_maps,
        fit_result.quality_mask,
        elapsed_fit,
    )
    logger.info("Saving results...")
    _save_results(
        fit_result.parameter_maps,
        fit_result.quality_mask,
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
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "data_path": str(data_path),
        "config": config.model_dump(),
    }
    if elapsed_seconds is not None:
        metadata["elapsed_seconds"] = round(elapsed_seconds, 2)
    meta_path = output_dir / "osipy_run.json"
    with meta_path.open("w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("  Saved metadata -> %s", meta_path)
