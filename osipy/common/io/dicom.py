"""DICOM file loading for osipy.

This module provides functions for loading DICOM series into
PerfusionDataset containers with metadata extraction.

Supports both single-series and multi-series DCE data where each
timepoint may be stored in a separate DICOM series directory.

References
----------
DICOM Standard: https://www.dicomstandard.org/
"""

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from osipy.common.dataset import PerfusionDataset
from osipy.common.exceptions import DataValidationError, IOError, MetadataError
from osipy.common.types import AcquisitionParams, Modality

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# DICOM Modality values that represent non-image objects (no pixel data).
# KO = Key Object Selection, SR = Structured Report, PR = Presentation State,
# AU = Audio, DOC = Document, PLAN = RT Plan, REG = Registration.
_NON_IMAGE_MODALITIES = frozenset({"KO", "SR", "PR", "AU", "DOC", "PLAN", "REG"})


def build_affine_from_dicom(
    dcm: Any,
    slice_thickness: float,
    transpose_slices: bool = True,
) -> np.ndarray:
    """Build NIfTI affine matrix from DICOM geometry tags.

    This function builds an affine matrix that maps voxel indices (i, j, k)
    to patient coordinates (x, y, z) in millimeters.

    Parameters
    ----------
    dcm : pydicom.Dataset
        DICOM dataset with ImageOrientationPatient and ImagePositionPatient.
    slice_thickness : float
        Slice thickness in mm.
    transpose_slices : bool, default=True
        If True, assumes slices will be transposed when loading (col, row order).
        This matches standard NIfTI conventions used by tools like dcm2niix.
        If False, assumes slices stored in DICOM native (row, col) order.

    Returns
    -------
    np.ndarray
        4x4 affine matrix mapping voxel indices to patient coordinates.

    Notes
    -----
    DICOM geometry conventions:
    - ImageOrientationPatient[0:3]: row direction cosines (direction of increasing column)
    - ImageOrientationPatient[3:6]: column direction cosines (direction of increasing row)
    - PixelSpacing[0]: row spacing (distance between rows, in column direction)
    - PixelSpacing[1]: column spacing (distance between columns, in row direction)

    When transpose_slices=True (default):
    - Data is stored as array[col, row, slice] after transposing each DICOM slice
    - This matches the convention used by most DICOM-to-NIfTI converters
    - Affine column 0 maps to row direction (along increasing column index)
    - Affine column 1 maps to column direction (along increasing row index)
    """
    # Get image orientation (direction cosines)
    # DICOM defines:
    #   IOP[0:3] = row cosines = direction of increasing column index
    #   IOP[3:6] = column cosines = direction of increasing row index
    if hasattr(dcm, "ImageOrientationPatient"):
        iop = [float(x) for x in dcm.ImageOrientationPatient]
        row_cosines = np.array(iop[:3])  # Direction for increasing col index
        col_cosines = np.array(iop[3:])  # Direction for increasing row index
    else:
        row_cosines = np.array([1.0, 0.0, 0.0])
        col_cosines = np.array([0.0, 1.0, 0.0])

    # Compute slice direction (cross product gives normal to image plane)
    slice_cosines = np.cross(row_cosines, col_cosines)

    # Get pixel spacing
    # DICOM defines:
    #   PixelSpacing[0] = row spacing (distance between rows)
    #   PixelSpacing[1] = column spacing (distance between columns)
    if hasattr(dcm, "PixelSpacing"):
        pixel_spacing = [float(x) for x in dcm.PixelSpacing]
    else:
        pixel_spacing = [1.0, 1.0]

    row_spacing = pixel_spacing[0]  # Distance between rows (movement in col direction)
    col_spacing = pixel_spacing[1]  # Distance between cols (movement in row direction)

    # Get image position (origin = position of first voxel center)
    if hasattr(dcm, "ImagePositionPatient"):
        origin = np.array([float(x) for x in dcm.ImagePositionPatient])
    else:
        origin = np.array([0.0, 0.0, 0.0])

    # Build affine: maps voxel (i, j, k) to patient coordinates (x, y, z)
    affine = np.eye(4)

    if transpose_slices:
        # Data stored as array[col, row, slice] (transposed DICOM slices)
        # - Increasing i (col index in original) → row_cosines direction
        # - Increasing j (row index in original) → col_cosines direction
        affine[:3, 0] = row_cosines * col_spacing  # i axis: column direction
        affine[:3, 1] = col_cosines * row_spacing  # j axis: row direction
    else:
        # Data stored as array[row, col, slice] (native DICOM order)
        # - Increasing i (row index) → col_cosines direction
        # - Increasing j (col index) → row_cosines direction
        affine[:3, 0] = col_cosines * row_spacing  # i axis: row direction
        affine[:3, 1] = row_cosines * col_spacing  # j axis: column direction

    affine[:3, 2] = slice_cosines * slice_thickness
    affine[:3, 3] = origin

    return affine


def _prompt_for_value(
    tag_name: str,
    expected_type: type = float,
    prompt_missing: bool = True,
) -> Any:
    """Prompt user for missing metadata value.

    Parameters
    ----------
    tag_name : str
        Name of the missing tag.
    expected_type : type
        Expected type of the value.
    prompt_missing : bool
        If True, prompt interactively. If False, return None.

    Returns
    -------
    Any
        User-provided value or None if not prompting.
    """
    if not prompt_missing:
        return None

    try:
        value = input(f"Missing {tag_name}. Please enter value: ")
        return expected_type(value)
    except (ValueError, EOFError):
        return None


def load_dicom(
    path: str | Path,
    prompt_missing: bool = True,
    modality: Modality | None = None,
) -> PerfusionDataset:
    """Load DICOM series as PerfusionDataset.

    Parameters
    ----------
    path : str | Path
        Path to DICOM directory or single file.
    prompt_missing : bool, default=True
        If True, prompt user for missing required metadata.
        If False, raise MetadataError for missing required tags.
    modality : Modality | None
        Perfusion modality. If None, attempts to infer from DICOM tags.

    Returns
    -------
    PerfusionDataset
        Loaded imaging data with metadata.

    Raises
    ------
    FileNotFoundError
        If path does not exist.
    IOError
        If no valid DICOM files found.
    MetadataError
        If required metadata missing and prompt_missing=False.
    DataValidationError
        If data dimensions are invalid.

    Examples
    --------
    >>> from osipy.common.io.dicom import load_dicom
    >>> dataset = load_dicom("dicom_folder/", prompt_missing=True)
    >>> print(dataset.acquisition_params.tr)
    5.0

    Notes
    -----
    This implementation uses pydicom for DICOM file handling.
    Vendor-specific private tags are attempted after standard tags.
    """
    try:
        import pydicom
        from pydicom.errors import InvalidDicomError
    except ImportError as e:
        msg = "pydicom is required for DICOM loading"
        raise ImportError(msg) from e

    path = Path(path)

    if not path.exists():
        msg = f"Path not found: {path}"
        raise FileNotFoundError(msg)

    # Collect DICOM files
    dicom_files: list[Path] = []
    if path.is_file():
        dicom_files = [path]
    else:
        # Search directory for DICOM files
        for f in path.rglob("*"):
            if f.is_file() and not f.name.startswith("."):
                try:
                    pydicom.dcmread(f, stop_before_pixels=True)
                    dicom_files.append(f)
                except InvalidDicomError:
                    continue

    if not dicom_files:
        msg = f"No valid DICOM files found in: {path}"
        raise IOError(msg)

    # Sort by instance number or filename
    dicom_files = sorted(dicom_files)

    # Read first file for metadata
    first_dcm = pydicom.dcmread(dicom_files[0])

    # Extract metadata with fallbacks
    tr: float | None = None
    te: float | None = None
    flip_angle: float | None = None
    field_strength: float | None = None

    # Try standard tags
    if hasattr(first_dcm, "RepetitionTime"):
        tr = float(first_dcm.RepetitionTime)
    if hasattr(first_dcm, "EchoTime"):
        te = float(first_dcm.EchoTime)
    if hasattr(first_dcm, "FlipAngle"):
        flip_angle = float(first_dcm.FlipAngle)
    if hasattr(first_dcm, "MagneticFieldStrength"):
        field_strength = float(first_dcm.MagneticFieldStrength)

    # Warn and prompt for missing required values
    missing_tags = []
    if tr is None:
        missing_tags.append("RepetitionTime (TR)")
        logger.warning("Missing TR value in DICOM header")
        tr = _prompt_for_value("TR (ms)", float, prompt_missing)
    if te is None:
        missing_tags.append("EchoTime (TE)")
        logger.warning("Missing TE value in DICOM header")
        te = _prompt_for_value("TE (ms)", float, prompt_missing)

    # If still missing required values and not prompting, raise error
    if not prompt_missing and (tr is None or te is None):
        msg = f"Missing required DICOM metadata: {', '.join(missing_tags)}"
        raise MetadataError(msg)

    # Read all slices with metadata for sorting
    slices_info: list[dict] = []
    for dcm_file in dicom_files:
        dcm = pydicom.dcmread(dcm_file)

        # Get slice location
        slice_loc = float(getattr(dcm, "SliceLocation", 0))

        # Get temporal position using various tags
        # Priority: TemporalPositionIdentifier > AcquisitionNumber > AcquisitionTime > InstanceNumber
        temporal_pos = None

        # Try TemporalPositionIdentifier (explicit temporal index)
        if hasattr(dcm, "TemporalPositionIdentifier"):
            temporal_pos = int(dcm.TemporalPositionIdentifier)
        # Try AcquisitionNumber (often encodes temporal position)
        elif hasattr(dcm, "AcquisitionNumber"):
            temporal_pos = int(dcm.AcquisitionNumber)
        # Try AcquisitionTime (format: HHMMSS.FFFFFF)
        elif hasattr(dcm, "AcquisitionTime"):
            acq_time = str(dcm.AcquisitionTime)
            try:
                # Convert time string to seconds for sorting
                hours = int(acq_time[:2]) if len(acq_time) >= 2 else 0
                mins = int(acq_time[2:4]) if len(acq_time) >= 4 else 0
                secs = float(acq_time[4:]) if len(acq_time) > 4 else 0
                temporal_pos = hours * 3600 + mins * 60 + secs
            except (ValueError, IndexError):
                temporal_pos = None
        # Fall back to InstanceNumber divided by estimated slices per volume
        elif hasattr(dcm, "InstanceNumber"):
            temporal_pos = int(dcm.InstanceNumber)

        if temporal_pos is None:
            temporal_pos = 0

        # Capture TriggerTime separately for actual timing (in ms)
        trigger_time = None
        if hasattr(dcm, "TriggerTime"):
            trigger_time = float(dcm.TriggerTime)

        # Capture ContentTime for computing temporal resolution
        content_time = None
        if hasattr(dcm, "ContentTime"):
            ct_str = str(dcm.ContentTime)
            try:
                ct_h = int(ct_str[:2]) if len(ct_str) >= 2 else 0
                ct_m = int(ct_str[2:4]) if len(ct_str) >= 4 else 0
                ct_s = float(ct_str[4:]) if len(ct_str) > 4 else 0.0
                content_time = ct_h * 3600 + ct_m * 60 + ct_s
            except (ValueError, IndexError):
                content_time = None

        slices_info.append(
            {
                "file": dcm_file,
                "slice_loc": slice_loc,
                "temporal_pos": temporal_pos,
                "trigger_time": trigger_time,
                "content_time": content_time,
                "pixel_array": dcm.pixel_array.astype(np.float64),
            }
        )

    # Determine if this is 3D or 4D data
    # Group by unique temporal positions
    temporal_positions = sorted({s["temporal_pos"] for s in slices_info})
    n_timepoints = len(temporal_positions)

    # Get unique slice locations
    slice_locations = sorted({s["slice_loc"] for s in slices_info})
    n_slices = len(slice_locations)

    logger.info(
        f"DICOM series: {len(slices_info)} files, {n_slices} slices, {n_timepoints} timepoints"
    )

    # Check if we have proper 4D structure
    expected_files = n_slices * n_timepoints
    is_4d = n_timepoints > 1 and len(slices_info) == expected_files

    if is_4d:
        # Build 4D array: (x, y, z, t)
        # Sort slices by temporal position, then by slice location
        slices_info.sort(key=lambda x: (x["temporal_pos"], x["slice_loc"]))

        # Reshape into 4D and collect timing info
        data_4d = []
        trigger_times_ms: list[float | None] = []
        for _t_idx, t_pos in enumerate(temporal_positions):
            # Get all slices for this timepoint
            time_slices = [s for s in slices_info if s["temporal_pos"] == t_pos]
            time_slices.sort(key=lambda x: x["slice_loc"])

            # Get TriggerTime from first slice of this timepoint
            trigger_times_ms.append(time_slices[0].get("trigger_time"))

            # Stack slices for this timepoint
            volume = np.stack([s["pixel_array"] for s in time_slices], axis=-1)
            data_4d.append(volume)

        # Stack timepoints
        data = np.stack(data_4d, axis=-1)
        logger.info(f"Loaded 4D DICOM data: {data.shape}")
    else:
        # 3D data - sort by slice location only
        slices_info.sort(key=lambda x: x["slice_loc"])
        slices_data = [s["pixel_array"] for s in slices_info]

        if len(slices_data) == 1:
            data = slices_data[0]
            if data.ndim == 2:
                data = data[..., np.newaxis]  # Add z dimension
        else:
            data = np.stack(slices_data, axis=-1)
        logger.info(f"Loaded 3D DICOM data: {data.shape}")

    # Validate dimensions
    if data.ndim not in (3, 4):
        msg = f"DICOM data must be 3D or 4D, got {data.ndim}D"
        raise DataValidationError(msg)

    # Build affine from DICOM geometry tags
    slice_thickness = float(getattr(first_dcm, "SliceThickness", 1.0))
    affine = build_affine_from_dicom(first_dcm, slice_thickness)

    # Generate time points for 4D data
    time_points = None
    if data.ndim == 4:
        n_timepoints = data.shape[3]
        # Use TriggerTime if available (most accurate for DCE)
        if is_4d and trigger_times_ms and all(t is not None for t in trigger_times_ms):
            time_points = np.array(trigger_times_ms) / 1000.0  # Convert ms to s
            logger.info(
                "Using TriggerTime for timing: %.2f-%.2fs",
                time_points[0],
                time_points[-1],
            )
        elif is_4d:
            # Compute time points from ContentTime differences across
            # temporal positions (mean ContentTime per volume)
            vol_times: list[float] = []
            for t_pos in temporal_positions:
                ct_vals = [
                    s["content_time"]
                    for s in slices_info
                    if s["temporal_pos"] == t_pos and s["content_time"] is not None
                ]
                if ct_vals:
                    vol_times.append(float(np.mean(ct_vals)))
            if len(vol_times) == n_timepoints and len(vol_times) > 1:
                time_points = np.array(vol_times) - vol_times[0]
                logger.info(
                    "Using ContentTime for timing: temporal resolution %.2fs",
                    float(np.mean(np.diff(time_points))),
                )

    # Determine modality if not provided
    if modality is None:
        modality = Modality.DCE  # Default assumption

    # Build acquisition params
    acquisition_params = AcquisitionParams(
        tr=tr,
        te=te,
        flip_angle=flip_angle,
        field_strength=field_strength,
    )

    # Log summary of detected parameters
    _parts = [f"matrix {data.shape[0]}x{data.shape[1]}"]
    if data.ndim == 4:
        _parts.append(f"{data.shape[2]} slices x {data.shape[3]} frames")
    elif data.ndim == 3:
        _parts.append(f"{data.shape[2]} slices")
    if tr is not None:
        _parts.append(f"TR={tr:.2f}ms")
    if te is not None:
        _parts.append(f"TE={te:.2f}ms")
    if field_strength is not None:
        _parts.append(f"{field_strength:.1f}T")
    if time_points is not None and len(time_points) > 1:
        dt = float(np.mean(np.diff(time_points)))
        _parts.append(f"temporal res={dt:.2f}s")
    logger.info("Acquisition: %s", ", ".join(_parts))

    return PerfusionDataset(
        data=data,
        affine=affine,
        modality=modality,
        time_points=time_points,
        acquisition_params=acquisition_params,
        source_path=path,
        source_format="dicom",
    )


def _extract_time_from_series_description(description: str) -> float | None:
    """Extract time value from series description.

    Looks for common patterns like:
    - TT=49.6s (trigger time)
    - T=60s or t=60sec
    - dyn_1min, dyn_2min
    - phase_1, phase_2 (returns index)

    Parameters
    ----------
    description : str
        DICOM SeriesDescription string.

    Returns
    -------
    float | None
        Extracted time in seconds, or None if no pattern matched.
    """
    if not description:
        return None

    # Pattern: TT=49.6s (trigger time in seconds)
    match = re.search(r"TT[=_]?(\d+\.?\d*)\s*s", description, re.IGNORECASE)
    if match:
        return float(match.group(1))

    # Pattern: T=60s or time=60sec
    match = re.search(r"[Tt](?:ime)?[=_](\d+\.?\d*)\s*(?:s|sec)", description)
    if match:
        return float(match.group(1))

    # Pattern: dyn_1min, dyn_2min (minutes)
    match = re.search(r"(\d+\.?\d*)\s*min", description, re.IGNORECASE)
    if match:
        return float(match.group(1)) * 60.0

    return None


def _detect_series_timepoints(
    series_dirs: list[Path],
) -> list[tuple[Path, float | None, dict[str, Any]]]:
    """Detect timepoint ordering for multiple DICOM series.

    Attempts to extract timepoint information from DICOM metadata
    using multiple strategies:
    1. SeriesDescription patterns (TT=X.Xs, etc.)
    2. AcquisitionTime differences
    3. SeriesNumber ordering

    Parameters
    ----------
    series_dirs : list[Path]
        List of DICOM series directories.

    Returns
    -------
    list[tuple[Path, float | None, dict]]
        List of (directory, time_seconds, metadata) tuples sorted by time.
    """
    try:
        import pydicom
    except ImportError as e:
        msg = "pydicom is required for DICOM loading"
        raise ImportError(msg) from e

    series_info: list[tuple[Path, float | None, dict[str, Any]]] = []

    for series_dir in series_dirs:
        # Find first DICOM file in series
        dcm_files = list(series_dir.glob("*.dcm"))
        if not dcm_files:
            # Try files without extension
            dcm_files = [
                f
                for f in series_dir.iterdir()
                if f.is_file() and not f.name.startswith(".")
            ]
        if not dcm_files:
            logger.warning(f"No DICOM files found in {series_dir}")
            continue

        try:
            dcm = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
        except Exception as e:
            logger.warning(f"Could not read DICOM from {series_dir}: {e}")
            continue

        # Skip non-image DICOM objects (Key Object Selection, Structured
        # Reports, Presentation States, etc.) which have no pixel data.
        dicom_modality = getattr(dcm, "Modality", "")
        if dicom_modality in _NON_IMAGE_MODALITIES:
            logger.info(
                "Skipping non-image series %s (Modality=%s, Description=%s)",
                series_dir.name,
                dicom_modality,
                getattr(dcm, "SeriesDescription", ""),
            )
            continue

        # Extract metadata for ordering
        description = getattr(dcm, "SeriesDescription", "")
        series_number = int(getattr(dcm, "SeriesNumber", 0))
        acq_time_str = getattr(dcm, "AcquisitionTime", "")

        # Try to get time from description first
        time_sec = _extract_time_from_series_description(description)

        # Parse AcquisitionTime as fallback for relative ordering
        acq_time_sec = None
        if acq_time_str:
            try:
                hours = int(acq_time_str[:2]) if len(acq_time_str) >= 2 else 0
                mins = int(acq_time_str[2:4]) if len(acq_time_str) >= 4 else 0
                secs = float(acq_time_str[4:]) if len(acq_time_str) > 4 else 0
                acq_time_sec = hours * 3600 + mins * 60 + secs
            except (ValueError, IndexError):
                pass

        metadata = {
            "series_description": description,
            "series_number": series_number,
            "acquisition_time_sec": acq_time_sec,
            "n_files": len(dcm_files),
            "tr": getattr(dcm, "RepetitionTime", None),
            "te": getattr(dcm, "EchoTime", None),
            "flip_angle": getattr(dcm, "FlipAngle", None),
            "field_strength": getattr(dcm, "MagneticFieldStrength", None),
        }

        series_info.append((series_dir, time_sec, metadata))

    # Sort by extracted time if available, otherwise by series number
    has_times = all(t[1] is not None for t in series_info)
    if has_times:
        series_info.sort(key=lambda x: x[1])  # type: ignore[arg-type, return-value]
    else:
        # Try AcquisitionTime
        has_acq_times = all(
            t[2].get("acquisition_time_sec") is not None for t in series_info
        )
        if has_acq_times:
            series_info.sort(key=lambda x: x[2]["acquisition_time_sec"])
        else:
            # Fall back to series number
            series_info.sort(key=lambda x: x[2]["series_number"])

    return series_info


def load_dicom_multi_series(
    series_dirs: list[str | Path],
    time_points: list[float] | None = None,
    prompt_missing: bool = True,
    modality: Modality | None = None,
) -> PerfusionDataset:
    """Load DCE data from multiple DICOM series directories.

    This function handles datasets where each DCE timepoint is stored
    in a separate DICOM series directory (common in TCIA datasets like
    QIN-Breast-DCE-MRI).

    Parameters
    ----------
    series_dirs : list[str | Path]
        List of paths to DICOM series directories, one per timepoint.
        Can be provided in any order - the function will auto-detect
        temporal ordering from DICOM metadata.
    time_points : list[float] | None, default=None
        Explicit time points in seconds for each series (after sorting).
        If None, times are extracted from DICOM metadata (SeriesDescription
        patterns like TT=X.Xs, AcquisitionTime, etc.).
    prompt_missing : bool, default=True
        If True, prompt user for missing required metadata.
        If False, raise MetadataError for missing required tags.
    modality : Modality | None
        Perfusion modality. Defaults to DCE if None.

    Returns
    -------
    PerfusionDataset
        4D dataset with shape (x, y, z, n_timepoints) and time_points array.

    Raises
    ------
    IOError
        If series directories don't exist or have inconsistent dimensions.
    DataValidationError
        If series have mismatched spatial dimensions.

    Examples
    --------
    Load QIN-Breast-DCE-MRI with auto-detected timepoints:

    >>> from osipy.common.io.dicom import load_dicom_multi_series
    >>> series = [
    ...     "data/series_tt49.6s/",
    ...     "data/series_tt69.7s/",
    ...     "data/series_tt89.9s/",
    ... ]
    >>> dataset = load_dicom_multi_series(series)
    >>> print(dataset.time_points)
    [49.6, 69.7, 89.9]

    Load with explicit time points:

    >>> dataset = load_dicom_multi_series(
    ...     series_dirs=series,
    ...     time_points=[0, 20.1, 40.2],  # Relative times in seconds
    ... )

    Notes
    -----
    The function automatically:
    - Detects temporal ordering from SeriesDescription, AcquisitionTime,
      or SeriesNumber
    - Validates that all series have matching spatial dimensions
    - Builds proper NIfTI affine from DICOM geometry tags
    - Extracts acquisition parameters (TR, TE, flip angle) from first series

    """
    try:
        import pydicom
    except ImportError as e:
        msg = "pydicom is required for DICOM loading"
        raise ImportError(msg) from e

    # Convert to Path objects and validate
    series_paths = [Path(d) for d in series_dirs]
    for p in series_paths:
        if not p.exists():
            msg = f"Series directory not found: {p}"
            raise FileNotFoundError(msg)
        if not p.is_dir():
            msg = f"Path is not a directory: {p}"
            raise IOError(msg)

    if len(series_paths) < 2:
        msg = "At least 2 series directories required for multi-series loading"
        raise DataValidationError(msg)

    # Detect temporal ordering
    logger.info(f"Detecting temporal ordering for {len(series_paths)} series...")
    series_info = _detect_series_timepoints(series_paths)

    if len(series_info) == 0:
        msg = "No loadable image series found in the provided directories"
        raise IOError(msg)

    if len(series_info) != len(series_paths):
        logger.info(
            "Using %d of %d series (non-image series were skipped)",
            len(series_info),
            len(series_paths),
        )

    # Log detected ordering
    for i, (path, time_sec, meta) in enumerate(series_info):
        desc = meta["series_description"][:40] if meta["series_description"] else "N/A"
        time_str = f"{time_sec:.1f}s" if time_sec is not None else "N/A"
        logger.info(f"  [{i}] {path.name[:30]}... | time={time_str} | {desc}")

    # Load each series as a 3D volume
    volumes: list[NDArray[np.floating[Any]]] = []
    detected_times: list[float] = []
    reference_shape: tuple[int, ...] | None = None
    reference_affine: np.ndarray | None = None
    first_dcm: Any = None

    for series_idx, (series_dir, time_sec, _meta) in enumerate(series_info):
        logger.info(
            f"Loading series {series_idx + 1}/{len(series_info)}: {series_dir.name[:40]}..."
        )

        # Collect DICOM files
        dcm_files = sorted(series_dir.glob("*.dcm"))
        if not dcm_files:
            dcm_files = sorted(
                [
                    f
                    for f in series_dir.iterdir()
                    if f.is_file() and not f.name.startswith(".")
                ]
            )

        # Read slices with location info
        slices_info: list[dict[str, Any]] = []
        for dcm_file in dcm_files:
            try:
                dcm = pydicom.dcmread(dcm_file)
                slice_loc = float(getattr(dcm, "SliceLocation", 0))
                slices_info.append(
                    {
                        "file": dcm_file,
                        "slice_loc": slice_loc,
                        "pixel_array": dcm.pixel_array.astype(np.float64),
                    }
                )
                if first_dcm is None:
                    first_dcm = dcm
            except Exception as e:
                logger.warning(f"Could not read {dcm_file}: {e}")
                continue

        if not slices_info:
            msg = f"No readable DICOM files in {series_dir}"
            raise IOError(msg)

        # Sort by slice location and stack
        slices_info.sort(key=lambda x: x["slice_loc"])
        slices_data = [
            s["pixel_array"].T for s in slices_info
        ]  # Transpose for NIfTI convention
        volume = np.stack(slices_data, axis=-1)

        # Validate consistent dimensions
        if reference_shape is None:
            reference_shape = volume.shape
            # Build affine from first series
            slice_thickness = float(getattr(first_dcm, "SliceThickness", 1.0))
            reference_affine = build_affine_from_dicom(first_dcm, slice_thickness)
        elif volume.shape != reference_shape:
            msg = (
                f"Series dimension mismatch: {series_dir.name} has shape "
                f"{volume.shape}, expected {reference_shape}"
            )
            raise DataValidationError(msg)

        volumes.append(volume)
        detected_times.append(time_sec if time_sec is not None else float(series_idx))

    # Stack into 4D array
    data_4d = np.stack(volumes, axis=-1)
    logger.info(f"Combined 4D data: {data_4d.shape}")

    # Use provided time_points or detected times
    if time_points is not None:
        if len(time_points) != data_4d.shape[3]:
            msg = (
                f"Provided {len(time_points)} time_points but have "
                f"{data_4d.shape[3]} volumes"
            )
            raise DataValidationError(msg)
        final_times = np.array(time_points)
    else:
        final_times = np.array(detected_times)

    # Zero-reference time so first timepoint is t=0.
    # Detected times (e.g. from SeriesDescription TT=49.6s or
    # AcquisitionTime) are often absolute scanner times, not
    # relative to the start of the dynamic acquisition.
    if len(final_times) > 0 and final_times[0] != 0.0:
        logger.info("Zero-referencing time vector (offset %.1fs)", final_times[0])
        final_times = final_times - final_times[0]

    # Extract acquisition params from first file
    tr: float | None = None
    te: float | None = None
    flip_angle: float | None = None
    field_strength: float | None = None

    if first_dcm is not None:
        if hasattr(first_dcm, "RepetitionTime"):
            tr = float(first_dcm.RepetitionTime)
        if hasattr(first_dcm, "EchoTime"):
            te = float(first_dcm.EchoTime)
        if hasattr(first_dcm, "FlipAngle"):
            flip_angle = float(first_dcm.FlipAngle)
        if hasattr(first_dcm, "MagneticFieldStrength"):
            field_strength = float(first_dcm.MagneticFieldStrength)

    # Handle missing metadata
    if tr is None and prompt_missing:
        tr = _prompt_for_value("TR (ms)", float, prompt_missing)
    if te is None and prompt_missing:
        te = _prompt_for_value("TE (ms)", float, prompt_missing)

    if not prompt_missing and (tr is None or te is None):
        msg = "Missing required DICOM metadata: TR and/or TE"
        raise MetadataError(msg)

    # Determine modality
    if modality is None:
        modality = Modality.DCE

    acquisition_params = AcquisitionParams(
        tr=tr,
        te=te,
        flip_angle=flip_angle,
        field_strength=field_strength,
    )

    return PerfusionDataset(
        data=data_4d,
        affine=reference_affine,
        modality=modality,
        time_points=final_times,
        acquisition_params=acquisition_params,
        source_path=series_paths[0].parent,
        source_format="dicom",
    )
