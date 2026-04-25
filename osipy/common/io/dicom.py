"""DICOM geometry + pixel-scaling helpers for osipy.

All series discovery and PerfusionDataset assembly live in
:mod:`osipy.common.io.discovery`. This module retains the stateless
helpers that module relies on (affine construction, Philips pixel
scaling, private-tag reads, SeriesDescription time extraction).

References
----------
DICOM Standard: https://www.dicomstandard.org/
"""

import logging
import re
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# DICOM Modality values that represent non-image objects (no pixel data).
# KO = Key Object Selection, SR = Structured Report, PR = Presentation State,
# AU = Audio, DOC = Document, PLAN = RT Plan, REG = Registration.
_NON_IMAGE_MODALITIES = frozenset({"KO", "SR", "PR", "AU", "DOC", "PLAN", "REG"})


def _read_private_tag(dcm: Any, group: int, element: int) -> Any:
    """Safely read a private DICOM tag value, or return None.

    Parameters
    ----------
    dcm : pydicom.Dataset
        DICOM dataset.
    group : int
        Private tag group (e.g., 0x2005).
    element : int
        Private tag element (e.g., 0x100E).

    Returns
    -------
    Any
        The tag's ``.value`` attribute, or None if the tag is missing.
    """
    tag = (group, element)
    if tag in dcm:
        elem = dcm[tag]
        return elem.value
    return None


def _apply_pixel_scaling(dcm: Any, pixel_array: Any) -> np.ndarray:
    """Rescale DICOM stored pixel values to physical values.

    Applied rules, in priority order:

    1. **Philips private quantitative rescale** — when the scanner is Philips
       and the private tags ``(2005,100E)`` ScaleSlope and ``(2005,100D)``
       ScaleIntercept are present, return
       ``FP = (stored - ScaleIntercept) / ScaleSlope``.
       Per the Philips DICOM conformance statement, this produces the
       quantitative floating-point value intended for analysis (the standard
       RescaleSlope/Intercept are intended for display on Philips and are
       superseded here).
    2. **Standard DICOM rescale** — otherwise (or if the Philips private
       tags are missing / invalid), apply
       ``value = stored * RescaleSlope + RescaleIntercept``.
    3. **No-op** — if neither set of tags is applicable, return the array
       cast to float64.

    Never raises: malformed tags fall back to the next rule with a warning.

    Parameters
    ----------
    dcm : pydicom.Dataset
        Source DICOM dataset (used to look up rescale tags).
    pixel_array : array-like
        Stored pixel array from ``dcm.pixel_array``.

    Returns
    -------
    np.ndarray
        Rescaled float64 array with the same shape as ``pixel_array``.
    """
    data = np.asarray(pixel_array, dtype=np.float64)

    manufacturer = str(getattr(dcm, "Manufacturer", "")).upper()
    if "PHILIPS" in manufacturer:
        scale_slope_raw = _read_private_tag(dcm, 0x2005, 0x100E)
        scale_intercept_raw = _read_private_tag(dcm, 0x2005, 0x100D)
        if scale_slope_raw is not None:
            try:
                scale_slope = float(scale_slope_raw)
                scale_intercept = (
                    float(scale_intercept_raw)
                    if scale_intercept_raw is not None
                    else 0.0
                )
            except (TypeError, ValueError):
                logger.warning(
                    "Philips private scale tags present but non-numeric "
                    "(slope=%r, intercept=%r); falling back to standard rescale.",
                    scale_slope_raw,
                    scale_intercept_raw,
                )
            else:
                if scale_slope == 0.0:
                    logger.warning(
                        "Philips ScaleSlope (2005,100E) is zero; "
                        "skipping quantitative rescale."
                    )
                else:
                    logger.debug(
                        "Applied Philips quantitative rescale: slope=%s, intercept=%s",
                        scale_slope,
                        scale_intercept,
                    )
                    return (data - scale_intercept) / scale_slope

    # Standard DICOM rescale fallback
    rescale_slope_raw = getattr(dcm, "RescaleSlope", None)
    rescale_intercept_raw = getattr(dcm, "RescaleIntercept", None)
    try:
        slope = float(rescale_slope_raw) if rescale_slope_raw is not None else 1.0
        intercept = (
            float(rescale_intercept_raw) if rescale_intercept_raw is not None else 0.0
        )
    except (TypeError, ValueError):
        logger.warning(
            "DICOM RescaleSlope/Intercept non-numeric (slope=%r, intercept=%r); "
            "returning unscaled pixel values.",
            rescale_slope_raw,
            rescale_intercept_raw,
        )
        return data

    if slope != 1.0 or intercept != 0.0:
        data = data * slope + intercept

    return data


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
