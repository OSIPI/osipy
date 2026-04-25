"""I/O utilities for osipy.

This module provides functions for loading and exporting
perfusion imaging data in various formats including NIfTI,
DICOM, and BIDS.

Key functions:
- `discover_dicom()`: Scan a directory for DICOM series
- `load_dicom_series()`: Load one or more discovered series
- `load_nifti()`: Load NIfTI files
- `load_bids()`: Load from BIDS dataset
- `export_bids()`: Export to BIDS derivatives

"""

from osipy.common.io.bids import (
    export_bids,
    get_bids_subjects,
    is_bids_dataset,
    load_asl_context,
    load_bids,
    load_bids_with_m0,
)
from osipy.common.io.dicom import build_affine_from_dicom
from osipy.common.io.discovery import SeriesInfo, discover_dicom, load_dicom_series
from osipy.common.io.nifti import load_nifti, save_nifti

__all__ = [
    "SeriesInfo",
    "build_affine_from_dicom",
    # DICOM discovery + loading
    "discover_dicom",
    "export_bids",
    "get_bids_subjects",
    "is_bids_dataset",
    "load_asl_context",
    # BIDS I/O
    "load_bids",
    "load_bids_with_m0",
    "load_dicom_series",
    # NIfTI I/O
    "load_nifti",
    "save_nifti",
]
