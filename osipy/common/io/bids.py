"""BIDS I/O for osipy.

This module provides functions for loading and exporting perfusion
data in BIDS-compliant format, including ASL-BIDS input support.

References
----------
BIDS Specification: https://bids-specification.readthedocs.io/
BIDS Derivatives: https://bids-specification.readthedocs.io/en/stable/derivatives/
ASL-BIDS: https://bids-specification.readthedocs.io/en/stable/modality-specific-files/magnetic-resonance-imaging-data.html#arterial-spin-labeling-perfusion-data
"""

import csv
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from osipy._version import __version__
from osipy.common.dataset import PerfusionDataset
from osipy.common.exceptions import IOError as OsipyIOError
from osipy.common.io.metadata.mapper import MetadataMapper
from osipy.common.parameter_map import ParameterMap
from osipy.common.types import Modality

logger = logging.getLogger(__name__)


def export_bids(
    parameter_maps: dict[str, ParameterMap],
    output_dir: str | Path,
    subject_id: str,
    session_id: str | None = None,
    metadata: dict | None = None,
) -> Path:
    """Export parameter maps in BIDS derivatives format.

    Parameters
    ----------
    parameter_maps : dict[str, ParameterMap]
        Parameter maps to export, keyed by parameter name.
    output_dir : str | Path
        BIDS derivatives root directory.
    subject_id : str
        Subject identifier (without 'sub-' prefix).
    session_id : str | None
        Session identifier (without 'ses-' prefix), optional.
    metadata : dict | None
        Additional metadata for sidecar JSON files.

    Returns
    -------
    Path
        Path to created subject directory.

    Raises
    ------
    ValueError
        If parameter_maps is empty or subject_id is invalid.

    Examples
    --------
    >>> from osipy.common.io.bids import export_bids
    >>> results = {"Ktrans": ktrans_map, "ve": ve_map}
    >>> path = export_bids(results, "derivatives/osipy", "01")
    >>> print(path)
    derivatives/osipy/sub-01/perf

    Notes
    -----
    Creates the following structure:
    ```
    derivatives/osipy/
    ├── dataset_description.json
    └── sub-XX/
        └── [ses-XX/]
            └── perf/
                ├── sub-XX_Ktrans.nii.gz
                ├── sub-XX_ve.nii.gz
                └── sub-XX_perf.json
    ```
    """
    if not parameter_maps:
        msg = "parameter_maps cannot be empty"
        raise OsipyIOError(msg)

    if not subject_id:
        msg = "subject_id cannot be empty"
        raise OsipyIOError(msg)

    output_dir = Path(output_dir)

    # Create dataset_description.json if it doesn't exist
    dataset_desc_path = output_dir / "dataset_description.json"
    if not dataset_desc_path.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_description = {
            "Name": "osipy perfusion analysis",
            "BIDSVersion": "1.9.0",
            "DatasetType": "derivative",
            "GeneratedBy": [
                {
                    "Name": "osipy",
                    "Version": __version__,
                    "CodeURL": "https://github.com/OSIPI/osipy",
                }
            ],
        }
        with dataset_desc_path.open("w") as f:
            json.dump(dataset_description, f, indent=2)

    # Build subject directory path
    subject_dir = output_dir / f"sub-{subject_id}"
    if session_id:
        subject_dir = subject_dir / f"ses-{session_id}"
    perf_dir = subject_dir / "perf"
    perf_dir.mkdir(parents=True, exist_ok=True)

    # Build filename prefix
    prefix = f"sub-{subject_id}"
    if session_id:
        prefix = f"{prefix}_ses-{session_id}"

    # Export each parameter map
    for param_name, param_map in parameter_maps.items():
        # Create NIfTI image
        nifti_img = nib.Nifti1Image(
            param_map.values.astype(np.float32),
            param_map.affine,
        )

        # Set description in header
        # NIfTI intent_name only supports ASCII, so encode/decode to sanitize
        intent_name = f"{param_name} ({param_map.units})"
        intent_name_ascii = intent_name.encode("ascii", errors="replace").decode(
            "ascii"
        )
        nifti_img.header.set_intent(
            "none",
            name=intent_name_ascii[:16],  # NIfTI intent_name max 16 chars
        )

        # Save NIfTI file
        filename = f"{prefix}_{param_name}.nii.gz"
        nifti_path = perf_dir / filename
        nib.save(nifti_img, nifti_path)

        logger.info(f"Exported {param_name} to {nifti_path}")

    # Create sidecar JSON with provenance
    sidecar = {
        "Description": "Perfusion parameter maps from osipy analysis",
        "GeneratedBy": {
            "Name": "osipy",
            "Version": __version__,
        },
        "Timestamp": datetime.now(UTC).isoformat(),
        "Parameters": {},
    }

    # Add parameter-specific metadata
    for param_name, param_map in parameter_maps.items():
        sidecar["Parameters"][param_name] = {
            "Units": param_map.units,
            "Model": param_map.model_name,
            "FittingMethod": param_map.fitting_method,
            "Reference": param_map.literature_reference,
            "ValidFraction": param_map.valid_fraction,
            "Statistics": param_map.statistics(),
        }

    # Add custom metadata if provided
    if metadata:
        sidecar.update(metadata)

    # Write sidecar JSON
    sidecar_path = perf_dir / f"{prefix}_perf.json"
    with sidecar_path.open("w") as f:
        json.dump(sidecar, f, indent=2)

    logger.info(f"Exported {len(parameter_maps)} parameter maps to {perf_dir}")

    return perf_dir


def load_bids(
    bids_dir: str | Path,
    subject: str,
    modality: Modality = Modality.ASL,
    session: str | None = None,
    run: int | None = None,
    interactive: bool = True,
) -> PerfusionDataset:
    """Load perfusion data from a BIDS dataset.

    Parameters
    ----------
    bids_dir : str | Path
        Path to BIDS dataset root directory.
    subject : str
        Subject ID (without 'sub-' prefix).
    modality : Modality, default=Modality.ASL
        Perfusion modality to load.
    session : str | None
        Session ID (without 'ses-' prefix), optional.
    run : int | None
        Run number, optional.
    interactive : bool, default=True
        Whether to prompt for missing required parameters.

    Returns
    -------
    PerfusionDataset
        Loaded perfusion data with metadata.

    Raises
    ------
    FileNotFoundError
        If BIDS directory or required files not found.
    IOError
        If BIDS dataset structure is invalid.
    MetadataError
        If required metadata is missing and interactive=False.

    Examples
    --------
    >>> from osipy.common.io.bids import load_bids
    >>> from osipy.common.types import Modality
    >>> dataset = load_bids("bids_dataset/", "01", Modality.ASL)
    >>> print(dataset.shape)
    (64, 64, 24, 80)

    """
    bids_dir = Path(bids_dir)

    if not bids_dir.exists():
        msg = f"BIDS directory not found: {bids_dir}"
        raise FileNotFoundError(msg)

    # Check for dataset_description.json
    dataset_desc = bids_dir / "dataset_description.json"
    if not dataset_desc.exists():
        logger.warning("Missing dataset_description.json - may not be valid BIDS")

    # Build path to subject data
    subject_dir = bids_dir / f"sub-{subject}"
    if session:
        subject_dir = subject_dir / f"ses-{session}"

    if not subject_dir.exists():
        msg = f"Subject directory not found: {subject_dir}"
        raise FileNotFoundError(msg)

    # Determine data type directory based on modality
    # Check multiple potential locations for compatibility with different BIDS structures
    if modality == Modality.ASL:
        # ASL data can be in perf/ or directly in subject directory
        data_dir = subject_dir / "perf"
        if not data_dir.exists():
            # Some ASL-BIDS datasets have files directly in subject dir
            data_dir = subject_dir
    elif modality == Modality.IVIM:
        data_dir = subject_dir / "dwi"
        if not data_dir.exists():
            data_dir = subject_dir
    else:
        # DCE/DSC typically in func or perf
        data_dir = subject_dir / "perf"
        if not data_dir.exists():
            data_dir = subject_dir / "func"
        if not data_dir.exists():
            data_dir = subject_dir

    if not data_dir.exists():
        msg = f"Data directory not found: {data_dir}"
        raise FileNotFoundError(msg)

    # Find the data files
    nifti_file, sidecar_file = _find_bids_files(
        data_dir, subject, session, modality, run
    )

    # Load sidecar JSON
    sidecar: dict[str, Any] = {}
    if sidecar_file and sidecar_file.exists():
        with sidecar_file.open() as f:
            sidecar = json.load(f)
    else:
        logger.warning(f"No sidecar JSON found for {nifti_file}")

    # Load NIfTI data
    nifti_img = nib.load(nifti_file)
    data = np.asarray(nifti_img.dataobj, dtype=np.float64)
    affine = nifti_img.affine

    # Generate time points if 4D
    time_points = None
    if data.ndim == 4:
        n_volumes = data.shape[3]
        tr = sidecar.get("RepetitionTimePreparation") or sidecar.get("RepetitionTime")
        if tr:
            # BIDS stores TR in seconds
            tr_sec = float(tr)
            time_points = np.arange(n_volumes) * tr_sec
        else:
            time_points = np.arange(n_volumes, dtype=np.float64)

    # Map metadata to acquisition params
    mapper = MetadataMapper(modality, interactive=interactive)
    acquisition_params = mapper.map_to_acquisition_params(bids_sidecar=sidecar)

    return PerfusionDataset(
        data=data,
        affine=affine,
        modality=modality,
        time_points=time_points,
        acquisition_params=acquisition_params,
        source_path=nifti_file,
        source_format="bids",
    )


def load_bids_with_m0(
    bids_dir: str | Path,
    subject: str,
    session: str | None = None,
    run: int | None = None,
    interactive: bool = True,
) -> tuple[PerfusionDataset, PerfusionDataset | None]:
    """Load ASL data with separate M0 calibration scan from BIDS.

    Parameters
    ----------
    bids_dir : str | Path
        Path to BIDS dataset root directory.
    subject : str
        Subject ID (without 'sub-' prefix).
    session : str | None
        Session ID (without 'ses-' prefix), optional.
    run : int | None
        Run number, optional.
    interactive : bool, default=True
        Whether to prompt for missing required parameters.

    Returns
    -------
    tuple[PerfusionDataset, PerfusionDataset | None]
        Tuple of (ASL data, M0 calibration data).
        M0 is None if not found.

    Examples
    --------
    >>> asl_data, m0_data = load_bids_with_m0("bids_dataset/", "01")
    >>> if m0_data is not None:
    ...     print(f"M0 shape: {m0_data.shape}")

    Notes
    -----
    BIDS ASL M0 scans can be:
    - Separate file: `*_m0scan.nii.gz`
    - Included in ASL timeseries (indicated by M0Type in sidecar)
    """
    bids_dir = Path(bids_dir)

    # Load main ASL data
    asl_dataset = load_bids(bids_dir, subject, Modality.ASL, session, run, interactive)

    # Try to find M0 scan
    m0_dataset = None

    subject_dir = bids_dir / f"sub-{subject}"
    if session:
        subject_dir = subject_dir / f"ses-{session}"

    # Check multiple locations for M0 file (perf/ and subject directory)
    perf_dir = subject_dir / "perf"
    search_dirs = [perf_dir, subject_dir] if perf_dir.exists() else [subject_dir]

    m0_file = None
    for search_dir in search_dirs:
        if search_dir.exists():
            m0_file = _find_m0_file(search_dir, subject, session, run)
            if m0_file and m0_file.exists():
                break

    if m0_file and m0_file.exists():
        # Load M0 NIfTI
        m0_img = nib.load(m0_file)
        m0_data = np.asarray(m0_img.dataobj, dtype=np.float64)

        # Handle 4D M0 data (e.g., multiple averages)
        # Average along 4th dimension to create 3D volume
        if m0_data.ndim == 4:
            logger.info(f"M0 scan is 4D with shape {m0_data.shape}, averaging")
            m0_data = m0_data.mean(axis=-1)

        m0_dataset = PerfusionDataset(
            data=m0_data,
            affine=m0_img.affine,
            modality=Modality.ASL,
            time_points=None,
            acquisition_params=asl_dataset.acquisition_params,
            source_path=m0_file,
            source_format="bids",
        )
        logger.info(f"Loaded M0 calibration scan: {m0_file}")
    else:
        logger.info("No separate M0 scan found")

    return asl_dataset, m0_dataset


def load_asl_context(
    bids_dir: str | Path,
    subject: str,
    session: str | None = None,
    run: int | None = None,
) -> list[str]:
    """Load ASL context (control/label order) from BIDS.

    Parameters
    ----------
    bids_dir : str | Path
        Path to BIDS dataset root directory.
    subject : str
        Subject ID (without 'sub-' prefix).
    session : str | None
        Session ID, optional.
    run : int | None
        Run number, optional.

    Returns
    -------
    list[str]
        List of volume types: 'control', 'label', 'm0scan', 'cbf', 'deltam'.

    Examples
    --------
    >>> context = load_asl_context("bids_dataset/", "01")
    >>> print(context[:4])
    ['control', 'label', 'control', 'label']

    Notes
    -----
    The aslcontext.tsv file is required by BIDS ASL and indicates
    the type of each volume in the ASL timeseries.
    """
    bids_dir = Path(bids_dir)

    subject_dir = bids_dir / f"sub-{subject}"
    if session:
        subject_dir = subject_dir / f"ses-{session}"

    # Check multiple locations for context file
    perf_dir = subject_dir / "perf"
    search_dirs = [perf_dir, subject_dir] if perf_dir.exists() else [subject_dir]

    # Find context file
    context_file = None
    for search_dir in search_dirs:
        if search_dir.exists():
            context_file = _find_context_file(search_dir, subject, session, run)
            if context_file and context_file.exists():
                break

    if not context_file or not context_file.exists():
        msg = f"ASL context file not found in {perf_dir}"
        raise FileNotFoundError(msg)

    return _parse_asl_context(context_file)


def _find_bids_files(
    data_dir: Path,
    subject: str,
    session: str | None,
    modality: Modality,
    run: int | None,
) -> tuple[Path, Path | None]:
    """Find BIDS NIfTI and sidecar files.

    Parameters
    ----------
    data_dir : Path
        Data type directory (perf/, dwi/, func/).
    subject : str
        Subject ID.
    session : str | None
        Session ID.
    modality : Modality
        Perfusion modality.
    run : int | None
        Run number.

    Returns
    -------
    tuple[Path, Path | None]
        (NIfTI file path, sidecar JSON path or None).
    """
    # Build filename pattern
    prefix = f"sub-{subject}"
    if session:
        prefix = f"{prefix}_ses-{session}"
    if run:
        prefix = f"{prefix}_run-{run:02d}"

    # Modality suffix
    suffix_map = {
        Modality.ASL: "_asl",
        Modality.IVIM: "_dwi",
        Modality.DCE: "_perf",
        Modality.DSC: "_perf",
    }
    suffix = suffix_map.get(modality, "_perf")

    # Look for NIfTI file
    nifti_pattern = f"{prefix}*{suffix}.nii*"
    nifti_files = list(data_dir.glob(nifti_pattern))

    if not nifti_files:
        # Try without run
        nifti_pattern = f"sub-{subject}*{suffix}.nii*"
        nifti_files = list(data_dir.glob(nifti_pattern))

    if not nifti_files:
        msg = f"No {modality.value} NIfTI files found in {data_dir}"
        raise FileNotFoundError(msg)

    # Prefer .nii.gz
    nifti_file = None
    for f in nifti_files:
        if f.name.endswith(".nii.gz"):
            nifti_file = f
            break
    if nifti_file is None:
        nifti_file = nifti_files[0]

    # Find matching sidecar
    if nifti_file.name.endswith(".nii.gz"):
        sidecar_name = nifti_file.name[:-7] + ".json"
    else:
        sidecar_name = nifti_file.stem + ".json"
    sidecar_file = data_dir / sidecar_name

    return nifti_file, sidecar_file if sidecar_file.exists() else None


def _find_m0_file(
    perf_dir: Path,
    subject: str,
    session: str | None,
    run: int | None,
) -> Path | None:
    """Find M0 calibration scan file.

    Parameters
    ----------
    perf_dir : Path
        Perfusion data directory.
    subject : str
        Subject ID.
    session : str | None
        Session ID.
    run : int | None
        Run number.

    Returns
    -------
    Path | None
        M0 file path or None if not found.
    """
    prefix = f"sub-{subject}"
    if session:
        prefix = f"{prefix}_ses-{session}"
    if run:
        prefix = f"{prefix}_run-{run:02d}"

    # Look for M0 scan
    m0_pattern = f"{prefix}*_m0scan.nii*"
    m0_files = list(perf_dir.glob(m0_pattern))

    if not m0_files:
        # Try without run
        m0_pattern = f"sub-{subject}*_m0scan.nii*"
        m0_files = list(perf_dir.glob(m0_pattern))

    if m0_files:
        # Prefer .nii.gz
        for f in m0_files:
            if f.name.endswith(".nii.gz"):
                return f
        return m0_files[0]

    return None


def _find_context_file(
    perf_dir: Path,
    subject: str,
    session: str | None,
    run: int | None,
) -> Path | None:
    """Find ASL context TSV file.

    Parameters
    ----------
    perf_dir : Path
        Perfusion data directory.
    subject : str
        Subject ID.
    session : str | None
        Session ID.
    run : int | None
        Run number.

    Returns
    -------
    Path | None
        Context file path or None if not found.
    """
    prefix = f"sub-{subject}"
    if session:
        prefix = f"{prefix}_ses-{session}"
    if run:
        prefix = f"{prefix}_run-{run:02d}"

    # Look for context file
    context_pattern = f"{prefix}*_aslcontext.tsv"
    context_files = list(perf_dir.glob(context_pattern))

    if not context_files:
        # Try without run
        context_pattern = f"sub-{subject}*_aslcontext.tsv"
        context_files = list(perf_dir.glob(context_pattern))

    return context_files[0] if context_files else None


def _parse_asl_context(context_file: Path) -> list[str]:
    """Parse ASL context TSV file.

    Parameters
    ----------
    context_file : Path
        Path to aslcontext.tsv file.

    Returns
    -------
    list[str]
        List of volume types.
    """
    context: list[str] = []

    with context_file.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            volume_type = row.get("volume_type", "").lower()
            context.append(volume_type)

    return context


def is_bids_dataset(path: str | Path) -> bool:
    """Check if a path is a valid BIDS dataset.

    Parameters
    ----------
    path : str | Path
        Path to check.

    Returns
    -------
    bool
        True if path appears to be a BIDS dataset.
    """
    path = Path(path)
    if not path.is_dir():
        return False

    # Check for dataset_description.json
    return (path / "dataset_description.json").exists()


def get_bids_subjects(bids_dir: str | Path) -> list[str]:
    """Get list of subjects in a BIDS dataset.

    Parameters
    ----------
    bids_dir : str | Path
        Path to BIDS dataset root.

    Returns
    -------
    list[str]
        List of subject IDs (without 'sub-' prefix).
    """
    bids_dir = Path(bids_dir)
    subjects = []

    for item in bids_dir.iterdir():
        if item.is_dir() and item.name.startswith("sub-"):
            subjects.append(item.name[4:])  # Remove 'sub-' prefix

    return sorted(subjects)
