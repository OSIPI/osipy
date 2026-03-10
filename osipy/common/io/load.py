"""Unified data loading interface for osipy.

This module provides a universal `load_perfusion()` function that
auto-detects file format and loads perfusion data from DICOM, NIfTI,
or BIDS sources.

"""

import logging
from pathlib import Path
from typing import Any, Literal

from osipy.common.dataset import PerfusionDataset
from osipy.common.exceptions import IOError as OsipyIOError
from osipy.common.types import Modality

logger = logging.getLogger(__name__)


def load_perfusion(
    path: str | Path,
    modality: Modality | str | None = None,
    format: Literal["auto", "nifti", "dicom", "bids"] = "auto",
    subject: str | None = None,
    session: str | None = None,
    interactive: bool = True,
    use_dcm2niix: bool = True,
    sidecar_json: str | Path | None = None,
    **kwargs: Any,
) -> PerfusionDataset:
    """Universal perfusion data loader with format auto-detection.

    This is the primary entry point for loading perfusion imaging data.
    It automatically detects the file format and delegates to the
    appropriate loader.

    Parameters
    ----------
    path : str | Path
        Path to data. Can be:
        - NIfTI file (.nii or .nii.gz)
        - DICOM directory or file
        - BIDS dataset directory
    modality : Modality | str | None
        Perfusion modality (DCE, DSC, ASL, IVIM). If None, attempts
        to infer from data or defaults to DCE.
    format : {'auto', 'nifti', 'dicom', 'bids'}, default='auto'
        Data format. If 'auto', format is detected from path.
    subject : str | None
        Subject ID for BIDS loading (without 'sub-' prefix).
        Required when format='bids'.
    session : str | None
        Session ID for BIDS loading (without 'ses-' prefix).
    interactive : bool, default=True
        Whether to prompt for missing required parameters.
    use_dcm2niix : bool, default=True
        Whether to use dcm2niix for DICOM conversion when available.
        Falls back to direct DICOM loading if dcm2niix not found.
    sidecar_json : str | Path | None
        Optional path to BIDS-style sidecar JSON with metadata.
        Useful when loading NIfTI files without embedded metadata.
    **kwargs : Any
        Additional arguments passed to format-specific loaders.

    Returns
    -------
    PerfusionDataset
        Loaded imaging data with metadata.

    Raises
    ------
    FileNotFoundError
        If path does not exist.
    IOError
        If format cannot be determined or data cannot be loaded.
    MetadataError
        If required metadata is missing and interactive=False.

    Examples
    --------
    Load NIfTI file:

    >>> from osipy.common.io import load_perfusion
    >>> dataset = load_perfusion("data/asl.nii.gz", modality="ASL")

    Load DICOM directory:

    >>> dataset = load_perfusion("data/dicom_series/", modality="DCE")

    Load from BIDS dataset:

    >>> dataset = load_perfusion(
    ...     "bids_dataset/",
    ...     format="bids",
    ...     subject="01",
    ...     modality="ASL",
    ... )

    Load NIfTI with sidecar JSON:

    >>> dataset = load_perfusion(
    ...     "data/asl.nii.gz",
    ...     sidecar_json="data/asl.json",
    ...     modality="ASL",
    ... )

    """
    path = Path(path)

    if not path.exists():
        msg = f"Path not found: {path}"
        raise FileNotFoundError(msg)

    # Parse modality
    modality_enum = _parse_modality(modality)

    # Auto-detect format if needed
    if format == "auto":
        format = _detect_format(path)
        logger.info(f"Detected format: {format}")

    # Load based on format
    if format == "nifti":
        return _load_nifti(
            path,
            modality_enum,
            interactive=interactive,
            sidecar_json=sidecar_json,
            **kwargs,
        )
    elif format == "dicom":
        return _load_dicom(
            path,
            modality_enum,
            interactive=interactive,
            use_dcm2niix=use_dcm2niix,
            **kwargs,
        )
    elif format == "bids":
        if subject is None:
            msg = "subject parameter required for BIDS loading"
            raise OsipyIOError(msg)
        return _load_bids(
            path,
            subject,
            modality_enum,
            session=session,
            interactive=interactive,
            **kwargs,
        )
    else:
        msg = f"Unknown format: {format}"
        raise OsipyIOError(msg)


def _parse_modality(modality: Modality | str | None) -> Modality:
    """Parse modality argument to Modality enum.

    Parameters
    ----------
    modality : Modality | str | None
        Modality value.

    Returns
    -------
    Modality
        Parsed modality enum.
    """
    if modality is None:
        return Modality.DCE  # Default

    if isinstance(modality, Modality):
        return modality

    modality_str = str(modality).upper()
    modality_map = {
        "DCE": Modality.DCE,
        "DCE-MRI": Modality.DCE,
        "DSC": Modality.DSC,
        "DSC-MRI": Modality.DSC,
        "ASL": Modality.ASL,
        "IVIM": Modality.IVIM,
    }

    if modality_str in modality_map:
        return modality_map[modality_str]

    msg = f"Unknown modality: {modality}. Valid options: DCE, DSC, ASL, IVIM"
    raise OsipyIOError(msg)


def _detect_format(path: Path) -> str:
    """Auto-detect data format from path.

    Parameters
    ----------
    path : Path
        Path to data.

    Returns
    -------
    str
        Detected format: 'nifti', 'dicom', or 'bids'.
    """
    # Check if it's a NIfTI file
    if path.is_file():
        if path.suffix == ".gz" and path.stem.endswith(".nii"):
            return "nifti"
        if path.suffix == ".nii":
            return "nifti"
        # Single DICOM file
        if path.suffix.lower() in [".dcm", ".ima"]:
            return "dicom"
        # Try to read as DICOM (often no extension)
        try:
            import pydicom

            pydicom.dcmread(path, stop_before_pixels=True)
            return "dicom"
        except Exception:
            pass
        msg = f"Cannot determine format for file: {path}"
        raise OsipyIOError(msg)

    # It's a directory
    if path.is_dir():
        # Check for BIDS dataset_description.json
        if (path / "dataset_description.json").exists():
            return "bids"

        # Check for DICOM files
        for item in path.iterdir():
            if item.is_file():
                if item.suffix.lower() in [".dcm", ".ima"]:
                    return "dicom"
                # Try to read as DICOM
                try:
                    import pydicom

                    pydicom.dcmread(item, stop_before_pixels=True)
                    return "dicom"
                except Exception:
                    continue

        # Check for subdirectories containing DICOM files (multi-series layout)
        subdirs = [d for d in path.iterdir() if d.is_dir()]
        if subdirs:
            for subdir in subdirs:
                for item in subdir.iterdir():
                    if item.is_file():
                        if item.suffix.lower() in [".dcm", ".ima"]:
                            return "dicom"
                        try:
                            import pydicom

                            pydicom.dcmread(item, stop_before_pixels=True)
                            return "dicom"
                        except Exception:
                            continue
                # Only check the first non-empty subdir
                break

        # Check for NIfTI files
        nifti_files = list(path.glob("*.nii*"))
        if nifti_files:
            return "nifti"

        msg = f"Cannot determine format for directory: {path}"
        raise OsipyIOError(msg)

    msg = f"Path is neither file nor directory: {path}"
    raise OsipyIOError(msg)


def _load_nifti(
    path: Path,
    modality: Modality,
    interactive: bool = True,
    sidecar_json: str | Path | None = None,
    **kwargs: Any,
) -> PerfusionDataset:
    """Load NIfTI file with optional sidecar metadata.

    Parameters
    ----------
    path : Path
        Path to NIfTI file.
    modality : Modality
        Perfusion modality.
    interactive : bool
        Whether to prompt for missing parameters.
    sidecar_json : str | Path | None
        Optional sidecar JSON path.
    **kwargs : Any
        Additional arguments.

    Returns
    -------
    PerfusionDataset
        Loaded dataset.
    """
    import json

    import nibabel as nib
    import numpy as np

    from osipy.common.io.metadata.mapper import MetadataMapper

    # Load NIfTI
    nifti_img = nib.load(path)
    data = np.asarray(nifti_img.dataobj, dtype=np.float64)
    affine = nifti_img.affine

    # Load sidecar if provided
    sidecar: dict[str, Any] = {}
    if sidecar_json:
        sidecar_path = Path(sidecar_json)
        if sidecar_path.exists():
            with sidecar_path.open() as f:
                sidecar = json.load(f)
    else:
        # Try to find automatic sidecar
        auto_sidecar = path.with_suffix("").with_suffix(".json")
        if path.name.endswith(".nii.gz"):
            auto_sidecar = Path(str(path)[:-7] + ".json")
        if auto_sidecar.exists():
            with auto_sidecar.open() as f:
                sidecar = json.load(f)
            logger.info(f"Found sidecar: {auto_sidecar}")

    # Generate time points
    time_points = None
    if data.ndim == 4:
        n_volumes = data.shape[3]
        tr = sidecar.get("RepetitionTimePreparation") or sidecar.get("RepetitionTime")
        if tr:
            tr_sec = float(tr) if float(tr) < 100 else float(tr) / 1000
            time_points = np.arange(n_volumes) * tr_sec
        else:
            # Use NIfTI header zoom
            zooms = nifti_img.header.get_zooms()
            if len(zooms) > 3 and zooms[3] > 0:
                time_points = np.arange(n_volumes) * zooms[3]
            else:
                time_points = np.arange(n_volumes, dtype=np.float64)

    # Map metadata
    mapper = MetadataMapper(modality, interactive=interactive)
    acquisition_params = mapper.map_to_acquisition_params(
        bids_sidecar=sidecar,
        user_overrides=kwargs.get("acquisition_params"),
    )

    return PerfusionDataset(
        data=data,
        affine=affine,
        modality=modality,
        time_points=time_points,
        acquisition_params=acquisition_params,
        source_path=path,
        source_format="nifti",
    )


def _load_dicom(
    path: Path,
    modality: Modality,
    interactive: bool = True,
    use_dcm2niix: bool = True,
    **kwargs: Any,
) -> PerfusionDataset:
    """Load DICOM data with vendor-specific parsing.

    Parameters
    ----------
    path : Path
        Path to DICOM directory or file.
    modality : Modality
        Perfusion modality.
    interactive : bool
        Whether to prompt for missing parameters.
    use_dcm2niix : bool
        Whether to try dcm2niix first.
    **kwargs : Any
        Additional arguments.

    Returns
    -------
    PerfusionDataset
        Loaded dataset.
    """
    from osipy.common.io.metadata.mapper import MetadataMapper
    from osipy.common.io.vendors.detection import extract_vendor_metadata

    # Try dcm2niix conversion first if requested
    if use_dcm2niix:
        try:
            from osipy.common.io.converters.dcm2niix import Dcm2niixConverter

            converter = Dcm2niixConverter()
            if converter.is_available():
                logger.info("Using dcm2niix for DICOM conversion")
                nifti_path, _sidecar = converter.convert(path)

                # Load the converted NIfTI
                return _load_nifti(
                    nifti_path,
                    modality,
                    interactive=interactive,
                    sidecar_json=None,  # sidecar already loaded
                    **kwargs,
                )
        except (ImportError, Exception) as e:
            logger.debug(f"dcm2niix not available: {e}")

    # Fall back to direct DICOM loading
    try:
        import pydicom
    except ImportError as e:
        msg = "pydicom is required for DICOM loading"
        raise ImportError(msg) from e

    # Check for multi-series directory layout before falling through
    # to the single-series rglob path
    if path.is_dir():
        series_subdirs = _detect_multi_series_layout(path)
        if series_subdirs is not None:
            from osipy.common.io.dicom import load_dicom_multi_series

            logger.info(
                "Detected multi-series DICOM layout with %d series",
                len(series_subdirs),
            )
            return load_dicom_multi_series(
                series_dirs=series_subdirs,
                prompt_missing=interactive,
                modality=modality,
            )

    # Delegate to the full DICOM loader which handles temporal
    # grouping, ContentTime-based timing, and proper 4D assembly
    from osipy.common.io.dicom import load_dicom

    dataset = load_dicom(path, prompt_missing=interactive, modality=modality)

    # Layer vendor metadata and MetadataMapper on top for
    # modality-specific acquisition params
    first_dcm = pydicom.dcmread(
        sorted(
            f for f in path.rglob("*") if f.is_file() and not f.name.startswith(".")
        )[0]
    )
    vendor_metadata = extract_vendor_metadata(first_dcm)
    logger.info("Detected vendor: %s", vendor_metadata.vendor)

    mapper = MetadataMapper(modality, interactive=interactive)
    acquisition_params = mapper.map_to_acquisition_params(
        vendor_metadata=vendor_metadata,
        user_overrides=kwargs.get("acquisition_params"),
    )

    return PerfusionDataset(
        data=dataset.data,
        affine=dataset.affine,
        modality=modality,
        time_points=dataset.time_points,
        acquisition_params=acquisition_params,
        source_path=path,
        source_format="dicom",
    )


def _detect_multi_series_layout(path: Path) -> list[Path] | None:
    """Detect if a directory contains a multi-series DICOM layout.

    A multi-series layout is a directory whose immediate subdirectories
    each contain DICOM files (e.g., QIN-Breast study-level directories
    with ``MR_*`` subdirs).

    Parameters
    ----------
    path : Path
        Directory to inspect.

    Returns
    -------
    list[Path] | None
        Sorted list of series subdirectories if multi-series layout
        detected (at least 2 subdirs with DICOM files), or None.
    """
    if not path.is_dir():
        return None

    # Check if directory has DICOM files directly in it → single-series
    for item in path.iterdir():
        if item.is_file() and item.suffix.lower() in [".dcm", ".ima"]:
            return None

    # Check subdirectories for DICOM files
    series_dirs: list[Path] = []
    for subdir in sorted(path.iterdir()):
        if not subdir.is_dir():
            continue
        for item in subdir.iterdir():
            if item.is_file():
                if item.suffix.lower() in [".dcm", ".ima"]:
                    series_dirs.append(subdir)
                    break
                try:
                    import pydicom

                    pydicom.dcmread(item, stop_before_pixels=True)
                    series_dirs.append(subdir)
                    break
                except Exception:
                    continue

    if len(series_dirs) >= 2:
        return series_dirs
    return None


def _load_bids(
    path: Path,
    subject: str,
    modality: Modality,
    session: str | None = None,
    interactive: bool = True,
    **kwargs: Any,
) -> PerfusionDataset:
    """Load from BIDS dataset.

    Parameters
    ----------
    path : Path
        BIDS dataset root directory.
    subject : str
        Subject ID.
    modality : Modality
        Perfusion modality.
    session : str | None
        Session ID.
    interactive : bool
        Whether to prompt for missing parameters.
    **kwargs : Any
        Additional arguments.

    Returns
    -------
    PerfusionDataset
        Loaded dataset.
    """
    from osipy.common.io.bids import load_bids

    run = kwargs.get("run")

    return load_bids(
        path,
        subject=subject,
        modality=modality,
        session=session,
        run=run,
        interactive=interactive,
    )
