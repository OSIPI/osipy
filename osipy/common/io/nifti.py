"""NIfTI file loading and saving for osipy.

This module provides functions for loading and saving NIfTI files,
including PerfusionDataset containers and ParameterMap objects.

References
----------
NIfTI-1 Data Format: https://nifti.nimh.nih.gov/
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import nibabel as nib
import numpy as np

from osipy.common.dataset import PerfusionDataset
from osipy.common.exceptions import DataValidationError
from osipy.common.exceptions import IOError as OsipyIOError
from osipy.common.types import AcquisitionParams, Modality

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from osipy.common.parameter_map import ParameterMap

logger = logging.getLogger(__name__)


def load_nifti(
    path: str | Path,
    modality: Modality | None = None,
    acquisition_params: AcquisitionParams | None = None,
    sidecar_json: str | Path | None = None,
    interactive: bool = False,
) -> PerfusionDataset:
    """Load NIfTI file as PerfusionDataset.

    Parameters
    ----------
    path : str | Path
        Path to NIfTI file (.nii or .nii.gz).
    modality : Modality | None
        Perfusion modality. Defaults to :class:`Modality.DCE` when unset.
    acquisition_params : AcquisitionParams | None
        Acquisition parameters. When provided, takes precedence over any
        values derived from the sidecar JSON. When None and a sidecar is
        available, parameters are mapped via ``MetadataMapper``.
    sidecar_json : str | Path | None
        Optional BIDS-style sidecar JSON. If None, a file sharing the
        NIfTI's stem with a ``.json`` extension is loaded automatically
        when present.
    interactive : bool, default False
        Forwarded to ``MetadataMapper`` — when True, prompts for any
        modality-required fields missing from the sidecar.

    Returns
    -------
    PerfusionDataset
        Loaded imaging data with metadata. ``time_points`` is derived
        from ``RepetitionTimePreparation`` / ``RepetitionTime`` in the
        sidecar when present, otherwise from the NIfTI header's temporal
        zoom, otherwise defaults to integer frame indices.

    Raises
    ------
    FileNotFoundError
        If the NIfTI file does not exist.
    IOError
        If the file is not a valid NIfTI.
    DataValidationError
        If the data is not 3D or 4D.

    Examples
    --------
    >>> from osipy.common.io.nifti import load_nifti
    >>> from osipy.common.types import Modality
    >>> dataset = load_nifti("dce_data.nii.gz", modality=Modality.DCE)
    >>> print(dataset.shape)
    (64, 64, 20, 30)

    Load with an explicit sidecar:

    >>> dataset = load_nifti(
    ...     "asl.nii.gz",
    ...     modality="ASL",
    ...     sidecar_json="asl.json",
    ... )
    """
    import json

    path = Path(path)

    if not path.exists():
        msg = f"File not found: {path}"
        raise FileNotFoundError(msg)
    if not str(path).endswith((".nii", ".nii.gz")):
        msg = f"Invalid NIfTI file extension: {path.suffix}"
        raise OsipyIOError(msg)

    try:
        img = nib.load(path)
    except Exception as e:
        msg = f"Failed to load NIfTI file: {e}"
        raise OsipyIOError(msg) from e

    data = np.asarray(img.dataobj, dtype=np.float64)
    affine = np.asarray(img.affine, dtype=np.float64)

    if data.ndim not in (3, 4):
        msg = f"NIfTI data must be 3D or 4D, got {data.ndim}D"
        raise DataValidationError(msg)

    if np.any(~np.isfinite(data)):
        logger.warning(
            "NIfTI data contains NaN or infinite values. "
            "Consider preprocessing data before analysis."
        )

    # Resolve sidecar: explicit > auto-adjacent.
    sidecar: dict[str, Any] = {}
    sidecar_path: Path | None = None
    if sidecar_json is not None:
        candidate = Path(sidecar_json)
        if candidate.exists():
            sidecar_path = candidate
    else:
        auto = (
            Path(str(path)[:-7] + ".json")
            if path.name.endswith(".nii.gz")
            else path.with_suffix(".json")
        )
        if auto.exists():
            sidecar_path = auto
    if sidecar_path is not None:
        with sidecar_path.open(encoding="utf-8") as f:
            sidecar = json.load(f)
        logger.info("Loaded NIfTI sidecar: %s", sidecar_path)

    # Build time vector.
    time_points = None
    if data.ndim == 4:
        n_volumes = data.shape[3]
        tr = sidecar.get("RepetitionTimePreparation") or sidecar.get("RepetitionTime")
        if tr is not None:
            tr_sec = float(tr) if float(tr) < 100 else float(tr) / 1000.0
            time_points = np.arange(n_volumes, dtype=np.float64) * tr_sec
        else:
            zooms = img.header.get_zooms()
            if len(zooms) > 3 and zooms[3] > 0:
                time_points = np.arange(n_volumes, dtype=np.float64) * float(zooms[3])
            else:
                time_points = np.arange(n_volumes, dtype=np.float64)

    if modality is None:
        modality = Modality.DCE

    # Build acquisition params — explicit user override wins over sidecar.
    if acquisition_params is None:
        if sidecar:
            from osipy.common.io.metadata.mapper import MetadataMapper

            mapper = MetadataMapper(modality, interactive=interactive)
            acquisition_params = mapper.map_to_acquisition_params(bids_sidecar=sidecar)
        else:
            acquisition_params = AcquisitionParams()

    return PerfusionDataset(
        data=data,
        affine=affine,
        modality=modality,
        time_points=time_points,
        acquisition_params=acquisition_params,
        source_path=path,
        source_format="nifti",
    )


def save_nifti(
    data: "NDArray[Any] | ParameterMap | PerfusionDataset",
    path: str | Path,
    affine: "NDArray[Any] | None" = None,
    dtype: np.dtype | None = None,
) -> Path:
    """Save data as a NIfTI file with proper orientation preservation.

    This function ensures the affine matrix is correctly applied to
    preserve spatial orientation when saving parameter maps or datasets.

    Parameters
    ----------
    data : NDArray | ParameterMap | PerfusionDataset
        Data to save. Can be:
        - A numpy array (requires affine parameter)
        - A ParameterMap (uses embedded affine)
        - A PerfusionDataset (uses embedded affine)
    path : str | Path
        Output path. Will add .nii.gz extension if not present.
    affine : NDArray | None
        4x4 affine transformation matrix. Required if data is a numpy array.
        Ignored if data is a ParameterMap or PerfusionDataset.
    dtype : np.dtype | None
        Output data type. Defaults to float32 for parameter maps.

    Returns
    -------
    Path
        Path to the saved file.

    Raises
    ------
    ValueError
        If data is a numpy array and affine is not provided.
    IOError
        If saving fails.

    Examples
    --------
    >>> from osipy.common.io import save_nifti
    >>> import numpy as np
    >>> data = np.random.rand(64, 64, 20)
    >>> affine = np.eye(4)
    >>> save_nifti(data, "output.nii.gz", affine=affine)

    >>> from osipy.common.parameter_map import ParameterMap
    >>> ktrans = ParameterMap(name="Ktrans", ...)
    >>> save_nifti(ktrans, "Ktrans.nii.gz")  # Uses embedded affine

    """
    from osipy.common.parameter_map import ParameterMap

    path = Path(path)

    # Ensure .nii or .nii.gz extension
    if not str(path).endswith((".nii", ".nii.gz")):
        path = Path(str(path) + ".nii.gz")

    # Extract data and affine based on input type
    if isinstance(data, ParameterMap):
        array_data = data.values
        output_affine = data.affine
        if dtype is None:
            dtype = np.float32
    elif isinstance(data, PerfusionDataset):
        array_data = data.data
        output_affine = data.affine
        if dtype is None:
            dtype = np.float32
    elif isinstance(data, np.ndarray):
        array_data = data
        if affine is None:
            msg = "affine parameter is required when saving numpy array"
            raise OsipyIOError(msg)
        output_affine = affine
        if dtype is None:
            dtype = np.float32
    else:
        msg = f"Unsupported data type: {type(data)}"
        raise TypeError(msg)

    # Validate affine
    output_affine = np.asarray(output_affine, dtype=np.float64)
    if output_affine.shape != (4, 4):
        msg = f"Affine must be 4x4, got shape {output_affine.shape}"
        raise OsipyIOError(msg)

    # Check for degenerate affine (zero determinant)
    det = np.linalg.det(output_affine[:3, :3])
    if np.abs(det) < 1e-10:
        logger.warning(
            f"Affine matrix has near-zero determinant ({det:.2e}). "
            "Output orientation may be incorrect."
        )

    # Convert data type
    array_data = np.asarray(array_data, dtype=dtype)

    # Create NIfTI image
    nifti_img = nib.Nifti1Image(array_data, output_affine)

    # Set header fields for better compatibility
    nifti_img.header.set_xyzt_units("mm", "sec")

    # Save
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nifti_img, path)
        logger.debug(f"Saved NIfTI: {path}")
    except Exception as e:
        msg = f"Failed to save NIfTI file: {e}"
        raise OsipyIOError(msg) from e

    return path
