"""dcm2niix integration for DICOM conversion.

This module provides a wrapper around the dcm2niix tool for
converting DICOM to NIfTI with BIDS-compliant sidecar generation.

References
----------
dcm2niix: https://github.com/rordenlab/dcm2niix
"""

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from osipy.common.exceptions import IOError

logger = logging.getLogger(__name__)


class Dcm2niixError(IOError):
    """Raised when dcm2niix conversion fails.

    Examples
    --------
    >>> raise Dcm2niixError("dcm2niix not found")
    Traceback (most recent call last):
        ...
    osipy.common.io.converters.dcm2niix.Dcm2niixError: dcm2niix not found
    """

    pass


class Dcm2niixConverter:
    """Wrapper for dcm2niix DICOM to NIfTI conversion.

    This class provides integration with dcm2niix for converting
    DICOM data to NIfTI format with automatic BIDS sidecar generation.

    Attributes
    ----------
    dcm2niix_path : str | None
        Path to dcm2niix executable, or None if not found.

    Examples
    --------
    >>> converter = Dcm2niixConverter()
    >>> if converter.is_available():
    ...     nifti_path, sidecar = converter.convert("dicom_dir/")
    ...     print(f"Converted to: {nifti_path}")

    Notes
    -----
    dcm2niix must be installed and available in PATH.
    Install via: https://github.com/rordenlab/dcm2niix#install
    """

    def __init__(self, dcm2niix_path: str | None = None):
        """Initialize the converter.

        Parameters
        ----------
        dcm2niix_path : str | None
            Path to dcm2niix executable. If None, searches PATH.
        """
        if dcm2niix_path:
            self.dcm2niix_path = dcm2niix_path
        else:
            self.dcm2niix_path = shutil.which("dcm2niix")

    def is_available(self) -> bool:
        """Check if dcm2niix is available.

        Returns
        -------
        bool
            True if dcm2niix is installed and accessible.
        """
        if self.dcm2niix_path is None:
            return False

        try:
            result = subprocess.run(
                [self.dcm2niix_path, "-h"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def convert(
        self,
        dicom_dir: str | Path,
        output_dir: str | Path | None = None,
        filename_format: str = "%p_%s",
        compress: bool = True,
        merge_2d: bool = True,
        single_file: bool = True,
    ) -> tuple[Path, dict[str, Any]]:
        """Convert DICOM to NIfTI using dcm2niix.

        Parameters
        ----------
        dicom_dir : str | Path
            Path to DICOM directory.
        output_dir : str | Path | None
            Output directory. If None, uses a temporary directory.
        filename_format : str, default='%p_%s'
            dcm2niix filename format. %p=protocol, %s=series.
        compress : bool, default=True
            Compress output with gzip (.nii.gz).
        merge_2d : bool, default=True
            Merge 2D slices into 3D volumes.
        single_file : bool, default=True
            Prefer single file output.

        Returns
        -------
        tuple[Path, dict[str, Any]]
            Tuple of (NIfTI file path, sidecar JSON content).

        Raises
        ------
        Dcm2niixError
            If dcm2niix is not available or conversion fails.
        FileNotFoundError
            If DICOM directory does not exist.

        Examples
        --------
        >>> converter = Dcm2niixConverter()
        >>> nifti_path, sidecar = converter.convert("dicom_series/")
        >>> print(f"TR: {sidecar.get('RepetitionTime')}")
        """
        if not self.is_available():
            msg = (
                "dcm2niix is not installed or not found in PATH. "
                "Install from: https://github.com/rordenlab/dcm2niix#install"
            )
            raise Dcm2niixError(msg)

        dicom_dir = Path(dicom_dir)
        if not dicom_dir.exists():
            msg = f"DICOM directory not found: {dicom_dir}"
            raise FileNotFoundError(msg)

        # Use temp directory if no output specified
        temp_dir = None
        if output_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="osipy_dcm2niix_")
            output_dir = Path(temp_dir)
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Build dcm2niix command
            cmd = [
                self.dcm2niix_path,
                "-f",
                filename_format,  # Filename format
                "-o",
                str(output_dir),  # Output directory
                "-z",
                "y" if compress else "n",  # Compress
                "-b",
                "y",  # BIDS sidecar
                "-ba",
                "y",  # Anonymize BIDS
                "-m",
                "y" if merge_2d else "n",  # Merge 2D
                "-s",
                "y" if single_file else "n",  # Single file
                str(dicom_dir),
            ]

            logger.debug(f"Running: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode != 0:
                msg = f"dcm2niix failed: {result.stderr}"
                raise Dcm2niixError(msg)

            logger.debug(f"dcm2niix output: {result.stdout}")

            # Find output files
            nifti_files = list(output_dir.glob("*.nii*"))
            if not nifti_files:
                msg = f"dcm2niix produced no NIfTI files in {output_dir}"
                raise Dcm2niixError(msg)

            # Prefer .nii.gz
            nifti_file = None
            for f in nifti_files:
                if f.name.endswith(".nii.gz"):
                    nifti_file = f
                    break
            if nifti_file is None:
                nifti_file = nifti_files[0]

            # Load sidecar JSON
            if nifti_file.name.endswith(".nii.gz"):
                sidecar_path = Path(str(nifti_file)[:-7] + ".json")
            else:
                sidecar_path = nifti_file.with_suffix(".json")

            sidecar: dict[str, Any] = {}
            if sidecar_path.exists():
                with sidecar_path.open() as f:
                    sidecar = json.load(f)
            else:
                logger.warning(f"No sidecar JSON generated: {sidecar_path}")

            return nifti_file, sidecar

        except subprocess.TimeoutExpired as err:
            msg = "dcm2niix conversion timed out"
            raise Dcm2niixError(msg) from err
