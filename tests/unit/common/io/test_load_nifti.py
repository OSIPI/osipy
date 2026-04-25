"""Tests for NIfTI loading with BIDS-style sidecar support.

Covers :func:`load_nifti` and its JSON-sidecar metadata merge, plus
the CLI runner's ``_detect_format`` helper — the only remaining
auto-detection surface.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from osipy.common.exceptions import DataValidationError
from osipy.common.exceptions import IOError as OsipyIOError
from osipy.common.io.nifti import load_nifti
from osipy.common.types import Modality


@pytest.fixture
def temp_nifti_file():
    """Create a temporary 4D NIfTI file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nifti_path = Path(tmpdir) / "test.nii.gz"
        data = np.random.randn(64, 64, 24, 40).astype(np.float32)
        nib.save(nib.Nifti1Image(data, np.eye(4)), nifti_path)
        yield nifti_path


@pytest.fixture
def temp_nifti_with_sidecar():
    """Create a NIfTI file with an adjacent BIDS sidecar."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nifti_path = Path(tmpdir) / "test_asl.nii.gz"
        data = np.random.randn(64, 64, 24, 80).astype(np.float32)
        nib.save(nib.Nifti1Image(data, np.eye(4)), nifti_path)

        sidecar = {
            "RepetitionTimePreparation": 4.5,
            "ArterialSpinLabelingType": "PCASL",
            "PostLabelingDelay": 1.8,
            "LabelingDuration": 1.8,
        }
        (Path(tmpdir) / "test_asl.json").write_text(json.dumps(sidecar))
        yield nifti_path


class TestLoadNiftiBasic:
    """Basic NIfTI loading — no sidecar."""

    def test_load_4d_defaults_to_dce(self, temp_nifti_file: Path) -> None:
        ds = load_nifti(temp_nifti_file)
        assert ds.shape == (64, 64, 24, 40)
        assert ds.modality == Modality.DCE
        assert ds.source_format == "nifti"

    def test_load_explicit_modality(self, temp_nifti_file: Path) -> None:
        ds = load_nifti(temp_nifti_file, modality=Modality.ASL)
        assert ds.modality == Modality.ASL

    def test_time_points_default_to_frame_index(self, temp_nifti_file: Path) -> None:
        ds = load_nifti(temp_nifti_file)
        assert ds.time_points is not None
        assert len(ds.time_points) == 40
        np.testing.assert_allclose(ds.time_points, np.arange(40))

    def test_3d_has_no_time_points(self, tmp_path: Path) -> None:
        nifti_path = tmp_path / "t1.nii.gz"
        nib.save(
            nib.Nifti1Image(np.zeros((8, 8, 4), np.float32), np.eye(4)), nifti_path
        )
        ds = load_nifti(nifti_path)
        assert ds.time_points is None


class TestLoadNiftiSidecar:
    """Sidecar JSON resolution — auto-adjacent + explicit override."""

    def test_auto_adjacent_sidecar_used(self, temp_nifti_with_sidecar: Path) -> None:
        ds = load_nifti(temp_nifti_with_sidecar, modality=Modality.ASL)
        # RepetitionTimePreparation=4.5 s → spacing of 4.5 s between frames
        assert ds.time_points is not None
        np.testing.assert_allclose(ds.time_points[1] - ds.time_points[0], 4.5)
        # MetadataMapper should have populated acquisition params from sidecar
        assert ds.acquisition_params.pld == 1800.0

    def test_explicit_sidecar_path(self, temp_nifti_file: Path, tmp_path: Path) -> None:
        sidecar_path = tmp_path / "custom.json"
        sidecar_path.write_text(
            json.dumps(
                {
                    "ArterialSpinLabelingType": "PCASL",
                    "PostLabelingDelay": 2.0,
                }
            )
        )
        ds = load_nifti(
            temp_nifti_file,
            modality=Modality.ASL,
            sidecar_json=sidecar_path,
        )
        assert ds.acquisition_params.pld == 2000.0

    def test_tr_ms_converted_to_sec(self, tmp_path: Path) -> None:
        """TRs given as milliseconds (>100) are rescaled to seconds."""
        nifti_path = tmp_path / "dyn.nii.gz"
        nib.save(
            nib.Nifti1Image(np.zeros((4, 4, 2, 3), dtype=np.float32), np.eye(4)),
            nifti_path,
        )
        (tmp_path / "dyn.json").write_text(json.dumps({"RepetitionTime": 6000.0}))
        ds = load_nifti(nifti_path)
        # 6000 > 100 → treated as ms; spacing becomes 6.0 s
        np.testing.assert_allclose(ds.time_points[1] - ds.time_points[0], 6.0)


class TestLoadNiftiErrors:
    """Error handling."""

    def test_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_nifti("/does/not/exist.nii.gz")

    def test_invalid_extension(self, tmp_path: Path) -> None:
        bad = tmp_path / "wrong.txt"
        bad.write_text("x")
        with pytest.raises(OsipyIOError):
            load_nifti(bad)

    def test_2d_rejected(self, tmp_path: Path) -> None:
        nifti_path = tmp_path / "2d.nii.gz"
        nib.save(
            nib.Nifti1Image(np.zeros((8, 8), dtype=np.float32), np.eye(4)),
            nifti_path,
        )
        with pytest.raises(DataValidationError):
            load_nifti(nifti_path)


class TestDetectFormat:
    """CLI runner format auto-detection."""

    def test_detect_nifti_file(self, temp_nifti_file: Path) -> None:
        from osipy.cli.runner import _detect_format

        assert _detect_format(temp_nifti_file) == "nifti"

    def test_detect_bids_directory(self, tmp_path: Path) -> None:
        from osipy.cli.runner import _detect_format

        (tmp_path / "dataset_description.json").write_text('{"Name":"t"}')
        assert _detect_format(tmp_path) == "bids"

    def test_detect_nifti_directory(self, tmp_path: Path) -> None:
        from osipy.cli.runner import _detect_format

        nib.save(
            nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)),
            tmp_path / "vol.nii.gz",
        )
        assert _detect_format(tmp_path) == "nifti"

    def test_detect_dicom_fallback(self, tmp_path: Path) -> None:
        """A directory with no NIfTI or BIDS marker falls through to 'dicom'."""
        from osipy.cli.runner import _detect_format

        (tmp_path / "001.dcm").write_bytes(b"x")
        assert _detect_format(tmp_path) == "dicom"
