"""Tests for unified load_perfusion interface.

Tests the universal data loader with format auto-detection.
"""

import json
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from osipy.common.exceptions import IOError as OsipyIOError
from osipy.common.io.load import (
    _detect_format,
    _parse_modality,
    load_perfusion,
)
from osipy.common.types import Modality


@pytest.fixture
def temp_nifti_file():
    """Create a temporary NIfTI file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nifti_path = Path(tmpdir) / "test.nii.gz"

        data = np.random.randn(64, 64, 24, 40).astype(np.float32)
        affine = np.eye(4)
        nifti_img = nib.Nifti1Image(data, affine)
        nib.save(nifti_img, nifti_path)

        yield nifti_path


@pytest.fixture
def temp_nifti_with_sidecar():
    """Create a NIfTI file with BIDS-style sidecar."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nifti_path = Path(tmpdir) / "test_asl.nii.gz"

        data = np.random.randn(64, 64, 24, 80).astype(np.float32)
        affine = np.eye(4)
        nifti_img = nib.Nifti1Image(data, affine)
        nib.save(nifti_img, nifti_path)

        # Create sidecar
        sidecar = {
            "RepetitionTimePreparation": 4.5,
            "ArterialSpinLabelingType": "PCASL",
            "PostLabelingDelay": 1.8,
            "LabelingDuration": 1.8,
        }
        sidecar_path = Path(tmpdir) / "test_asl.json"
        with sidecar_path.open("w") as f:
            json.dump(sidecar, f)

        yield nifti_path


@pytest.fixture
def temp_bids_dataset():
    """Create a temporary BIDS dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bids_root = Path(tmpdir)

        # Create dataset_description.json
        desc = {
            "Name": "Test Dataset",
            "BIDSVersion": "1.9.0",
        }
        with (bids_root / "dataset_description.json").open("w") as f:
            json.dump(desc, f)

        # Create subject data
        perf_dir = bids_root / "sub-01" / "perf"
        perf_dir.mkdir(parents=True)

        data = np.random.randn(64, 64, 24, 80).astype(np.float32)
        nifti_img = nib.Nifti1Image(data, np.eye(4))
        nib.save(nifti_img, perf_dir / "sub-01_asl.nii.gz")

        sidecar = {
            "ArterialSpinLabelingType": "PCASL",
            "PostLabelingDelay": 1.8,
            "LabelingDuration": 1.8,
        }
        with (perf_dir / "sub-01_asl.json").open("w") as f:
            json.dump(sidecar, f)

        yield bids_root


class TestParseModality:
    """Tests for modality parsing."""

    def test_parse_enum(self):
        """Test parsing Modality enum."""
        assert _parse_modality(Modality.ASL) == Modality.ASL
        assert _parse_modality(Modality.DCE) == Modality.DCE

    def test_parse_string(self):
        """Test parsing string modality."""
        assert _parse_modality("ASL") == Modality.ASL
        assert _parse_modality("dce") == Modality.DCE
        assert _parse_modality("DSC-MRI") == Modality.DSC
        assert _parse_modality("IVIM") == Modality.IVIM

    def test_parse_none(self):
        """Test default modality."""
        assert _parse_modality(None) == Modality.DCE

    def test_parse_invalid(self):
        """Test invalid modality."""
        with pytest.raises(OsipyIOError):
            _parse_modality("INVALID")


class TestDetectFormat:
    """Tests for format auto-detection."""

    def test_detect_nifti_file(self, temp_nifti_file):
        """Test NIfTI file detection."""
        assert _detect_format(temp_nifti_file) == "nifti"

    def test_detect_nifti_uncompressed(self, tmp_path):
        """Test uncompressed NIfTI detection."""
        nifti_path = tmp_path / "test.nii"
        data = np.random.randn(64, 64, 24).astype(np.float32)
        nifti_img = nib.Nifti1Image(data, np.eye(4))
        nib.save(nifti_img, nifti_path)

        assert _detect_format(nifti_path) == "nifti"

    def test_detect_bids_directory(self, temp_bids_dataset):
        """Test BIDS directory detection."""
        assert _detect_format(temp_bids_dataset) == "bids"

    def test_detect_nifti_directory(self, tmp_path):
        """Test directory with NIfTI files."""
        nifti_path = tmp_path / "test.nii.gz"
        data = np.random.randn(64, 64, 24).astype(np.float32)
        nifti_img = nib.Nifti1Image(data, np.eye(4))
        nib.save(nifti_img, nifti_path)

        assert _detect_format(tmp_path) == "nifti"


class TestLoadPerfusionNifti:
    """Tests for loading NIfTI files."""

    def test_load_nifti_basic(self, temp_nifti_file):
        """Test basic NIfTI loading."""
        dataset = load_perfusion(
            temp_nifti_file,
            modality="DCE",
            interactive=False,
        )

        assert dataset.shape == (64, 64, 24, 40)
        assert dataset.modality == Modality.DCE
        assert dataset.source_format == "nifti"

    def test_load_nifti_with_sidecar(self, temp_nifti_with_sidecar):
        """Test NIfTI loading with sidecar JSON."""
        dataset = load_perfusion(
            temp_nifti_with_sidecar,
            modality="ASL",
            interactive=False,
        )

        assert dataset.shape == (64, 64, 24, 80)
        assert dataset.modality == Modality.ASL
        # Params should be extracted from sidecar
        assert dataset.acquisition_params.pld == 1800.0

    def test_load_nifti_explicit_sidecar(self, temp_nifti_file, tmp_path):
        """Test loading with explicitly specified sidecar."""
        # Create sidecar in different location
        sidecar_path = tmp_path / "custom_sidecar.json"
        sidecar = {
            "ArterialSpinLabelingType": "PCASL",
            "PostLabelingDelay": 2.0,
        }
        with sidecar_path.open("w") as f:
            json.dump(sidecar, f)

        dataset = load_perfusion(
            temp_nifti_file,
            modality="ASL",
            sidecar_json=sidecar_path,
            interactive=False,
        )

        assert dataset.acquisition_params.pld == 2000.0

    def test_load_nifti_explicit_format(self, temp_nifti_file):
        """Test loading with explicit format specification."""
        dataset = load_perfusion(
            temp_nifti_file,
            modality="DCE",
            format="nifti",
            interactive=False,
        )

        assert dataset.source_format == "nifti"


class TestLoadPerfusionBids:
    """Tests for loading from BIDS datasets."""

    def test_load_bids(self, temp_bids_dataset):
        """Test loading from BIDS dataset."""
        dataset = load_perfusion(
            temp_bids_dataset,
            format="bids",
            subject="01",
            modality="ASL",
            interactive=False,
        )

        assert dataset.shape == (64, 64, 24, 80)
        assert dataset.modality == Modality.ASL
        assert dataset.source_format == "bids"

    def test_load_bids_auto_detect(self, temp_bids_dataset):
        """Test BIDS auto-detection."""
        dataset = load_perfusion(
            temp_bids_dataset,
            subject="01",
            modality="ASL",
            interactive=False,
        )

        assert dataset.source_format == "bids"

    def test_load_bids_requires_subject(self, temp_bids_dataset):
        """Test that subject is required for BIDS."""
        with pytest.raises(OsipyIOError, match="subject"):
            load_perfusion(
                temp_bids_dataset,
                format="bids",
                modality="ASL",
            )


class TestLoadPerfusionErrors:
    """Tests for error handling."""

    def test_file_not_found(self):
        """Test error for non-existent path."""
        with pytest.raises(FileNotFoundError):
            load_perfusion("/nonexistent/path")

    def test_unknown_format_file(self, tmp_path):
        """Test error for unknown file format."""
        # Create a file with unknown extension
        unknown_file = tmp_path / "test.xyz"
        unknown_file.write_text("unknown content")

        with pytest.raises(OsipyIOError):
            load_perfusion(unknown_file)


class TestLoadPerfusionTimePoints:
    """Tests for time point generation."""

    def test_time_points_from_sidecar(self, temp_nifti_with_sidecar):
        """Test time points generated from sidecar TR."""
        dataset = load_perfusion(
            temp_nifti_with_sidecar,
            modality="ASL",
            interactive=False,
        )

        assert dataset.time_points is not None
        assert len(dataset.time_points) == 80
        # TR from sidecar is 4.5s
        np.testing.assert_allclose(dataset.time_points[1] - dataset.time_points[0], 4.5)

    def test_time_points_3d_data(self, tmp_path):
        """Test that 3D data has no time points."""
        nifti_path = tmp_path / "test_3d.nii.gz"
        data = np.random.randn(64, 64, 24).astype(np.float32)
        nifti_img = nib.Nifti1Image(data, np.eye(4))
        nib.save(nifti_img, nifti_path)

        dataset = load_perfusion(
            nifti_path,
            modality="DCE",
            interactive=False,
        )

        assert dataset.time_points is None
