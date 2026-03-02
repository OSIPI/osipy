"""Unit tests for osipy.common.io.bids module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from osipy.common.exceptions import IOError as OsipyIOError
from osipy.common.io.bids import export_bids
from osipy.common.parameter_map import ParameterMap


class TestExportBids:
    """Tests for export_bids function."""

    @pytest.fixture
    def sample_parameter_maps(self) -> dict[str, ParameterMap]:
        """Create sample parameter maps."""
        shape = (64, 64, 20)
        affine = np.eye(4)

        return {
            "Ktrans": ParameterMap(
                name="Ktrans",
                symbol="Ktrans",
                units="1/min",
                values=np.random.rand(*shape).astype(np.float32) * 0.5,
                affine=affine,
                quality_mask=np.ones(shape, dtype=bool),
                model_name="ExtendedTofts",
                fitting_method="lm",
                literature_reference="Tofts PS et al. (1999). JMRI.",
            ),
            "ve": ParameterMap(
                name="ve",
                symbol="ve",
                units="",
                values=np.random.rand(*shape).astype(np.float32),
                affine=affine,
                quality_mask=np.ones(shape, dtype=bool),
                model_name="ExtendedTofts",
                fitting_method="lm",
                literature_reference="Tofts PS et al. (1999). JMRI.",
            ),
        }

    def test_export_creates_directory(
        self, sample_parameter_maps: dict[str, ParameterMap]
    ) -> None:
        """Test that export creates subject directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "derivatives" / "osipy"
            result_path = export_bids(
                sample_parameter_maps,
                output_dir,
                subject_id="01",
            )

            assert result_path.exists()
            assert (result_path / "sub-01_Ktrans.nii.gz").exists()
            assert (result_path / "sub-01_ve.nii.gz").exists()

    def test_export_creates_sidecar(
        self, sample_parameter_maps: dict[str, ParameterMap]
    ) -> None:
        """Test that export creates sidecar JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "derivatives" / "osipy"
            result_path = export_bids(
                sample_parameter_maps,
                output_dir,
                subject_id="01",
            )

            sidecar_path = result_path / "sub-01_perf.json"
            assert sidecar_path.exists()

            with sidecar_path.open() as f:
                sidecar = json.load(f)

            assert "Parameters" in sidecar
            assert "Ktrans" in sidecar["Parameters"]
            assert sidecar["Parameters"]["Ktrans"]["Units"] == "1/min"

    def test_export_creates_dataset_description(
        self, sample_parameter_maps: dict[str, ParameterMap]
    ) -> None:
        """Test that export creates dataset_description.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "derivatives" / "osipy"
            export_bids(
                sample_parameter_maps,
                output_dir,
                subject_id="01",
            )

            desc_path = output_dir / "dataset_description.json"
            assert desc_path.exists()

            with desc_path.open() as f:
                desc = json.load(f)

            assert desc["DatasetType"] == "derivative"
            assert desc["BIDSVersion"] == "1.9.0"

    def test_export_with_session(
        self, sample_parameter_maps: dict[str, ParameterMap]
    ) -> None:
        """Test export with session ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "derivatives" / "osipy"
            result_path = export_bids(
                sample_parameter_maps,
                output_dir,
                subject_id="01",
                session_id="pre",
            )

            # Check path includes session
            assert "ses-pre" in str(result_path)
            assert (result_path / "sub-01_ses-pre_Ktrans.nii.gz").exists()

    def test_export_empty_maps_raises(self) -> None:
        """Test that empty parameter_maps raises ValueError."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(OsipyIOError, match="cannot be empty"),
        ):
            export_bids({}, tmpdir, subject_id="01")

    def test_export_empty_subject_id_raises(
        self, sample_parameter_maps: dict[str, ParameterMap]
    ) -> None:
        """Test that empty subject_id raises ValueError."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(OsipyIOError, match="cannot be empty"),
        ):
            export_bids(sample_parameter_maps, tmpdir, subject_id="")
