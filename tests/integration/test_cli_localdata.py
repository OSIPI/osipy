"""End-to-end CLI pipeline tests against local data.

These tests run ``run_pipeline()`` against real data in the ``data/``
directory at the repository root.  Every test is marked ``localdata``
and skips gracefully when the expected data is absent, so they never
break CI or machines without the data.

Run with::

    pytest -m localdata

"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from osipy.cli.config import PipelineConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _skip_unless(path: Path) -> None:
    """``pytest.skip`` when *path* does not exist."""
    if not path.exists():
        pytest.skip(f"Data not found: {path}")


def _output_files(output_dir: Path) -> list[str]:
    """Return sorted basenames of files written to *output_dir*."""
    return sorted(p.name for p in output_dir.iterdir() if p.is_file())


def _assert_nifti_valid(path: Path, *, min_ndim: int = 3) -> None:
    """Assert *path* is a loadable NIfTI with at least *min_ndim* dims."""
    import nibabel as nib

    assert path.exists(), f"Expected output file not found: {path}"
    img = nib.load(path)
    data = np.asarray(img.dataobj)
    assert data.ndim >= min_ndim, f"{path.name}: ndim={data.ndim} < {min_ndim}"
    assert data.size > 0, f"{path.name}: empty array"
    # No all-NaN slices
    finite = np.isfinite(data)
    assert finite.any(), f"{path.name}: entirely NaN/Inf"


# ---------------------------------------------------------------------------
# DCE - DICOM clinical data
# ---------------------------------------------------------------------------

# The study dir that contains the VFA + perfusion series
_DCE_STUDY_DIR = (
    DATA_DIR / "dce" / "Clinical_P1" / "Visit1" / "09-15-1904-BRAINRESEARCH-89964"
)


@pytest.mark.localdata
@pytest.mark.slow
class TestDCEDicomPipeline:
    """Run the DCE-MRI pipeline against Clinical_P1 DICOM data."""

    def test_dce_pipeline_produces_maps(self, tmp_path: Path) -> None:
        """Pipeline produces Ktrans, ve, vp NIfTIs."""
        _skip_unless(_DCE_STUDY_DIR)

        from osipy.cli.runner import run_pipeline

        config = PipelineConfig(
            modality="dce",
            pipeline={
                "model": "extended_tofts",
                "t1_mapping_method": "vfa",
                "aif_source": "population",
                "population_aif": "parker",
                "acquisition": {
                    "tr": 5.0,
                    "flip_angles": [5, 10, 15, 20, 25, 30],
                    "baseline_frames": 5,
                    "relaxivity": 4.5,
                },
            },
        )

        out = tmp_path / "dce_out"
        run_pipeline(config, _DCE_STUDY_DIR, output_dir=out)

        # At minimum we expect parameter maps + metadata
        assert (out / "osipy_run.json").exists()
        assert (out / "ktrans.nii.gz").exists()
        assert (out / "ve.nii.gz").exists()
        assert (out / "quality_mask.nii.gz").exists()

        _assert_nifti_valid(out / "ktrans.nii.gz")
        _assert_nifti_valid(out / "ve.nii.gz")

    def test_dce_discover_series(self) -> None:
        """_discover_dce_dicom_series finds VFA + perfusion dirs."""
        _skip_unless(_DCE_STUDY_DIR)

        from osipy.cli.runner import _discover_dce_dicom_series

        result = _discover_dce_dicom_series(_DCE_STUDY_DIR)
        assert result is not None, "Discovery returned None on real data"
        vfa_dirs, perfusion_dir = result
        assert len(vfa_dirs) >= 2, f"Expected >=2 VFA dirs, got {len(vfa_dirs)}"
        assert "perfusion" in perfusion_dir.name.lower()


# ---------------------------------------------------------------------------
# IVIM - NIfTI data
# ---------------------------------------------------------------------------

_IVIM_BRAIN_NII = DATA_DIR / "ivim" / "brain" / "brain.nii.gz"
_IVIM_BRAIN_BVAL = DATA_DIR / "ivim" / "brain" / "brain.bval"
_IVIM_ABDOMEN_NII = DATA_DIR / "ivim" / "abdomen" / "abdomen.nii.gz"
_IVIM_ABDOMEN_BVAL = DATA_DIR / "ivim" / "abdomen" / "abdomen.bval"


@pytest.mark.localdata
@pytest.mark.slow
class TestIVIMPipeline:
    """Run the IVIM pipeline against real NIfTI DWI data."""

    def test_ivim_brain_segmented(self, tmp_path: Path) -> None:
        """Segmented fit on brain DWI produces parameter maps."""
        _skip_unless(_IVIM_BRAIN_NII)
        _skip_unless(_IVIM_BRAIN_BVAL)

        from osipy.cli.runner import run_pipeline

        config = PipelineConfig(
            modality="ivim",
            pipeline={
                "fitting_method": "segmented",
                "b_threshold": 200.0,
                "normalize_signal": True,
            },
            data={"format": "nifti", "b_values_file": str(_IVIM_BRAIN_BVAL)},
        )

        out = tmp_path / "ivim_brain_out"
        run_pipeline(config, _IVIM_BRAIN_NII, output_dir=out)

        assert (out / "osipy_run.json").exists()
        assert (out / "d.nii.gz").exists()
        assert (out / "f.nii.gz").exists()
        assert (out / "d_star.nii.gz").exists()

        _assert_nifti_valid(out / "d.nii.gz")
        _assert_nifti_valid(out / "f.nii.gz")

    def test_ivim_abdomen_segmented(self, tmp_path: Path) -> None:
        """Segmented fit on abdomen DWI produces parameter maps."""
        _skip_unless(_IVIM_ABDOMEN_NII)
        _skip_unless(_IVIM_ABDOMEN_BVAL)

        from osipy.cli.runner import run_pipeline

        config = PipelineConfig(
            modality="ivim",
            pipeline={
                "fitting_method": "segmented",
                "b_threshold": 200.0,
                "normalize_signal": True,
            },
            data={"format": "nifti", "b_values_file": str(_IVIM_ABDOMEN_BVAL)},
        )

        out = tmp_path / "ivim_abdomen_out"
        run_pipeline(config, _IVIM_ABDOMEN_NII, output_dir=out)

        assert (out / "osipy_run.json").exists()
        assert (out / "d.nii.gz").exists()

        _assert_nifti_valid(out / "d.nii.gz")


# ---------------------------------------------------------------------------
# ASL - BIDS / ExploreASL data
# ---------------------------------------------------------------------------

_ASL_EXPLORE_DIR = DATA_DIR / "asl" / "ExploreASL_TestDataSet" / "rawdata"
_ASL_OSIPI_DATASET1 = DATA_DIR / "asl" / "OSIPI_TESTING" / "OSIPI_Dataset1" / "rawdata"


@pytest.mark.localdata
@pytest.mark.slow
class TestASLPipeline:
    """Run the ASL pipeline against real BIDS ASL data."""

    def test_asl_explore_dataset(self, tmp_path: Path) -> None:
        """ExploreASL test dataset loads and produces CBF."""
        _skip_unless(_ASL_EXPLORE_DIR)

        # Read the sidecar to get actual ASL parameters
        import json

        sidecar = _ASL_EXPLORE_DIR / "sub-Sub1" / "perf" / "sub-Sub1_asl.json"
        _skip_unless(sidecar)

        with sidecar.open() as f:
            meta = json.load(f)

        from osipy.cli.runner import run_pipeline

        config = PipelineConfig(
            modality="asl",
            pipeline={
                "labeling_scheme": meta.get(
                    "ArterialSpinLabelingType", "pcasl"
                ).lower(),
                "pld": meta.get("PostLabelingDelay", 2.025) * 1000,
                "label_duration": meta.get("LabelingDuration", 1.65) * 1000,
                "m0_method": "single",
                "label_control_order": "label_first",
            },
            data={"format": "bids", "subject": "Sub1"},
        )

        out = tmp_path / "asl_explore_out"
        run_pipeline(config, _ASL_EXPLORE_DIR, output_dir=out)

        assert (out / "osipy_run.json").exists()
        assert (out / "cbf.nii.gz").exists()
        _assert_nifti_valid(out / "cbf.nii.gz")

    def test_asl_osipi_dataset1(self, tmp_path: Path) -> None:
        """OSIPI Dataset1 loads and produces CBF."""
        _skip_unless(_ASL_OSIPI_DATASET1)

        import json

        sidecar = _ASL_OSIPI_DATASET1 / "sub-001" / "perf" / "sub-001_asl.json"
        _skip_unless(sidecar)

        with sidecar.open() as f:
            meta = json.load(f)

        from osipy.cli.runner import run_pipeline

        config = PipelineConfig(
            modality="asl",
            pipeline={
                "labeling_scheme": meta.get(
                    "ArterialSpinLabelingType", "pcasl"
                ).lower(),
                "pld": meta.get("PostLabelingDelay", 2.025) * 1000,
                "label_duration": meta.get("LabelingDuration", 1.8) * 1000,
                "m0_method": "single",
                "label_control_order": "label_first",
            },
            data={"format": "bids", "subject": "001"},
        )

        out = tmp_path / "asl_osipi1_out"
        run_pipeline(config, _ASL_OSIPI_DATASET1, output_dir=out)

        assert (out / "osipy_run.json").exists()
        assert (out / "cbf.nii.gz").exists()
        _assert_nifti_valid(out / "cbf.nii.gz")
