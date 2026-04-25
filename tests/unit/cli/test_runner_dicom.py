"""Tests for the DICOM pathway in the CLI pipeline runner.

Covers:
- ``_detect_format`` on directories with nested DICOM subdirectories.
- ``_select_dataset_series`` preference order
  (dynamic_frame cluster > dynamic > t1_look_locker > imaged).
- ``_load_data`` dispatch to ``discover_dicom`` + ``load_dicom_series``.
- ``_discover_dce_dicom`` extraction of VFA + dynamic from discovery output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from osipy.common.io.discovery import SeriesInfo

if TYPE_CHECKING:
    from pathlib import Path


def _stub(**overrides):
    """Build a SeriesInfo with minimal defaults; override any fields."""
    defaults = {
        "uid": "1.2.3.1",
        "study_instance_uid": "1.2.3",
        "files": [],
        "description": "",
        "series_number": None,
        "dicom_modality": "MR",
        "manufacturer": None,
        "flip_angle": None,
        "tr": None,
        "te": None,
        "field_strength": None,
        "rows": 64,
        "columns": 64,
        "n_temporal_positions": 0,
        "n_acquisition_numbers": 0,
        "n_slice_locations": 1,
    }
    defaults.update(overrides)
    return SeriesInfo(**defaults)


# ---------------------------------------------------------------------------
# _detect_format
# ---------------------------------------------------------------------------


class TestDetectFormatDicomDirectory:
    """Runner's ``_detect_format`` on directories."""

    def test_flat_directory_with_dcm_detected_as_dicom(self, tmp_path: Path) -> None:
        from osipy.cli.runner import _detect_format

        (tmp_path / "001.dcm").write_bytes(b"fake")
        assert _detect_format(tmp_path) == "dicom"

    def test_nested_subdirs_detected_as_dicom(self, tmp_path: Path) -> None:
        """Study dir with per-series subdirs containing .dcm still maps
        to 'dicom' (discover_dicom walks recursively)."""
        from osipy.cli.runner import _detect_format

        (tmp_path / "SeriesA").mkdir()
        (tmp_path / "SeriesB").mkdir()
        (tmp_path / "SeriesA" / "001.dcm").write_bytes(b"fake")
        (tmp_path / "SeriesB" / "001.dcm").write_bytes(b"fake")
        assert _detect_format(tmp_path) == "dicom"


# ---------------------------------------------------------------------------
# _select_dataset_series
# ---------------------------------------------------------------------------


class TestSelectDatasetSeries:
    """Preference order for picking a series from discovery output."""

    def test_dynamic_frame_cluster_wins(self) -> None:
        from osipy.cli.runner import _select_dataset_series

        dyn = _stub(
            uid="u.dyn",
            description="DCE",
            role_hint="dynamic",
            n_temporal_positions=30,
        )
        frames = [
            _stub(uid=f"u.frame.{i}", role_hint="dynamic_frame", group_key="g1")
            for i in range(5)
        ]
        chosen = _select_dataset_series([dyn, *frames])
        assert isinstance(chosen, list)
        assert len(chosen) == 5

    def test_dynamic_preferred_over_t1_look_locker(self) -> None:
        from osipy.cli.runner import _select_dataset_series

        dyn = _stub(uid="u.dyn", role_hint="dynamic")
        ll = _stub(uid="u.ll", role_hint="t1_look_locker")
        chosen = _select_dataset_series([ll, dyn])
        assert chosen is dyn

    def test_t1_look_locker_picked_without_dynamic(self) -> None:
        from osipy.cli.runner import _select_dataset_series

        ll = _stub(uid="u.ll", role_hint="t1_look_locker")
        unk = _stub(uid="u.unk", role_hint="unknown")
        chosen = _select_dataset_series([unk, ll])
        assert chosen is ll

    def test_falls_through_to_imaged_series(self) -> None:
        from osipy.cli.runner import _select_dataset_series

        vfa = _stub(uid="u.vfa", role_hint="vfa")
        unk = _stub(uid="u.unk", role_hint="unknown")
        chosen = _select_dataset_series([unk, vfa])
        assert chosen is vfa

    def test_last_resort_returns_first(self) -> None:
        from osipy.cli.runner import _select_dataset_series

        a = _stub(uid="u.a", role_hint="unknown")
        b = _stub(uid="u.b", role_hint="unknown")
        chosen = _select_dataset_series([a, b])
        assert chosen is a


# ---------------------------------------------------------------------------
# _load_data
# ---------------------------------------------------------------------------


class TestLoadDataDispatch:
    """``_load_data`` routes to the right loader based on format."""

    def test_dicom_dispatches_to_discover_and_load_series(self, tmp_path: Path) -> None:
        from osipy.cli.config import DataConfig, PipelineConfig
        from osipy.cli.runner import _load_data

        (tmp_path / "001.dcm").write_bytes(b"fake")
        config = PipelineConfig(modality="dce", data=DataConfig(format="dicom"))
        dyn = _stub(uid="u.dyn", role_hint="dynamic")

        with (
            patch(
                "osipy.common.io.discovery.discover_dicom",
                return_value=[dyn],
            ) as mock_disc,
            patch(
                "osipy.common.io.discovery.load_dicom_series",
                return_value=MagicMock(),
            ) as mock_load,
        ):
            _load_data(config, tmp_path, "DCE")

            from osipy.common.types import Modality

            mock_disc.assert_called_once_with(tmp_path)
            mock_load.assert_called_once()
            # load_dicom_series called with the single discovered series
            args, kwargs = mock_load.call_args
            assert args[0] is dyn
            assert kwargs["modality"] == Modality.DCE

    def test_nifti_dispatches_to_load_nifti(self, tmp_path: Path) -> None:
        from osipy.cli.config import DataConfig, PipelineConfig
        from osipy.cli.runner import _load_data

        nii = tmp_path / "x.nii.gz"
        nii.write_bytes(b"x")
        config = PipelineConfig(modality="dce", data=DataConfig(format="nifti"))

        with patch(
            "osipy.common.io.nifti.load_nifti",
            return_value=MagicMock(),
        ) as mock_ln:
            from osipy.common.types import Modality

            _load_data(config, nii, "DCE")
            mock_ln.assert_called_once_with(nii, modality=Modality.DCE)

    def test_bids_requires_subject(self, tmp_path: Path) -> None:
        import pytest

        from osipy.cli.config import DataConfig, PipelineConfig
        from osipy.cli.runner import _load_data

        config = PipelineConfig(
            modality="asl",
            data=DataConfig(format="bids"),
        )
        with pytest.raises(ValueError, match="subject"):
            _load_data(config, tmp_path, "ASL")


# ---------------------------------------------------------------------------
# _discover_dce_dicom
# ---------------------------------------------------------------------------


class TestDiscoverDceDicom:
    """DCE pipeline's VFA + dynamic extraction."""

    def test_returns_vfa_and_mode_a_dynamic(self, tmp_path: Path) -> None:
        from osipy.cli.runner import _discover_dce_dicom

        vfa5 = _stub(uid="u.v5", role_hint="vfa", flip_angle=5.0)
        vfa10 = _stub(uid="u.v10", role_hint="vfa", flip_angle=10.0)
        dyn = _stub(
            uid="u.dyn",
            role_hint="dynamic",
            flip_angle=30.0,
            n_temporal_positions=20,
            description="DCE",
        )
        with patch(
            "osipy.common.io.discovery.discover_dicom",
            return_value=[vfa10, vfa5, dyn],
        ):
            result = _discover_dce_dicom(tmp_path)

        assert result is not None
        vfa_list, dce = result
        assert [s.flip_angle for s in vfa_list] == [5.0, 10.0]
        assert dce is dyn

    def test_returns_vfa_and_mode_b_frame_cluster(self, tmp_path: Path) -> None:
        from osipy.cli.runner import _discover_dce_dicom

        vfa = _stub(uid="u.v10", role_hint="vfa", flip_angle=10.0)
        frames = [
            _stub(
                uid=f"u.frame.{i}",
                role_hint="dynamic_frame",
                group_key="g1",
                description=f"DCE_TT={i}.0s",
            )
            for i in range(4)
        ]
        with patch(
            "osipy.common.io.discovery.discover_dicom",
            return_value=[vfa, *frames],
        ):
            result = _discover_dce_dicom(tmp_path)

        assert result is not None
        vfa_list, dce = result
        assert len(vfa_list) == 1
        assert isinstance(dce, list)
        assert len(dce) == 4

    def test_none_when_no_vfa(self, tmp_path: Path) -> None:
        from osipy.cli.runner import _discover_dce_dicom

        dyn = _stub(uid="u.dyn", role_hint="dynamic")
        with patch(
            "osipy.common.io.discovery.discover_dicom",
            return_value=[dyn],
        ):
            assert _discover_dce_dicom(tmp_path) is None

    def test_none_when_no_dynamic(self, tmp_path: Path) -> None:
        from osipy.cli.runner import _discover_dce_dicom

        vfa = _stub(uid="u.v", role_hint="vfa", flip_angle=5.0)
        with patch(
            "osipy.common.io.discovery.discover_dicom",
            return_value=[vfa],
        ):
            assert _discover_dce_dicom(tmp_path) is None

    def test_none_for_file_path(self, tmp_path: Path) -> None:
        """Non-directory paths (NIfTI file inputs) bypass DICOM discovery."""
        from osipy.cli.runner import _discover_dce_dicom

        nii = tmp_path / "x.nii.gz"
        nii.write_bytes(b"x")
        assert _discover_dce_dicom(nii) is None
