"""Tests for classifier coherence and loader dispatch in ``discovery.py``.

Covers:
- ``_classify`` VFA pass: demands coherent (same-shape, ≥2 distinct FA)
  clusters; lone mis-shaped candidates stay ``unknown``.
- ``load_dicom_series`` list dispatch: all-``vfa`` list goes to the
  flip-angle stack path; other lists still go to the Mode-B multi-series
  path.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from osipy.common.exceptions import DataValidationError
from osipy.common.io.discovery import SeriesInfo, _classify


def _stub(**overrides) -> SeriesInfo:
    """Build a SeriesInfo with minimal defaults; override any fields."""
    defaults: dict = {
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
        "rows": 256,
        "columns": 256,
        "n_temporal_positions": 0,
        "n_acquisition_numbers": 0,
        "n_slice_locations": 16,
    }
    defaults.update(overrides)
    return SeriesInfo(**defaults)


# ---------------------------------------------------------------------------
# _classify — VFA coherent-cluster rule
# ---------------------------------------------------------------------------


class TestClassifyVfaCluster:
    """VFA tagging requires a coherent same-shape cluster."""

    def test_coherent_cluster_tagged_as_vfa(self) -> None:
        """≥2 same-study same-shape series with distinct FAs → role_hint='vfa'."""
        series = [
            _stub(uid="u.a", flip_angle=2.0),
            _stub(uid="u.b", flip_angle=5.0),
            _stub(uid="u.c", flip_angle=15.0),
        ]
        _classify(series)
        assert [s.role_hint for s in series] == ["vfa", "vfa", "vfa"]

    def test_misshaped_candidate_stays_unknown(self) -> None:
        """A lone out-of-shape series with a unique FA must not be tagged 'vfa'.

        Reproduces the Clinical_P1 failure mode: a same-shape VFA cluster
        plus a structural (e.g. 128×128 tensor) each with distinct FAs.
        Only the coherent cluster should be tagged.
        """
        series = [
            _stub(
                uid="u.vfa5",
                flip_angle=5.0,
                rows=256,
                columns=256,
                n_slice_locations=16,
            ),
            _stub(
                uid="u.vfa10",
                flip_angle=10.0,
                rows=256,
                columns=256,
                n_slice_locations=16,
            ),
            _stub(
                uid="u.struct",
                flip_angle=90.0,
                rows=128,
                columns=128,
                n_slice_locations=40,
            ),
        ]
        _classify(series)
        by_uid = {s.uid: s.role_hint for s in series}
        assert by_uid["u.vfa5"] == "vfa"
        assert by_uid["u.vfa10"] == "vfa"
        assert by_uid["u.struct"] == "unknown"

    def test_single_flip_angle_cluster_not_vfa(self) -> None:
        """Same-shape series sharing a single FA is not VFA (need ≥2 FAs)."""
        series = [
            _stub(uid="u.a", flip_angle=10.0),
            _stub(uid="u.b", flip_angle=10.0),
        ]
        _classify(series)
        assert all(s.role_hint == "unknown" for s in series)

    def test_lone_vfa_candidate_not_tagged(self) -> None:
        """A single candidate with a unique FA cannot form a cluster."""
        series = [_stub(uid="u.solo", flip_angle=20.0)]
        _classify(series)
        assert series[0].role_hint == "unknown"

    def test_multiple_shape_groups_both_tagged(self) -> None:
        """Two independent coherent VFA clusters are both tagged."""
        series = [
            _stub(
                uid="u.a1", flip_angle=2.0, rows=256, columns=256, n_slice_locations=16
            ),
            _stub(
                uid="u.a2", flip_angle=10.0, rows=256, columns=256, n_slice_locations=16
            ),
            _stub(
                uid="u.b1", flip_angle=3.0, rows=128, columns=128, n_slice_locations=20
            ),
            _stub(
                uid="u.b2", flip_angle=15.0, rows=128, columns=128, n_slice_locations=20
            ),
        ]
        _classify(series)
        assert all(s.role_hint == "vfa" for s in series)

    def test_cross_study_not_grouped(self) -> None:
        """Same-shape VFAs from different studies shouldn't cluster together."""
        series = [
            _stub(uid="u.v1", study_instance_uid="study.A", flip_angle=5.0),
            _stub(uid="u.v2", study_instance_uid="study.B", flip_angle=15.0),
        ]
        _classify(series)
        # Each is alone within its study → no cluster → stays unknown.
        assert all(s.role_hint == "unknown" for s in series)


# ---------------------------------------------------------------------------
# load_dicom_series — VFA list dispatch
# ---------------------------------------------------------------------------


class TestLoadDicomSeriesListDispatch:
    """``load_dicom_series(list)`` dispatches on role_hint of list members."""

    def test_all_vfa_list_goes_to_stack_path(self) -> None:
        from osipy.common.io.discovery import load_dicom_series

        vfa_a = _stub(uid="u.a", role_hint="vfa", flip_angle=5.0)
        vfa_b = _stub(uid="u.b", role_hint="vfa", flip_angle=15.0)

        with (
            patch("osipy.common.io.discovery._load_vfa_stack") as mock_vfa,
            patch("osipy.common.io.discovery._load_multi_series") as mock_multi,
        ):
            load_dicom_series([vfa_a, vfa_b])

        mock_vfa.assert_called_once()
        mock_multi.assert_not_called()

    def test_dynamic_frame_list_goes_to_multi_series(self) -> None:
        from osipy.common.io.discovery import load_dicom_series

        frames = [
            _stub(uid=f"u.f{i}", role_hint="dynamic_frame", group_key="g1")
            for i in range(3)
        ]

        with (
            patch("osipy.common.io.discovery._load_vfa_stack") as mock_vfa,
            patch("osipy.common.io.discovery._load_multi_series") as mock_multi,
        ):
            load_dicom_series(frames)

        mock_multi.assert_called_once()
        mock_vfa.assert_not_called()

    def test_mixed_list_goes_to_multi_series(self) -> None:
        """Fallback when the list isn't uniformly VFA."""
        from osipy.common.io.discovery import load_dicom_series

        series = [
            _stub(uid="u.vfa", role_hint="vfa", flip_angle=5.0),
            _stub(uid="u.df", role_hint="dynamic_frame", group_key="g1"),
        ]
        with (
            patch("osipy.common.io.discovery._load_vfa_stack") as mock_vfa,
            patch("osipy.common.io.discovery._load_multi_series") as mock_multi,
        ):
            load_dicom_series(series)

        mock_multi.assert_called_once()
        mock_vfa.assert_not_called()

    def test_empty_list_raises(self) -> None:
        from osipy.common.io.discovery import load_dicom_series

        with pytest.raises(DataValidationError, match="empty series list"):
            load_dicom_series([])


# ---------------------------------------------------------------------------
# _load_vfa_stack — stacking semantics (no DICOM I/O, uses stubbed 3D volumes)
# ---------------------------------------------------------------------------


class TestLoadVfaStack:
    """Exercise ``_load_vfa_stack`` with mocked single-series loads."""

    def _stub_volume(self, fa: float, shape=(32, 32, 16)):
        """Build a 3D PerfusionDataset stub for one flip angle."""
        from osipy.common.dataset import PerfusionDataset
        from osipy.common.types import AcquisitionParams, Modality

        return PerfusionDataset(
            data=np.full(shape, fa, dtype=np.float32),
            affine=np.eye(4),
            modality=Modality.DCE,
            acquisition_params=AcquisitionParams(
                tr=5.0, te=2.0, flip_angle=fa, field_strength=3.0
            ),
        )

    def test_stacks_sorted_by_flip_angle(self, tmp_path) -> None:
        from osipy.common.io.discovery import _load_vfa_stack
        from osipy.common.types import Modality

        dummy = tmp_path / "placeholder.dcm"
        dummy.write_bytes(b"")
        vfa_a = _stub(uid="u.a", role_hint="vfa", flip_angle=15.0, files=[dummy])
        vfa_b = _stub(uid="u.b", role_hint="vfa", flip_angle=5.0, files=[dummy])
        vfa_c = _stub(uid="u.c", role_hint="vfa", flip_angle=10.0, files=[dummy])

        by_fa = {
            15.0: self._stub_volume(15.0),
            5.0: self._stub_volume(5.0),
            10.0: self._stub_volume(10.0),
        }

        def _fake_load(s, _mod):
            return by_fa[float(s.flip_angle)]

        with patch(
            "osipy.common.io.discovery._load_single_series",
            side_effect=_fake_load,
        ):
            ds = _load_vfa_stack([vfa_a, vfa_b, vfa_c], Modality.DCE)

        assert ds.data.shape == (32, 32, 16, 3)
        assert list(ds.acquisition_params.flip_angles) == [5.0, 10.0, 15.0]
        # Each 3D slab along the last axis is filled with its FA value
        # (from the stub), so this confirms the sort order.
        assert ds.data[0, 0, 0, 0] == pytest.approx(5.0)
        assert ds.data[0, 0, 0, 1] == pytest.approx(10.0)
        assert ds.data[0, 0, 0, 2] == pytest.approx(15.0)
        assert ds.acquisition_params.tr == 5.0
        assert ds.acquisition_params.te == 2.0
        assert ds.time_points is not None
        assert len(ds.time_points) == 3

    def test_raises_on_single_series(self) -> None:
        from osipy.common.io.discovery import _load_vfa_stack
        from osipy.common.types import Modality

        with pytest.raises(DataValidationError, match=">= 2"):
            _load_vfa_stack(
                [_stub(uid="u.solo", role_hint="vfa", flip_angle=10.0)],
                Modality.DCE,
            )

    def test_raises_on_missing_flip_angle(self) -> None:
        from osipy.common.io.discovery import _load_vfa_stack
        from osipy.common.types import Modality

        with pytest.raises(DataValidationError, match="FlipAngle"):
            _load_vfa_stack(
                [
                    _stub(uid="u.a", role_hint="vfa", flip_angle=5.0),
                    _stub(uid="u.b", role_hint="vfa", flip_angle=None),
                ],
                Modality.DCE,
            )

    def test_raises_on_shape_mismatch(self, tmp_path) -> None:
        from osipy.common.io.discovery import _load_vfa_stack
        from osipy.common.types import Modality

        dummy = tmp_path / "placeholder.dcm"
        dummy.write_bytes(b"")
        vfa_a = _stub(uid="u.a", role_hint="vfa", flip_angle=5.0, files=[dummy])
        vfa_b = _stub(uid="u.b", role_hint="vfa", flip_angle=10.0, files=[dummy])

        vols = [
            self._stub_volume(5.0, shape=(32, 32, 16)),
            self._stub_volume(10.0, shape=(32, 32, 15)),
        ]

        def _fake_load(s, _mod):
            return vols[0] if s.uid == "u.a" else vols[1]

        with (
            patch(
                "osipy.common.io.discovery._load_single_series",
                side_effect=_fake_load,
            ),
            pytest.raises(DataValidationError, match="shape mismatch"),
        ):
            _load_vfa_stack([vfa_a, vfa_b], Modality.DCE)
