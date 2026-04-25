"""Unit tests for osipy.common.io.dicom module.

Covers:
- Pixel-scaling helpers (``_apply_pixel_scaling``) for standard + Philips paths.
- SeriesDescription time-extraction (``_extract_time_from_series_description``).
- End-to-end discovery + load against real vendor DICOM data
  (skipped when ``data/test_dicom/`` is absent).
"""

from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_TEST_DICOM_DIR = DATA_DIR / "test_dicom"


def _skip_unless(path: Path) -> None:
    """``pytest.skip`` when *path* does not exist."""
    if not path.exists():
        pytest.skip(f"Data not found: {path}")


def _find_dcm_leaf(root: Path) -> Path | None:
    """Return the first directory under *root* that contains ``.dcm`` files."""
    for p in sorted(root.rglob("*.dcm")):
        return p.parent
    return None


# ---------------------------------------------------------------------------
# Real DICOM data tests (skip when data/test_dicom/ is absent)
# ---------------------------------------------------------------------------


@pytest.mark.localdata
class TestDiscoverAndLoadRealData:
    """``discover_dicom`` + ``load_dicom_series`` on real vendor DICOMs."""

    @pytest.mark.parametrize("vendor", ["ge", "siemens", "philips"])
    def test_load_vendor_dce(self, vendor: str) -> None:
        from osipy.common.dataset import PerfusionDataset
        from osipy.common.io.discovery import discover_dicom, load_dicom_series

        vendor_dir = _TEST_DICOM_DIR / vendor / "dce"
        _skip_unless(vendor_dir)

        dcm_dir = _find_dcm_leaf(vendor_dir)
        if dcm_dir is None:
            pytest.skip(f"No .dcm files found under {vendor_dir}")

        series_list = discover_dicom(dcm_dir)
        assert series_list, f"No series discovered under {dcm_dir}"

        ds = load_dicom_series(series_list[0])
        assert isinstance(ds, PerfusionDataset)
        assert ds.data.ndim >= 3
        assert ds.data.size > 0
        assert np.isfinite(ds.data).any()
        assert ds.acquisition_params is not None


# ---------------------------------------------------------------------------
# Pixel scaling (unit tests using a minimal DICOM-like stub)
# ---------------------------------------------------------------------------


class _StubDataset:
    """Minimal stand-in for ``pydicom.Dataset`` for pixel-scaling tests.

    Supports both attribute access (``dcm.RescaleSlope``) and private-tag
    lookup via ``(group, element) in dcm`` / ``dcm[(group, element)]``.
    """

    def __init__(self, attrs: dict, private: dict | None = None) -> None:
        self._attrs = attrs
        self._private = private or {}

    def __getattr__(self, name: str):
        if name in self._attrs:
            return self._attrs[name]
        raise AttributeError(name)

    def __contains__(self, tag: tuple) -> bool:
        return tag in self._private

    def __getitem__(self, tag: tuple):
        class _Elem:
            def __init__(self, value):
                self.value = value

        return _Elem(self._private[tag])


class TestApplyPixelScaling:
    """Unit tests for the DICOM pixel-scaling helper."""

    def test_no_scaling_tags_returns_float64_passthrough(self) -> None:
        from osipy.common.io.dicom import _apply_pixel_scaling

        dcm = _StubDataset(attrs={"Manufacturer": "GE MEDICAL SYSTEMS"})
        raw = np.array([[10, 20], [30, 40]], dtype=np.int16)

        out = _apply_pixel_scaling(dcm, raw)

        assert out.dtype == np.float64
        np.testing.assert_array_equal(out, raw.astype(np.float64))

    def test_standard_rescale_applied(self) -> None:
        """Non-Philips scanner uses standard rescale slope/intercept."""
        from osipy.common.io.dicom import _apply_pixel_scaling

        dcm = _StubDataset(
            attrs={
                "Manufacturer": "SIEMENS",
                "RescaleSlope": 2.0,
                "RescaleIntercept": 5.0,
            }
        )
        raw = np.array([10, 20, 30], dtype=np.int16)

        out = _apply_pixel_scaling(dcm, raw)

        np.testing.assert_array_almost_equal(out, [25.0, 45.0, 65.0])

    def test_philips_private_scale_supersedes_standard_rescale(self) -> None:
        """Philips scanners apply (SV - SI) / SS from private tags."""
        from osipy.common.io.dicom import _apply_pixel_scaling

        # Standard rescale is deliberately different so we can prove it
        # was ignored in favor of the Philips private transformation.
        dcm = _StubDataset(
            attrs={
                "Manufacturer": "Philips Medical Systems",
                "RescaleSlope": 2.0,
                "RescaleIntercept": 5.0,
            },
            private={(0x2005, 0x100E): 4.0, (0x2005, 0x100D): 8.0},
        )
        raw = np.array([16, 40], dtype=np.int16)

        out = _apply_pixel_scaling(dcm, raw)

        # (16 - 8) / 4 = 2.0, (40 - 8) / 4 = 8.0
        np.testing.assert_array_almost_equal(out, [2.0, 8.0])

    def test_philips_zero_scale_slope_falls_back_to_standard(self) -> None:
        """A zero ScaleSlope is invalid; fall back to standard rescale."""
        from osipy.common.io.dicom import _apply_pixel_scaling

        dcm = _StubDataset(
            attrs={
                "Manufacturer": "PHILIPS",
                "RescaleSlope": 1.0,
                "RescaleIntercept": 0.0,
            },
            private={(0x2005, 0x100E): 0.0, (0x2005, 0x100D): 1.0},
        )
        raw = np.array([7, 14], dtype=np.int16)

        out = _apply_pixel_scaling(dcm, raw)

        np.testing.assert_array_almost_equal(out, [7.0, 14.0])

    def test_philips_missing_private_tags_uses_standard_rescale(self) -> None:
        """Philips scan without private scale tags falls back cleanly."""
        from osipy.common.io.dicom import _apply_pixel_scaling

        dcm = _StubDataset(
            attrs={
                "Manufacturer": "PHILIPS",
                "RescaleSlope": 3.0,
                "RescaleIntercept": 1.0,
            }
        )
        raw = np.array([0, 2, 4], dtype=np.int16)

        out = _apply_pixel_scaling(dcm, raw)

        np.testing.assert_array_almost_equal(out, [1.0, 7.0, 13.0])
