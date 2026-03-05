"""Unit tests for M0 calibration.

Tests for osipy/asl/calibration/m0.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from osipy.asl.calibration.m0 import (
    M0CalibrationParams,
    apply_m0_calibration,
    compute_m0_from_pd,
)


class TestM0CalibrationParams:
    """Tests for M0CalibrationParams dataclass."""

    def test_default_params(self) -> None:
        """Test default parameter values."""
        params = M0CalibrationParams()
        assert params.method == "single"
        assert params.reference_region == "csf"
        assert params.t1_tissue == 1330.0  # ms
        assert params.t2_star_tissue is None
        assert params.tr_m0 == 6000.0  # ms
        assert params.te_m0 == 13.0  # ms

    def test_custom_params(self) -> None:
        """Test custom parameter values."""
        params = M0CalibrationParams(
            method="voxelwise",
            reference_region="white_matter",
            t1_tissue=1400.0,
            t2_star_tissue=50.0,
            tr_m0=8000.0,
            te_m0=10.0,
        )
        assert params.method == "voxelwise"
        assert params.reference_region == "white_matter"
        assert params.t2_star_tissue == 50.0


class TestApplyM0Calibration:
    """Tests for apply_m0_calibration function."""

    @pytest.fixture
    def synthetic_data(self) -> dict:
        """Create synthetic ASL and M0 data."""
        np.random.seed(42)

        # Spatial dimensions
        nx, ny, nz = 16, 16, 4

        # M0 image (equilibrium magnetization)
        m0_image = np.random.uniform(800, 1200, (nx, ny, nz))

        # ASL difference image
        asl_data = np.random.uniform(5, 20, (nx, ny, nz))

        # Mask
        mask = np.ones((nx, ny, nz), dtype=bool)
        mask[:2, :, :] = False

        return {
            "asl_data": asl_data,
            "m0_image": m0_image,
            "mask": mask,
            "shape": (nx, ny, nz),
        }

    def test_apply_m0_calibration_single(self, synthetic_data: dict) -> None:
        """Test M0 calibration with single (mean) method."""
        params = M0CalibrationParams(method="single")

        calibrated, m0_corrected = apply_m0_calibration(
            asl_data=synthetic_data["asl_data"],
            m0_image=synthetic_data["m0_image"],
            params=params,
            mask=synthetic_data["mask"],
        )

        assert calibrated.shape == synthetic_data["shape"]
        assert m0_corrected.shape == synthetic_data["shape"]
        # Calibrated values should be much smaller (divided by M0)
        assert np.mean(calibrated) < np.mean(synthetic_data["asl_data"])

    def test_apply_m0_calibration_voxelwise(self, synthetic_data: dict) -> None:
        """Test M0 calibration with voxelwise method."""
        params = M0CalibrationParams(method="voxelwise")

        calibrated, m0_corrected = apply_m0_calibration(
            asl_data=synthetic_data["asl_data"],
            m0_image=synthetic_data["m0_image"],
            params=params,
            mask=synthetic_data["mask"],
        )

        assert calibrated.shape == synthetic_data["shape"]

        # Check that division was voxelwise
        expected = synthetic_data["asl_data"] / m0_corrected
        expected = np.where(synthetic_data["mask"] & (m0_corrected > 0), expected, 0)
        np.testing.assert_array_almost_equal(
            calibrated[synthetic_data["mask"]],
            expected[synthetic_data["mask"]],
            decimal=5,
        )

    def test_apply_m0_calibration_reference_region(self, synthetic_data: dict) -> None:
        """Test M0 calibration with reference region method."""
        params = M0CalibrationParams(
            method="reference_region",
            reference_region="csf",
        )

        calibrated, _m0_corrected = apply_m0_calibration(
            asl_data=synthetic_data["asl_data"],
            m0_image=synthetic_data["m0_image"],
            params=params,
            mask=synthetic_data["mask"],
        )

        assert calibrated.shape == synthetic_data["shape"]

    def test_apply_m0_calibration_white_matter_ref(self, synthetic_data: dict) -> None:
        """Test M0 calibration with white matter reference."""
        params = M0CalibrationParams(
            method="reference_region",
            reference_region="white_matter",
        )

        calibrated, _m0_corrected = apply_m0_calibration(
            asl_data=synthetic_data["asl_data"],
            m0_image=synthetic_data["m0_image"],
            params=params,
            mask=synthetic_data["mask"],
        )

        assert calibrated.shape == synthetic_data["shape"]

    def test_apply_m0_calibration_without_mask(self, synthetic_data: dict) -> None:
        """Test M0 calibration without mask."""
        calibrated, _m0_corrected = apply_m0_calibration(
            asl_data=synthetic_data["asl_data"],
            m0_image=synthetic_data["m0_image"],
        )

        assert calibrated.shape == synthetic_data["shape"]

    def test_apply_m0_calibration_t1_correction(self, synthetic_data: dict) -> None:
        """Test that T1 recovery correction is applied."""
        # Short TR should have larger correction factor
        params_short_tr = M0CalibrationParams(
            tr_m0=2000.0,  # Short TR
            t1_tissue=1330.0,
        )

        params_long_tr = M0CalibrationParams(
            tr_m0=10000.0,  # Long TR (full recovery)
            t1_tissue=1330.0,
        )

        _, m0_short = apply_m0_calibration(
            asl_data=synthetic_data["asl_data"],
            m0_image=synthetic_data["m0_image"],
            params=params_short_tr,
        )

        _, m0_long = apply_m0_calibration(
            asl_data=synthetic_data["asl_data"],
            m0_image=synthetic_data["m0_image"],
            params=params_long_tr,
        )

        # Short TR should have higher corrected M0 (compensating for less recovery)
        assert np.mean(m0_short) > np.mean(m0_long)

    def test_apply_m0_calibration_t2_star_correction(
        self, synthetic_data: dict
    ) -> None:
        """Test that T2* correction is applied when specified."""
        params_no_t2 = M0CalibrationParams(
            t2_star_tissue=None,
            te_m0=20.0,
        )

        params_with_t2 = M0CalibrationParams(
            t2_star_tissue=50.0,  # 50ms T2*
            te_m0=20.0,
        )

        _, m0_no_t2 = apply_m0_calibration(
            asl_data=synthetic_data["asl_data"],
            m0_image=synthetic_data["m0_image"],
            params=params_no_t2,
        )

        _, m0_with_t2 = apply_m0_calibration(
            asl_data=synthetic_data["asl_data"],
            m0_image=synthetic_data["m0_image"],
            params=params_with_t2,
        )

        # With T2* correction, M0 should be higher (compensating for decay)
        assert np.mean(m0_with_t2) > np.mean(m0_no_t2)

    def test_apply_m0_calibration_shape_mismatch(self, synthetic_data: dict) -> None:
        """Test that shape mismatch raises error."""
        from osipy.common.exceptions import DataValidationError

        wrong_m0 = np.ones((8, 8, 2)) * 1000

        with pytest.raises(DataValidationError, match="Shape mismatch"):
            apply_m0_calibration(
                asl_data=synthetic_data["asl_data"],
                m0_image=wrong_m0,
            )

    def test_calibrated_removes_nan_inf(self, synthetic_data: dict) -> None:
        """Test that NaN and Inf values are handled."""
        # Create M0 with zero value
        m0_with_zero = synthetic_data["m0_image"].copy()
        m0_with_zero[5, 5, 2] = 0

        params = M0CalibrationParams(method="voxelwise")

        calibrated, _ = apply_m0_calibration(
            asl_data=synthetic_data["asl_data"],
            m0_image=m0_with_zero,
            params=params,
            mask=synthetic_data["mask"],
        )

        # Should have no NaN or Inf
        assert not np.any(np.isnan(calibrated))
        assert not np.any(np.isinf(calibrated))


class TestComputeM0FromPD:
    """Tests for compute_m0_from_pd function."""

    def test_compute_m0_basic(self) -> None:
        """Test basic M0 computation from PD."""
        pd_image = np.ones((10, 10, 5)) * 500

        m0 = compute_m0_from_pd(
            pd_image=pd_image,
            t1_tissue=1330.0,
            t2_tissue=80.0,
            tr=6000.0,
            te=13.0,
        )

        assert m0.shape == pd_image.shape
        # M0 should be larger than PD (corrections for T1/T2)
        assert np.mean(m0) >= np.mean(pd_image)

    def test_compute_m0_long_tr(self) -> None:
        """Test M0 computation with long TR (minimal T1 correction)."""
        pd_image = np.ones((5, 5, 3)) * 500

        # Very long TR - T1 correction factor should be ~1
        m0 = compute_m0_from_pd(
            pd_image=pd_image,
            t1_tissue=1330.0,
            tr=30000.0,  # 30s TR
            te=0.0,  # No T2 decay
        )

        # Should be approximately equal to PD
        np.testing.assert_array_almost_equal(m0, pd_image, decimal=0)

    def test_compute_m0_short_tr_increases_m0(self) -> None:
        """Test that shorter TR increases estimated M0."""
        pd_image = np.ones((5, 5, 3)) * 500

        m0_long_tr = compute_m0_from_pd(
            pd_image=pd_image,
            tr=10000.0,
            te=10.0,
        )

        m0_short_tr = compute_m0_from_pd(
            pd_image=pd_image,
            tr=2000.0,
            te=10.0,
        )

        # Shorter TR needs larger correction
        assert np.mean(m0_short_tr) > np.mean(m0_long_tr)

    def test_compute_m0_te_effect(self) -> None:
        """Test T2 decay correction effect."""
        pd_image = np.ones((5, 5, 3)) * 500

        m0_short_te = compute_m0_from_pd(
            pd_image=pd_image,
            tr=10000.0,
            te=5.0,
            t2_tissue=80.0,
        )

        m0_long_te = compute_m0_from_pd(
            pd_image=pd_image,
            tr=10000.0,
            te=30.0,
            t2_tissue=80.0,
        )

        # Longer TE needs larger T2 correction
        assert np.mean(m0_long_te) > np.mean(m0_short_te)
