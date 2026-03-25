"""Unit tests for ASL QC SNR metric.

Tests for osipy/asl/qc/snr.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from osipy.asl.qc.snr import (
    ASLSNRParams,
    ASLSNRResult,
    compute_snr,
)
from osipy.common.exceptions import DataValidationError
from osipy.common.parameter_map import ParameterMap


class TestASLSNRParams:
    """Tests for ASLSNRParams dataclass."""

    def test_default_params(self) -> None:
        """Test default parameter values."""
        params = ASLSNRParams()
        assert params.min_noise_voxels == 10
        assert params.snr_threshold == 2.0

    def test_custom_params(self) -> None:
        """Test custom parameter values."""
        params = ASLSNRParams(min_noise_voxels=20, snr_threshold=5.0)
        assert params.min_noise_voxels == 20
        assert params.snr_threshold == 5.0


class TestComputeSNR:
    """Tests for compute_snr function."""

    @pytest.fixture
    def synthetic_data(self) -> dict:
        """Create synthetic ASL CBF data with noise region."""
        np.random.seed(42)
        shape = (16, 16, 8)

        # Simulate CBF map: brain voxels ~60 mL/100g/min + small noise
        cbf = np.random.normal(loc=60.0, scale=5.0, size=shape)

        # Noise region: top slice — background with std ~10
        noise_mask = np.zeros(shape, dtype=bool)
        noise_mask[0, :, :] = True
        cbf[noise_mask] = np.random.normal(loc=0.0, scale=10.0, size=noise_mask.sum())

        # Brain mask: everything except noise slice
        brain_mask = ~noise_mask

        return {
            "cbf": cbf,
            "noise_mask": noise_mask,
            "brain_mask": brain_mask,
            "shape": shape,
        }

    def test_returns_asl_snr_result(self, synthetic_data: dict) -> None:
        """Test that output is ASLSNRResult."""
        result = compute_snr(
            cbf_map=synthetic_data["cbf"],
            noise_mask=synthetic_data["noise_mask"],
        )
        assert isinstance(result, ASLSNRResult)

    def test_snr_map_is_parameter_map(self, synthetic_data: dict) -> None:
        """Test that snr_map is a ParameterMap with correct metadata."""
        result = compute_snr(
            cbf_map=synthetic_data["cbf"],
            noise_mask=synthetic_data["noise_mask"],
        )
        assert isinstance(result.snr_map, ParameterMap)
        assert result.snr_map.name == "SNR"
        assert result.snr_map.units == "dimensionless"

    def test_snr_map_shape_matches_input(self, synthetic_data: dict) -> None:
        """Test SNR map has same shape as input CBF map."""
        result = compute_snr(
            cbf_map=synthetic_data["cbf"],
            noise_mask=synthetic_data["noise_mask"],
        )
        assert result.snr_map.values.shape == synthetic_data["shape"]

    def test_noise_std_positive(self, synthetic_data: dict) -> None:
        """Test that noise_std is a positive float."""
        result = compute_snr(
            cbf_map=synthetic_data["cbf"],
            noise_mask=synthetic_data["noise_mask"],
        )
        assert isinstance(result.noise_std, float)
        assert result.noise_std > 0.0

    def test_mean_snr_positive(self, synthetic_data: dict) -> None:
        """Test that mean SNR over brain is positive."""
        result = compute_snr(
            cbf_map=synthetic_data["cbf"],
            noise_mask=synthetic_data["noise_mask"],
        )
        assert result.mean_snr > 0.0

    def test_snr_non_negative_in_brain(self, synthetic_data: dict) -> None:
        """Test SNR values are non-negative in brain region."""
        result = compute_snr(
            cbf_map=synthetic_data["cbf"],
            noise_mask=synthetic_data["noise_mask"],
            brain_mask=synthetic_data["brain_mask"],
        )
        brain_snr = result.snr_map.values[synthetic_data["brain_mask"]]
        assert np.all(brain_snr >= 0.0), "SNR should be non-negative in brain"

    def test_noise_region_zeroed_in_snr_map(self, synthetic_data: dict) -> None:
        """Test that noise region is zeroed out in SNR map."""
        result = compute_snr(
            cbf_map=synthetic_data["cbf"],
            noise_mask=synthetic_data["noise_mask"],
        )
        noise_snr = result.snr_map.values[synthetic_data["noise_mask"]]
        assert np.all(noise_snr == 0.0), "Noise region should be zero in SNR map"

    def test_quality_mask_shape_matches_input(self, synthetic_data: dict) -> None:
        """Test quality mask has same shape as input CBF map."""
        result = compute_snr(
            cbf_map=synthetic_data["cbf"],
            noise_mask=synthetic_data["noise_mask"],
        )
        assert result.quality_mask.shape == synthetic_data["shape"]

    def test_quality_mask_bool_dtype(self, synthetic_data: dict) -> None:
        """Test quality mask is boolean."""
        result = compute_snr(
            cbf_map=synthetic_data["cbf"],
            noise_mask=synthetic_data["noise_mask"],
        )
        assert result.quality_mask.dtype == bool

    def test_snr_increases_with_higher_signal(self) -> None:
        """Test SNR increases when signal strength increases."""
        shape = (10, 10, 4)
        noise_mask = np.zeros(shape, dtype=bool)
        noise_mask[0, :, :] = True

        # Noise region with fixed std
        noise = np.random.normal(0.0, 5.0, size=shape)
        noise[~noise_mask] = 0.0

        # Low signal
        cbf_low = noise.copy()
        cbf_low[~noise_mask] = 30.0

        # High signal
        cbf_high = noise.copy()
        cbf_high[~noise_mask] = 120.0

        result_low = compute_snr(cbf_low, noise_mask)
        result_high = compute_snr(cbf_high, noise_mask)

        assert result_high.mean_snr > result_low.mean_snr

    def test_snr_threshold_quality_mask(self, synthetic_data: dict) -> None:
        """Test that quality_mask respects snr_threshold."""
        # Very high threshold → most voxels fail
        params_strict = ASLSNRParams(snr_threshold=1000.0)
        result_strict = compute_snr(
            cbf_map=synthetic_data["cbf"],
            noise_mask=synthetic_data["noise_mask"],
            params=params_strict,
        )

        # Very low threshold → almost all voxels pass
        params_loose = ASLSNRParams(snr_threshold=0.0)
        result_loose = compute_snr(
            cbf_map=synthetic_data["cbf"],
            noise_mask=synthetic_data["noise_mask"],
            params=params_loose,
        )

        n_pass_strict = int(np.sum(result_strict.quality_mask))
        n_pass_loose = int(np.sum(result_loose.quality_mask))
        assert n_pass_strict <= n_pass_loose

    def test_shape_mismatch_raises_error(self) -> None:
        """Test that mismatched shapes raise DataValidationError."""
        cbf = np.ones((10, 10, 5))
        noise_mask = np.zeros((10, 10, 4), dtype=bool)  # wrong z-dimension
        noise_mask[0, :, :] = True

        with pytest.raises(DataValidationError, match="shape"):
            compute_snr(cbf, noise_mask)

    def test_too_few_noise_voxels_raises_error(self) -> None:
        """Test that too few noise voxels raise DataValidationError."""
        cbf = np.ones((10, 10, 5)) * 60.0
        noise_mask = np.zeros((10, 10, 5), dtype=bool)
        noise_mask[0, 0, 0] = True  # only 1 noise voxel

        params = ASLSNRParams(min_noise_voxels=10)
        with pytest.raises(DataValidationError, match="noise_mask contains only"):
            compute_snr(cbf, noise_mask, params=params)

    def test_with_explicit_brain_mask(self, synthetic_data: dict) -> None:
        """Test compute_snr with an explicit brain mask."""
        result = compute_snr(
            cbf_map=synthetic_data["cbf"],
            noise_mask=synthetic_data["noise_mask"],
            brain_mask=synthetic_data["brain_mask"],
        )
        # Quality mask should only be True inside brain_mask
        outside_brain = result.quality_mask & ~synthetic_data["brain_mask"]
        assert not np.any(outside_brain), "Quality mask must not exceed brain_mask"

    def test_without_brain_mask_uses_complement(self, synthetic_data: dict) -> None:
        """Test that omitting brain_mask uses complement of noise_mask."""
        result_no_mask = compute_snr(
            cbf_map=synthetic_data["cbf"],
            noise_mask=synthetic_data["noise_mask"],
            brain_mask=None,
        )
        result_with_mask = compute_snr(
            cbf_map=synthetic_data["cbf"],
            noise_mask=synthetic_data["noise_mask"],
            brain_mask=~synthetic_data["noise_mask"],
        )
        np.testing.assert_array_equal(
            result_no_mask.quality_mask, result_with_mask.quality_mask
        )
