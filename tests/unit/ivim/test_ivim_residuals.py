"""Unit tests for IVIM signal reconstruction and RMSE residual maps."""

import numpy as np

from osipy.ivim import (
    compute_rmse_map,
    fit_ivim,
    reconstruct_ivim_signal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

B_VALUES = np.array([0, 10, 20, 50, 100, 200, 400, 800], dtype=float)
S0, D, DS, F = 1000.0, 1.0e-3, 10.0e-3, 0.1


def _clean_signal_1d(b: np.ndarray = B_VALUES) -> np.ndarray:
    """Noise-free IVIM signal for known parameters."""
    return S0 * ((1 - F) * np.exp(-b * D) + F * np.exp(-b * DS))


# ---------------------------------------------------------------------------
# reconstruct_ivim_signal
# ---------------------------------------------------------------------------


class TestReconstructIVIMSignal:
    """Tests for reconstruct_ivim_signal()."""

    def test_at_b_zero_equals_s0(self) -> None:
        """At b=0 the reconstructed signal must equal S0 exactly."""
        signal_1d = _clean_signal_1d()
        signal = np.broadcast_to(signal_1d, (3, 3, len(B_VALUES))).copy()

        result = fit_ivim(signal, B_VALUES)
        recon = reconstruct_ivim_signal(
            result.d_map, result.d_star_map,
            result.f_map, result.s0_map,
            B_VALUES,
        )

        # b=0 is the first b-value → first element along last axis
        b0_recon = recon[..., 0]
        s0_fitted = result.s0_map.values
        # Reconstructed S(b=0) must match the S0 parameter map
        np.testing.assert_allclose(b0_recon, s0_fitted, rtol=1e-5)

    def test_signal_shape(self) -> None:
        """Output shape must be (*spatial_shape, n_b)."""
        signal_1d = _clean_signal_1d()
        signal = np.broadcast_to(signal_1d, (4, 4, len(B_VALUES))).copy()

        result = fit_ivim(signal, B_VALUES)
        recon = reconstruct_ivim_signal(
            result.d_map, result.d_star_map,
            result.f_map, result.s0_map,
            B_VALUES,
        )

        spatial = result.d_map.values.shape
        assert recon.shape == spatial + (len(B_VALUES),)

    def test_signal_is_finite(self) -> None:
        """All reconstructed values must be finite."""
        signal_1d = _clean_signal_1d()
        signal = np.broadcast_to(signal_1d, (2, 2, len(B_VALUES))).copy()

        result = fit_ivim(signal, B_VALUES)
        recon = reconstruct_ivim_signal(
            result.d_map, result.d_star_map,
            result.f_map, result.s0_map,
            B_VALUES,
        )
        assert np.all(np.isfinite(recon))


# ---------------------------------------------------------------------------
# compute_rmse_map
# ---------------------------------------------------------------------------


class TestComputeRMSEMap:
    """Tests for compute_rmse_map()."""

    def test_near_zero_rmse_on_clean_data(self) -> None:
        """Noise-free signal should give very small RMSE at well-fitted voxels."""
        signal_1d = _clean_signal_1d()
        signal = np.broadcast_to(signal_1d, (3, 3, len(B_VALUES))).copy()

        result = fit_ivim(signal, B_VALUES)
        rmse = compute_rmse_map(
            signal,
            result.d_map, result.d_star_map,
            result.f_map, result.s0_map,
            B_VALUES,
        )

        # Only check well-fitted voxels
        good = result.quality_mask
        assert bool(good.any()), (
            "Expected at least one well-fitted voxel for noise-free synthetic data"
        )
        rmse_good = rmse.values[good]
        assert float(np.nanmax(rmse_good)) < 50.0  # < 5% of S0=1000

    def test_rmse_map_shape_matches_spatial(self) -> None:
        """RMSE map spatial shape must match parameter map shapes."""
        signal_1d = _clean_signal_1d()
        signal = np.broadcast_to(signal_1d, (4, 4, len(B_VALUES))).copy()

        result = fit_ivim(signal, B_VALUES)
        rmse = compute_rmse_map(
            signal,
            result.d_map, result.d_star_map,
            result.f_map, result.s0_map,
            B_VALUES,
        )

        spatial = result.d_map.values.shape
        assert rmse.values.shape == spatial

    def test_rmse_map_metadata(self) -> None:
        """RMSE ParameterMap must carry correct name, symbol, and units."""
        signal_1d = _clean_signal_1d()
        signal = signal_1d.reshape(1, 1, -1)

        result = fit_ivim(signal, B_VALUES)
        rmse = compute_rmse_map(
            signal,
            result.d_map, result.d_star_map,
            result.f_map, result.s0_map,
            B_VALUES,
        )

        assert rmse.name == "RMSE"
        assert rmse.symbol == "RMSE"
        assert rmse.units == "a.u."

    def test_unmasked_voxels_are_nan(self) -> None:
        """Voxels excluded by mask must be NaN in the RMSE map."""
        signal_1d = _clean_signal_1d()
        signal = np.broadcast_to(signal_1d, (4, 4, len(B_VALUES))).copy()

        # Mask only the top-left 2×2 block
        mask = np.zeros((4, 4), dtype=bool)
        mask[:2, :2] = True

        result = fit_ivim(signal, B_VALUES, mask=mask)
        rmse = compute_rmse_map(
            signal,
            result.d_map, result.d_star_map,
            result.f_map, result.s0_map,
            B_VALUES,
            mask=mask,
        )

        # Bottom-right region (outside mask) should be NaN
        outside_vals = rmse.values[2:, 2:].flatten()
        assert np.all(np.isnan(outside_vals))


# ---------------------------------------------------------------------------
# Integration: fit_ivim returns rmse_map
# ---------------------------------------------------------------------------


class TestFitIVIMIntegration:
    """Integration tests: rmse_map field in IVIMFitResult."""

    def test_rmse_map_not_none(self) -> None:
        """fit_ivim() must populate rmse_map on the result."""
        signal_1d = _clean_signal_1d()
        signal = np.broadcast_to(signal_1d, (3, 3, len(B_VALUES))).copy()

        result = fit_ivim(signal, B_VALUES)

        assert result.rmse_map is not None

    def test_rmse_map_shape_in_result(self) -> None:
        """rmse_map from fit_ivim() must have correct spatial shape."""
        signal_1d = _clean_signal_1d()
        signal = np.broadcast_to(signal_1d, (4, 4, len(B_VALUES))).copy()

        result = fit_ivim(signal, B_VALUES)
        spatial = result.d_map.values.shape

        assert result.rmse_map is not None
        assert result.rmse_map.values.shape == spatial
