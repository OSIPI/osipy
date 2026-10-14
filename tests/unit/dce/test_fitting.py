"""Unit tests for DCE high-level fitting functions.

Tests for fit_model and DCEFitResult.
"""

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from osipy.common.aif import ParkerAIF
from osipy.common.exceptions import FittingError
from osipy.common.parameter_map import ParameterMap
from osipy.dce.fitting import (
    DCEFitResult,
    _compute_r_squared_vectorized,
    fit_model,
)
from osipy.dce.models import ToftsModel, ToftsParams


def generate_synthetic_dce_data(
    shape: tuple[int, ...],
    time: np.ndarray,
    ktrans: float = 0.2,
    ve: float = 0.3,
    vp: float = 0.0,
    noise_std: float = 0.01,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic DCE concentration data.

    Parameters
    ----------
    shape : tuple
        Spatial shape (x, y) or (x, y, z).
    time : ndarray
        Time points in seconds.
    ktrans, ve, vp : float
        Model parameters.
    noise_std : float
        Standard deviation of Gaussian noise.
    rng : Generator, optional
        Random number generator.

    Returns
    -------
    concentration : ndarray
        Concentration data with shape (*shape, n_timepoints).
    aif : ndarray
        Arterial input function.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Generate AIF
    parker = ParkerAIF()
    aif_result = parker(time)
    aif = aif_result.concentration

    # Generate tissue concentration using Extended Tofts
    t_min = time / 60.0
    kep = ktrans / ve if ve > 0 else 0

    ct = np.zeros(len(time))
    for i in range(len(time)):
        tau = t_min[: i + 1]
        integrand = aif[: i + 1] * np.exp(-kep * (t_min[i] - tau))
        ct[i] = ktrans * np.trapezoid(integrand, tau) + vp * aif[i]

    # Broadcast to spatial shape
    full_shape = (*shape, len(time))
    concentration = np.broadcast_to(ct, full_shape).copy()

    # Add noise
    noise = rng.normal(0, noise_std, full_shape)
    concentration += noise

    return concentration, aif


class TestFitModelBasic:
    """Basic tests for fit_model function."""

    def test_fit_shape_mismatch_raises(self) -> None:
        """Test that shape mismatch raises error."""
        time = np.linspace(0, 300, 60)
        aif = np.random.rand(50)  # Wrong length
        concentration = np.random.rand(4, 4, 60)

        with pytest.raises(FittingError, match="does not match"):
            fit_model("tofts", concentration, aif, time)


class TestDCEFitResult:
    """Tests for DCEFitResult dataclass."""

    def test_result_structure(self) -> None:
        """Test result structure and defaults."""
        values = np.ones((2, 2, 1))
        affine = np.eye(4)

        param_maps = {
            "Ktrans": ParameterMap(
                name="Ktrans",
                symbol="Ktrans",
                units="1/min",
                values=values,
                affine=affine,
            ),
        }
        quality_mask = np.ones((2, 2, 1), dtype=bool)

        result = DCEFitResult(
            parameter_maps=param_maps,
            quality_mask=quality_mask,
            model_name="Test Model",
        )

        assert result.parameter_maps == param_maps
        assert np.array_equal(result.quality_mask, quality_mask)
        assert result.model_name == "Test Model"
        assert result.r_squared_map is None
        assert result.residual_map is None
        assert result.fitting_stats == {}


class TestToftsIntegration:
    """Integration tests for Tofts model fitting."""

    def test_tofts_model_direct_use(self) -> None:
        """Test using ToftsModel directly."""
        model = ToftsModel()
        t = np.linspace(0, 300, 60)

        # Generate Parker AIF
        parker = ParkerAIF()
        aif_result = parker(t)
        aif = aif_result.concentration

        # Predict with known parameters
        params = ToftsParams(ktrans=0.2, ve=0.3)
        ct = model.predict(t, aif, params)

        assert ct.shape == t.shape
        assert np.all(np.isfinite(ct))
        assert ct[0] < ct[30]  # Should increase over time


class TestPatlakIntegration:
    """Integration tests for Patlak model."""

    def test_patlak_model_basic(self) -> None:
        """Test Patlak model characteristics."""
        from osipy.dce.models import PatlakModel, PatlakParams

        model = PatlakModel()
        t = np.linspace(0, 300, 60)

        parker = ParkerAIF()
        aif = parker(t).concentration

        params = PatlakParams(ktrans=0.1, vp=0.02)
        ct = model.predict(t, aif, params)

        assert ct.shape == t.shape
        assert np.all(np.isfinite(ct))


class TestTwoCompartmentIntegration:
    """Integration tests for 2CXM."""

    def test_2cxm_model_basic(self) -> None:
        """Test 2CXM model characteristics."""
        from osipy.dce.models import TwoCompartmentModel, TwoCompartmentParams

        model = TwoCompartmentModel()
        t = np.linspace(0, 300, 60)

        parker = ParkerAIF()
        aif = parker(t).concentration

        params = TwoCompartmentParams(fp=40.0, ps=4.0, ve=0.2, vp=0.02)
        ct = model.predict(t, aif, params)

        assert ct.shape == t.shape
        assert np.all(np.isfinite(ct))


class TestDelayFitting:
    """Tests for arterial delay fitting via fit_delay parameter."""

    def test_fit_delay_false_no_delay_param(self) -> None:
        """When fit_delay=False, no delay parameter in result."""
        time = np.linspace(0, 300, 60)
        concentration, aif = generate_synthetic_dce_data(
            shape=(4, 4, 1),
            time=time,
            ktrans=0.2,
            ve=0.3,
            noise_std=0.001,
        )

        result = fit_model(
            "tofts",
            concentration,
            aif,
            time,
            fit_delay=False,
        )

        assert "delay" not in result.parameter_maps

    def test_fit_delay_true_recovers_delay(self) -> None:
        """When fit_delay=True, delay parameter is recovered."""
        from osipy.common.aif.delay import shift_aif

        time = np.linspace(0, 300, 60)
        true_delay = 5.0  # seconds

        # Generate AIF and shift it
        parker = ParkerAIF()
        aif_result = parker(time)
        aif = aif_result.concentration
        shifted_aif = shift_aif(aif, time, true_delay)

        # Generate concentration with the shifted AIF
        model = ToftsModel()
        true_params = ToftsParams(ktrans=0.2, ve=0.3)
        ct_single = model.predict(time, shifted_aif, true_params)

        # Broadcast to 4x4x1 spatial volume
        shape = (4, 4, 1)
        concentration = np.broadcast_to(
            ct_single,
            (*shape, len(time)),
        ).copy()
        rng = np.random.default_rng(42)
        concentration += rng.normal(0, 0.001, concentration.shape)

        result = fit_model(
            "tofts",
            concentration,
            aif,
            time,
            fit_delay=True,
        )

        assert "delay" in result.parameter_maps
        delay_map = result.parameter_maps["delay"]

        quality = result.quality_mask
        if np.any(quality):
            recovered_delay = np.median(delay_map.values[quality])
            np.testing.assert_allclose(
                recovered_delay,
                true_delay,
                atol=2.0,
            )

    def test_fit_delay_zero_matches_no_delay(self) -> None:
        """Fitting non-delayed data with fit_delay=True gives delay~0."""
        time = np.linspace(0, 300, 60)
        concentration, aif = generate_synthetic_dce_data(
            shape=(4, 4, 1),
            time=time,
            ktrans=0.2,
            ve=0.3,
            noise_std=0.001,
        )

        result = fit_model(
            "tofts",
            concentration,
            aif,
            time,
            fit_delay=True,
        )

        assert "delay" in result.parameter_maps
        delay_map = result.parameter_maps["delay"]

        quality = result.quality_mask
        if np.any(quality):
            recovered_delay = np.median(delay_map.values[quality])
            # Extra free parameter may absorb small residual;
            # verify delay stays near zero (within one time step)
            assert abs(recovered_delay) < 6.0, (
                f"Delay should be near 0 for non-delayed data, "
                f"got {recovered_delay:.2f}"
            )


class TestComputeRSquaredVectorized:
    """Tests for _compute_r_squared_vectorized."""

    def test_logs_warning_on_predict_failure(self) -> None:
        """A failing predict_array_batch logs a warning instead of silently returning zeros."""
        ct_4d = np.ones((2, 2, 1, 10), dtype=np.float64)
        quality_mask = np.ones((2, 2, 1), dtype=bool)

        bound_model = MagicMock()
        bound_model.parameters = ["ktrans", "ve"]
        bound_model.predict_array_batch.side_effect = ValueError("shape mismatch")

        param_maps = {
            "ktrans": ParameterMap(
                name="ktrans",
                symbol="Ktrans",
                units="1/min",
                values=np.full((2, 2, 1), 0.2),
                affine=np.eye(4),
            ),
            "ve": ParameterMap(
                name="ve",
                symbol="ve",
                units="mL/100mL",
                values=np.full((2, 2, 1), 0.3),
                affine=np.eye(4),
            ),
        }

        with patch("osipy.dce.fitting.logger") as mock_logger:
            result = _compute_r_squared_vectorized(
                ct_4d, bound_model, param_maps, quality_mask, np
            )

        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0]
        assert "R²" in call_args[0]
        assert isinstance(call_args[-1], ValueError)
        assert np.all(result == 0.0)
