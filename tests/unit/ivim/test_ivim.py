"""Unit tests for IVIM module."""

import numpy as np
import pytest

from osipy.ivim import (
    FittingMethod,
    IVIMBiexponentialModel,
    IVIMFitParams,
    IVIMFitResult,
    IVIMParams,
    IVIMSimplifiedModel,
    fit_ivim,
)


def generate_ivim_signal(
    b_values: np.ndarray,
    s0: float = 1000.0,
    d: float = 1.0e-3,
    d_star: float = 10.0e-3,
    f: float = 0.1,
    noise_std: float = 0.0,
) -> np.ndarray:
    """Generate synthetic IVIM signal."""
    signal = s0 * ((1 - f) * np.exp(-b_values * d) + f * np.exp(-b_values * d_star))
    if noise_std > 0:
        rng = np.random.default_rng(42)
        signal += rng.normal(0, noise_std, signal.shape)
    return signal


class TestIVIMBiexponentialModel:
    """Tests for IVIM bi-exponential model."""

    def test_model_properties(self) -> None:
        """Test model property accessors."""
        model = IVIMBiexponentialModel()
        assert model.name == "IVIM Bi-exponential"
        assert "D" in model.parameters
        assert "D*" in model.parameters
        assert "f" in model.parameters

    def test_predict_with_params_object(self) -> None:
        """Test prediction with IVIMParams object."""
        model = IVIMBiexponentialModel()
        b_values = np.array([0, 10, 20, 50, 100, 200, 400, 800])
        params = IVIMParams(s0=1000, d=1.0e-3, d_star=10.0e-3, f=0.1)

        signal = model.predict(b_values, params)

        assert signal.shape == b_values.shape
        assert signal[0] == pytest.approx(1000.0)  # At b=0, S = S0
        assert np.all(np.diff(signal) <= 0)  # Signal should decrease

    def test_predict_with_dict(self) -> None:
        """Test prediction with dictionary parameters."""
        model = IVIMBiexponentialModel()
        b_values = np.array([0, 50, 100, 200, 400, 800])
        params = {"S0": 800, "D": 1.2e-3, "D*": 15e-3, "f": 0.15}

        signal = model.predict(b_values, params)

        assert signal.shape == b_values.shape
        assert np.all(np.isfinite(signal))

    def test_get_bounds(self) -> None:
        """Test parameter bounds."""
        model = IVIMBiexponentialModel()
        bounds = model.get_bounds()

        assert "D" in bounds
        assert "D*" in bounds
        assert "f" in bounds
        assert bounds["D"][0] < bounds["D"][1]
        assert bounds["f"][0] >= 0
        assert bounds["f"][1] <= 1


class TestIVIMSimplifiedModel:
    """Tests for simplified IVIM model."""

    def test_model_properties(self) -> None:
        """Test model properties."""
        model = IVIMSimplifiedModel(b_threshold=200)
        assert model.name == "IVIM Simplified"
        assert "D" in model.parameters
        assert "f" in model.parameters
        assert "D*" not in model.parameters

    def test_predict(self) -> None:
        """Test simplified model prediction."""
        model = IVIMSimplifiedModel(b_threshold=200)
        b_values = np.array([0, 50, 100, 200, 400, 800])
        params = IVIMParams(s0=1000, d=1.0e-3, f=0.1)

        signal = model.predict(b_values, params)

        assert signal.shape == b_values.shape


class TestFitIVIM:
    """Tests for IVIM fitting function."""

    def test_fit_segmented_2d(self) -> None:
        """Test segmented fitting on 2D data."""
        b_values = np.array([0, 10, 20, 50, 100, 200, 400, 800])

        # Generate synthetic data
        signal_1d = generate_ivim_signal(
            b_values, s0=1000, d=1.0e-3, d_star=10e-3, f=0.1
        )
        signal = np.broadcast_to(signal_1d, (4, 4, len(b_values))).copy()
        signal += np.random.randn(*signal.shape) * 10

        params = IVIMFitParams(method=FittingMethod.SEGMENTED, b_threshold=200)
        result = fit_ivim(signal, b_values, params=params)

        assert isinstance(result, IVIMFitResult)
        # Shape now 3D due to our fix
        assert result.d_map.values.shape[:2] == (4, 4)
        assert result.d_star_map.values.shape[:2] == (4, 4)
        assert result.f_map.values.shape[:2] == (4, 4)

    def test_fit_full_model(self) -> None:
        """Test full bi-exponential fitting."""
        b_values = np.array([0, 10, 20, 50, 100, 200, 400, 800])

        signal_1d = generate_ivim_signal(
            b_values, s0=800, d=1.2e-3, d_star=12e-3, f=0.15, noise_std=5
        )
        signal = signal_1d.reshape(1, 1, -1)

        params = IVIMFitParams(method=FittingMethod.FULL)
        result = fit_ivim(signal, b_values, params=params)

        # Shape now 3D due to our fix
        assert result.d_map.values.shape[:2] == (1, 1)
        assert result.quality_mask[0, 0, 0]

    def test_fit_with_mask(self) -> None:
        """Test fitting with mask."""
        b_values = np.array([0, 50, 100, 200, 400, 800])
        shape = (6, 6, len(b_values))

        signal = generate_ivim_signal(b_values, s0=1000, d=1e-3, f=0.1)
        signal = np.broadcast_to(signal, shape).copy()

        mask = np.zeros((6, 6), dtype=bool)
        mask[2:4, 2:4] = True

        result = fit_ivim(signal, b_values, mask=mask)

        # Only masked voxels should be fitted
        assert np.sum(result.quality_mask) <= np.sum(mask)

    def test_r_squared(self) -> None:
        """Test R squared computation."""
        b_values = np.array([0, 10, 20, 50, 100, 200, 400, 800])

        # Low noise -> high R squared
        signal_1d = generate_ivim_signal(
            b_values, s0=1000, d=1e-3, d_star=10e-3, f=0.1, noise_std=1
        )
        signal = signal_1d.reshape(1, 1, -1)

        result = fit_ivim(signal, b_values)

        assert result.r_squared is not None
        # Access quality_mask with proper indexing (now 3D)
        if result.quality_mask[0, 0, 0]:
            assert result.r_squared[0, 0] > 0.9


class TestIVIMParams:
    """Tests for IVIMParams dataclass."""

    def test_default_values(self) -> None:
        """Test default parameter values."""
        params = IVIMParams()
        assert params.s0 == 1.0
        assert params.d == 1.0e-3
        assert params.d_star == 10.0e-3
        assert params.f == 0.1

    def test_custom_values(self) -> None:
        """Test custom parameter values."""
        params = IVIMParams(s0=500, d=0.8e-3, d_star=15e-3, f=0.2)
        assert params.s0 == 500
        assert params.d == 0.8e-3


class TestIVIMFitParams:
    """Tests for fitting parameters."""

    def test_default_values(self) -> None:
        """Test default fitting parameters."""
        params = IVIMFitParams()
        assert params.method == FittingMethod.SEGMENTED
        assert params.b_threshold == 200.0

    def test_custom_values(self) -> None:
        """Test custom fitting parameters."""
        params = IVIMFitParams(
            method=FittingMethod.FULL,
            b_threshold=150.0,
            max_iterations=1000,
        )
        assert params.method == FittingMethod.FULL
        assert params.b_threshold == 150.0
