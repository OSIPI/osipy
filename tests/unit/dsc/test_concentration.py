"""Unit tests for DSC signal-to-concentration conversion."""

import numpy as np
import pytest

from osipy.common.exceptions import DataValidationError
from osipy.dsc.concentration import (
    DSCAcquisitionParams,
    delta_r2_to_concentration,
    signal_to_delta_r2,
)


class TestSignalToDeltaR2:
    """Tests for signal to ΔR2* conversion."""

    def test_basic_conversion(self) -> None:
        """Test basic signal to ΔR2* conversion."""
        te = 30.0  # ms
        n_timepoints = 60

        # Create synthetic signal with bolus passage
        baseline_signal = 1000.0
        signal = np.ones(n_timepoints) * baseline_signal

        # Simulate signal drop during bolus
        for i in range(15, 40):
            # Gaussian-like signal drop
            t_rel = (i - 25) / 5
            signal[i] = baseline_signal * np.exp(-np.exp(-(t_rel**2)) * 0.5)

        delta_r2 = signal_to_delta_r2(signal, te, baseline_frames=10)

        assert delta_r2.shape == signal.shape
        # Baseline should be ~0
        assert np.allclose(delta_r2[:10], 0, atol=0.1)
        # Peak should be positive (signal drop -> positive ΔR2*)
        assert np.max(delta_r2) > 0

    def test_4d_data(self) -> None:
        """Test conversion on 4D data."""
        te = 25.0
        shape = (4, 4, 2, 50)

        signal = np.random.rand(*shape) * 500 + 500
        # Add signal drop
        signal[..., 15:30] *= 0.7

        delta_r2 = signal_to_delta_r2(signal, te, baseline_frames=10)

        assert delta_r2.shape == shape
        assert np.all(np.isfinite(delta_r2))

    def test_custom_baseline_indices(self) -> None:
        """Test with custom baseline indices."""
        te = 30.0
        signal = np.random.rand(50) * 800 + 200

        baseline_idx = np.array([5, 6, 7, 8, 9])
        delta_r2 = signal_to_delta_r2(signal, te, baseline_indices=baseline_idx)

        assert delta_r2.shape == signal.shape

    def test_invalid_te_raises(self) -> None:
        """Test that invalid TE raises error."""
        signal = np.random.rand(30)

        with pytest.raises(DataValidationError):
            signal_to_delta_r2(signal, te=-5.0)


class TestDeltaR2ToConcentration:
    """Tests for ΔR2* to concentration conversion."""

    def test_basic_conversion(self) -> None:
        """Test basic ΔR2* to concentration."""
        delta_r2 = np.array([0, 5, 10, 8, 3, 1])  # s⁻¹
        r2_star = 32.0  # s⁻¹ mM⁻¹

        concentration = delta_r2_to_concentration(delta_r2, r2_star)

        expected = delta_r2 / r2_star
        assert np.allclose(concentration, expected)

    def test_field_strength_scaling(self) -> None:
        """Test relaxivity scaling with field strength."""
        delta_r2 = np.ones(10) * 10  # s⁻¹
        r2_star = 32.0

        conc_1_5T = delta_r2_to_concentration(delta_r2, r2_star, field_strength=1.5)
        conc_3T = delta_r2_to_concentration(delta_r2, r2_star, field_strength=3.0)

        # Higher field = higher relaxivity = lower apparent concentration
        assert np.all(conc_3T < conc_1_5T)

    def test_4d_data(self) -> None:
        """Test conversion on 4D data."""
        shape = (8, 8, 4, 40)
        delta_r2 = np.random.rand(*shape) * 15

        concentration = delta_r2_to_concentration(delta_r2)

        assert concentration.shape == shape
        assert np.all(concentration >= 0)


class TestDSCAcquisitionParams:
    """Tests for DSC acquisition parameters."""

    def test_default_values(self) -> None:
        """Test default parameter values."""
        params = DSCAcquisitionParams()
        assert params.te == 30.0
        assert params.tr == 1500.0
        assert params.field_strength == 1.5
        assert params.r2_star == 32.0

    def test_custom_values(self) -> None:
        """Test custom parameter values."""
        params = DSCAcquisitionParams(
            te=25.0,
            tr=1000.0,
            field_strength=3.0,
            r2_star=50.0,
        )
        assert params.te == 25.0
        assert params.field_strength == 3.0
