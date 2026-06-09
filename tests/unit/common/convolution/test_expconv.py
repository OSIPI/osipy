"""Unit tests for exponential convolution functions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from osipy.common.convolution import expconv
from osipy.common.exceptions import DataValidationError


class TestExpconv:
    """Tests for expconv() recursive exponential convolution."""

    def test_expconv_empty_input(self):
        """Test expconv with empty arrays."""
        result = expconv(np.array([]), 1.0, np.array([]))
        assert len(result) == 0

    def test_expconv_zero_time_constant(self):
        """Test expconv with zero time constant returns zeros."""
        t = np.linspace(0, 10, 101)
        f = np.ones_like(t)

        result = expconv(f, 0.0, t)

        assert len(result) == len(t)
        assert_allclose(result, np.zeros_like(t))

    def test_expconv_negative_time_constant(self):
        """Test expconv with negative time constant returns zeros."""
        t = np.linspace(0, 10, 101)
        f = np.ones_like(t)

        result = expconv(f, -1.0, t)

        assert_allclose(result, np.zeros_like(t))

    def test_expconv_constant_input_analytical(self):
        """Test expconv of constant input against analytical solution.

        For f(t) = C (constant), the analytical convolution with exp(-t/T) is:
            (f * h)(t) = C * T * (1 - exp(-t/T))
        """
        t = np.linspace(0, 20, 201)
        T = 5.0
        C = 2.0

        f = np.full_like(t, C)

        result = expconv(f, T, t)

        # Analytical solution
        analytical = C * T * (1 - np.exp(-t / T))

        # Compare (skip very early points)
        mask = t > 0.5
        assert_allclose(result[mask], analytical[mask], rtol=0.1)

    def test_expconv_exponential_input(self):
        """Test expconv with exponential input."""
        t = np.linspace(0, 20, 201)
        T1 = 2.0  # Input time constant
        T2 = 5.0  # Convolution time constant

        f = np.exp(-t / T1)

        result = expconv(f, T2, t)

        # Check basic properties
        assert len(result) == len(t)
        assert result[0] == 0.0  # Starts at zero
        assert np.all(result >= -1e-10)  # Non-negative

        # Should have a peak then decay
        peak_idx = np.argmax(result)
        assert peak_idx > 0
        assert peak_idx < len(t) - 1

    def test_expconv_mismatched_lengths(self):
        """Test that mismatched lengths raise error."""
        t = np.linspace(0, 10, 101)
        f = np.ones(100)  # Wrong length

        with pytest.raises(DataValidationError, match="same length"):
            expconv(f, 1.0, t)

    def test_expconv_large_time_constant(self):
        """Test expconv with large time constant approaches integral."""
        t = np.linspace(0, 10, 101)
        T = 1000.0  # Very large T

        f = np.ones_like(t)

        result = expconv(f, T, t)

        # For T >> t, convolution ≈ T * integral(f) ≈ T * t * mean(f)
        # Actually: integral_0^t f(u) du for constant f = t
        # So result ≈ t (approximately, for large T)
        # Check that result grows roughly linearly
        assert result[-1] > result[len(t) // 2]


class TestExpconvFlouri:
    """Tests verifying Flouri et al. (2016) formula implementation."""

    def test_expconv_linear_input(self):
        """Test expconv with linearly increasing input.

        For f(t) = a*t, the analytical convolution with exp(-t/T) is:
            (f * h)(t) = a * T * (t - T * (1 - exp(-t/T)))
        """
        t = np.linspace(0, 20, 201)
        T = 5.0
        a = 0.5  # Slope

        f = a * t

        result = expconv(f, T, t)

        # Analytical solution
        analytical = a * T * (t - T * (1 - np.exp(-t / T)))

        # Compare
        mask = t > 1.0
        assert_allclose(result[mask], analytical[mask], rtol=0.15)

    def test_expconv_preserves_integral(self):
        """Test that convolution integral is preserved.

        integral(f * h) = integral(f) * integral(h)
        For h = exp(-t/T), integral_0^inf = T
        """
        t = np.linspace(0, 50, 501)  # Long enough for decay
        T = 5.0

        f = np.exp(-t / 2)
        f_integral = 2.0  # integral_0^inf exp(-t/2) dt

        result = expconv(f, T, t)

        # Numerical integral of result
        dt = t[1] - t[0]
        result_integral = np.trapezoid(result, dx=dt)

        # Expected: f_integral * T
        expected_integral = f_integral * T

        assert abs(result_integral - expected_integral) / expected_integral < 0.2
