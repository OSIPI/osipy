"""Unit tests for GPU-optimized convolution operations.

Tests for osipy.common.convolution module.
"""

from __future__ import annotations

import numpy as np
import pytest

scipy_signal = pytest.importorskip(
    "scipy.signal", reason="scipy required for reference test"
)

from osipy.common.convolution import (
    convolve_aif,
    deconvolve_svd,
    deconvolve_svd_batch,
)
from osipy.common.exceptions import DataValidationError


class TestConvolveAif:
    """Tests for convolve_aif function."""

    def test_impulse_response(self) -> None:
        """Convolution with impulse should return scaled AIF."""
        n = 100
        aif = np.exp(-0.1 * np.arange(n))
        impulse = np.zeros(n)
        impulse[0] = 1.0
        dt = 1.0

        result = convolve_aif(aif, impulse, dt=dt)

        # Result should be close to AIF * dt
        np.testing.assert_array_almost_equal(result, aif * dt, decimal=5)

    def test_shape_preserved(self) -> None:
        """Output shape should match input shape."""
        n = 50
        aif = np.random.randn(n)
        irf = np.random.randn(n)

        result = convolve_aif(aif, irf)

        assert result.shape == aif.shape

    def test_shape_mismatch_raises(self) -> None:
        """Should raise error for shape mismatch."""
        aif = np.random.randn(50)
        irf = np.random.randn(60)

        with pytest.raises(DataValidationError, match="mismatch"):
            convolve_aif(aif, irf)

    def test_dt_scaling(self) -> None:
        """Result should scale with dt."""
        n = 50
        aif = np.random.randn(n)
        irf = np.random.randn(n)

        result_dt1 = convolve_aif(aif, irf, dt=1.0)
        result_dt2 = convolve_aif(aif, irf, dt=2.0)

        np.testing.assert_array_almost_equal(result_dt2, 2 * result_dt1)

    def test_matches_scipy_convolve(self) -> None:
        """Should produce similar results to scipy.signal.convolve."""
        n = 100
        aif = np.exp(-0.05 * np.arange(n))
        irf = np.exp(-0.1 * np.arange(n))
        dt = 0.5

        our_result = convolve_aif(aif, irf, dt=dt)
        scipy_result = scipy_signal.convolve(aif, irf, mode="full")[:n] * dt

        # Allow some numerical difference due to FFT vs direct
        np.testing.assert_array_almost_equal(our_result, scipy_result, decimal=5)

    def test_2d_input(self) -> None:
        """Should handle 2D input (multiple voxels)."""
        n_time = 50
        n_voxels = 10
        aif = np.exp(-0.1 * np.arange(n_time))[:, np.newaxis] * np.ones(n_voxels)
        irf = np.exp(-0.05 * np.arange(n_time))[:, np.newaxis] * np.random.rand(
            n_voxels
        )

        result = convolve_aif(aif, irf)

        assert result.shape == (n_time, n_voxels)


class TestConvolveAifBatch:
    """Tests for convolve_aif function."""

    def test_single_aif_multiple_irf(self) -> None:
        """Should convolve single AIF with multiple IRFs."""
        n_time = 50
        n_voxels = 100
        aif = np.exp(-0.1 * np.arange(n_time))
        irfs = np.random.rand(n_time, n_voxels) * np.exp(
            -0.05 * np.arange(n_time)[:, np.newaxis]
        )

        result = convolve_aif(aif, irfs)

        assert result.shape == (n_time, n_voxels)

    def test_matches_loop_version(self) -> None:
        """Batch version should match loop over individual convolutions."""
        n_time = 30
        n_voxels = 10
        aif = np.exp(-0.1 * np.arange(n_time))
        irfs = np.random.rand(n_time, n_voxels) * 0.1
        dt = 0.5

        batch_result = convolve_aif(aif, irfs, dt=dt)

        # Compare with loop version
        for v in range(n_voxels):
            expected = convolve_aif(
                aif[:, np.newaxis] * np.ones((1, 1)),
                irfs[:, v : v + 1],
                dt=dt,
            )
            np.testing.assert_array_almost_equal(
                batch_result[:, v], expected[:, 0], decimal=5
            )

    def test_time_mismatch_raises(self) -> None:
        """Should raise error for time dimension mismatch."""
        aif = np.random.randn(50)
        irfs = np.random.randn(60, 10)

        with pytest.raises(DataValidationError, match="Time dimension"):
            convolve_aif(aif, irfs)


class TestDeconvolveSvd:
    """Tests for deconvolve_svd function."""

    def test_recovers_impulse(self) -> None:
        """Should recover impulse from convolution."""
        n = 50
        aif = np.exp(-0.1 * np.arange(n)) + 0.1
        true_irf = np.zeros(n)
        true_irf[0] = 1.0
        dt = 1.0

        ct = convolve_aif(aif, true_irf, dt=dt)
        irf, _ = deconvolve_svd(ct, aif, dt=dt, threshold=0.01)

        # First element should be close to 1, others close to 0
        assert abs(irf[0] - 1.0) < 0.2
        assert np.mean(np.abs(irf[1:])) < 0.2

    def test_returns_tuple(self) -> None:
        """Should return (irf, residue) tuple."""
        n = 30
        ct = np.random.randn(n)
        aif = np.exp(-0.1 * np.arange(n)) + 0.1

        result = deconvolve_svd(ct, aif)

        assert isinstance(result, tuple)
        assert len(result) == 2
        irf, residue = result
        assert irf.shape == (n,)
        assert residue.shape == (n,)

    def test_residue_is_cumulative_irf(self) -> None:
        """Residue should be cumulative sum of IRF."""
        n = 30
        ct = np.random.randn(n)
        aif = np.exp(-0.1 * np.arange(n)) + 0.1
        dt = 0.5

        irf, residue = deconvolve_svd(ct, aif, dt=dt)

        expected_residue = np.cumsum(irf) * dt
        np.testing.assert_array_almost_equal(residue, expected_residue)


class TestDeconvolveSvdBatch:
    """Tests for deconvolve_svd_batch function."""

    def test_batch_shape(self) -> None:
        """Should handle batch input correctly."""
        n_time = 30
        n_voxels = 20
        ct = np.random.randn(n_time, n_voxels)
        aif = np.exp(-0.1 * np.arange(n_time)) + 0.1

        irf, residue = deconvolve_svd_batch(ct, aif)

        assert irf.shape == (n_time, n_voxels)
        assert residue.shape == (n_time, n_voxels)

    def test_matches_loop_version(self) -> None:
        """Batch version should match loop over individual deconvolutions."""
        n_time = 20
        n_voxels = 5
        ct = np.random.randn(n_time, n_voxels)
        aif = np.exp(-0.1 * np.arange(n_time)) + 0.1
        dt = 0.5
        threshold = 0.1

        batch_irf, _batch_residue = deconvolve_svd_batch(
            ct, aif, dt=dt, threshold=threshold
        )

        for v in range(n_voxels):
            single_irf, _single_residue = deconvolve_svd(
                ct[:, v], aif, dt=dt, threshold=threshold
            )
            np.testing.assert_array_almost_equal(batch_irf[:, v], single_irf, decimal=5)


class TestGpuConvolution:
    """GPU integration tests for convolution functions."""

    def test_gpu_convolve_matches_cpu(self) -> None:
        """GPU convolution should match CPU result."""
        cp = pytest.importorskip("cupy")

        n = 100
        aif = np.exp(-0.1 * np.arange(n))
        irf = np.exp(-0.05 * np.arange(n))

        cpu_result = convolve_aif(aif, irf)
        gpu_result = convolve_aif(cp.asarray(aif), cp.asarray(irf))

        np.testing.assert_array_almost_equal(
            cpu_result, cp.asnumpy(gpu_result), decimal=10
        )

    def test_gpu_deconvolve_matches_cpu(self) -> None:
        """GPU deconvolution should match CPU result."""
        cp = pytest.importorskip("cupy")

        n = 50
        ct = np.random.randn(n)
        aif = np.exp(-0.1 * np.arange(n)) + 0.1

        cpu_irf, cpu_residue = deconvolve_svd(ct, aif)
        gpu_irf, gpu_residue = deconvolve_svd(cp.asarray(ct), cp.asarray(aif))

        np.testing.assert_array_almost_equal(cpu_irf, cp.asnumpy(gpu_irf), decimal=8)
        np.testing.assert_array_almost_equal(
            cpu_residue, cp.asnumpy(gpu_residue), decimal=8
        )
