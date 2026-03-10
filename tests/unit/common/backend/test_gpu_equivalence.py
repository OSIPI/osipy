"""GPU equivalence tests for osipy.

Tests to verify that GPU computations produce numerically equivalent
results to CPU implementations. Parameter values must be identical to
4 significant figures.
"""

from __future__ import annotations

import numpy as np
import pytest

from osipy.common.backend.config import GPUConfig, get_backend, set_backend


class TestConvolutionEquivalence:
    """Test GPU convolution matches CPU implementation."""

    def test_convolve_aif_equivalence(self) -> None:
        """GPU and CPU convolve_aif should produce identical results."""
        cp = pytest.importorskip("cupy")
        from osipy.common.convolution import convolve_aif

        n = 100
        aif = np.exp(-0.1 * np.arange(n))
        irf = np.exp(-0.05 * np.arange(n)) * 0.5
        dt = 1.0

        # CPU result
        cpu_result = convolve_aif(aif, irf, dt=dt)

        # GPU result
        gpu_result = convolve_aif(cp.asarray(aif), cp.asarray(irf), dt=dt)
        gpu_result_np = cp.asnumpy(gpu_result)

        # Check equivalence to 4 significant figures
        np.testing.assert_array_almost_equal(
            cpu_result,
            gpu_result_np,
            decimal=4,
            err_msg="GPU convolution does not match CPU to 4 significant figures",
        )

    def test_convolve_aif_batch_equivalence(self) -> None:
        """GPU batch convolution should match CPU."""
        cp = pytest.importorskip("cupy")
        from osipy.common.convolution import convolve_aif

        n_time = 50
        n_voxels = 100
        aif = np.exp(-0.1 * np.arange(n_time))
        irfs = np.random.rand(n_time, n_voxels) * np.exp(
            -0.05 * np.arange(n_time)[:, np.newaxis]
        )
        dt = 0.5

        cpu_result = convolve_aif(aif, irfs, dt=dt)
        gpu_result = convolve_aif(cp.asarray(aif), cp.asarray(irfs), dt=dt)
        gpu_result_np = cp.asnumpy(gpu_result)

        np.testing.assert_array_almost_equal(cpu_result, gpu_result_np, decimal=4)

    def test_deconvolve_svd_equivalence(self) -> None:
        """GPU SVD deconvolution should match CPU."""
        cp = pytest.importorskip("cupy")
        from osipy.common.convolution import deconvolve_svd

        n = 50
        ct = np.random.randn(n)
        aif = np.exp(-0.1 * np.arange(n)) + 0.1
        dt = 1.0
        threshold = 0.1

        cpu_irf, cpu_residue = deconvolve_svd(ct, aif, dt=dt, threshold=threshold)
        gpu_irf, gpu_residue = deconvolve_svd(
            cp.asarray(ct), cp.asarray(aif), dt=dt, threshold=threshold
        )

        np.testing.assert_array_almost_equal(cpu_irf, cp.asnumpy(gpu_irf), decimal=4)
        np.testing.assert_array_almost_equal(
            cpu_residue, cp.asnumpy(gpu_residue), decimal=4
        )


class TestModelPredictionEquivalence:
    """Test GPU model predictions match CPU."""

    def test_tofts_predict_equivalence(self) -> None:
        """GPU Tofts model prediction should match CPU."""
        cp = pytest.importorskip("cupy")
        from osipy.dce.models.tofts import ToftsModel

        model = ToftsModel()
        t = np.linspace(0, 300, 50)
        aif = np.exp(-t / 60) * 2
        params_arr = np.array([0.1, 0.2])  # [Ktrans, ve]

        cpu_result = model.predict(t, aif, params_arr, np)
        gpu_result = model.predict(
            cp.asarray(t), cp.asarray(aif), cp.asarray(params_arr), cp
        )

        np.testing.assert_array_almost_equal(
            cpu_result, cp.asnumpy(gpu_result), decimal=4
        )

    def test_extended_tofts_predict_equivalence(self) -> None:
        """GPU Extended Tofts prediction should match CPU."""
        cp = pytest.importorskip("cupy")
        from osipy.dce.models.extended_tofts import ExtendedToftsModel

        model = ExtendedToftsModel()
        t = np.linspace(0, 300, 50)
        aif = np.exp(-t / 60) * 2
        params_arr = np.array([0.1, 0.2, 0.02])  # [Ktrans, ve, vp]

        cpu_result = model.predict(t, aif, params_arr, np)
        gpu_result = model.predict(
            cp.asarray(t), cp.asarray(aif), cp.asarray(params_arr), cp
        )

        np.testing.assert_array_almost_equal(
            cpu_result, cp.asnumpy(gpu_result), decimal=4
        )

    def test_ivim_biexponential_equivalence(self) -> None:
        """GPU IVIM prediction should match CPU."""
        cp = pytest.importorskip("cupy")
        from osipy.ivim.models.biexponential import IVIMBiexponentialModel

        model = IVIMBiexponentialModel()
        b_values = np.array([0, 50, 100, 200, 400, 600, 800, 1000], dtype=np.float64)
        params_arr = np.array([1.0, 1e-3, 10e-3, 0.1])  # [S0, D, D*, f]

        cpu_result = model.predict(b_values, params_arr, np)
        gpu_result = model.predict(cp.asarray(b_values), cp.asarray(params_arr), cp)

        np.testing.assert_array_almost_equal(
            cpu_result, cp.asnumpy(gpu_result), decimal=4
        )


class TestASLQuantificationEquivalence:
    """Test GPU ASL quantification matches CPU."""

    def test_pcasl_quantification_equivalence(self) -> None:
        """GPU pCASL CBF quantification should match CPU."""
        cp = pytest.importorskip("cupy")
        from osipy.asl.quantification.cbf import (
            ASLQuantificationParams,
            _quantify_pcasl,
        )

        delta_m = np.random.rand(8, 8, 4) * 100
        m0 = np.full((8, 8, 4), 1000.0)
        params = ASLQuantificationParams()

        cpu_result = _quantify_pcasl(delta_m, m0, params)
        gpu_result = _quantify_pcasl(cp.asarray(delta_m), cp.asarray(m0), params)

        np.testing.assert_array_almost_equal(
            cpu_result, cp.asnumpy(gpu_result), decimal=4
        )

    def test_pasl_quantification_equivalence(self) -> None:
        """GPU PASL CBF quantification should match CPU."""
        cp = pytest.importorskip("cupy")
        from osipy.asl.labeling import LabelingScheme
        from osipy.asl.quantification.cbf import ASLQuantificationParams, _quantify_pasl

        delta_m = np.random.rand(8, 8, 4) * 100
        m0 = np.full((8, 8, 4), 1000.0)
        params = ASLQuantificationParams(labeling_scheme=LabelingScheme.PASL)

        cpu_result = _quantify_pasl(delta_m, m0, params)
        gpu_result = _quantify_pasl(cp.asarray(delta_m), cp.asarray(m0), params)

        np.testing.assert_array_almost_equal(
            cpu_result, cp.asnumpy(gpu_result), decimal=4
        )


class TestBatchProcessorEquivalence:
    """Test GPU batch processing matches CPU."""

    def test_batch_processor_equivalence(self) -> None:
        """GPU batch processing should match CPU results."""
        cp = pytest.importorskip("cupy")
        from osipy.common.backend.batch import BatchProcessor

        data = np.random.randn(1000, 10).astype(np.float64)

        def compute(x: np.ndarray) -> np.ndarray:
            xp = cp.get_array_module(x) if hasattr(cp, "get_array_module") else np
            return xp.sin(x) + xp.cos(x) * 2

        # CPU processing
        cpu_processor = BatchProcessor(batch_size=100, use_gpu=False)
        cpu_result = cpu_processor.map(data, compute)

        # GPU processing
        gpu_processor = BatchProcessor(batch_size=100, use_gpu=True)
        gpu_result = gpu_processor.map(data, compute)

        np.testing.assert_array_almost_equal(
            cpu_result.data, gpu_result.data, decimal=10
        )


class TestVectorizedFitterEquivalence:
    """Test GPU vectorized fitting matches CPU."""

    def test_fit_image_equivalence(self) -> None:
        """GPU image fitting should match CPU results."""
        cp = pytest.importorskip("cupy")
        from osipy.common.fitting.least_squares import LevenbergMarquardtFitter
        from osipy.dce.models.binding import BoundDCEModel
        from osipy.dce.models.tofts import ToftsModel

        # Create synthetic test data
        np.random.seed(42)
        nx, ny, nz, nt = 4, 4, 2, 30
        t = np.linspace(0, 300, nt)
        aif = np.exp(-t / 60) * 2

        # Generate synthetic concentration curves
        model = ToftsModel()
        ct_4d = np.zeros((nx, ny, nz, nt))
        true_params = {"Ktrans": 0.1, "ve": 0.2}
        ct_base = model.predict(t, aif, true_params)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    noise = np.random.randn(nt) * 0.01
                    ct_4d[i, j, k, :] = ct_base + noise

        mask = np.ones((nx, ny, nz), dtype=bool)

        # CPU fitting
        original_config = get_backend()
        try:
            set_backend(GPUConfig(force_cpu=True))
            cpu_bound = BoundDCEModel(model, t, aif)
            cpu_fitter = LevenbergMarquardtFitter(max_iterations=100)
            cpu_result = cpu_fitter.fit_image(cpu_bound, ct_4d, mask)
        finally:
            set_backend(original_config)

        # GPU fitting (if available)
        from osipy.common.backend.config import is_gpu_available

        if not is_gpu_available():
            pytest.skip("GPU not available for fitting equivalence test")

        gpu_bound = BoundDCEModel(model, cp.asarray(t), cp.asarray(aif))
        gpu_fitter = LevenbergMarquardtFitter(max_iterations=100)
        gpu_result = gpu_fitter.fit_image(gpu_bound, ct_4d, mask)

        # Compare parameter maps within tight tolerance
        for param in model.parameters:
            cpu_values = cpu_result[param].values
            gpu_values = gpu_result[param].values

            # Check valid voxels only
            valid_mask = np.isfinite(cpu_values) & np.isfinite(gpu_values)
            if np.any(valid_mask):
                np.testing.assert_allclose(
                    cpu_values[valid_mask],
                    gpu_values[valid_mask],
                    rtol=1e-4,
                    err_msg=f"Parameter {param} differs between CPU and GPU",
                )

    def test_batch_initial_guess_equivalence(self) -> None:
        """Batch initial guesses should match per-voxel initial guesses."""
        from osipy.dce.models.extended_tofts import ExtendedToftsModel
        from osipy.dce.models.tofts import ToftsModel

        np.random.seed(42)
        nt = 30
        n_voxels = 10
        t = np.linspace(0, 300, nt)
        aif = np.exp(-t / 60) * 2

        for ModelClass in [ToftsModel, ExtendedToftsModel]:
            model = ModelClass()
            ct_batch = np.random.rand(nt, n_voxels) * 0.5

            # Batch initial guess
            batch_guess = model.get_initial_guess_batch(ct_batch, aif, t, np)

            # Per-voxel initial guesses
            for v in range(n_voxels):
                voxel_guess = model.get_initial_guess(ct_batch[:, v], aif, t)
                voxel_array = model.params_to_array(voxel_guess)
                np.testing.assert_allclose(
                    batch_guess[:, v],
                    voxel_array,
                    rtol=1e-10,
                    err_msg=f"{model.name} batch guess differs at voxel {v}",
                )
