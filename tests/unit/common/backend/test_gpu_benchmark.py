"""GPU benchmark tests for osipy.

Tests to verify GPU performance meets requirements. GPU-accelerated fitting
should be 10x faster than CPU for >10000 voxels, and all voxel-wise fitting
operations should be vectorized to leverage GPU.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from typing import TYPE_CHECKING

from osipy.common.backend.config import is_gpu_available

if TYPE_CHECKING:
    from collections.abc import Callable

if not is_gpu_available():
    pytest.skip("GPU not available for benchmark", allow_module_level=True)


def benchmark(
    func: Callable[[], None],
    n_iterations: int = 3,
    warmup: int = 1,
) -> float:
    """Benchmark a function and return average execution time.

    Parameters
    ----------
    func : Callable
        Function to benchmark.
    n_iterations : int
        Number of iterations to average.
    warmup : int
        Number of warmup iterations to discard.

    Returns
    -------
    float
        Average execution time in seconds.
    """
    # Warmup
    for _ in range(warmup):
        func()

    # Benchmark
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)

    return sum(times) / len(times)


class TestConvolutionBenchmark:
    """Benchmark GPU convolution performance."""

    def test_batch_convolution_speedup(self) -> None:
        """GPU batch convolution should be faster than CPU for large data."""
        from osipy.common.convolution import convolve_aif

        n_time = 100
        n_voxels = 10000  # Large dataset

        aif_np = np.exp(-0.1 * np.arange(n_time)).astype(np.float64)
        irfs_np = np.random.rand(n_time, n_voxels).astype(np.float64) * 0.1

        aif_gpu = cp.asarray(aif_np)
        irfs_gpu = cp.asarray(irfs_np)

        def cpu_compute() -> None:
            convolve_aif(aif_np, irfs_np)

        def gpu_compute() -> None:
            convolve_aif(aif_gpu, irfs_gpu)
            cp.cuda.Stream.null.synchronize()

        cpu_time = benchmark(cpu_compute, n_iterations=3)
        gpu_time = benchmark(gpu_compute, n_iterations=3)

        speedup = cpu_time / gpu_time

        # Log performance
        print(f"\nBatch convolution ({n_voxels} voxels):")
        print(f"  CPU time: {cpu_time * 1000:.2f} ms")
        print(f"  GPU time: {gpu_time * 1000:.2f} ms")
        print(f"  Speedup: {speedup:.1f}x")

        # GPU should be faster for large datasets
        assert gpu_time < cpu_time, (
            f"GPU ({gpu_time:.3f}s) not faster than CPU ({cpu_time:.3f}s)"
        )


class TestBatchProcessorBenchmark:
    """Benchmark GPU batch processing performance."""

    def test_batch_processor_speedup(self) -> None:
        """GPU batch processing should be faster than CPU."""
        from osipy.common.backend.batch import BatchProcessor

        n_samples = 100000
        data = np.random.randn(n_samples, 50).astype(np.float64)

        def heavy_compute(x: np.ndarray) -> np.ndarray:
            xp = (
                cp.get_array_module(x) if hasattr(x, "__cuda_array_interface__") else np
            )
            return xp.sin(x) * xp.exp(-(x**2) / 10) + xp.cos(x * 2)

        cpu_processor = BatchProcessor(batch_size=10000, use_gpu=False)
        gpu_processor = BatchProcessor(batch_size=10000, use_gpu=True)

        def cpu_compute() -> None:
            cpu_processor.map(data, heavy_compute)

        def gpu_compute() -> None:
            gpu_processor.map(data, heavy_compute)
            cp.cuda.Stream.null.synchronize()

        cpu_time = benchmark(cpu_compute, n_iterations=3)
        gpu_time = benchmark(gpu_compute, n_iterations=3)

        speedup = cpu_time / gpu_time

        print(f"\nBatch processing ({n_samples} samples):")
        print(f"  CPU time: {cpu_time * 1000:.2f} ms")
        print(f"  GPU time: {gpu_time * 1000:.2f} ms")
        print(f"  Speedup: {speedup:.1f}x")

        assert gpu_time < cpu_time


class TestFittingBenchmark:
    """Benchmark GPU fitting performance."""

    def test_fitting_speedup_requirement(self) -> None:
        """GPU fitting should be 10x faster than CPU for >10000 voxels."""
        from osipy.common.backend.config import (
            GPUConfig,
            get_backend,
            set_backend,
        )
        from osipy.common.fitting.least_squares import LevenbergMarquardtFitter
        from osipy.dce.models.binding import BoundDCEModel
        from osipy.dce.models.tofts import ToftsModel

        # Create test data with >10000 voxels for meaningful GPU speedup
        np.random.seed(42)
        n_voxels_side = 25  # 25^3 = 15625 voxels
        nx, ny, nz = n_voxels_side, n_voxels_side, n_voxels_side
        nt = 30

        t = np.linspace(0, 300, nt)
        aif = np.exp(-t / 60) * 2

        model = ToftsModel()
        ct_base = model.predict(t, aif, {"Ktrans": 0.1, "ve": 0.2})
        noise = np.random.randn(nx, ny, nz, nt) * 0.01
        ct_4d = ct_base[np.newaxis, np.newaxis, np.newaxis, :] + noise

        mask = np.ones((nx, ny, nz), dtype=bool)
        n_total_voxels = int(np.sum(mask))

        print(f"\nFitting benchmark ({n_total_voxels} voxels):")

        # CPU fitting
        original_config = get_backend()
        try:
            set_backend(GPUConfig(force_cpu=True))
            cpu_fitter = LevenbergMarquardtFitter(max_iterations=50)
            bound_model = BoundDCEModel(model, t, aif)

            start = time.perf_counter()
            cpu_fitter.fit_image(bound_model, ct_4d, mask=mask)
            cpu_time = time.perf_counter() - start

            print(f"  CPU time: {cpu_time:.2f} s")
        finally:
            set_backend(original_config)

        # GPU fitting
        gpu_fitter = LevenbergMarquardtFitter(max_iterations=50)
        bound_model_gpu = BoundDCEModel(model, t, aif)

        start = time.perf_counter()
        gpu_fitter.fit_image(bound_model_gpu, ct_4d, mask=mask)
        cp.cuda.Stream.null.synchronize()
        gpu_time = time.perf_counter() - start

        print(f"  GPU time: {gpu_time:.2f} s")

        speedup = cpu_time / gpu_time
        print(f"  Speedup: {speedup:.1f}x")

        assert gpu_time < cpu_time, (
            f"GPU ({gpu_time:.3f}s) not faster than CPU ({cpu_time:.3f}s)"
        )


class TestModelPredictionBenchmark:
    """Benchmark GPU model prediction performance."""

    def test_tofts_prediction_speedup(self) -> None:
        """GPU batch Tofts prediction should be faster than CPU."""
        from osipy.dce.models.tofts import ToftsModel

        model = ToftsModel()
        n_time = 100
        n_voxels = 50000

        t_np = np.linspace(0, 300, n_time).astype(np.float64)
        aif_np = (np.exp(-t_np / 60) * 2).astype(np.float64)

        # Batch params: (n_params, n_voxels) — Tofts has 2 params (Ktrans, ve)
        params_np = np.column_stack(
            [
                np.random.uniform(0.01, 0.5, n_voxels),  # Ktrans
                np.random.uniform(0.05, 0.5, n_voxels),  # ve
            ]
        ).T.astype(np.float64)

        t_gpu = cp.asarray(t_np)
        aif_gpu = cp.asarray(aif_np)
        params_gpu = cp.asarray(params_np)

        def cpu_predict() -> None:
            model.predict_batch(t_np, aif_np, params_np, np)

        def gpu_predict() -> None:
            model.predict_batch(t_gpu, aif_gpu, params_gpu, cp)
            cp.cuda.Stream.null.synchronize()

        cpu_time = benchmark(cpu_predict, n_iterations=5)
        gpu_time = benchmark(gpu_predict, n_iterations=5)

        speedup = cpu_time / gpu_time

        print(f"\nTofts batch prediction ({n_voxels} voxels):")
        print(f"  CPU time: {cpu_time * 1000:.2f} ms")
        print(f"  GPU time: {gpu_time * 1000:.2f} ms")
        print(f"  Speedup: {speedup:.1f}x")

        assert gpu_time < cpu_time, (
            f"GPU ({gpu_time:.3f}s) not faster than CPU ({cpu_time:.3f}s)"
        )


class TestASLQuantificationBenchmark:
    """Benchmark GPU ASL quantification performance."""

    def test_cbf_quantification_speedup(self) -> None:
        """GPU CBF quantification should be faster than CPU for large volumes."""
        from osipy.asl.quantification.cbf import (
            ASLQuantificationParams,
            _quantify_pcasl,
        )

        # Large brain volume for meaningful GPU speedup
        nx, ny, nz = 128, 128, 64
        n_voxels = nx * ny * nz

        delta_m_np = np.random.rand(nx, ny, nz).astype(np.float64) * 100
        m0_np = np.full((nx, ny, nz), 1000.0, dtype=np.float64)
        params = ASLQuantificationParams()

        delta_m_gpu = cp.asarray(delta_m_np)
        m0_gpu = cp.asarray(m0_np)

        def cpu_compute() -> None:
            _quantify_pcasl(delta_m_np, m0_np, params)

        def gpu_compute() -> None:
            _quantify_pcasl(delta_m_gpu, m0_gpu, params)
            cp.cuda.Stream.null.synchronize()

        cpu_time = benchmark(cpu_compute, n_iterations=5)
        gpu_time = benchmark(gpu_compute, n_iterations=5)

        speedup = cpu_time / gpu_time

        print(f"\nCBF quantification ({n_voxels} voxels):")
        print(f"  CPU time: {cpu_time * 1000:.2f} ms")
        print(f"  GPU time: {gpu_time * 1000:.2f} ms")
        print(f"  Speedup: {speedup:.1f}x")

        assert gpu_time < cpu_time, "GPU should be faster for large volumes"
