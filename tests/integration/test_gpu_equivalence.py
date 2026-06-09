"""Integration tests for GPU/CPU numerical equivalence.

Tests that GPU-accelerated operations produce numerically equivalent
results to CPU implementations across all modalities.

This test suite verifies:
- GPU array handling (to_gpu, to_numpy)
- DCE T1 mapping GPU equivalence
- DSC deconvolution GPU array handling
- Convolution module GPU support
- Numerical tolerance across operations
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from osipy.common.backend import (
    GPUConfig,
    get_array_module,
    get_backend,
    is_gpu_available,
    set_backend,
    to_gpu,
    to_numpy,
)

# Skip all tests if GPU is not available
pytestmark = pytest.mark.skipif(
    not is_gpu_available(), reason="GPU (CuPy) not available"
)


class TestGPUArrayHandling:
    """Test basic GPU array handling utilities."""

    def test_to_gpu_creates_cupy_array(self):
        """Test that to_gpu creates a CuPy array."""
        import cupy as cp

        data = np.random.randn(100, 100).astype(np.float32)
        gpu_data = to_gpu(data)

        assert isinstance(gpu_data, cp.ndarray)
        assert gpu_data.shape == data.shape
        assert gpu_data.dtype == data.dtype

    def test_to_numpy_from_cupy(self):
        """Test that to_numpy converts CuPy array back to NumPy."""
        import cupy as cp

        gpu_data = cp.random.randn(100, 100).astype(cp.float32)
        cpu_data = to_numpy(gpu_data)

        assert isinstance(cpu_data, np.ndarray)
        assert cpu_data.shape == gpu_data.shape

    def test_to_numpy_passthrough_for_numpy(self):
        """Test that to_numpy returns NumPy arrays unchanged."""
        data = np.random.randn(50, 50)
        result = to_numpy(data)

        assert result is data  # Same object

    def test_get_array_module_numpy(self):
        """Test get_array_module returns numpy for NumPy arrays."""
        data = np.random.randn(10, 10)
        xp = get_array_module(data)

        assert xp is np

    def test_get_array_module_cupy(self):
        """Test get_array_module returns cupy for CuPy arrays."""
        import cupy as cp

        data = cp.random.randn(10, 10)
        xp = get_array_module(data)

        assert xp is cp


class TestDCET1MappingGPUEquivalence:
    """Test DCE T1 mapping GPU/CPU equivalence."""

    @pytest.fixture
    def vfa_data(self):
        """Create synthetic VFA data."""
        np.random.seed(42)
        nx, ny, nz = 16, 16, 4
        n_flip_angles = 5

        # Known T1 values
        t1_true = np.random.uniform(800, 1500, (nx, ny, nz))

        # Flip angles and TR
        flip_angles = np.array([2, 5, 10, 15, 20], dtype=np.float64)
        tr = 5.0  # ms

        # Generate SPGR signal
        signal = np.zeros((nx, ny, nz, n_flip_angles))
        m0 = 1000.0

        for i, fa in enumerate(flip_angles):
            fa_rad = np.radians(fa)
            e1 = np.exp(-tr / t1_true)
            sin_a = np.sin(fa_rad)
            cos_a = np.cos(fa_rad)
            signal[..., i] = m0 * sin_a * (1 - e1) / (1 - e1 * cos_a)

        # Add noise
        signal += np.random.randn(*signal.shape) * 5

        return {
            "signal": signal.astype(np.float64),
            "flip_angles": flip_angles,
            "tr": tr,
            "t1_true": t1_true,
        }

    def test_vfa_t1_gpu_cpu_equivalence(self, vfa_data):
        """Test VFA T1 mapping produces equivalent results on GPU and CPU."""
        from osipy.common.dataset import PerfusionDataset
        from osipy.common.types import DCEAcquisitionParams, Modality
        from osipy.dce.t1_mapping.vfa import compute_t1_vfa

        # Create dataset with CPU data
        # Note: VFA data is 4D but each volume is at different flip angle, not time
        # For T1 mapping, time_points correspond to flip angle acquisitions
        params = DCEAcquisitionParams(
            flip_angles=vfa_data["flip_angles"].tolist(),
            tr=vfa_data["tr"],
        )

        # Time points for VFA (one per flip angle)
        time_points = np.arange(len(vfa_data["flip_angles"]), dtype=np.float64)

        dataset_cpu = PerfusionDataset(
            data=vfa_data["signal"],
            affine=np.eye(4),
            modality=Modality.DCE,
            time_points=time_points,
            acquisition_params=params,
        )

        # Compute on CPU
        result_cpu = compute_t1_vfa(dataset_cpu, method="linear")
        t1_cpu = result_cpu.t1_map.values

        # Create dataset with GPU data
        gpu_signal = to_gpu(vfa_data["signal"])

        dataset_gpu = PerfusionDataset(
            data=to_numpy(gpu_signal),  # Dataset requires NumPy, GPU accel is internal
            affine=np.eye(4),
            modality=Modality.DCE,
            time_points=time_points,
            acquisition_params=params,
        )

        # Force GPU by converting data internally
        dataset_gpu._data = gpu_signal

        # Compute on GPU
        result_gpu = compute_t1_vfa(dataset_gpu, method="linear")
        t1_gpu = result_gpu.t1_map.values  # Should be NumPy from to_numpy in function

        # Compare results - should be very close
        valid_mask = np.isfinite(t1_cpu) & np.isfinite(t1_gpu)
        assert_allclose(
            t1_cpu[valid_mask],
            t1_gpu[valid_mask],
            rtol=1e-5,
            atol=1e-3,
            err_msg="VFA T1 GPU/CPU results differ",
        )


class TestDSCDeconvolutionGPUHandling:
    """Test DSC deconvolution GPU array handling."""

    @pytest.fixture
    def dsc_data(self):
        """Create synthetic DSC data."""
        np.random.seed(123)
        n_time = 60
        t = np.linspace(0, 60, n_time)  # seconds

        # Gamma variate AIF
        alpha, beta = 3.0, 0.5
        aif = (t**alpha) * np.exp(-t / beta)
        aif = aif / np.max(aif)

        # Exponential residue function
        residue = np.exp(-t / 5.0)

        # Create tissue concentration via convolution
        dt = t[1] - t[0]
        from osipy.common.convolution import convolve_aif

        tissue = convolve_aif(aif, residue, dt=dt)

        # Create 4D data (small volume)
        nx, ny, nz = 4, 4, 2
        concentration = np.zeros((nx, ny, nz, n_time))
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    scale = 0.5 + 0.5 * np.random.rand()
                    concentration[i, j, k, :] = tissue * scale

        return {
            "concentration": concentration.astype(np.float64),
            "aif": aif.astype(np.float64),
            "time": t.astype(np.float64),
        }

    def test_svd_deconvolution_accepts_gpu_arrays(self, dsc_data):
        """Test that SVD deconvolution accepts GPU arrays without error."""
        from osipy.dsc.deconvolution import get_deconvolver

        # Convert to GPU arrays
        gpu_conc = to_gpu(dsc_data["concentration"])
        gpu_aif = to_gpu(dsc_data["aif"])
        gpu_time = to_gpu(dsc_data["time"])

        # Should not raise - GPU arrays handled internally
        deconvolver = get_deconvolver("oSVD")
        result = deconvolver.deconvolve(gpu_conc, gpu_aif, gpu_time)

        # Results should stay on GPU (xp pattern), convert to numpy for validation
        cbf_np = to_numpy(result.cbf)
        mtt_np = to_numpy(result.mtt)
        assert isinstance(cbf_np, np.ndarray)
        assert isinstance(mtt_np, np.ndarray)
        assert cbf_np.shape == dsc_data["concentration"].shape[:-1]

    def test_svd_gpu_cpu_results_equivalent(self, dsc_data):
        """Test SVD deconvolution gives same results for GPU and CPU input."""
        from osipy.dsc.deconvolution import get_deconvolver

        deconvolver = get_deconvolver("cSVD")

        # CPU computation
        result_cpu = deconvolver.deconvolve(
            dsc_data["concentration"],
            dsc_data["aif"],
            dsc_data["time"],
        )

        # GPU input computation
        gpu_conc = to_gpu(dsc_data["concentration"])
        gpu_aif = to_gpu(dsc_data["aif"])
        gpu_time = to_gpu(dsc_data["time"])

        result_gpu = deconvolver.deconvolve(gpu_conc, gpu_aif, gpu_time)

        # GPU SVD (cuSOLVER) and CPU SVD (LAPACK) use different algorithms
        # and parallel reductions, so results differ at ~1e-12 level for
        # well-conditioned matrices. Use rtol=1e-6 for robust comparison.
        assert_allclose(
            result_cpu.cbf,
            to_numpy(result_gpu.cbf),
            rtol=1e-6,
            err_msg="SVD CBF GPU/CPU results differ",
        )
        assert_allclose(
            result_cpu.mtt,
            to_numpy(result_gpu.mtt),
            rtol=1e-6,
            err_msg="SVD MTT GPU/CPU results differ",
        )


class TestConvolutionGPUEquivalence:
    """Test convolution module GPU/CPU equivalence."""

    def test_expconv_gpu_cpu_equivalence(self):
        """Test exponential convolution GPU/CPU equivalence."""
        from osipy.common.convolution import expconv

        np.random.seed(42)
        n = 200
        t = np.linspace(0, 20, n)
        f = np.exp(-t / 3)
        T = 5.0

        # CPU computation
        result_cpu = expconv(f, T, t)

        # GPU computation
        f_gpu = to_gpu(f)
        t_gpu = to_gpu(t)
        result_gpu = expconv(f_gpu, T, t_gpu)

        # Convert result to CPU for comparison
        result_gpu_np = to_numpy(result_gpu)

        assert_allclose(
            result_cpu,
            result_gpu_np,
            rtol=1e-10,
            err_msg="expconv GPU/CPU results differ",
        )


class TestFitToftsGPUHandling:
    """Test DCE model prediction with GPU array handling."""

    @pytest.fixture
    def dce_fitting_data(self):
        """Create synthetic DCE data for fitting tests."""
        np.random.seed(42)
        n_time = 60
        t = np.linspace(0, 360, n_time)  # seconds

        # Parker AIF - get concentration array
        from osipy.common.aif.population import ParkerAIF

        parker = ParkerAIF()
        aif_obj = parker(t)
        aif = aif_obj.concentration  # Get the concentration array

        return {
            "aif": aif.astype(np.float64),
            "time": t.astype(np.float64),
        }

    def test_tofts_model_prediction_with_gpu_arrays(self, dce_fitting_data):
        """Test that ToftsModel.predict works with GPU arrays."""
        from osipy.dce.models import ToftsModel

        model = ToftsModel()
        params_arr = np.array([0.1, 0.2])  # [Ktrans, ve]

        # CPU prediction
        ct_cpu = model.predict(
            dce_fitting_data["time"],
            dce_fitting_data["aif"],
            params_arr,
            np,
        )

        # GPU prediction
        import cupy as cp

        gpu_time = to_gpu(dce_fitting_data["time"])
        gpu_aif = to_gpu(dce_fitting_data["aif"])

        ct_gpu = model.predict(gpu_time, gpu_aif, cp.asarray(params_arr), cp)
        ct_gpu_np = to_numpy(ct_gpu)

        # Results should be equivalent
        assert_allclose(
            ct_cpu,
            ct_gpu_np,
            rtol=1e-6,
            err_msg="Tofts model GPU/CPU predictions differ",
        )

    def test_extended_tofts_prediction_with_gpu_arrays(self, dce_fitting_data):
        """Test that ExtendedToftsModel.predict works with GPU arrays."""
        from osipy.dce.models import ExtendedToftsModel

        model = ExtendedToftsModel()
        params_arr = np.array([0.1, 0.2, 0.05])  # [Ktrans, ve, vp]

        # CPU prediction
        ct_cpu = model.predict(
            dce_fitting_data["time"],
            dce_fitting_data["aif"],
            params_arr,
            np,
        )

        # GPU prediction
        import cupy as cp

        gpu_time = to_gpu(dce_fitting_data["time"])
        gpu_aif = to_gpu(dce_fitting_data["aif"])

        ct_gpu = model.predict(gpu_time, gpu_aif, cp.asarray(params_arr), cp)
        ct_gpu_np = to_numpy(ct_gpu)

        # Results should be equivalent
        assert_allclose(
            ct_cpu,
            ct_gpu_np,
            rtol=1e-6,
            err_msg="Extended Tofts model GPU/CPU predictions differ",
        )


class TestBackendConfiguration:
    """Test backend configuration functionality."""

    def test_force_cpu_mode(self):
        """Test that force_cpu mode works correctly."""
        import cupy as cp

        # Create GPU data
        gpu_data = cp.random.randn(10, 10)

        # Save current config
        original_config = get_backend()

        try:
            # Force CPU mode
            set_backend(GPUConfig(force_cpu=True))

            # to_numpy should still work
            cpu_data = to_numpy(gpu_data)
            assert isinstance(cpu_data, np.ndarray)

            # In force_cpu mode, get_array_module returns numpy
            # This is intentional - it forces all operations to use CPU
            xp = get_array_module(gpu_data)
            assert xp is np  # Force CPU mode returns numpy

        finally:
            # Restore original config
            set_backend(original_config)

    def test_is_gpu_available(self):
        """Test GPU availability check."""
        # Since this test file is skipped if GPU not available,
        # this should always return True here
        assert is_gpu_available() is True
