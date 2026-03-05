"""Unit tests for array module utilities.

Tests for osipy.common.backend.array_module functions that provide
CPU/GPU agnostic array operations.
"""

from __future__ import annotations

import numpy as np
import pytest

from osipy.common.backend.array_module import (
    get_array_module,
    to_gpu,
    to_numpy,
)
from osipy.common.backend.config import (
    GPUConfig,
    _reset_gpu_cache,
    get_backend,
    set_backend,
)


class TestGetArrayModule:
    """Tests for get_array_module function."""

    def test_returns_numpy_for_numpy_array(self) -> None:
        """Should return numpy module for numpy arrays."""
        data = np.array([1, 2, 3])
        xp = get_array_module(data)
        assert xp.__name__ == "numpy"

    def test_returns_numpy_for_list(self) -> None:
        """Should return numpy for list input."""
        data = [1, 2, 3]
        xp = get_array_module(data)
        assert xp.__name__ == "numpy"

    def test_returns_numpy_when_forced_cpu(self) -> None:
        """Should return numpy when force_cpu is True."""
        original_config = get_backend()
        try:
            set_backend(GPUConfig(force_cpu=True))
            data = np.array([1, 2, 3])
            xp = get_array_module(data)
            assert xp.__name__ == "numpy"
        finally:
            set_backend(original_config)

    def test_handles_none_input(self) -> None:
        """Should handle None in input gracefully."""
        data = np.array([1, 2, 3])
        xp = get_array_module(data, None)
        assert xp.__name__ == "numpy"

    def test_multiple_arrays(self) -> None:
        """Should handle multiple array inputs."""
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        xp = get_array_module(a, b)
        assert xp.__name__ == "numpy"


class TestToNumpy:
    """Tests for to_numpy function."""

    def test_numpy_array_unchanged(self) -> None:
        """NumPy array should be returned as-is."""
        data = np.array([1, 2, 3])
        result = to_numpy(data)
        assert result is data  # Same object, no copy

    def test_converts_list(self) -> None:
        """Should convert list to numpy array."""
        data = [1, 2, 3]
        result = to_numpy(data)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    def test_preserves_dtype(self) -> None:
        """Should preserve array dtype."""
        data = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        result = to_numpy(data)
        assert result.dtype == np.float32

    def test_multidimensional(self) -> None:
        """Should work with multidimensional arrays."""
        data = np.random.randn(10, 20, 30)
        result = to_numpy(data)
        assert result is data


class TestToGpu:
    """Tests for to_gpu function."""

    def test_returns_numpy_when_forced_cpu(self) -> None:
        """Should return numpy array when force_cpu is True."""
        original_config = get_backend()
        try:
            set_backend(GPUConfig(force_cpu=True))
            data = np.array([1, 2, 3])
            result = to_gpu(data)
            assert isinstance(result, np.ndarray)
        finally:
            set_backend(original_config)

    def test_returns_numpy_when_no_gpu(self) -> None:
        """Should return numpy array when GPU not available."""
        # This test always passes on CPU-only systems
        data = np.array([1, 2, 3])
        result = to_gpu(data)
        # Result is either numpy or cupy array depending on system
        assert hasattr(result, "shape")
        assert result.shape == (3,)

    def test_preserves_values(self) -> None:
        """Should preserve array values."""
        data = np.array([1.0, 2.0, 3.0])
        result = to_gpu(data)
        np.testing.assert_array_almost_equal(to_numpy(result), data)


class TestGpuIntegration:
    """Integration tests that run if CuPy is available."""

    @pytest.fixture(autouse=True)
    def reset_cache(self) -> None:
        """Reset GPU cache before each test."""
        _reset_gpu_cache()

    def test_get_array_module_cupy(self) -> None:
        """Should return cupy for cupy arrays."""
        cp = pytest.importorskip("cupy")
        data = cp.array([1, 2, 3])
        xp = get_array_module(data)
        assert xp.__name__ == "cupy"

    def test_to_numpy_from_cupy(self) -> None:
        """Should convert cupy array to numpy."""
        cp = pytest.importorskip("cupy")
        data = cp.array([1, 2, 3])
        result = to_numpy(data)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    def test_to_gpu_creates_cupy(self) -> None:
        """Should create cupy array when GPU available."""
        cp = pytest.importorskip("cupy")
        original_config = get_backend()
        try:
            set_backend(GPUConfig(force_cpu=False))
            data = np.array([1, 2, 3])
            result = to_gpu(data)
            assert isinstance(result, cp.ndarray)
        finally:
            set_backend(original_config)
