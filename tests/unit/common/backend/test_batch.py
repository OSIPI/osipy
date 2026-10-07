"""Unit tests for batch processing utilities.

Tests for osipy.common.backend.batch module.
"""

from __future__ import annotations

import numpy as np
import pytest

from osipy.common.backend import batch as batch_module
from osipy.common.backend.batch import BatchProcessor, BatchResult, batch_apply
from osipy.common.backend.config import GPUConfig, set_backend
from osipy.common.exceptions import DataValidationError, GPUTransferError


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_default_values(self) -> None:
        """Should have sensible defaults."""
        data = np.array([1, 2, 3])
        result = BatchResult(data=data)
        assert result.used_gpu is False
        assert result.fallback_occurred is False
        assert result.batches_processed == 0

    def test_custom_values(self) -> None:
        """Should accept custom values."""
        data = np.array([1, 2, 3])
        result = BatchResult(
            data=data,
            used_gpu=True,
            fallback_occurred=True,
            batches_processed=5,
        )
        assert result.used_gpu is True
        assert result.fallback_occurred is True
        assert result.batches_processed == 5


class TestBatchProcessor:
    """Tests for BatchProcessor class."""

    @pytest.fixture
    def processor(self) -> BatchProcessor:
        """Create a batch processor for testing."""
        return BatchProcessor(batch_size=100, use_gpu=False)

    def test_default_batch_size(self) -> None:
        """Should use default batch size from config."""
        processor = BatchProcessor()
        assert processor.batch_size is not None
        assert processor.batch_size > 0

    def test_custom_batch_size(self) -> None:
        """Should accept custom batch size."""
        processor = BatchProcessor(batch_size=500)
        assert processor.batch_size == 500

    def test_invalid_memory_margin(self) -> None:
        """Should reject invalid memory safety margin."""
        with pytest.raises(DataValidationError, match="memory_safety_margin"):
            BatchProcessor(memory_safety_margin=0.6)

    def test_map_simple_function(self, processor: BatchProcessor) -> None:
        """Should apply function to all data."""
        data = np.arange(1000).astype(np.float64)

        def square(x: np.ndarray) -> np.ndarray:
            return x**2

        result = processor.map(data, square)

        np.testing.assert_array_almost_equal(result.data, data**2)
        assert result.batches_processed > 0

    def test_map_preserves_shape(self, processor: BatchProcessor) -> None:
        """Should preserve array shape."""
        data = np.random.randn(500, 10).astype(np.float64)

        def identity(x: np.ndarray) -> np.ndarray:
            return x

        result = processor.map(data, identity)

        assert result.data.shape == data.shape

    def test_map_handles_empty_array(self, processor: BatchProcessor) -> None:
        """Should handle empty arrays."""
        data = np.array([]).astype(np.float64)

        def identity(x: np.ndarray) -> np.ndarray:
            return x

        result = processor.map(data, identity)

        assert result.data.shape == (0,)
        assert result.batches_processed == 0

    def test_map_different_axis(self, processor: BatchProcessor) -> None:
        """Should batch along specified axis."""
        data = np.random.randn(10, 500).astype(np.float64)

        def square(x: np.ndarray) -> np.ndarray:
            return x**2

        result = processor.map(data, square, axis=1)

        np.testing.assert_array_almost_equal(result.data, data**2)

    def test_map_multiple_batches(self) -> None:
        """Should process multiple batches."""
        processor = BatchProcessor(batch_size=100, use_gpu=False)
        data = np.arange(550).astype(np.float64)

        def square(x: np.ndarray) -> np.ndarray:
            return x**2

        result = processor.map(data, square)

        np.testing.assert_array_almost_equal(result.data, data**2)
        assert result.batches_processed == 6  # ceil(550/100)

    def test_uses_cpu_when_forced(self) -> None:
        """Should use CPU when force_cpu is True."""
        original_config = GPUConfig()
        try:
            set_backend(GPUConfig(force_cpu=True))
            processor = BatchProcessor(batch_size=100, use_gpu=True)
            data = np.arange(200).astype(np.float64)

            result = processor.map(data, lambda x: x**2)

            assert result.used_gpu is False
        finally:
            set_backend(original_config)


class TestGpuTransferFallback:
    """GH-175/GH-176: BatchProcessor must keep its deliberate CPU fallback
    when a GPU transfer fails, even though to_gpu() itself now raises
    instead of silently degrading."""

    def test_falls_back_to_cpu_on_gpu_transfer_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A GPUTransferError from to_gpu() should be caught and the
        batch retried on CPU when auto_fallback is True (the default)."""
        monkeypatch.setattr(batch_module, "is_gpu_available", lambda: True)

        def _raise(arr: np.ndarray) -> np.ndarray:
            raise GPUTransferError("GPU transfer failed (CUDA out of memory)")

        monkeypatch.setattr(batch_module, "to_gpu", _raise)

        processor = BatchProcessor(batch_size=50, use_gpu=True)
        data = np.arange(100).astype(np.float64)

        with pytest.warns(UserWarning, match="GPU transfer failed"):
            result = processor.map(data, lambda x: x**2)

        np.testing.assert_array_almost_equal(result.data, data**2)
        assert result.fallback_occurred is True
        assert result.used_gpu is False

    def test_reraises_gpu_transfer_error_when_auto_fallback_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With auto_fallback disabled, the GPUTransferError must propagate
        instead of being swallowed."""
        monkeypatch.setattr(batch_module, "is_gpu_available", lambda: True)

        def _raise(arr: np.ndarray) -> np.ndarray:
            raise GPUTransferError("GPU transfer failed (CUDA out of memory)")

        monkeypatch.setattr(batch_module, "to_gpu", _raise)

        processor = BatchProcessor(batch_size=50, use_gpu=True, auto_fallback=False)
        data = np.arange(100).astype(np.float64)

        with pytest.raises(GPUTransferError, match="GPU transfer failed"):
            processor.map(data, lambda x: x**2)


class TestBatchApply:
    """Tests for batch_apply convenience function."""

    def test_applies_function(self) -> None:
        """Should apply function to data."""
        data = np.arange(100).astype(np.float64)
        result = batch_apply(data, lambda x: x * 2, batch_size=50)
        np.testing.assert_array_almost_equal(result, data * 2)

    def test_custom_batch_size(self) -> None:
        """Should use custom batch size."""
        data = np.arange(100).astype(np.float64)
        result = batch_apply(data, lambda x: x, batch_size=25)
        np.testing.assert_array_equal(result, data)

    def test_custom_axis(self) -> None:
        """Should batch along custom axis."""
        data = np.random.randn(10, 100).astype(np.float64)
        result = batch_apply(data, lambda x: x**2, batch_size=25, axis=1)
        np.testing.assert_array_almost_equal(result, data**2)


class TestGpuBatchProcessing:
    """Integration tests for GPU batch processing."""

    def test_gpu_batch_processing(self) -> None:
        """Should process batches on GPU when available."""
        pytest.importorskip("cupy")
        from osipy.common.backend.config import is_gpu_available

        if not is_gpu_available():
            pytest.skip("GPU not available")

        processor = BatchProcessor(batch_size=1000, use_gpu=True)
        data = np.random.randn(5000, 10).astype(np.float64)

        def square(x: np.ndarray) -> np.ndarray:
            return x**2

        result = processor.map(data, square)

        np.testing.assert_array_almost_equal(result.data, data**2)
        assert result.used_gpu is True

    def test_gpu_to_cpu_equivalence(self) -> None:
        """GPU and CPU should produce same results."""
        pytest.importorskip("cupy")
        from osipy.common.backend.config import is_gpu_available

        if not is_gpu_available():
            pytest.skip("GPU not available")

        data = np.random.randn(1000, 10).astype(np.float64)

        def compute(x: np.ndarray) -> np.ndarray:
            return np.sin(x) + np.cos(x) * 2

        # CPU result
        cpu_processor = BatchProcessor(batch_size=100, use_gpu=False)
        cpu_result = cpu_processor.map(data, compute)

        # GPU result
        gpu_processor = BatchProcessor(batch_size=100, use_gpu=True)
        gpu_result = gpu_processor.map(data, compute)

        np.testing.assert_array_almost_equal(
            cpu_result.data, gpu_result.data, decimal=10
        )
