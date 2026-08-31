"""Unit tests for BaseFitter.fit_image() GPU/CPU dispatch.

Tests for osipy.common.fitting.base — specifically the GH-175 regression:
fit_image() must not keep treating a chunk as GPU-bound when the GPU
transfer actually failed. to_gpu() now raises GPUTransferError instead of
silently falling back to NumPy, so fit_image() should let that error
propagate rather than continue on a wrong assumption about the device.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np
import pytest

from osipy.common.backend.config import GPUConfig, get_backend, set_backend
from osipy.common.exceptions import GPUTransferError
from osipy.common.fitting import base as fitting_base
from osipy.common.fitting.base import BaseFitter


class _FakeModel:
    """Minimal FittableModel stand-in — just enough for fit_image/create_parameter_maps."""

    name = "fake_model"
    parameters: ClassVar[list[str]] = ["a"]
    parameter_units: ClassVar[dict[str, str]] = {"a": ""}
    reference = ""


class _OneParamFitter(BaseFitter):
    """Trivial fitter: returns zeros for every chunk, records call count."""

    fitting_method_name = "fake"
    chunk_size = 4

    def __init__(self) -> None:
        self.batch_calls = 0

    def fit_batch(
        self,
        model: Any,
        observed_batch: np.ndarray,
        bounds_override: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.batch_calls += 1
        n_voxels = observed_batch.shape[1]
        params = np.zeros((1, n_voxels))
        r2 = np.ones(n_voxels)
        converged = np.ones(n_voxels, dtype=bool)
        return params, r2, converged


@pytest.fixture
def restore_backend():
    """Restore the global backend config after each test."""
    original = get_backend()
    yield
    set_backend(original)


class TestFitImageGpuFallback:
    """GH-175: fit_image must not keep assuming GPU after a failed transfer."""

    def test_gpu_transfer_failure_propagates(
        self, monkeypatch: pytest.MonkeyPatch, restore_backend
    ) -> None:
        """If to_gpu() fails, fit_image must raise rather than silently
        continue on CPU while still believing it is on GPU.
        """
        monkeypatch.setattr(fitting_base, "is_gpu_available", lambda: True)

        def _raise(arr: np.ndarray) -> np.ndarray:
            raise GPUTransferError("GPU transfer failed (CUDA out of memory)")

        monkeypatch.setattr(fitting_base, "to_gpu", _raise)

        set_backend(GPUConfig(force_cpu=False, n_workers=2))

        fitter = _OneParamFitter()
        model = _FakeModel()
        data = np.ones((4, 4, 1, 5))  # 16 voxels, chunk_size=4 -> 4 chunks

        with pytest.raises(GPUTransferError, match="GPU transfer failed"):
            fitter.fit_image(model, data)

        assert fitter.batch_calls == 0

    def test_successful_gpu_transfer_keeps_gpu_path(
        self, monkeypatch: pytest.MonkeyPatch, restore_backend, caplog
    ) -> None:
        """Sanity check: when to_gpu() succeeds (returns an object that
        looks like a GPU array), the CPU threading path must NOT be
        used."""

        class _FakeGpuArray(np.ndarray):
            # Marks it as "on GPU" for the hasattr() check in fit_image().
            __cuda_array_interface__: ClassVar[dict[str, Any]] = {}

        def _fake_to_gpu(arr: np.ndarray) -> _FakeGpuArray:
            return arr.view(_FakeGpuArray)

        monkeypatch.setattr(fitting_base, "is_gpu_available", lambda: True)
        monkeypatch.setattr(fitting_base, "to_gpu", _fake_to_gpu)
        monkeypatch.setattr(fitting_base, "get_gpu_batch_size", lambda: 0)

        set_backend(GPUConfig(force_cpu=False, n_workers=2))

        fitter = _OneParamFitter()
        model = _FakeModel()
        data = np.ones((4, 4, 1, 5))

        with caplog.at_level(logging.INFO, logger=fitting_base.logger.name):
            fitter.fit_image(model, data)

        assert "Using" not in caplog.text or "threads" not in caplog.text
