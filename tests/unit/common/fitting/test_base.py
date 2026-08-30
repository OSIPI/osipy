"""Unit tests for BaseFitter.fit_image() GPU/CPU dispatch.

Tests for osipy.common.fitting.base — specifically the GH-175 regression:
fit_image() must not keep treating a chunk as GPU-bound after to_gpu()
has silently fallen back to CPU.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np
import pytest

from osipy.common.backend.config import GPUConfig, get_backend, set_backend
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
    """GH-175: use_gpu must be re-derived after to_gpu() may have fallen back."""

    def test_gpu_transfer_failure_reenables_cpu_threading(
        self, monkeypatch: pytest.MonkeyPatch, restore_backend, caplog
    ) -> None:
        """If to_gpu() falls back to NumPy, fit_image must use the CPU
        multi-threaded path instead of silently running single-threaded.
        """
        # Pretend GPU is available and requested...
        monkeypatch.setattr(fitting_base, "is_gpu_available", lambda: True)
        # ...but to_gpu() falls back to plain NumPy (as it does on a real
        # transfer failure, after emitting its own warning).
        monkeypatch.setattr(fitting_base, "to_gpu", lambda arr: np.asarray(arr))

        set_backend(GPUConfig(force_cpu=False, n_workers=2))

        fitter = _OneParamFitter()
        model = _FakeModel()
        data = np.ones((4, 4, 1, 5))  # 16 voxels, chunk_size=4 -> 4 chunks

        with caplog.at_level(logging.INFO, logger=fitting_base.logger.name):
            result = fitter.fit_image(model, data)

        assert "Using 2 threads for 4 chunks" in caplog.text
        assert "a" in result
        assert fitter.batch_calls == 4

    def test_successful_gpu_transfer_keeps_gpu_path(
        self, monkeypatch: pytest.MonkeyPatch, restore_backend, caplog
    ) -> None:
        """Sanity check: when to_gpu() *doesn't* fall back (returns an
        object that looks like a GPU array), the CPU threading path must
        NOT be used — this guards against an overly broad fix."""

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
