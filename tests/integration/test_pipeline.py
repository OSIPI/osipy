"""Integration tests for automated perfusion analysis pipeline.

Tests for osipy/pipeline/ module.
"""

from __future__ import annotations

import numpy as np
import pytest

from osipy.common.types import AnalysisResult, Modality
from osipy.pipeline import (
    AnalysisResult,
    IVIMPipeline,
    IVIMPipelineConfig,
    PipelineResult,
    run_analysis,
)


class TestRunAnalysis:
    """Tests for unified run_analysis function."""

    def test_run_analysis_returns_analysis_result(self) -> None:
        """Test that run_analysis returns AnalysisResult (the contract type)."""
        data = np.random.rand(8, 8, 4, 6)
        b_values = np.array([0, 50, 100, 200, 400, 800])

        result = run_analysis(
            data,
            modality=Modality.IVIM,
            b_values=b_values,
        )

        assert isinstance(result, AnalysisResult)
        assert result.modality == Modality.IVIM

    def test_run_analysis_string_modality(self) -> None:
        """Test run_analysis accepts string modality."""
        data = np.random.rand(8, 8, 4, 6)
        b_values = np.array([0, 50, 100, 200, 400, 800])

        result = run_analysis(
            data,
            modality="ivim",
            b_values=b_values,
        )

        assert result.modality == Modality.IVIM

    def test_analysis_result_parameter_maps_is_dict(self) -> None:
        """Test that parameter_maps is always a non-empty dict."""
        data = np.random.rand(4, 4, 2, 6)
        b_values = np.array([0, 50, 100, 200, 400, 800])

        result = run_analysis(data, modality="ivim", b_values=b_values)

        assert isinstance(result.parameter_maps, dict)
        assert len(result.parameter_maps) > 0

    def test_analysis_result_quality_mask_never_none(self) -> None:
        """Test that quality_mask is always present and is a bool array."""
        data = np.random.rand(4, 4, 2, 6)
        b_values = np.array([0, 50, 100, 200, 400, 800])

        result = run_analysis(data, modality="ivim", b_values=b_values)

        assert result.quality_mask is not None
        assert result.quality_mask.dtype == np.bool_

    def test_analysis_result_provenance_fields(self) -> None:
        """Test that provenance always contains the required audit fields."""
        data = np.random.rand(4, 4, 2, 6)
        b_values = np.array([0, 50, 100, 200, 400, 800])

        result = run_analysis(data, modality="ivim", b_values=b_values)

        assert "osipy_version" in result.provenance
        assert "captured_at" in result.provenance
        assert "modality" in result.provenance
        assert "config" in result.provenance
        assert result.provenance["modality"] == "ivim"

    def test_uniform_save_interface_all_modalities(self) -> None:
        """AnalysisResult enables the same save code for every modality.

        This is the canonical test for the contract: downstream code that
        only knows AnalysisResult should work identically for IVIM.
        """
        data = np.random.rand(4, 4, 2, 6)
        b_values = np.array([0, 50, 100, 200, 400, 800])
        result = run_analysis(data, modality="ivim", b_values=b_values)

        # This code works WITHOUT knowing the modality ─ that's the whole point
        saved = {}
        for name, pmap in result.parameter_maps.items():
            arr = pmap.values if hasattr(pmap, "values") else pmap
            saved[name] = arr
        assert result.quality_mask is not None
        assert len(saved) > 0


class TestIVIMPipelineIntegration:
    """Integration tests for IVIM pipeline."""

    def test_ivim_pipeline_full_run(self) -> None:
        """Test full IVIM pipeline execution."""
        shape = (8, 8, 4)
        b_values = np.array([0, 50, 100, 200, 400, 800])
        n_bvalues = len(b_values)

        d = 1.0e-3
        d_star = 10e-3
        f = 0.1

        signal = np.zeros((*shape, n_bvalues))
        for i, b in enumerate(b_values):
            signal[..., i] = f * np.exp(-b * d_star) + (1 - f) * np.exp(-b * d)

        signal += np.random.randn(*signal.shape) * 0.01
        signal = np.maximum(signal, 0.01)

        config = IVIMPipelineConfig()
        pipeline = IVIMPipeline(config)
        result = pipeline.run(signal, b_values=b_values)

        # pipeline.run() still returns IVIMPipelineResult (unchanged)
        assert result is not None
        assert hasattr(result, "fit_result")


class TestPipelineMemoryEfficiency:
    """Tests for memory-efficient processing."""

    def test_pipeline_handles_large_data(self) -> None:
        """Test pipeline can handle larger datasets without memory issues."""
        data = np.random.rand(32, 32, 8, 6).astype(np.float32)
        b_values = np.array([0, 50, 100, 200, 400, 800])

        config = IVIMPipelineConfig()
        pipeline = IVIMPipeline(config)

        result = pipeline.run(data, b_values=b_values)
        assert result is not None


class TestAutomatedPipelineHeadless:
    """Tests for headless/automated pipeline execution."""

    def test_headless_execution_no_display(self) -> None:
        """Test pipeline runs without display requirements."""
        import os

        original_display = os.environ.get("DISPLAY")
        try:
            os.environ.pop("DISPLAY", None)

            data = np.random.rand(4, 4, 2, 4)
            b_values = np.array([0, 100, 400, 800])

            result = run_analysis(data, modality="ivim", b_values=b_values)
            assert result is not None

        finally:
            if original_display:
                os.environ["DISPLAY"] = original_display

    def test_pipeline_deterministic_results(self) -> None:
        """Test pipeline produces deterministic results."""
        np.random.seed(42)
        data = np.random.rand(4, 4, 2, 4)
        b_values = np.array([0, 100, 400, 800])

        np.random.seed(42)
        result1 = run_analysis(data.copy(), modality="ivim", b_values=b_values)

        np.random.seed(42)
        result2 = run_analysis(data.copy(), modality="ivim", b_values=b_values)

        for key in result1.parameter_maps:
            if key in result2.parameter_maps:
                pm1 = result1.parameter_maps[key]
                pm2 = result2.parameter_maps[key]
                arr1 = pm1.values if hasattr(pm1, "values") else pm1
                arr2 = pm2.values if hasattr(pm2, "values") else pm2
                np.testing.assert_array_almost_equal(arr1, arr2)
