"""Tests for fitter registry, aliases, and bounds override."""

import pytest

from osipy.common.exceptions import DataValidationError
from osipy.common.fitting import get_fitter, list_fitters
from osipy.common.fitting.base import BaseFitter
from osipy.common.fitting.bayesian import BayesianFitter
from osipy.common.fitting.least_squares import LevenbergMarquardtFitter


class TestFitterRegistry:
    def test_list_fitters_returns_sorted(self):
        result = list_fitters()
        # Core fitters always present
        assert "bayesian" in result
        assert "lm" in result
        # Verify sorted order
        assert result == sorted(result)

    def test_list_fitters_includes_svd_after_dsc_import(self):
        """SVD fitters appear after DSC deconvolution module is imported."""
        import osipy.dsc.deconvolution.svd_fitters  # noqa: F401

        result = list_fitters()
        for name in ["sSVD", "cSVD", "oSVD", "tikhonov"]:
            assert name in result, f"Fitter '{name}' not registered"

    def test_get_fitter_lm(self):
        fitter = get_fitter("lm")
        assert isinstance(fitter, LevenbergMarquardtFitter)

    def test_get_fitter_bayesian(self):
        fitter = get_fitter("bayesian")
        assert isinstance(fitter, BayesianFitter)

    def test_get_fitter_unknown_raises(self):
        with pytest.raises(DataValidationError, match="Unknown fitter"):
            get_fitter("nonexistent")


class TestMergeBounds:
    def test_merge_none_returns_original(self):
        model_bounds = {"Ktrans": (0.0, 1.0), "ve": (0.0, 1.0)}
        result = BaseFitter._merge_bounds(model_bounds, None)
        assert result == model_bounds

    def test_merge_overrides_specific_param(self):
        model_bounds = {"Ktrans": (0.0, 1.0), "ve": (0.0, 1.0)}
        overrides = {"Ktrans": (0.0, 5.0)}
        result = BaseFitter._merge_bounds(model_bounds, overrides)
        assert result["Ktrans"] == (0.0, 5.0)
        assert result["ve"] == (0.0, 1.0)

    def test_merge_does_not_mutate_original(self):
        model_bounds = {"Ktrans": (0.0, 1.0), "ve": (0.0, 1.0)}
        overrides = {"Ktrans": (0.0, 5.0)}
        BaseFitter._merge_bounds(model_bounds, overrides)
        assert model_bounds["Ktrans"] == (0.0, 1.0)
