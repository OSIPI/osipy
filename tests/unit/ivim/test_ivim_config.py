"""Unit tests for the registry-driven IVIM config models and wiring.

Covers the discriminated-union config models (fitting strategy + signal
model), the ``_build_signal_model`` selection helper, and the
``initial_guess`` threading through ``BoundIVIMModel``.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from osipy.ivim.config import (
    IVIM_FITTING_CONFIGS,
    IVIM_MODEL_CONFIGS,
    IVIM_SIGNAL_MODEL_REGISTRY,
    BayesianFittingConfig,
    BiexponentialModelConfig,
    FullFittingConfig,
    SegmentedFittingConfig,
    SimplifiedModelConfig,
)


class TestIVIMFittingConfigs:
    """Discriminated-union fitting-strategy MethodConfig models."""

    def test_registry_keys(self) -> None:
        """All three strategies are exposed as config models."""
        assert set(IVIM_FITTING_CONFIGS) == {"segmented", "full", "bayesian"}

    def test_shared_knobs_present_on_all(self) -> None:
        """Every strategy carries the shared fitting knobs."""
        for cfg_cls in IVIM_FITTING_CONFIGS.values():
            fields = cfg_cls.model_fields
            assert "max_iterations" in fields
            assert "tolerance" in fields
            assert "bounds" in fields
            assert "initial_guess" in fields

    def test_b_threshold_only_on_segmented_and_bayesian(self) -> None:
        """The full strategy has no segmentation threshold knob."""
        assert "b_threshold" in SegmentedFittingConfig.model_fields
        assert "b_threshold" in BayesianFittingConfig.model_fields
        assert "b_threshold" not in FullFittingConfig.model_fields

    def test_prior_knobs_only_on_bayesian(self) -> None:
        """Bayesian-only knobs do not leak onto the other strategies."""
        assert "prior_scale" in BayesianFittingConfig.model_fields
        assert "prior_scale" not in SegmentedFittingConfig.model_fields
        assert "prior_scale" not in FullFittingConfig.model_fields

    def test_extra_keys_forbidden(self) -> None:
        """MethodConfig rejects unknown keys."""
        with pytest.raises(ValidationError):
            SegmentedFittingConfig(nonsense=1)

    def test_bounds_validation(self) -> None:
        """Bounds must be [lower, upper] with lower <= upper."""
        with pytest.raises(ValidationError, match="must be"):
            SegmentedFittingConfig(bounds={"D": [1e-4]})
        with pytest.raises(ValidationError, match="Lower bound > upper bound"):
            SegmentedFittingConfig(bounds={"D": [3e-3, 1e-4]})


class TestIVIMModelConfigs:
    """Discriminated-union signal-model MethodConfig models."""

    def test_registry_keys(self) -> None:
        """Both signal models are exposed as config models."""
        assert set(IVIM_MODEL_CONFIGS) == {"biexponential", "simplified"}

    def test_signal_model_registry_constructs(self) -> None:
        """The registry maps names to constructible model classes."""
        assert set(IVIM_SIGNAL_MODEL_REGISTRY) == {"biexponential", "simplified"}
        biexp = IVIM_SIGNAL_MODEL_REGISTRY["biexponential"]()
        assert biexp.parameters == ["S0", "D", "D*", "f"]

    def test_simplified_exposes_b_threshold(self) -> None:
        """The simplified model config carries its perfusion-cutoff knob."""
        assert "b_threshold" in SimplifiedModelConfig.model_fields
        assert "b_threshold" not in BiexponentialModelConfig.model_fields

    def test_simplified_extra_keys_forbidden(self) -> None:
        """Cross-model keys are rejected."""
        with pytest.raises(ValidationError):
            BiexponentialModelConfig(b_threshold=180.0)


class TestBuildSignalModel:
    """The ``_build_signal_model`` selection helper."""

    def test_default_biexponential(self) -> None:
        from osipy.ivim.fitting.estimators import IVIMFitParams, _build_signal_model

        model = _build_signal_model(IVIMFitParams())
        assert model.parameters == ["S0", "D", "D*", "f"]

    def test_simplified_uses_b_threshold(self) -> None:
        from osipy.ivim.fitting.estimators import IVIMFitParams, _build_signal_model

        params = IVIMFitParams(signal_model="simplified", b_threshold=175.0)
        model = _build_signal_model(params)
        assert model.parameters == ["S0", "D", "f"]
        assert model.b_threshold == 175.0


class TestInitialGuessThreading:
    """``initial_guess`` flows into ``BoundIVIMModel.get_initial_guess_batch``."""

    def test_override_seeds_named_params(self) -> None:
        from osipy.ivim.models import IVIMBiexponentialModel
        from osipy.ivim.models.binding import BoundIVIMModel

        b = np.array([0, 10, 50, 100, 200, 400, 800], dtype=float)
        obs = np.ones((len(b), 3)) * 0.5

        model = IVIMBiexponentialModel()
        bound = BoundIVIMModel(
            model, b, b_threshold=200.0, initial_guess={"D": 2.5e-3, "f": 0.33}
        )
        guess = bound.get_initial_guess_batch(obs, np)
        free = bound.parameters
        assert np.allclose(guess[free.index("D"), :], 2.5e-3)
        assert np.allclose(guess[free.index("f"), :], 0.33)

    def test_no_override_preserves_data_driven_guess(self) -> None:
        from osipy.ivim.models import IVIMBiexponentialModel
        from osipy.ivim.models.binding import BoundIVIMModel

        b = np.array([0, 10, 50, 100, 200, 400, 800], dtype=float)
        obs = np.ones((len(b), 3)) * 0.5

        model = IVIMBiexponentialModel()
        g0 = BoundIVIMModel(model, b, b_threshold=200.0).get_initial_guess_batch(
            obs, np
        )
        g1 = BoundIVIMModel(
            model, b, b_threshold=200.0, initial_guess={"D": 2.5e-3}
        ).get_initial_guess_batch(obs, np)
        free = BoundIVIMModel(model, b, b_threshold=200.0).parameters
        # Only D changed; S0 and f keep the data-driven values.
        assert np.allclose(g0[free.index("S0"), :], g1[free.index("S0"), :])
        assert np.allclose(g0[free.index("f"), :], g1[free.index("f"), :])
