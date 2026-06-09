"""Integration tests for IVIM pipeline.

Tests the complete IVIM analysis workflow from multi-b-value diffusion
data through parameter estimation for D, D*, and f.

User Story 4: Researcher analyzes multi-b-value diffusion data to separate
true diffusion (D) from pseudo-diffusion (D*) and estimate perfusion
fraction (f) using segmented IVIM fitting.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestIVIMPipelineIntegration:
    """Integration tests for IVIM pipeline."""

    @pytest.fixture
    def synthetic_ivim_data(self) -> dict:
        """Create synthetic IVIM data for integration testing."""
        np.random.seed(42)

        # Dimensions
        nx, ny, nz = 16, 16, 4

        # b-values (typical IVIM protocol)
        b_values = np.array([0, 10, 20, 30, 50, 80, 100, 150, 200, 400, 600, 800])

        # Ground truth parameters
        s0_true = np.random.uniform(900, 1100, (nx, ny, nz))
        d_true = np.random.uniform(0.8e-3, 1.5e-3, (nx, ny, nz))  # mm²/s
        d_star_true = np.random.uniform(8e-3, 25e-3, (nx, ny, nz))  # mm²/s
        f_true = np.random.uniform(0.05, 0.25, (nx, ny, nz))

        # Generate signal using bi-exponential model
        # S(b) = S0 × ((1-f) × exp(-b×D) + f × exp(-b×D*))
        signal = np.zeros((nx, ny, nz, len(b_values)))
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    s0 = s0_true[i, j, k]
                    d = d_true[i, j, k]
                    d_star = d_star_true[i, j, k]
                    f = f_true[i, j, k]

                    s = s0 * (
                        (1 - f) * np.exp(-b_values * d) + f * np.exp(-b_values * d_star)
                    )
                    signal[i, j, k, :] = s

        # Add Rician noise (approximated as Gaussian for high SNR)
        snr = 50
        noise_std = s0_true.mean() / snr
        signal += np.random.randn(*signal.shape) * noise_std
        signal = np.maximum(signal, 0)  # Ensure non-negative

        mask = np.ones((nx, ny, nz), dtype=bool)

        return {
            "signal": signal,
            "b_values": b_values,
            "s0_true": s0_true,
            "d_true": d_true,
            "d_star_true": d_star_true,
            "f_true": f_true,
            "mask": mask,
            "shape": (nx, ny, nz),
        }

    def test_segmented_ivim_fitting(self, synthetic_ivim_data: dict) -> None:
        """Test segmented (two-step) IVIM fitting."""
        from osipy.ivim.fitting import FittingMethod, fit_ivim

        # Use small subset for speed
        signal = synthetic_ivim_data["signal"][:4, :4, :2, :]
        mask = synthetic_ivim_data["mask"][:4, :4, :2]

        result = fit_ivim(
            signal=signal,
            b_values=synthetic_ivim_data["b_values"],
            method=FittingMethod.SEGMENTED,
            mask=mask,
        )

        assert result is not None
        assert result.d_map is not None
        assert result.d_star_map is not None
        assert result.f_map is not None

    def test_simultaneous_ivim_fitting(self, synthetic_ivim_data: dict) -> None:
        """Test simultaneous bi-exponential IVIM fitting."""
        from osipy.ivim.fitting import FittingMethod, fit_ivim

        signal = synthetic_ivim_data["signal"][:4, :4, :2, :]
        mask = synthetic_ivim_data["mask"][:4, :4, :2]

        result = fit_ivim(
            signal=signal,
            b_values=synthetic_ivim_data["b_values"],
            method=FittingMethod.FULL,
            mask=mask,
        )

        assert result is not None
        assert result.d_map is not None

    def test_bayesian_ivim_fitting(self, synthetic_ivim_data: dict) -> None:
        """Test Bayesian IVIM fitting with uncertainty."""
        from osipy.ivim.fitting import FittingMethod, IVIMFitParams, fit_ivim

        signal = synthetic_ivim_data["signal"][:4, :4, :2, :]
        mask = synthetic_ivim_data["mask"][:4, :4, :2]

        params = IVIMFitParams(
            bayesian_params={"compute_uncertainty": True},
        )

        result = fit_ivim(
            signal=signal,
            b_values=synthetic_ivim_data["b_values"],
            method=FittingMethod.BAYESIAN,
            mask=mask,
            params=params,
        )

        assert result is not None
        assert result.d_map is not None
        assert result.f_map is not None

    def test_physiological_bounds(self, synthetic_ivim_data: dict) -> None:
        """Test that physiological bounds are applied."""
        from osipy.ivim.fitting import FittingMethod, fit_ivim

        signal = synthetic_ivim_data["signal"][:4, :4, :2, :]
        mask = synthetic_ivim_data["mask"][:4, :4, :2]

        result = fit_ivim(
            signal=signal,
            b_values=synthetic_ivim_data["b_values"],
            method=FittingMethod.SEGMENTED,
            mask=mask,
        )

        # D should be in physiological range (0-5 x10^-3 mm²/s)
        d_values = result.d_map.values[mask]
        # Values are stored in x10^-3 mm²/s
        assert np.all(d_values >= 0), "D should be non-negative"
        assert np.all(d_values <= 5), "D should be <= 5 x10^-3 mm²/s"

        # f should be 0-1 (or 0-0.5 for strict bounds)
        f_values = result.f_map.values[mask]
        assert np.all(f_values >= 0), "f should be non-negative"
        assert np.all(f_values <= 1), "f should be <= 1"

    def test_full_ivim_pipeline_segmented(self, synthetic_ivim_data: dict) -> None:
        """Test complete IVIM pipeline with segmented fitting."""
        from osipy.ivim.fitting import FittingMethod, IVIMFitParams, fit_ivim

        # Use subset for speed
        nx, ny, nz = 4, 4, 2
        signal = synthetic_ivim_data["signal"][:nx, :ny, :nz, :]
        mask = synthetic_ivim_data["mask"][:nx, :ny, :nz]
        d_true = synthetic_ivim_data["d_true"][:nx, :ny, :nz]
        synthetic_ivim_data["f_true"][:nx, :ny, :nz]

        # Step 1: Fit IVIM model
        params = IVIMFitParams(
            b_threshold=200.0,  # b-value threshold for segmented fitting
            bounds={
                "d": (0.1e-3, 4.0e-3),
                "d_star": (5.0e-3, 100.0e-3),
                "f": (0.0, 0.5),
            },
        )

        result = fit_ivim(
            signal=signal,
            b_values=synthetic_ivim_data["b_values"],
            method=FittingMethod.SEGMENTED,
            mask=mask,
            params=params,
        )

        assert result.d_map is not None, "D map not generated"
        assert result.d_star_map is not None, "D* map not generated"
        assert result.f_map is not None, "f map not generated"

        # Step 2: Check parameter recovery (with tolerance)
        # D should correlate with true values
        d_estimated = result.d_map.values[mask] / 1e3  # Convert to mm²/s
        d_truth = d_true[mask]

        # At least positive correlation
        if np.std(d_estimated) > 0 and np.std(d_truth) > 0:
            corr = np.corrcoef(d_estimated.flatten(), d_truth.flatten())[0, 1]
            assert corr > 0, "D estimates should correlate with truth"

    def test_full_ivim_pipeline_bayesian(self, synthetic_ivim_data: dict) -> None:
        """Test complete IVIM pipeline with Bayesian fitting."""
        from osipy.ivim.fitting import FittingMethod, IVIMFitParams, fit_ivim

        nx, ny, nz = 4, 4, 2
        signal = synthetic_ivim_data["signal"][:nx, :ny, :nz, :]
        mask = synthetic_ivim_data["mask"][:nx, :ny, :nz]

        params = IVIMFitParams(
            bayesian_params={"compute_uncertainty": True},
        )

        result = fit_ivim(
            signal=signal,
            b_values=synthetic_ivim_data["b_values"],
            method=FittingMethod.BAYESIAN,
            mask=mask,
            params=params,
        )

        # Check all outputs present
        assert result.d_map is not None
        assert result.d_star_map is not None
        assert result.f_map is not None
        assert result.quality_mask is not None

    def test_bi_exponential_model(self) -> None:
        """Test bi-exponential model implementation."""
        from osipy.ivim.models import IVIMBiexponentialModel

        model = IVIMBiexponentialModel()

        b_values = np.array([0, 50, 100, 200, 500, 800])
        s0 = 1000.0
        d = 1.0e-3
        d_star = 15.0e-3
        f = 0.1

        params = {"S0": s0, "D": d, "D*": d_star, "f": f}
        signal = model.predict(b_values, params)

        assert len(signal) == len(b_values)
        assert signal[0] == pytest.approx(s0, rel=1e-10)  # At b=0
        assert signal[-1] < signal[0]  # Decay


class TestIVIMOutputValidation:
    """Test IVIM output format and units."""

    @pytest.fixture
    def synthetic_ivim_data(self) -> dict:
        """Create synthetic IVIM data for output validation testing."""
        np.random.seed(42)

        nx, ny, nz = 16, 16, 4
        b_values = np.array([0, 10, 20, 30, 50, 80, 100, 150, 200, 400, 600, 800])
        s0_true = np.random.uniform(900, 1100, (nx, ny, nz))
        d_true = np.random.uniform(0.8e-3, 1.5e-3, (nx, ny, nz))
        d_star_true = np.random.uniform(8e-3, 25e-3, (nx, ny, nz))
        f_true = np.random.uniform(0.05, 0.25, (nx, ny, nz))

        signal = np.zeros((nx, ny, nz, len(b_values)))
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    s0 = s0_true[i, j, k]
                    d = d_true[i, j, k]
                    d_star = d_star_true[i, j, k]
                    f = f_true[i, j, k]
                    s = s0 * (
                        (1 - f) * np.exp(-b_values * d) + f * np.exp(-b_values * d_star)
                    )
                    signal[i, j, k, :] = s

        snr = 50
        noise_std = s0_true.mean() / snr
        signal += np.random.randn(*signal.shape) * noise_std
        signal = np.maximum(signal, 0)
        mask = np.ones((nx, ny, nz), dtype=bool)

        return {
            "signal": signal,
            "b_values": b_values,
            "s0_true": s0_true,
            "d_true": d_true,
            "d_star_true": d_star_true,
            "f_true": f_true,
            "mask": mask,
            "shape": (nx, ny, nz),
        }

    def test_d_units(self, synthetic_ivim_data: dict) -> None:
        """Test that D is in correct units (x10^-3 mm²/s)."""
        from osipy.ivim.fitting import FittingMethod, fit_ivim

        signal = synthetic_ivim_data["signal"][:4, :4, :2, :]
        mask = synthetic_ivim_data["mask"][:4, :4, :2]

        result = fit_ivim(
            signal=signal,
            b_values=synthetic_ivim_data["b_values"],
            method=FittingMethod.SEGMENTED,
            mask=mask,
        )

        # D should be reported in x10^-3 mm²/s
        assert "mm" in result.d_map.units.lower()

    def test_d_star_larger_than_d(self, synthetic_ivim_data: dict) -> None:
        """Test that D* > D (physiological constraint)."""
        from osipy.ivim.fitting import FittingMethod, fit_ivim

        signal = synthetic_ivim_data["signal"][:4, :4, :2, :]
        mask = synthetic_ivim_data["mask"][:4, :4, :2]

        result = fit_ivim(
            signal=signal,
            b_values=synthetic_ivim_data["b_values"],
            method=FittingMethod.SEGMENTED,
            mask=mask,
        )

        d = result.d_map.values[mask]
        d_star = result.d_star_map.values[mask]

        # D* should generally be larger than D
        # Allow some fitting failures
        ratio_correct = np.sum(d_star > d) / len(d)
        assert ratio_correct > 0.8, "D* should be > D for most voxels"

    def test_f_dimensionless(self, synthetic_ivim_data: dict) -> None:
        """Test that f is dimensionless."""
        from osipy.ivim.fitting import FittingMethod, fit_ivim

        signal = synthetic_ivim_data["signal"][:4, :4, :2, :]
        mask = synthetic_ivim_data["mask"][:4, :4, :2]

        result = fit_ivim(
            signal=signal,
            b_values=synthetic_ivim_data["b_values"],
            method=FittingMethod.SEGMENTED,
            mask=mask,
        )

        # f should be dimensionless (empty units or "fraction")
        assert result.f_map.units == "" or "fraction" in result.f_map.units.lower()

    def test_parameter_map_structure(self, synthetic_ivim_data: dict) -> None:
        """Test that parameter maps have required structure."""
        from osipy.ivim.fitting import FittingMethod, fit_ivim

        signal = synthetic_ivim_data["signal"][:4, :4, :2, :]
        mask = synthetic_ivim_data["mask"][:4, :4, :2]

        result = fit_ivim(
            signal=signal,
            b_values=synthetic_ivim_data["b_values"],
            method=FittingMethod.SEGMENTED,
            mask=mask,
        )

        # Check ParameterMap structure
        for param_map in [result.d_map, result.d_star_map, result.f_map]:
            assert hasattr(param_map, "name")
            assert hasattr(param_map, "symbol")
            assert hasattr(param_map, "units")
            assert hasattr(param_map, "values")
            assert hasattr(param_map, "affine")
            assert param_map.affine.shape == (4, 4)


class TestIVIMPipelineRegistryConfig:
    """Tests for the registry-driven IVIM config wiring (config -> pipeline)."""

    @staticmethod
    def _synthetic_biexp(seed: int = 7):
        """Generate a small noisy bi-exponential IVIM dataset."""
        rng = np.random.default_rng(seed)
        b = np.array([0, 10, 20, 30, 50, 80, 100, 150, 200, 400, 600, 800], dtype=float)
        nx, ny, nz = 4, 4, 2
        s0 = rng.uniform(900, 1100, (nx, ny, nz))
        d = rng.uniform(0.8e-3, 1.5e-3, (nx, ny, nz))
        dstar = rng.uniform(8e-3, 25e-3, (nx, ny, nz))
        f = rng.uniform(0.05, 0.25, (nx, ny, nz))
        sig = s0[..., None] * (
            (1 - f[..., None]) * np.exp(-b * d[..., None])
            + f[..., None] * np.exp(-b * dstar[..., None])
        )
        sig = sig + rng.standard_normal(sig.shape) * 5.0
        return sig, b

    def test_load_config_to_pipeline_round_trip(self, tmp_path) -> None:
        """A nested IVIM YAML loads and drives the pipeline end-to-end."""
        from osipy.cli.config import load_config
        from osipy.ivim.fitting import FittingMethod
        from osipy.pipeline.ivim_pipeline import IVIMPipeline, IVIMPipelineConfig

        sig, b = self._synthetic_biexp()
        cfg_path = tmp_path / "ivim.yaml"
        cfg_path.write_text(
            "modality: ivim\n"
            "pipeline:\n"
            "  fitting:\n"
            "    method: segmented\n"
            "    b_threshold: 180.0\n"
            "    initial_guess:\n"
            "      D: 1.0e-3\n"
            "      f: 0.1\n"
            "  model:\n"
            "    model: biexponential\n"
            "  normalize_signal: true\n"
        )
        mc = load_config(cfg_path).get_modality_config()

        pipeline_cfg = IVIMPipelineConfig(
            fitting_method=FittingMethod(mc.fitting.method),
            signal_model=mc.model.model,
            b_threshold=mc.fitting.b_threshold,
            normalize_signal=mc.normalize_signal,
            initial_guess=mc.fitting.initial_guess,
        )
        result = IVIMPipeline(pipeline_cfg).run(sig, b)
        assert int(result.fit_result.quality_mask.sum()) > 0
        assert result.config.signal_model == "biexponential"
        assert result.config.initial_guess == {"D": 1.0e-3, "f": 0.1}

    def test_simplified_model_selectable_and_fits(self) -> None:
        """The simplified model is selectable via config and produces fits."""
        from osipy.ivim.fitting import FittingMethod
        from osipy.pipeline.ivim_pipeline import IVIMPipeline, IVIMPipelineConfig

        sig, b = self._synthetic_biexp(seed=11)
        cfg = IVIMPipelineConfig(
            fitting_method=FittingMethod.SEGMENTED,
            signal_model="simplified",
        )
        result = IVIMPipeline(cfg).run(sig, b)
        qmask = result.fit_result.quality_mask
        assert int(qmask.sum()) > 0
        d_vals = result.fit_result.d_map.values[qmask]
        assert np.all(d_vals > 0)
        # simplified model has no D*; the pipeline still produces a D* map
        # (zeros) so downstream consumers see a uniform interface.
        assert result.fit_result.d_star_map is not None

    def test_all_fitting_methods_run_through_pipeline(self) -> None:
        """segmented / full / bayesian all run via the pipeline config."""
        from osipy.ivim.fitting import FittingMethod
        from osipy.pipeline.ivim_pipeline import IVIMPipeline, IVIMPipelineConfig

        sig, b = self._synthetic_biexp(seed=3)
        for method in (
            FittingMethod.SEGMENTED,
            FittingMethod.FULL,
            FittingMethod.BAYESIAN,
        ):
            cfg = IVIMPipelineConfig(fitting_method=method)
            result = IVIMPipeline(cfg).run(sig, b)
            assert int(result.fit_result.quality_mask.sum()) > 0

    def test_initial_guess_seeds_the_fit(self) -> None:
        """A user initial_guess reaches the optimizer (seeds the starting D/f)."""
        from unittest.mock import patch

        import osipy.ivim.models.binding as binding_module
        from osipy.ivim.fitting import FittingMethod
        from osipy.pipeline.ivim_pipeline import IVIMPipeline, IVIMPipelineConfig

        sig, b = self._synthetic_biexp(seed=5)
        guess = {"D": 2.0e-3, "f": 0.3}
        cfg = IVIMPipelineConfig(
            fitting_method=FittingMethod.SEGMENTED,
            initial_guess=guess,
        )

        seen: dict[str, object] = {}
        original_init = binding_module.BoundIVIMModel.__init__

        def spy_init(self, *args, **kwargs):
            seen["initial_guess"] = kwargs.get("initial_guess")
            return original_init(self, *args, **kwargs)

        with patch.object(binding_module.BoundIVIMModel, "__init__", spy_init):
            IVIMPipeline(cfg).run(sig, b)

        # The configured initial guess flows through to BoundIVIMModel.
        assert seen.get("initial_guess") == guess

    def test_initial_guess_changes_starting_point(self) -> None:
        """get_initial_guess_batch honors the override in place of data-driven seed."""
        from osipy.ivim.models import IVIMBiexponentialModel
        from osipy.ivim.models.binding import BoundIVIMModel

        sig, b = self._synthetic_biexp(seed=9)
        obs = sig.reshape(-1, sig.shape[-1]).T  # (n_b, n_voxels)

        model = IVIMBiexponentialModel()
        bm_default = BoundIVIMModel(model, b, b_threshold=200.0)
        bm_override = BoundIVIMModel(
            model, b, b_threshold=200.0, initial_guess={"D": 2.5e-3, "f": 0.33}
        )
        g0 = bm_default.get_initial_guess_batch(obs, np)
        g1 = bm_override.get_initial_guess_batch(obs, np)

        free = bm_default.parameters
        d_idx = free.index("D")
        f_idx = free.index("f")
        assert np.allclose(g1[d_idx, :], 2.5e-3)
        assert np.allclose(g1[f_idx, :], 0.33)
        # Unspecified parameters keep their data-driven guess.
        s0_idx = free.index("S0")
        assert np.allclose(g0[s0_idx, :], g1[s0_idx, :])
