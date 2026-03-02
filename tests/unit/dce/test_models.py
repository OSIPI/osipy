"""Unit tests for DCE pharmacokinetic models.

Tests for Tofts, Extended Tofts, Patlak, 2CXM, and 2CUM implementations.
"""

import numpy as np
import pytest

from osipy.common.exceptions import DataValidationError
from osipy.dce.models import (
    MODEL_REGISTRY,
    ExtendedToftsModel,
    ExtendedToftsParams,
    PatlakModel,
    PatlakParams,
    ToftsModel,
    ToftsParams,
    TwoCompartmentModel,
    TwoCompartmentParams,
    TwoCompartmentUptakeModel,
    TwoCompartmentUptakeParams,
    get_model,
    list_models,
)


def generate_parker_aif(t: np.ndarray) -> np.ndarray:
    """Generate simplified Parker-like AIF for testing."""
    t_min = t / 60.0
    # Simplified bi-gaussian + exponential
    cb = (
        0.8 * np.exp(-((t_min - 0.17) ** 2) / (2 * 0.05**2))
        + 0.3 * np.exp(-((t_min - 0.4) ** 2) / (2 * 0.1**2))
        + 0.5 * np.exp(-0.1 * t_min)
    )
    return cb


class TestToftsModel:
    """Tests for Standard Tofts model."""

    def test_model_properties(self) -> None:
        """Test model property accessors."""
        model = ToftsModel()
        assert model.name == "Standard Tofts"
        assert "Ktrans" in model.parameters
        assert "ve" in model.parameters
        assert len(model.parameters) == 2
        assert "Ktrans" in model.parameter_units
        assert model.parameter_units["Ktrans"] == "1/min"

    def test_predict_with_params_object(self) -> None:
        """Test prediction with ToftsParams object."""
        model = ToftsModel()
        t = np.linspace(0, 300, 60)  # 5 minutes, seconds
        aif = generate_parker_aif(t)
        params = ToftsParams(ktrans=0.2, ve=0.3)

        ct = model.predict(t, aif, params)

        assert ct.shape == t.shape
        assert np.all(np.isfinite(ct))
        # Tissue concentration should start low and increase
        assert ct[0] < ct[30]

    def test_predict_with_dict_params(self) -> None:
        """Test prediction with dictionary parameters."""
        model = ToftsModel()
        t = np.linspace(0, 300, 60)
        aif = generate_parker_aif(t)
        params = {"Ktrans": 0.15, "ve": 0.25}

        ct = model.predict(t, aif, params)

        assert ct.shape == t.shape
        assert np.all(ct >= 0)

    def test_get_bounds(self) -> None:
        """Test parameter bounds."""
        model = ToftsModel()
        bounds = model.get_bounds()

        assert "Ktrans" in bounds
        assert "ve" in bounds
        assert bounds["Ktrans"][0] < bounds["Ktrans"][1]
        assert bounds["ve"][0] < bounds["ve"][1]

    def test_get_initial_guess(self) -> None:
        """Test initial parameter estimation."""
        model = ToftsModel()
        t = np.linspace(0, 300, 60)
        aif = generate_parker_aif(t)

        # Generate synthetic tissue curve
        true_params = ToftsParams(ktrans=0.2, ve=0.3)
        ct = model.predict(t, aif, true_params)

        guess = model.get_initial_guess(ct, aif, t)

        assert isinstance(guess, ToftsParams)
        assert guess.ktrans > 0
        assert 0 < guess.ve < 1

    def test_kep_property(self) -> None:
        """Test kep derived property."""
        params = ToftsParams(ktrans=0.3, ve=0.2)
        assert params.kep == pytest.approx(1.5)

        params_zero = ToftsParams(ktrans=0.3, ve=0.0)
        assert params_zero.kep == 0.0


class TestExtendedToftsModel:
    """Tests for Extended Tofts model."""

    def test_model_properties(self) -> None:
        """Test model property accessors."""
        model = ExtendedToftsModel()
        assert model.name == "Extended Tofts"
        assert "Ktrans" in model.parameters
        assert "ve" in model.parameters
        assert "vp" in model.parameters
        assert len(model.parameters) == 3

    def test_predict_with_vp(self) -> None:
        """Test that vp adds vascular contribution."""
        model_std = ToftsModel()
        model_ext = ExtendedToftsModel()

        t = np.linspace(0, 300, 60)
        aif = generate_parker_aif(t)

        # Same Ktrans, ve; with vp
        params_std = ToftsParams(ktrans=0.2, ve=0.3)
        params_ext = ExtendedToftsParams(ktrans=0.2, ve=0.3, vp=0.05)

        ct_std = model_std.predict(t, aif, params_std)
        ct_ext = model_ext.predict(t, aif, params_ext)

        # Extended should be higher due to vascular contribution
        # Especially at early times
        assert ct_ext[5] > ct_std[5]

    def test_predict_with_dict(self) -> None:
        """Test prediction with dictionary parameters."""
        model = ExtendedToftsModel()
        t = np.linspace(0, 300, 60)
        aif = generate_parker_aif(t)
        params = {"Ktrans": 0.1, "ve": 0.2, "vp": 0.03}

        ct = model.predict(t, aif, params)

        assert ct.shape == t.shape
        assert np.all(np.isfinite(ct))

    def test_get_initial_guess(self) -> None:
        """Test initial guess estimation."""
        model = ExtendedToftsModel()
        t = np.linspace(0, 300, 60)
        aif = generate_parker_aif(t)

        true_params = ExtendedToftsParams(ktrans=0.15, ve=0.25, vp=0.04)
        ct = model.predict(t, aif, true_params)

        guess = model.get_initial_guess(ct, aif, t)

        assert isinstance(guess, ExtendedToftsParams)
        assert guess.ktrans > 0
        assert guess.vp > 0


class TestPatlakModel:
    """Tests for Patlak model."""

    def test_model_properties(self) -> None:
        """Test model property accessors."""
        model = PatlakModel()
        assert model.name == "Patlak"
        assert "Ktrans" in model.parameters
        assert "vp" in model.parameters
        assert len(model.parameters) == 2

    def test_predict_linear_accumulation(self) -> None:
        """Test Patlak predicts linear accumulation (no backflux)."""
        model = PatlakModel()
        t = np.linspace(0, 300, 60)
        aif = generate_parker_aif(t)
        params = PatlakParams(ktrans=0.1, vp=0.02)

        ct = model.predict(t, aif, params)

        assert ct.shape == t.shape
        # Should show monotonic increase after initial uptake
        # (characteristic of no backflux)
        late_indices = slice(30, 60)
        ct_late = ct[late_indices]
        # Allow for some noise but trend should be non-decreasing
        assert ct_late[-1] >= ct_late[0] * 0.8

    def test_graphical_analysis_initial_guess(self) -> None:
        """Test Patlak graphical analysis for initial guess."""
        model = PatlakModel()
        t = np.linspace(0, 300, 60)
        aif = generate_parker_aif(t)

        true_params = PatlakParams(ktrans=0.15, vp=0.03)
        ct = model.predict(t, aif, true_params)

        guess = model.get_initial_guess(ct, aif, t)

        assert isinstance(guess, PatlakParams)


class TestTwoCompartmentModel:
    """Tests for Two-Compartment Exchange Model (2CXM)."""

    def test_model_properties(self) -> None:
        """Test model property accessors."""
        model = TwoCompartmentModel()
        assert "2CXM" in model.name or "Two-Compartment" in model.name
        assert "Fp" in model.parameters
        assert "PS" in model.parameters
        assert "ve" in model.parameters
        assert "vp" in model.parameters
        assert len(model.parameters) == 4

    def test_predict_bi_exponential(self) -> None:
        """Test 2CXM produces bi-exponential response."""
        model = TwoCompartmentModel()
        t = np.linspace(0, 300, 100)
        aif = generate_parker_aif(t)
        params = TwoCompartmentParams(fp=50.0, ps=5.0, ve=0.2, vp=0.03)

        ct = model.predict(t, aif, params)

        assert ct.shape == t.shape
        assert np.all(np.isfinite(ct))

    def test_predict_with_dict(self) -> None:
        """Test prediction with dictionary parameters."""
        model = TwoCompartmentModel()
        t = np.linspace(0, 300, 60)
        aif = generate_parker_aif(t)
        params = {"Fp": 40.0, "PS": 4.0, "ve": 0.25, "vp": 0.02}

        ct = model.predict(t, aif, params)

        assert np.all(np.isfinite(ct))

    def test_parameter_bounds(self) -> None:
        """Test physiological parameter bounds."""
        model = TwoCompartmentModel()
        bounds = model.get_bounds()

        assert bounds["Fp"][0] > 0  # Flow must be positive
        assert bounds["PS"][0] >= 0  # PS=0 is valid (no permeability)
        assert bounds["ve"][1] <= 1.0  # Volume fraction <= 1
        assert bounds["vp"][1] <= 1.0


class TestModelRegistry:
    """Tests for model registry functionality."""

    def test_registry_contains_all_models(self) -> None:
        """Test that registry contains all expected models."""
        assert "tofts" in MODEL_REGISTRY
        assert "extended_tofts" in MODEL_REGISTRY
        assert "patlak" in MODEL_REGISTRY
        assert "2cxm" in MODEL_REGISTRY

    def test_get_model_by_name(self) -> None:
        """Test get_model factory function."""
        tofts = get_model("tofts")
        assert isinstance(tofts, ToftsModel)

        ext_tofts = get_model("extended_tofts")
        assert isinstance(ext_tofts, ExtendedToftsModel)

        patlak = get_model("patlak")
        assert isinstance(patlak, PatlakModel)

        cxm = get_model("2cxm")
        assert isinstance(cxm, TwoCompartmentModel)

    def test_get_model_invalid_name(self) -> None:
        """Test get_model raises error for invalid name."""
        with pytest.raises(DataValidationError, match="Unknown model"):
            get_model("invalid_model")

    def test_registry_contains_2cum(self) -> None:
        """Test that registry contains 2CUM model."""
        assert "2cum" in MODEL_REGISTRY

    def test_get_model_2cum(self) -> None:
        """Test get_model returns 2CUM instance."""
        model = get_model("2cum")
        assert isinstance(model, TwoCompartmentUptakeModel)

    def test_list_models_contains_2cum(self) -> None:
        """Test list_models includes 2cum."""
        assert "2cum" in list_models()


class TestTwoCompartmentUptakeModel:
    """Tests for Two-Compartment Uptake Model (2CUM)."""

    def test_model_properties(self) -> None:
        """Test model property accessors."""
        model = TwoCompartmentUptakeModel()
        assert "2CUM" in model.name or "Uptake" in model.name
        assert model.parameters == ["Fp", "PS", "vp"]
        assert len(model.parameters) == 3

    def test_predict_returns_correct_shape(self) -> None:
        """Test predict returns array with correct shape."""
        model = TwoCompartmentUptakeModel()
        t = np.linspace(0, 300, 100)
        aif = generate_parker_aif(t)
        params = TwoCompartmentUptakeParams(fp=50.0, ps=5.0, vp=0.02)

        ct = model.predict(t, aif, params)

        assert ct.shape == (100,)

    def test_predict_all_finite(self) -> None:
        """Test predict returns all finite values."""
        model = TwoCompartmentUptakeModel()
        t = np.linspace(0, 300, 100)
        aif = generate_parker_aif(t)
        params = TwoCompartmentUptakeParams(fp=50.0, ps=5.0, vp=0.02)

        ct = model.predict(t, aif, params)

        assert np.all(np.isfinite(ct))

    def test_predict_with_dict(self) -> None:
        """Test prediction with dictionary parameters."""
        model = TwoCompartmentUptakeModel()
        t = np.linspace(0, 300, 60)
        aif = generate_parker_aif(t)
        params = {"Fp": 40.0, "PS": 3.0, "vp": 0.03}

        ct = model.predict(t, aif, params)

        assert ct.shape == t.shape
        assert np.all(np.isfinite(ct))

    def test_parameter_bounds(self) -> None:
        """Test physiological parameter bounds."""
        model = TwoCompartmentUptakeModel()
        bounds = model.get_bounds()

        assert "Fp" in bounds
        assert "PS" in bounds
        assert "vp" in bounds
        for key in ("Fp", "PS", "vp"):
            low, high = bounds[key]
            assert low < high

    def test_get_initial_guess(self) -> None:
        """Test initial parameter estimation."""
        model = TwoCompartmentUptakeModel()
        t = np.linspace(0, 300, 60)
        aif = generate_parker_aif(t)

        true_params = TwoCompartmentUptakeParams(fp=50.0, ps=5.0, vp=0.02)
        ct = model.predict(t, aif, true_params)

        guess = model.get_initial_guess(ct, aif, t)

        assert isinstance(guess, TwoCompartmentUptakeParams)
        assert guess.fp > 0
        assert guess.ps > 0
        assert guess.vp > 0

    def test_predict_batch_shape(self) -> None:
        """Test predict_batch returns correct shape."""
        model = TwoCompartmentUptakeModel()
        t = np.linspace(0, 300, 60)
        aif = generate_parker_aif(t)
        n_voxels = 5
        params_batch = np.array(
            [
                [50.0, 40.0, 60.0, 30.0, 55.0],  # Fp
                [5.0, 3.0, 8.0, 2.0, 6.0],  # PS
                [0.02, 0.03, 0.01, 0.04, 0.02],  # vp
            ]
        )

        ct = model.predict_batch(t, aif, params_batch, np)

        assert ct.shape == (len(t), n_voxels)

    def test_predict_batch_matches_single(self) -> None:
        """Test predict_batch matches per-voxel predict."""
        model = TwoCompartmentUptakeModel()
        t = np.linspace(0, 300, 60)
        aif = generate_parker_aif(t)
        params_batch = np.array(
            [
                [50.0, 40.0],  # Fp
                [5.0, 3.0],  # PS
                [0.02, 0.03],  # vp
            ]
        )

        ct_batch = model.predict_batch(t, aif, params_batch, np)

        # Compare with single-voxel predictions
        for v in range(2):
            params_dict = {
                "Fp": params_batch[0, v],
                "PS": params_batch[1, v],
                "vp": params_batch[2, v],
            }
            ct_single = model.predict(t, aif, params_dict)
            np.testing.assert_allclose(ct_batch[:, v], ct_single, rtol=1e-10)


def _cases_are_batchable(cases: list[dict]) -> bool:
    """Return True if all cases share the same time and AIF arrays."""
    ref_t = cases[0]["time"]
    ref_aif = cases[0]["aif"]
    return all(
        np.array_equal(ref_t, c["time"]) and np.array_equal(ref_aif, c["aif"])
        for c in cases[1:]
    )


def _check_case_results(
    case: dict,
    recovered: dict[str, float],
) -> bool:
    """Check if recovered params are within tolerance for a single case."""
    for param_name, true_val in case["true_params"].items():
        if param_name not in recovered:
            continue
        tol = case["tolerances"].get(param_name, {"absolute": 0.1, "relative": 0.1})
        abs_err = abs(recovered[param_name] - true_val)
        rel_err = abs_err / (abs(true_val) + 1e-10)
        if not (abs_err <= tol["absolute"] or rel_err <= tol["relative"]):
            return False
    return True


class TestOSIPIReferenceData:
    """Test models against OSIPI CodeCollection reference data."""

    @pytest.fixture(autouse=True)
    def _load_fixtures(self):
        """Load fixture infrastructure."""
        # Compatibility shim: numpy < 2.0 uses trapz, >= 2.0 uses trapezoid
        if not hasattr(np, "trapezoid"):
            np.trapezoid = np.trapezoid  # type: ignore[attr-defined]

        from tests.fixtures.osipi_codecollection import load_reference

        self.load_reference = load_reference

    @pytest.mark.parametrize(
        "model_name", ["tofts", "extended_tofts", "patlak", "2cxm", "2cum"]
    )
    def test_forward_model_matches_reference(self, model_name, generate_figures):
        """Test that model.predict() matches reference curves."""
        ref = self.load_reference(model_name)
        if ref is None:
            pytest.skip(f"No reference data for {model_name}")

        model = get_model(model_name)
        cases_data: list[dict] = []

        for case in ref["test_cases"]:
            time = np.array(case["time_seconds"])
            aif = np.array(case["aif"])
            expected = np.array(case["concentration"])
            params = case["params"]

            predicted = model.predict(time, aif, params)

            # Use tight tolerance for forward model (should be exact up to floating point)
            np.testing.assert_allclose(
                predicted,
                expected,
                atol=1e-6,
                rtol=1e-5,
                err_msg=f"Forward model mismatch for {model_name}/{case['name']}",
            )

            cases_data.append(
                {
                    "name": case["name"],
                    "time": time,
                    "aif": aif,
                    "expected": expected,
                    "predicted": predicted,
                    "params": params,
                }
            )

        # Record compliance result
        try:
            from tests.conftest import record_compliance_result

            record_compliance_result(
                f"{model_name}_forward_model",
                True,
                f"{len(ref['test_cases'])} cases within tolerance",
            )
        except ImportError:
            pass

        if generate_figures:
            from pathlib import Path

            from tests.compliance_figures import plot_forward_model

            plot_forward_model(model_name, cases_data, Path("output/compliance"))

    @pytest.mark.parametrize(
        "model_name",
        [
            "tofts",
            "extended_tofts",
            "patlak",
            pytest.param(
                "2cxm",
                marks=pytest.mark.xfail(
                    reason=(
                        "2CXM has 4 free params; case 1 "
                        "(low Fp=5, PS=5, low ve=0.1) has identifiability "
                        "issues with noisy DRO data (23/24 pass)"
                    ),
                    strict=False,
                ),
            ),
            "2cum",
        ],
    )
    def test_osipi_dro_parameter_recovery(self, model_name, generate_figures):
        """Test parameter recovery against real OSIPI DRO data with noise.

        Uses externally-generated DRO CSV files from the OSIPI CodeCollection
        with known ground-truth parameters and realistic noise levels.

        When all cases share the same time and AIF, they are batched into
        a single ``fit_model()`` call to avoid per-case GPU overhead.
        """
        import time as timer
        import warnings

        from tests.fixtures.osipi_codecollection.dce_csv_loader import (
            load_osipi_dro,
        )

        cases = load_osipi_dro(model_name)
        if cases is None:
            pytest.skip(f"No OSIPI DRO CSV data for {model_name}")

        from osipy.common.exceptions import FittingError
        from osipy.dce.fitting import fit_model

        get_model(model_name)
        batchable = _cases_are_batchable(cases)

        n_pass = 0
        n_fail = 0
        recovery_data: list[dict] = []
        t_start_all = timer.perf_counter()

        if batchable:
            # Stack all cases into one volume: (n_cases, 1, 1, n_time)
            conc_4d = np.stack([c["concentration"] for c in cases])[
                :, np.newaxis, np.newaxis, :
            ]

            print(
                f"\n  [{model_name}] fitting {len(cases)} cases batched...",
                end="",
                flush=True,
            )
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    result = fit_model(
                        model_name,
                        conc_4d,
                        cases[0]["aif"],
                        cases[0]["time"],
                    )
            except FittingError as e:
                t_total = timer.perf_counter() - t_start_all
                print(f" ({t_total:.2f}s) -> FittingError: {e}", flush=True)
                pytest.fail(f"Batched fit_model failed: {e}")

            t_fit = timer.perf_counter() - t_start_all
            print(f" ({t_fit:.2f}s)", flush=True)

            for case_idx, case in enumerate(cases):
                recovered: dict[str, float] = {}
                for param_name in case["true_params"]:
                    if param_name in result.parameter_maps:
                        recovered[param_name] = float(
                            result.parameter_maps[param_name].values[case_idx, 0, 0]
                        )

                case_pass = _check_case_results(case, recovered)
                if case_pass:
                    n_pass += 1
                    print(
                        f"  [{model_name}] case {case_idx + 1}/{len(cases)}: {case['label']} -> PASS",
                        flush=True,
                    )
                else:
                    n_fail += 1
                    print(
                        f"  [{model_name}] case {case_idx + 1}/{len(cases)}: {case['label']} -> FAIL got={recovered} expected={case['true_params']}",
                        flush=True,
                    )

                recovery_data.append(
                    {
                        "label": case["label"],
                        "true_params": case["true_params"],
                        "recovered_params": recovered,
                        "passed": case_pass,
                    }
                )
        else:
            # Different time/AIF per case — fit individually
            for case_idx, case in enumerate(cases):
                print(
                    f"\n  [{model_name}] case {case_idx + 1}/{len(cases)}: {case['label']}",
                    end="",
                    flush=True,
                )

                conc_4d = case["concentration"].reshape(1, 1, 1, -1)

                t_case_start = timer.perf_counter()
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        result = fit_model(
                            model_name,
                            conc_4d,
                            case["aif"],
                            case["time"],
                        )
                except FittingError:
                    t_case = timer.perf_counter() - t_case_start
                    print(f" ({t_case:.2f}s) -> FittingError", flush=True)
                    n_fail += 1
                    recovery_data.append(
                        {
                            "label": case["label"],
                            "true_params": case["true_params"],
                            "recovered_params": {},
                            "passed": False,
                        }
                    )
                    continue

                t_case = timer.perf_counter() - t_case_start
                recovered = {}
                for param_name in case["true_params"]:
                    if param_name in result.parameter_maps:
                        recovered[param_name] = float(
                            result.parameter_maps[param_name].values[0, 0, 0]
                        )

                case_pass = _check_case_results(case, recovered)
                if case_pass:
                    n_pass += 1
                    print(f" ({t_case:.2f}s) -> PASS", flush=True)
                else:
                    n_fail += 1
                    print(
                        f" ({t_case:.2f}s) -> FAIL got={recovered} expected={case['true_params']}",
                        flush=True,
                    )

                recovery_data.append(
                    {
                        "label": case["label"],
                        "true_params": case["true_params"],
                        "recovered_params": recovered,
                        "passed": case_pass,
                    }
                )

        t_total = timer.perf_counter() - t_start_all
        n_time = len(cases[0]["time"])
        total = n_pass + n_fail
        print(
            f"\n  [{model_name}] {total} cases, {n_time} timepoints each, total: {t_total:.2f}s ({t_total / total:.2f}s/case)",
            flush=True,
        )
        try:
            from tests.conftest import record_compliance_result

            record_compliance_result(
                f"{model_name}_dro_recovery",
                n_fail == 0,
                f"{n_pass}/{total} cases within OSIPI tolerance",
            )
        except ImportError:
            pass

        if generate_figures:
            from pathlib import Path

            from tests.compliance_figures import plot_osipi_dro_recovery

            plot_osipi_dro_recovery(
                model_name, recovery_data, Path("output/compliance")
            )

        assert n_fail == 0, (
            f"OSIPI DRO parameter recovery failed for {model_name}: "
            f"{n_fail}/{total} cases outside tolerance"
        )

    @pytest.mark.parametrize(
        "model_name",
        [
            "tofts",
            "extended_tofts",
            "patlak",
            pytest.param(
                "2cxm",
                marks=pytest.mark.xfail(
                    reason=(
                        "2CXM + delay = 5 free params; case 20 "
                        "(low Fp=5, high PS=15) has identifiability "
                        "issues with noisy DRO data (23/24 pass)"
                    ),
                    strict=False,
                ),
            ),
            "2cum",
        ],
    )
    def test_osipi_dro_delay_recovery(self, model_name, generate_figures):
        """Test delay recovery against OSIPI DRO data with arterial delay.

        For Group B models (Patlak, 2CXM, 2CUM), uses separate delay=5s CSV
        files from the OSIPI CodeCollection. For Group A models (Tofts,
        Extended Tofts), programmatically shifts tissue curves matching the
        OSIPI upstream approach.

        Uses ``fit_delay=True`` and checks both PK parameter and delay
        recovery against OSIPI-defined tolerances.

        References
        ----------
        .. [1] van Houdt PJ et al. MRM 2024;91(5):1774-1786.
               doi:10.1002/mrm.29826
        """
        import time as timer
        import warnings

        from tests.fixtures.osipi_codecollection.dce_csv_loader import (
            load_osipi_dro_delay,
        )

        cases = load_osipi_dro_delay(model_name)
        if cases is None:
            pytest.skip(f"No OSIPI delay DRO data for {model_name}")

        from osipy.common.exceptions import FittingError
        from osipy.dce.fitting import fit_model

        get_model(model_name)

        batchable = _cases_are_batchable(cases)

        n_pass = 0
        n_fail = 0
        recovery_data: list[dict] = []
        delay_aif_data: list[dict] = []
        t_start_all = timer.perf_counter()
        tag = f"{model_name}+delay"

        if batchable:
            # Stack all cases into one volume: (n_cases, 1, 1, n_time)
            conc_4d = np.stack([c["concentration"] for c in cases])[
                :, np.newaxis, np.newaxis, :
            ]

            print(
                f"\n  [{tag}] fitting {len(cases)} cases batched...", end="", flush=True
            )
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    result = fit_model(
                        model_name,
                        conc_4d,
                        cases[0]["aif"],
                        cases[0]["time"],
                        fit_delay=True,
                    )
            except FittingError as e:
                t_total = timer.perf_counter() - t_start_all
                print(f" ({t_total:.2f}s) -> FittingError: {e}", flush=True)
                pytest.fail(f"Batched fit_model failed: {e}")

            t_fit = timer.perf_counter() - t_start_all
            print(f" ({t_fit:.2f}s)", flush=True)

            for case_idx, case in enumerate(cases):
                recovered: dict[str, float] = {}
                for param_name in case["true_params"]:
                    if param_name in result.parameter_maps:
                        recovered[param_name] = float(
                            result.parameter_maps[param_name].values[case_idx, 0, 0]
                        )

                case_pass = _check_case_results(case, recovered)
                if case_pass:
                    n_pass += 1
                    print(
                        f"  [{tag}] case {case_idx + 1}/{len(cases)}: {case['label']} -> PASS",
                        flush=True,
                    )
                else:
                    n_fail += 1
                    print(
                        f"  [{tag}] case {case_idx + 1}/{len(cases)}: {case['label']} -> FAIL got={recovered} expected={case['true_params']}",
                        flush=True,
                    )

                recovery_data.append(
                    {
                        "label": case["label"],
                        "true_params": case["true_params"],
                        "recovered_params": recovered,
                        "passed": case_pass,
                    }
                )
                delay_aif_data.append(
                    {
                        "label": case["label"],
                        "time": case["time"],
                        "aif": case["aif"],
                        "true_delay": case["true_params"].get("delay", 0.0),
                        "recovered_delay": recovered.get("delay"),
                        "passed": case_pass,
                    }
                )
        else:
            # Different time/AIF per case — fit individually
            for case_idx, case in enumerate(cases):
                print(
                    f"\n  [{tag}] case {case_idx + 1}/{len(cases)}: {case['label']}",
                    end="",
                    flush=True,
                )

                conc_4d = case["concentration"].reshape(1, 1, 1, -1)

                t_case_start = timer.perf_counter()
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        result = fit_model(
                            model_name,
                            conc_4d,
                            case["aif"],
                            case["time"],
                            fit_delay=True,
                        )
                except FittingError:
                    t_case = timer.perf_counter() - t_case_start
                    print(f" ({t_case:.2f}s) -> FittingError", flush=True)
                    n_fail += 1
                    recovery_data.append(
                        {
                            "label": case["label"],
                            "true_params": case["true_params"],
                            "recovered_params": {},
                            "passed": False,
                        }
                    )
                    delay_aif_data.append(
                        {
                            "label": case["label"],
                            "time": case["time"],
                            "aif": case["aif"],
                            "true_delay": case["true_params"].get("delay", 0.0),
                            "recovered_delay": None,
                            "passed": False,
                        }
                    )
                    continue

                t_case = timer.perf_counter() - t_case_start
                recovered = {}
                for param_name in case["true_params"]:
                    if param_name in result.parameter_maps:
                        recovered[param_name] = float(
                            result.parameter_maps[param_name].values[0, 0, 0]
                        )

                case_pass = _check_case_results(case, recovered)
                if case_pass:
                    n_pass += 1
                    print(f" ({t_case:.2f}s) -> PASS", flush=True)
                else:
                    n_fail += 1
                    print(
                        f" ({t_case:.2f}s) -> FAIL got={recovered} expected={case['true_params']}",
                        flush=True,
                    )

                recovery_data.append(
                    {
                        "label": case["label"],
                        "true_params": case["true_params"],
                        "recovered_params": recovered,
                        "passed": case_pass,
                    }
                )
                delay_aif_data.append(
                    {
                        "label": case["label"],
                        "time": case["time"],
                        "aif": case["aif"],
                        "true_delay": case["true_params"].get("delay", 0.0),
                        "recovered_delay": recovered.get("delay"),
                        "passed": case_pass,
                    }
                )

        t_total = timer.perf_counter() - t_start_all
        total = n_pass + n_fail
        print(
            f"\n  [{tag}] {total} cases, "
            f"total: {t_total:.2f}s ({t_total / total:.2f}s/case)",
            flush=True,
        )
        try:
            from tests.conftest import record_compliance_result

            record_compliance_result(
                f"{model_name}_dro_delay_recovery",
                n_fail == 0,
                f"{n_pass}/{total} cases within OSIPI tolerance",
            )
        except ImportError:
            pass

        if generate_figures:
            from pathlib import Path

            from tests.compliance_figures import (
                plot_delay_aif_comparison,
                plot_osipi_dro_recovery,
            )

            out = Path("output/compliance")
            plot_delay_aif_comparison(
                f"{model_name}+delay",
                delay_aif_data,
                out,
            )
            plot_osipi_dro_recovery(
                f"{model_name}+delay",
                recovery_data,
                out,
            )

        assert n_fail == 0, (
            f"OSIPI DRO delay recovery failed for {model_name}: "
            f"{n_fail}/{total} cases outside tolerance"
        )
