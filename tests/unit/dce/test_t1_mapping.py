"""Unit tests for T1 mapping functions.

Tests for compute_t1_vfa, compute_t1_look_locker, compute_t1_map,
signal models, binding adapters, and Jacobian accuracy.
"""

import numpy as np
import pytest

from osipy.common.dataset import PerfusionDataset
from osipy.common.exceptions import DataValidationError
from osipy.common.models.fittable import FittableModel
from osipy.common.types import DCEAcquisitionParams, Modality
from osipy.dce.t1_mapping.binding import BoundLookLockerModel, BoundSPGRModel
from osipy.dce.t1_mapping.models import LookLockerSignalModel, SPGRSignalModel
from osipy.dce.t1_mapping.vfa import compute_t1_vfa

# ---------------------------------------------------------------------------
# Helpers for synthetic data generation
# ---------------------------------------------------------------------------


def _generate_spgr_signal(
    t1: float, m0: float, flip_angles_deg: list[float], tr: float
) -> np.ndarray:
    """Generate noise-free SPGR signal from known parameters."""
    signals = []
    for fa_deg in flip_angles_deg:
        fa_rad = np.deg2rad(fa_deg)
        e1 = np.exp(-tr / t1)
        s = m0 * np.sin(fa_rad) * (1 - e1) / (1 - np.cos(fa_rad) * e1)
        signals.append(s)
    return np.array(signals)


def _generate_look_locker_signal(
    t1_star: float, a: float, b: float, ti_times: np.ndarray
) -> np.ndarray:
    """Generate noise-free Look-Locker signal from known parameters."""
    return a - b * np.exp(-ti_times / t1_star)


def _make_vfa_dataset(
    t1: float,
    m0: float,
    flip_angles: list[float],
    tr: float,
    shape: tuple[int, int, int] = (3, 3, 2),
    noise_std: float = 0.0,
) -> PerfusionDataset:
    """Create a VFA dataset with uniform T1/M0 and optional noise."""
    signals = _generate_spgr_signal(t1, m0, flip_angles, tr)
    n_fa = len(flip_angles)
    nx, ny, nz = shape

    data = np.zeros((nx, ny, nz, n_fa))
    for i in range(n_fa):
        data[:, :, :, i] = signals[i]

    if noise_std > 0:
        rng = np.random.default_rng(42)
        data += rng.normal(0, noise_std, data.shape)
        data = np.maximum(data, 0)  # Signal is non-negative

    return PerfusionDataset(
        data=data,
        affine=np.eye(4),
        modality=Modality.DCE,
        time_points=np.arange(n_fa, dtype=np.float64),
        acquisition_params=DCEAcquisitionParams(tr=tr, flip_angles=flip_angles),
    )


def _make_ll_dataset(
    t1_star: float,
    a: float,
    b: float,
    ti_times: np.ndarray,
    shape: tuple[int, int, int] = (3, 3, 2),
    noise_std: float = 0.0,
) -> PerfusionDataset:
    """Create a Look-Locker dataset with uniform parameters and optional noise."""
    signals = _generate_look_locker_signal(t1_star, a, b, ti_times)
    n_ti = len(ti_times)
    nx, ny, nz = shape

    data = np.zeros((nx, ny, nz, n_ti))
    for i in range(n_ti):
        data[:, :, :, i] = signals[i]

    if noise_std > 0:
        rng = np.random.default_rng(42)
        data += rng.normal(0, noise_std, data.shape)

    return PerfusionDataset(
        data=data,
        affine=np.eye(4),
        modality=Modality.DCE,
        time_points=ti_times / 1000.0,  # ms -> s
        acquisition_params=DCEAcquisitionParams(tr=3000.0, flip_angles=[8.0]),
    )


# ---------------------------------------------------------------------------
# Basic dataset and parameter tests (preserved from original)
# ---------------------------------------------------------------------------


class TestT1MappingBasic:
    """Basic tests for T1 mapping utilities."""

    def test_dataset_creation_for_vfa(self) -> None:
        """Test creating a dataset for VFA T1 mapping."""
        flip_angles = [2.0, 5.0, 10.0, 15.0]
        n_fa = len(flip_angles)

        data = np.random.rand(4, 4, 2, n_fa) * 100
        affine = np.eye(4)
        time = np.arange(n_fa, dtype=float)

        params = DCEAcquisitionParams(tr=5.0, flip_angles=flip_angles)

        dataset = PerfusionDataset(
            data=data,
            affine=affine,
            modality=Modality.DCE,
            time_points=time,
            acquisition_params=params,
        )

        assert dataset.shape == (4, 4, 2, n_fa)
        assert dataset.is_dynamic
        assert dataset.modality == Modality.DCE

    def test_dataset_creation_for_look_locker(self) -> None:
        """Test creating a dataset for Look-Locker T1 mapping."""
        n_ti = 10
        data = np.random.rand(4, 4, 2, n_ti) * 100

        affine = np.eye(4)
        ti_times = np.linspace(50, 2000, n_ti)

        params = DCEAcquisitionParams(tr=3000.0, flip_angles=[8.0])

        dataset = PerfusionDataset(
            data=data,
            affine=affine,
            modality=Modality.DCE,
            time_points=ti_times,
            acquisition_params=params,
        )

        assert dataset.shape == (4, 4, 2, n_ti)
        assert len(dataset.time_points) == n_ti


class TestDCEAcquisitionParamsForT1:
    """Tests for DCE acquisition parameters used in T1 mapping."""

    def test_vfa_params(self) -> None:
        params = DCEAcquisitionParams(
            tr=4.5,
            flip_angles=[2.0, 5.0, 10.0, 15.0, 20.0],
        )
        assert params.tr == 4.5
        assert len(params.flip_angles) == 5

    def test_look_locker_params(self) -> None:
        params = DCEAcquisitionParams(tr=3000.0, flip_angles=[8.0], te=2.0)
        assert params.tr == 3000.0
        assert len(params.flip_angles) == 1
        assert params.te == 2.0


class TestFlipAnglesNormalization:
    """Tests for DCEAcquisitionParams.__post_init__ flip_angles handling."""

    def test_accepts_list(self) -> None:
        p = DCEAcquisitionParams(tr=5.0, flip_angles=[2.0, 5.0, 10.0, 15.0])
        assert p.flip_angles == [2.0, 5.0, 10.0, 15.0]
        assert isinstance(p.flip_angles, list)

    def test_accepts_numpy_array(self) -> None:
        p = DCEAcquisitionParams(tr=5.0, flip_angles=np.array([2.0, 5.0, 10.0, 15.0]))
        assert p.flip_angles == [2.0, 5.0, 10.0, 15.0]
        assert isinstance(p.flip_angles, list)

    def test_accepts_tuple(self) -> None:
        p = DCEAcquisitionParams(tr=5.0, flip_angles=(2, 5, 10, 15))
        assert p.flip_angles == [2.0, 5.0, 10.0, 15.0]

    def test_accepts_int_list(self) -> None:
        p = DCEAcquisitionParams(tr=5.0, flip_angles=[2, 5, 10, 15])
        assert all(isinstance(x, float) for x in p.flip_angles)

    def test_allows_empty_list_for_fixed_t1_workflows(self) -> None:
        p = DCEAcquisitionParams(tr=5.0, flip_angles=[], t1_assumed=1400.0)
        assert p.flip_angles == []

    def test_allows_empty_array_for_fixed_t1_workflows(self) -> None:
        p = DCEAcquisitionParams(tr=5.0, flip_angles=np.array([]), t1_assumed=1400.0)
        assert p.flip_angles == []

    def test_rejects_none(self) -> None:
        with pytest.raises(ValueError, match="cannot be None"):
            DCEAcquisitionParams(tr=5.0, flip_angles=None)

    def test_rejects_scalar(self) -> None:
        with pytest.raises(ValueError, match="single scalar"):
            DCEAcquisitionParams(tr=5.0, flip_angles=5.0)

    def test_rejects_non_numeric(self) -> None:
        with pytest.raises(ValueError):
            DCEAcquisitionParams(tr=5.0, flip_angles=["a", "b", "c"])

    def test_rejects_nan(self) -> None:
        with pytest.raises(ValueError, match="NaN or infinite"):
            DCEAcquisitionParams(tr=5.0, flip_angles=[2.0, float("nan"), 10.0])

    def test_rejects_inf(self) -> None:
        with pytest.raises(ValueError, match="NaN or infinite"):
            DCEAcquisitionParams(tr=5.0, flip_angles=[2.0, float("inf"), 10.0])

    def test_rejects_2d_array(self) -> None:
        with pytest.raises(ValueError, match="1D sequence"):
            DCEAcquisitionParams(
                tr=5.0, flip_angles=np.array([[2.0, 5.0], [10.0, 15.0]])
            )


class TestSyntheticT1Data:
    """Tests for generating synthetic T1 data."""

    def test_generate_vfa_signal(self) -> None:
        signals = _generate_spgr_signal(1000.0, 100.0, [2.0, 5.0, 10.0, 15.0], 5.0)
        assert np.all(signals > 0)
        assert signals[0] < signals[1]

    def test_generate_look_locker_signal(self) -> None:
        ti_values = np.linspace(100, 2000, 10)
        signals = _generate_look_locker_signal(1000.0, 100.0, 200.0, ti_values)
        # Short TI: A - B*exp(-TI/T1*) < 0 when B > A
        assert signals[0] < signals[-1]


# ---------------------------------------------------------------------------
# Signal model tests
# ---------------------------------------------------------------------------


class TestSPGRSignalModel:
    """Tests for SPGRSignalModel."""

    def test_properties(self) -> None:
        model = SPGRSignalModel()
        assert model.name == "SPGR"
        assert model.parameters == ["T1", "M0"]
        assert model.parameter_units == {"T1": "ms", "M0": "a.u."}
        assert "Deoni" in model.reference

    def test_bounds(self) -> None:
        model = SPGRSignalModel()
        bounds = model.get_bounds()
        assert "T1" in bounds
        assert "M0" in bounds
        assert bounds["T1"][0] > 0
        assert bounds["M0"][0] >= 0


class TestLookLockerSignalModel:
    """Tests for LookLockerSignalModel."""

    def test_properties(self) -> None:
        model = LookLockerSignalModel()
        assert model.name == "Look-Locker"
        assert model.parameters == ["T1_star", "A", "B"]
        assert model.parameter_units == {"T1_star": "ms", "A": "a.u.", "B": "a.u."}
        assert "Look" in model.reference

    def test_bounds(self) -> None:
        model = LookLockerSignalModel()
        bounds = model.get_bounds()
        assert "T1_star" in bounds
        assert bounds["T1_star"][0] > 0


# ---------------------------------------------------------------------------
# Bound model / FittableModel protocol conformance tests
# ---------------------------------------------------------------------------


class TestBoundSPGRModelProtocol:
    """Test that BoundSPGRModel conforms to FittableModel protocol."""

    @pytest.fixture()
    def bound_model(self) -> BoundSPGRModel:
        model = SPGRSignalModel()
        flip_angles_rad = np.deg2rad([2.0, 5.0, 10.0, 15.0, 20.0])
        return BoundSPGRModel(model, flip_angles_rad, tr=5.0)

    def test_is_fittable_model(self, bound_model: BoundSPGRModel) -> None:
        assert isinstance(bound_model, FittableModel)

    def test_predict_array_batch(self, bound_model: BoundSPGRModel) -> None:
        n_voxels = 4
        params = np.array([[1000.0] * n_voxels, [100.0] * n_voxels])
        pred = bound_model.predict_array_batch(params, np)
        assert pred.shape == (5, n_voxels)
        assert np.all(np.isfinite(pred))
        assert np.all(pred > 0)

    def test_get_initial_guess_batch(self, bound_model: BoundSPGRModel) -> None:
        n_voxels = 4
        observed = np.random.rand(5, n_voxels) * 100
        guess = bound_model.get_initial_guess_batch(observed, np)
        assert guess.shape == (2, n_voxels)  # [T1, M0]
        assert np.all(np.isfinite(guess))

    def test_compute_jacobian_batch(self, bound_model: BoundSPGRModel) -> None:
        n_voxels = 4
        params = np.array([[1000.0] * n_voxels, [100.0] * n_voxels])
        pred = bound_model.predict_array_batch(params, np)
        jac = bound_model.compute_jacobian_batch(params, pred, np)
        assert jac is not None
        assert jac.shape == (2, 5, n_voxels)  # (n_params, n_fa, n_voxels)

    def test_ensure_device(self, bound_model: BoundSPGRModel) -> None:
        bound_model.ensure_device(np)  # Should not raise


class TestBoundLookLockerModelProtocol:
    """Test that BoundLookLockerModel conforms to FittableModel protocol."""

    @pytest.fixture()
    def bound_model(self) -> BoundLookLockerModel:
        model = LookLockerSignalModel()
        ti_times = np.linspace(50, 2000, 10)
        return BoundLookLockerModel(model, ti_times)

    def test_is_fittable_model(self, bound_model: BoundLookLockerModel) -> None:
        assert isinstance(bound_model, FittableModel)

    def test_predict_array_batch(self, bound_model: BoundLookLockerModel) -> None:
        n_voxels = 4
        params = np.array(
            [
                [800.0] * n_voxels,  # T1_star
                [100.0] * n_voxels,  # A
                [200.0] * n_voxels,  # B
            ]
        )
        pred = bound_model.predict_array_batch(params, np)
        assert pred.shape == (10, n_voxels)
        assert np.all(np.isfinite(pred))

    def test_get_initial_guess_batch(self, bound_model: BoundLookLockerModel) -> None:
        n_voxels = 4
        observed = np.random.rand(10, n_voxels) * 100
        guess = bound_model.get_initial_guess_batch(observed, np)
        assert guess.shape == (3, n_voxels)  # [T1_star, A, B]
        assert np.all(np.isfinite(guess))

    def test_compute_jacobian_batch(self, bound_model: BoundLookLockerModel) -> None:
        n_voxels = 4
        params = np.array(
            [
                [800.0] * n_voxels,
                [100.0] * n_voxels,
                [200.0] * n_voxels,
            ]
        )
        pred = bound_model.predict_array_batch(params, np)
        jac = bound_model.compute_jacobian_batch(params, pred, np)
        assert jac is not None
        assert jac.shape == (3, 10, n_voxels)  # (n_params, n_ti, n_voxels)

    def test_ensure_device(self, bound_model: BoundLookLockerModel) -> None:
        bound_model.ensure_device(np)  # Should not raise


# ---------------------------------------------------------------------------
# Jacobian accuracy tests (analytical vs numerical finite differences)
# ---------------------------------------------------------------------------


class TestJacobianAccuracy:
    """Compare analytical Jacobians against numerical finite differences."""

    def test_spgr_jacobian_accuracy(self) -> None:
        """SPGR analytical Jacobian matches numerical within tolerance."""
        model = SPGRSignalModel()
        flip_angles_rad = np.deg2rad([2.0, 5.0, 10.0, 15.0, 20.0])
        bound = BoundSPGRModel(model, flip_angles_rad, tr=5.0)

        params = np.array(
            [
                [500.0, 1000.0, 2000.0],  # T1
                [80.0, 100.0, 150.0],  # M0
            ]
        )

        pred = bound.predict_array_batch(params, np)
        analytical_jac = bound.compute_jacobian_batch(params, pred, np)

        # Numerical Jacobian via finite differences
        eps = 1e-5
        numerical_jac = np.zeros_like(analytical_jac)
        for p in range(params.shape[0]):
            params_plus = params.copy()
            params_plus[p, :] += eps
            pred_plus = bound.predict_array_batch(params_plus, np)
            numerical_jac[p, :, :] = (pred_plus - pred) / eps

        np.testing.assert_allclose(
            analytical_jac,
            numerical_jac,
            rtol=1e-3,
            atol=1e-6,
            err_msg="SPGR analytical Jacobian doesn't match numerical",
        )

    def test_look_locker_jacobian_accuracy(self) -> None:
        """Look-Locker analytical Jacobian matches numerical within tolerance."""
        model = LookLockerSignalModel()
        ti_times = np.linspace(50, 2000, 10)
        bound = BoundLookLockerModel(model, ti_times)

        params = np.array(
            [
                [500.0, 800.0, 1500.0],  # T1_star
                [80.0, 100.0, 120.0],  # A
                [160.0, 200.0, 240.0],  # B
            ]
        )

        pred = bound.predict_array_batch(params, np)
        analytical_jac = bound.compute_jacobian_batch(params, pred, np)

        # Numerical Jacobian via finite differences
        eps = 1e-5
        numerical_jac = np.zeros_like(analytical_jac)
        for p in range(params.shape[0]):
            params_plus = params.copy()
            params_plus[p, :] += eps
            pred_plus = bound.predict_array_batch(params_plus, np)
            numerical_jac[p, :, :] = (pred_plus - pred) / eps

        np.testing.assert_allclose(
            analytical_jac,
            numerical_jac,
            rtol=1e-3,
            atol=1e-6,
            err_msg="Look-Locker analytical Jacobian doesn't match numerical",
        )


# ---------------------------------------------------------------------------
# End-to-end fitting tests (synthetic data -> fit -> recover parameters)
# ---------------------------------------------------------------------------


class TestVFAFitting:
    """End-to-end VFA T1 fitting tests."""

    def test_vfa_linear_recovers_t1(self) -> None:
        """VFA linear fit recovers known T1 from synthetic data."""
        from osipy.dce.t1_mapping.vfa import compute_t1_vfa

        t1_true = 1000.0
        m0_true = 100.0
        flip_angles = [2.0, 5.0, 10.0, 15.0, 20.0]
        tr = 5.0

        dataset = _make_vfa_dataset(t1_true, m0_true, flip_angles, tr)
        result = compute_t1_vfa(dataset, method="linear")

        # Linear fit on noise-free data should be very accurate
        t1_fitted = result.t1_map.values
        valid = result.quality_mask
        assert np.any(valid), "No valid voxels fitted"

        t1_mean = np.nanmean(t1_fitted[valid])
        np.testing.assert_allclose(
            t1_mean, t1_true, rtol=0.01, err_msg="VFA linear T1 recovery failed"
        )

    def test_vfa_nonlinear_recovers_t1(self) -> None:
        """VFA nonlinear fit recovers known T1 from synthetic data."""
        from osipy.dce.t1_mapping.vfa import compute_t1_vfa

        t1_true = 1000.0
        m0_true = 100.0
        flip_angles = [2.0, 5.0, 10.0, 15.0, 20.0]
        tr = 5.0

        dataset = _make_vfa_dataset(t1_true, m0_true, flip_angles, tr)
        result = compute_t1_vfa(dataset, method="nonlinear")

        t1_fitted = result.t1_map.values
        valid = result.quality_mask
        assert np.any(valid), "No valid voxels fitted"

        t1_mean = np.nanmean(t1_fitted[valid])
        np.testing.assert_allclose(
            t1_mean, t1_true, rtol=0.10, err_msg="VFA nonlinear T1 recovery failed"
        )

    def test_vfa_array_interface(self) -> None:
        """VFA fitting works with individual arrays (no dataset)."""
        from osipy.dce.t1_mapping.vfa import compute_t1_vfa

        t1_true = 800.0
        m0_true = 150.0
        flip_angles = [2.0, 5.0, 10.0, 15.0, 20.0]
        tr = 5.0

        signals = _generate_spgr_signal(t1_true, m0_true, flip_angles, tr)
        data_4d = np.zeros((2, 2, 1, len(flip_angles)))
        for i in range(len(flip_angles)):
            data_4d[:, :, :, i] = signals[i]

        result = compute_t1_vfa(
            signal=data_4d, flip_angles=flip_angles, tr=tr, method="linear"
        )

        t1_mean = np.nanmean(result.t1_map.values[result.quality_mask])
        np.testing.assert_allclose(t1_mean, t1_true, rtol=0.01)

    def test_vfa_invalid_method_raises(self) -> None:
        """Unknown VFA method raises DataValidationError."""
        from osipy.dce.t1_mapping.vfa import compute_t1_vfa

        dataset = _make_vfa_dataset(1000.0, 100.0, [2.0, 5.0, 10.0], 5.0)
        with pytest.raises(DataValidationError, match="Unknown VFA method"):
            compute_t1_vfa(dataset, method="bogus")

    def test_vfa_rejects_empty_flip_angles(self) -> None:
        """compute_t1_vfa rejects empty flip_angles even though the
        dataclass itself allows it (for fixed-T1/t1_assumed workflows)."""
        dataset = _make_vfa_dataset(1000.0, 100.0, [2.0, 5.0, 10.0], 5.0)
        dataset.acquisition_params.flip_angles = []

        with pytest.raises(DataValidationError, match="at least one flip angle"):
            compute_t1_vfa(dataset, method="linear")


class TestLookLockerFitting:
    """End-to-end Look-Locker T1 fitting tests."""

    def test_look_locker_recovers_t1(self) -> None:
        """Look-Locker fit recovers known T1 from synthetic data."""
        from osipy.dce.t1_mapping.look_locker import compute_t1_look_locker

        # True parameters
        t1_star_true = 800.0
        a_true = 100.0
        b_true = 200.0  # B/A = 2, so T1 = T1* * (2 - 1) = 800 ms
        t1_true = t1_star_true * (b_true / a_true - 1)  # 800 ms
        ti_times = np.linspace(50, 3000, 20)

        dataset = _make_ll_dataset(t1_star_true, a_true, b_true, ti_times)
        result = compute_t1_look_locker(dataset, ti_times=ti_times)

        t1_fitted = result.t1_map.values
        valid = result.quality_mask
        assert np.any(valid), "No valid voxels fitted"

        t1_mean = np.nanmean(t1_fitted[valid])
        np.testing.assert_allclose(
            t1_mean, t1_true, rtol=0.10, err_msg="Look-Locker T1 recovery failed"
        )

        # Invalid voxels (where B/A <= 1) should produce NaN in the T1 map
        invalid = ~valid
        if np.any(invalid):
            assert np.all(np.isnan(t1_fitted[invalid]))

    def test_look_locker_uses_time_points_if_no_ti(self) -> None:
        """Look-Locker extracts TI from dataset.time_points when ti_times=None."""
        from osipy.dce.t1_mapping.look_locker import compute_t1_look_locker

        t1_star_true = 800.0
        a_true = 100.0
        b_true = 200.0
        ti_times = np.linspace(50, 3000, 15)

        dataset = _make_ll_dataset(t1_star_true, a_true, b_true, ti_times)

        # time_points stored in seconds, Look-Locker converts to ms internally
        result = compute_t1_look_locker(dataset, ti_times=None)
        assert result.t1_map is not None
