"""Tests for L-curve selection of the Tikhonov parameter, on the OSIPI DSC DRO."""

import numpy as np
import pytest

from osipy.dsc.deconvolution.lcurve import select_lambda, solve_tikhonov_lcurve
from osipy.dsc.deconvolution.svd import _build_toeplitz_matrix_xp
from tests.fixtures.osipi_codecollection.dsc_csv_loader import load_osipi_dsc_dro


@pytest.fixture(scope="module")
def dro():
    """All 14 DRO cases share one AIF, so one SVD serves every tissue curve."""
    cases = load_osipi_dsc_dro()
    if cases is None:
        pytest.skip("OSIPI DSC DRO data not found")
    t = cases[0]["time"]
    A = np.asarray(_build_toeplitz_matrix_xp(cases[0]["c_aif"], t.size, np))
    A = A * float(t[1] - t[0])
    U, S, Vh = np.linalg.svd(A)
    C = np.stack([c["c_tis"] for c in cases], axis=1)
    return {"cases": cases, "A": A, "U": U, "S": S, "Vh": Vh, "C": C}


def add_noise(y, frac, seed=0):
    return y + np.random.default_rng(seed).normal(0, frac * y.max(), y.shape)


class TestSelectLambda:
    def test_more_noise_selects_more_regularization(self, dro):
        U, S, y = dro["U"], dro["S"], dro["C"][:, 3]
        Y = np.stack(
            [add_noise(y, f) if f else y for f in (0, 0.02, 0.05, 0.1, 0.2)], axis=1
        )
        lam = select_lambda(S, U.T @ Y, np)
        assert np.all(np.diff(lam) > 0), lam

    def test_scaling(self, dro):
        """Lambda ignores concentration units and scales with the matrix (dt, AIF units)."""
        S, UtC = dro["S"], dro["U"].T @ dro["C"]
        lam = select_lambda(S, UtC, np)
        assert np.allclose(select_lambda(S, 1000 * UtC, np), lam, rtol=1e-9)
        assert np.allclose(select_lambda(100 * S, 100 * UtC, np), 100 * lam, rtol=1e-6)

    def test_voxels_are_independent(self, dro):
        S, UtC = dro["S"], dro["U"].T @ dro["C"]
        batch = select_lambda(S, UtC, np)
        alone = [
            select_lambda(S, UtC[:, j : j + 1], np)[0] for j in range(UtC.shape[1])
        ]
        assert np.allclose(batch, alone, rtol=1e-12)

    def test_zero_voxel(self, dro):
        """Background voxels are all zero: no warnings, zero residue, others unchanged."""
        U, S, Vh, C = dro["U"], dro["S"], dro["Vh"], dro["C"]
        R, _ = solve_tikhonov_lcurve(U, S, Vh, C, np)
        R0, _ = solve_tikhonov_lcurve(
            U, S, Vh, np.column_stack([C, np.zeros(len(C))]), np
        )
        assert np.all(R0[:, -1] == 0)
        assert np.allclose(R0[:, :-1], R)


class TestOnDRO:
    def test_cbf_within_osipi_tolerance_and_beats_default_tsvd(self, dro):
        U, S, Vh, C, cases = dro["U"], dro["S"], dro["Vh"], dro["C"], dro["cases"]
        true = np.array([c["cbf"] for c in cases])
        tol = cases[0]["tolerances"]["CBF"]

        cbf = solve_tikhonov_lcurve(U, S, Vh, C, np)[0].max(axis=0) * 6000
        err = np.abs(cbf - true)
        assert np.all((err <= tol["absolute"]) | (err / true <= tol["relative"])), cbf

        f = np.where(0.2 * S.max() < S, 1 / S, 0.0)
        tsvd = (Vh.T @ (f[:, None] * (U.T @ C))).max(axis=0) * 6000
        assert err.mean() < np.abs(tsvd - true).mean()

    def test_high_noise_takes_the_corner_not_the_small_lambda_bend(self, dro):
        """At 20% noise the global curvature maximum gives CBF ~745 on this case."""
        U, S, Vh, c = dro["U"], dro["S"], dro["Vh"], dro["cases"][3]
        y = add_noise(c["c_tis"], 0.2)
        cbf = solve_tikhonov_lcurve(U, S, Vh, y[:, None], np)[0].max() * 6000
        assert abs(cbf - c["cbf"]) <= c["tolerances"]["CBF"]["absolute"], cbf

    def test_residual_matches_added_noise(self, dro):
        """At the corner the fit stops at the noise floor (discrepancy principle)."""
        A, U, S, Vh, y = dro["A"], dro["U"], dro["S"], dro["Vh"], dro["C"][:, 3]
        noisy = add_noise(y, 0.1)
        R, _ = solve_tikhonov_lcurve(U, S, Vh, noisy[:, None], np)
        ratio = np.linalg.norm(A @ R[:, 0] - noisy) / np.linalg.norm(noisy - y)
        assert 0.7 < ratio < 1.3, ratio
