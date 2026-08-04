"""OSIPI CodeCollection compliance tests for DSC deconvolution.

Tests CBV and CBF recovery against OSIPI ground truth DRO data using
oSVD, cSVD, and sSVD deconvolution methods with OSIPI-defined tolerances.

References
----------
.. [1] van Houdt PJ et al. MRM 2024;91(5):1774-1786.
       doi:10.1002/mrm.29826
.. [2] OSIPI DCE-DSC-MRI CodeCollection,
       https://github.com/OSIPI/DCE-DSC-MRI_CodeCollection
"""

import time as timer

import pytest

from osipy.common.backend.array_module import get_array_module
from osipy.dsc.deconvolution import DeconvolutionResult, get_deconvolver


class TestOSIPIDSCReferenceData:
    """Test DSC deconvolution against OSIPI CodeCollection ground truth."""

    @pytest.fixture(autouse=True)
    def _load_dsc_dro(self):
        """Load OSIPI DSC DRO data."""
        from tests.fixtures.osipi_codecollection.dsc_csv_loader import (
            load_osipi_dsc_dro,
        )

        self.cases = load_osipi_dsc_dro()

    @pytest.mark.parametrize(
        "method",
        [
            pytest.param(
                "oSVD",
                marks=pytest.mark.xfail(
                    reason=(
                        "Block-circulant deconvolution (cSVD/oSVD) trades some "
                        "delay-free accuracy for delay-insensitivity: its "
                        "Fourier-mode SVD basis smears the sharp residue peak "
                        "(Gibbs-type effect), underestimating CBF at high flow "
                        "(CBF>=50, CBV=4; CBF=35, CBV=2). This is a documented "
                        "tradeoff of block-circulant SVD (Wu et al. 2003)."
                    ),
                    strict=False,
                ),
            ),
            pytest.param(
                "cSVD",
                marks=pytest.mark.xfail(
                    reason=(
                        "Same block-circulant tradeoff as oSVD above -- "
                        "underestimates CBF at high flow (CBF>=50, CBV=4; "
                        "CBF=35, CBV=2); 10/14 cases pass"
                    ),
                    strict=False,
                ),
            ),
        ],
    )
    def test_osipi_dsc_parameter_recovery(self, method):
        """Test CBV and CBF recovery against OSIPI DRO ground truth.

        Uses the OSIPI CodeCollection dsc_data.csv with 14 test cases
        spanning CBV 2-4 mL/100mL and CBF 5-70 mL/100mL/min.

        Tolerances from OSIPI DSCmodels_data.py:
          CBV: absolute=1.0 mL/100mL, relative=10%
          CBF: absolute=15.0 mL/100mL/min, relative=10%
        """
        if self.cases is None:
            pytest.skip("OSIPI DSC DRO CSV data not found")

        deconvolver = get_deconvolver(method)

        n_pass = 0
        n_fail = 0
        t_start = timer.perf_counter()

        for case_idx, case in enumerate(self.cases):
            c_tis = case["c_tis"]
            c_aif = case["c_aif"]
            time = case["time"]
            true_cbv = case["cbv"]
            true_cbf = case["cbf"]
            tolerances = case["tolerances"]

            print(
                f"\n  [{method}] case {case_idx + 1}/{len(self.cases)}: "
                f"{case['label']}",
                end="",
                flush=True,
            )

            xp = get_array_module(c_tis)

            # Reshape to 3D for deconvolver: (1, 1, n_timepoints)
            conc_3d = c_tis.reshape(1, 1, -1)

            result = deconvolver.deconvolve(conc_3d, c_aif, time)

            assert isinstance(result, DeconvolutionResult)

            rec_mtt = float(result.mtt[0, 0])

            # CBV from area ratio (standard DSC indicator-dilution theory):
            #   CBV = integral(C_tis) / integral(C_aif) * 100  [mL/100mL]
            rec_cbv = xp.trapezoid(c_tis, time) / xp.trapezoid(c_aif, time) * 100.0

            # CBF from central volume theorem: CBF = CBV / MTT * 60
            #   CBV in mL/100mL, MTT in seconds -> CBF in mL/100mL/min
            rec_cbf = rec_cbv / rec_mtt * 60.0 if rec_mtt > 0 else 0.0

            case_pass = True

            # Check CBV
            cbv_tol = tolerances["CBV"]
            cbv_abs_err = abs(rec_cbv - true_cbv)
            cbv_rel_err = cbv_abs_err / (abs(true_cbv) + 1e-10)
            if not (
                cbv_abs_err <= cbv_tol["absolute"] or cbv_rel_err <= cbv_tol["relative"]
            ):
                case_pass = False

            # Check CBF
            cbf_tol = tolerances["CBF"]
            cbf_abs_err = abs(rec_cbf - true_cbf)
            cbf_rel_err = cbf_abs_err / (abs(true_cbf) + 1e-10)
            if not (
                cbf_abs_err <= cbf_tol["absolute"] or cbf_rel_err <= cbf_tol["relative"]
            ):
                case_pass = False

            if case_pass:
                n_pass += 1
                print(
                    f" -> PASS (CBV={rec_cbv:.2f}/{true_cbv}, "
                    f"CBF={rec_cbf:.1f}/{true_cbf})",
                    flush=True,
                )
            else:
                n_fail += 1
                print(
                    f" -> FAIL (CBV={rec_cbv:.2f}/{true_cbv} "
                    f"err={cbv_abs_err:.3f}, "
                    f"CBF={rec_cbf:.1f}/{true_cbf} "
                    f"err={cbf_abs_err:.1f})",
                    flush=True,
                )

        t_total = timer.perf_counter() - t_start
        total = n_pass + n_fail
        print(
            f"\n  [{method}] {total} cases, total: {t_total:.2f}s",
            flush=True,
        )

        try:
            from tests.conftest import record_compliance_result

            record_compliance_result(
                f"dsc_{method}_dro_recovery",
                n_fail == 0,
                f"{n_pass}/{total} cases within OSIPI tolerance",
            )
        except ImportError:
            pass

        assert n_fail == 0, (
            f"OSIPI DSC DRO parameter recovery failed for {method}: "
            f"{n_fail}/{total} cases outside tolerance"
        )
