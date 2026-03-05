"""Integration tests for ASL pipeline.

Tests the complete ASL analysis workflow from labeling difference
computation through CBF quantification and ATT estimation.

User Story 3: MRI physicist processes pCASL data at multiple PLDs
to generate CBF maps in ml/100g/min with ATT estimates.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestASLPipelineIntegration:
    """Integration tests for ASL pipeline."""

    @pytest.fixture
    def synthetic_asl_data(self) -> dict:
        """Create synthetic ASL data for integration testing."""
        np.random.seed(42)

        # Dimensions
        nx, ny, nz = 16, 16, 4

        # Create ground truth CBF and ATT maps
        cbf_true = np.random.uniform(40, 80, (nx, ny, nz))  # ml/100g/min
        att_true = np.random.uniform(800, 1500, (nx, ny, nz))  # ms

        # Acquisition parameters for pCASL
        pld = 1800.0  # ms
        label_duration = 1800.0  # ms
        t1_blood = 1650.0  # ms at 3T
        labeling_efficiency = 0.85
        partition_coefficient = 0.9

        # M0 image
        m0 = np.random.uniform(900, 1100, (nx, ny, nz))

        # Generate difference magnetization (ΔM)
        # Using simplified Buxton model
        pld_s = pld / 1000.0
        tau_s = label_duration / 1000.0
        t1b_s = t1_blood / 1000.0

        # ΔM = 2 * M0 * CBF * T1b * α * (1 - exp(-τ/T1b)) * exp(-PLD/T1b) / λ / 6000
        delta_m = (
            2
            * m0
            * cbf_true
            * t1b_s
            * labeling_efficiency
            * (1 - np.exp(-tau_s / t1b_s))
            * np.exp(-pld_s / t1b_s)
            / partition_coefficient
            / 6000.0
        )

        # Add noise
        delta_m += np.random.randn(nx, ny, nz) * 2

        # Create multi-PLD data
        plds = np.array([500, 1000, 1500, 2000, 2500])  # ms
        multi_pld_delta_m = np.zeros((nx, ny, nz, len(plds)))

        for idx, pld_val in enumerate(plds):
            pld_s = pld_val / 1000.0
            dm = (
                2
                * m0
                * cbf_true
                * t1b_s
                * labeling_efficiency
                * (1 - np.exp(-tau_s / t1b_s))
                * np.exp(-pld_s / t1b_s)
                / partition_coefficient
                / 6000.0
            )
            # Modify by ATT (reduced signal if ATT > PLD)
            att_factor = np.clip((pld_val - att_true) / 500.0, 0, 1)
            dm *= att_factor
            multi_pld_delta_m[..., idx] = dm + np.random.randn(nx, ny, nz) * 1

        mask = np.ones((nx, ny, nz), dtype=bool)

        return {
            "delta_m": delta_m,
            "multi_pld_delta_m": multi_pld_delta_m,
            "plds": plds,
            "m0": m0,
            "cbf_true": cbf_true,
            "att_true": att_true,
            "mask": mask,
            "pld": pld,
            "label_duration": label_duration,
            "t1_blood": t1_blood,
            "labeling_efficiency": labeling_efficiency,
            "shape": (nx, ny, nz),
        }

    def test_pcasl_difference_computation(self) -> None:
        """Test pCASL label-control difference."""
        from osipy.asl.quantification import compute_pcasl_difference

        np.random.seed(42)

        # Create label and control volumes
        shape = (16, 16, 4, 30)  # 30 dynamics
        control = np.random.uniform(900, 1100, shape)
        label = control - np.random.uniform(5, 20, shape)  # Label has lower signal

        delta_m = compute_pcasl_difference(
            label=label,
            control=control,
        )

        # The function averages across dynamics, returning 3D
        assert delta_m.shape == shape[:-1]
        # ΔM should be positive (control - label for pCASL)
        assert np.mean(delta_m) > 0

    def test_pasl_difference_computation(self) -> None:
        """Test PASL label-control difference."""
        from osipy.asl.quantification import compute_pcasl_difference

        np.random.seed(42)

        shape = (16, 16, 4, 20)
        control = np.random.uniform(900, 1100, shape)
        label = control - np.random.uniform(5, 20, shape)

        delta_m = compute_pcasl_difference(
            label=label,
            control=control,
        )

        assert delta_m is not None
        assert delta_m.shape == shape[:-1]

    def test_cbf_quantification_single_pld(self, synthetic_asl_data: dict) -> None:
        """Test single-PLD CBF quantification."""
        from osipy.asl.labeling import LabelingScheme
        from osipy.asl.quantification import quantify_cbf
        from osipy.asl.quantification.cbf import ASLQuantificationParams

        params = ASLQuantificationParams(
            labeling_scheme=LabelingScheme.PCASL,
            pld=synthetic_asl_data["pld"],
            label_duration=synthetic_asl_data["label_duration"],
            t1_blood=synthetic_asl_data["t1_blood"],
            labeling_efficiency=synthetic_asl_data["labeling_efficiency"],
        )

        result = quantify_cbf(
            delta_m=synthetic_asl_data["delta_m"],
            m0=synthetic_asl_data["m0"],
            params=params,
            mask=synthetic_asl_data["mask"],
        )

        assert result.cbf_map is not None
        assert result.cbf_map.units == "mL/100g/min"

        # CBF should be in physiological range
        cbf_values = result.cbf_map.values[synthetic_asl_data["mask"]]
        assert np.mean(cbf_values) > 0, "Mean CBF should be positive"
        assert np.mean(cbf_values) < 150, "Mean CBF should be < 150 mL/100g/min"

    def test_cbf_quantification_multi_pld(self, synthetic_asl_data: dict) -> None:
        """Test multi-PLD CBF quantification with ATT estimation."""
        from osipy.asl.quantification.multi_pld import (
            MultiPLDParams,
            quantify_multi_pld,
        )

        params = MultiPLDParams(
            plds=synthetic_asl_data["plds"],
            label_duration=synthetic_asl_data["label_duration"],
            t1_blood=synthetic_asl_data["t1_blood"],
            labeling_efficiency=synthetic_asl_data["labeling_efficiency"],
        )

        # Use smaller subset for speed
        delta_m = synthetic_asl_data["multi_pld_delta_m"][:4, :4, :2, :]
        mask = synthetic_asl_data["mask"][:4, :4, :2]
        m0 = synthetic_asl_data["m0"][:4, :4, :2]

        result = quantify_multi_pld(
            delta_m=delta_m,
            m0=m0,
            params=params,
            mask=mask,
        )

        assert result is not None
        assert result.cbf_map is not None
        assert result.att_map is not None

    def test_m0_calibration(self, synthetic_asl_data: dict) -> None:
        """Test M0 calibration."""
        from osipy.asl.calibration import M0CalibrationParams, apply_m0_calibration

        # Create M0 image
        m0_image = synthetic_asl_data["m0"]

        params = M0CalibrationParams(
            method="voxelwise",
            tr_m0=6000.0,
            t1_tissue=1330.0,
        )

        calibrated, m0_corrected = apply_m0_calibration(
            asl_data=synthetic_asl_data["delta_m"],
            m0_image=m0_image,
            params=params,
            mask=synthetic_asl_data["mask"],
        )

        assert calibrated.shape == synthetic_asl_data["delta_m"].shape
        assert m0_corrected.shape == m0_image.shape

    def test_background_suppression_correction(self) -> None:
        """Test background suppression correction."""
        from osipy.asl.calibration.m0 import compute_m0_from_pd

        pd_image = np.random.uniform(400, 600, (8, 8, 4))

        m0 = compute_m0_from_pd(
            pd_image=pd_image,
            t1_tissue=1330.0,
            t2_tissue=80.0,
            tr=6000.0,
            te=13.0,
        )

        assert m0.shape == pd_image.shape
        # M0 should be >= PD (with corrections)
        assert np.mean(m0) >= np.mean(pd_image)

    def test_full_asl_pipeline_single_pld(self, synthetic_asl_data: dict) -> None:
        """Test complete single-PLD ASL pipeline."""
        from osipy.asl.calibration import apply_m0_calibration
        from osipy.asl.labeling import LabelingScheme
        from osipy.asl.quantification import quantify_cbf
        from osipy.asl.quantification.cbf import ASLQuantificationParams

        # Use subset for speed
        nx, ny, nz = 4, 4, 2
        delta_m = synthetic_asl_data["delta_m"][:nx, :ny, :nz]
        m0 = synthetic_asl_data["m0"][:nx, :ny, :nz]
        mask = synthetic_asl_data["mask"][:nx, :ny, :nz]

        # Step 1: M0 calibration
        calibrated, m0_corrected = apply_m0_calibration(
            asl_data=delta_m,
            m0_image=m0,
        )

        assert calibrated is not None, "M0 calibration failed"

        # Step 2: CBF quantification
        params = ASLQuantificationParams(
            labeling_scheme=LabelingScheme.PCASL,
            pld=synthetic_asl_data["pld"],
            label_duration=synthetic_asl_data["label_duration"],
        )

        result = quantify_cbf(
            delta_m=delta_m,
            m0=m0_corrected,
            params=params,
            mask=mask,
        )

        assert result.cbf_map is not None, "CBF quantification failed"
        assert result.cbf_map.units == "mL/100g/min"

    def test_full_asl_pipeline_multi_pld(self, synthetic_asl_data: dict) -> None:
        """Test complete multi-PLD ASL pipeline with ATT."""
        from osipy.asl.quantification.multi_pld import (
            MultiPLDParams,
            quantify_multi_pld,
        )

        # Use subset for speed
        nx, ny, nz = 4, 4, 2
        delta_m = synthetic_asl_data["multi_pld_delta_m"][:nx, :ny, :nz, :]
        m0 = synthetic_asl_data["m0"][:nx, :ny, :nz]
        mask = synthetic_asl_data["mask"][:nx, :ny, :nz]

        params = MultiPLDParams(
            plds=synthetic_asl_data["plds"],
            label_duration=synthetic_asl_data["label_duration"],
        )

        result = quantify_multi_pld(
            delta_m=delta_m,
            m0=m0,
            params=params,
            mask=mask,
        )

        assert result.cbf_map is not None, "Multi-PLD CBF failed"
        assert result.att_map is not None, "ATT estimation failed"


class TestASLOutputValidation:
    """Test ASL output format and physiological validity."""

    @pytest.fixture
    def synthetic_asl_data(self) -> dict:
        """Create synthetic ASL data for output validation testing."""
        np.random.seed(42)

        nx, ny, nz = 16, 16, 4
        cbf_true = np.random.uniform(40, 80, (nx, ny, nz))
        att_true = np.random.uniform(800, 1500, (nx, ny, nz))
        pld = 1800.0
        label_duration = 1800.0
        t1_blood = 1650.0
        labeling_efficiency = 0.85
        partition_coefficient = 0.9
        m0 = np.random.uniform(900, 1100, (nx, ny, nz))
        pld_s = pld / 1000.0
        tau_s = label_duration / 1000.0
        t1b_s = t1_blood / 1000.0

        delta_m = (
            2
            * m0
            * cbf_true
            * t1b_s
            * labeling_efficiency
            * (1 - np.exp(-tau_s / t1b_s))
            * np.exp(-pld_s / t1b_s)
            / partition_coefficient
            / 6000.0
        )
        delta_m += np.random.randn(nx, ny, nz) * 2

        plds = np.array([500, 1000, 1500, 2000, 2500])
        multi_pld_delta_m = np.zeros((nx, ny, nz, len(plds)))
        for idx, pld_val in enumerate(plds):
            pld_s = pld_val / 1000.0
            dm = (
                2
                * m0
                * cbf_true
                * t1b_s
                * labeling_efficiency
                * (1 - np.exp(-tau_s / t1b_s))
                * np.exp(-pld_s / t1b_s)
                / partition_coefficient
                / 6000.0
            )
            att_factor = np.clip((pld_val - att_true) / 500.0, 0, 1)
            dm *= att_factor
            multi_pld_delta_m[..., idx] = dm + np.random.randn(nx, ny, nz) * 1

        mask = np.ones((nx, ny, nz), dtype=bool)

        return {
            "delta_m": delta_m,
            "multi_pld_delta_m": multi_pld_delta_m,
            "plds": plds,
            "m0": m0,
            "cbf_true": cbf_true,
            "att_true": att_true,
            "mask": mask,
            "pld": pld,
            "label_duration": label_duration,
            "t1_blood": t1_blood,
            "labeling_efficiency": labeling_efficiency,
            "shape": (nx, ny, nz),
        }

    def test_cbf_units(self, synthetic_asl_data: dict) -> None:
        """Test that CBF is in mL/100g/min (ASL Lexicon)."""
        from osipy.asl.labeling import LabelingScheme
        from osipy.asl.quantification import quantify_cbf
        from osipy.asl.quantification.cbf import ASLQuantificationParams

        params = ASLQuantificationParams(
            labeling_scheme=LabelingScheme.PCASL,
        )

        result = quantify_cbf(
            delta_m=synthetic_asl_data["delta_m"][:4, :4, :2],
            m0=synthetic_asl_data["m0"][:4, :4, :2],
            params=params,
        )

        assert result.cbf_map.units == "mL/100g/min"

    def test_att_units(self, synthetic_asl_data: dict) -> None:
        """Test that ATT is in milliseconds."""
        from osipy.asl.quantification.multi_pld import (
            MultiPLDParams,
            quantify_multi_pld,
        )

        params = MultiPLDParams(
            plds=synthetic_asl_data["plds"],
            label_duration=synthetic_asl_data["label_duration"],
        )

        result = quantify_multi_pld(
            delta_m=synthetic_asl_data["multi_pld_delta_m"][:4, :4, :2, :],
            m0=synthetic_asl_data["m0"][:4, :4, :2],
            params=params,
        )

        assert result.att_map is not None
        assert (
            "ms" in result.att_map.units.lower()
            or "millisec" in result.att_map.units.lower()
        )

    def test_cbf_physiological_range(self, synthetic_asl_data: dict) -> None:
        """Test CBF values are in physiological range."""
        from osipy.asl.labeling import LabelingScheme
        from osipy.asl.quantification import quantify_cbf
        from osipy.asl.quantification.cbf import ASLQuantificationParams

        params = ASLQuantificationParams(
            labeling_scheme=LabelingScheme.PCASL,
            pld=synthetic_asl_data["pld"],
            label_duration=synthetic_asl_data["label_duration"],
        )

        result = quantify_cbf(
            delta_m=synthetic_asl_data["delta_m"],
            m0=synthetic_asl_data["m0"],
            params=params,
            mask=synthetic_asl_data["mask"],
        )

        cbf = result.cbf_map.values[synthetic_asl_data["mask"]]

        # Gray matter CBF: 40-80 ml/100g/min typical
        # Allow wider range for test data
        assert np.percentile(cbf, 5) > -10, "CBF should not be significantly negative"
        assert np.percentile(cbf, 95) < 200, "CBF should be < 200 ml/100g/min"
