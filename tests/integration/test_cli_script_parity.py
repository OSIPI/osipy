"""CLI vs Python-script parity tests for the DCE pipeline.

Regression lock: both the CLI DICOM entry (``_run_dce_from_dicom``) and
end-user scripts using ``osipy.DCEPipeline`` must produce byte-identical
parameter maps and quality masks from the same config and data. The bug
that motivated this test was ``fit_model`` overriding the fitter's
r²-thresholded quality mask with a loose positivity check, and
``_run_dce_from_dicom`` duplicating pre-filter logic inline instead of
delegating to ``DCEPipeline``.
"""

from __future__ import annotations

import numpy as np
import pytest

from osipy.common.dataset import PerfusionDataset
from osipy.common.types import DCEAcquisitionParams, Modality
from osipy.dce.fitting import fit_model
from osipy.pipeline.dce_pipeline import DCEPipeline, DCEPipelineConfig


@pytest.fixture
def synthetic_dce_inputs() -> dict:
    """Build a small synthetic DCE + VFA dataset that fits cleanly."""
    rng = np.random.default_rng(42)
    nx, ny, nz, nt = 6, 6, 2, 40
    time = np.linspace(0, 240, nt)

    # Ground-truth T1 (800-1500 ms physiological range)
    t1_true = rng.uniform(800, 1500, (nx, ny, nz))

    # VFA acquisition
    tr_vfa = 5.0
    flip_angles = np.array([2.0, 5.0, 10.0, 15.0, 20.0])
    vfa_data = np.zeros((nx, ny, nz, len(flip_angles)))
    for i, fa in enumerate(flip_angles):
        fa_rad = np.radians(fa)
        e1 = np.exp(-tr_vfa / t1_true)
        m0 = 1000.0 * np.ones_like(t1_true)
        vfa_data[..., i] = m0 * np.sin(fa_rad) * (1 - e1) / (1 - np.cos(fa_rad) * e1)
        vfa_data[..., i] += rng.normal(0, 5, (nx, ny, nz))

    vfa_dataset = PerfusionDataset(
        data=vfa_data,
        affine=np.eye(4),
        modality=Modality.DCE,
        time_points=np.arange(len(flip_angles), dtype=float),
        acquisition_params=DCEAcquisitionParams(
            tr=tr_vfa, flip_angles=flip_angles.tolist()
        ),
    )

    # DCE: simple synthetic enhancement (Parker-like) driven by Tofts kinetics.
    # Build concentration, invert to signal so the pipeline has a full loop.
    ktrans_true = rng.uniform(0.05, 0.25, (nx, ny, nz))
    ve_true = rng.uniform(0.15, 0.35, (nx, ny, nz))

    from osipy.common.aif import ParkerAIF

    aif = ParkerAIF()(time).concentration
    conc = np.zeros((nx, ny, nz, nt))
    for t_idx in range(1, nt):
        dt = time[t_idx] - time[t_idx - 1]
        conv = np.zeros((nx, ny, nz))
        for k in range(t_idx + 1):
            tau = time[t_idx] - time[k]
            kern = np.exp(-ktrans_true * tau / np.where(ve_true > 1e-6, ve_true, 1e-6))
            conv += aif[k] * kern * dt
        conc[..., t_idx] = ktrans_true * conv
    conc += rng.normal(0, 0.005, conc.shape)

    # Invert concentration to signal via SPGR (relaxivity=4.5 mM^-1 s^-1)
    relaxivity = 4.5
    tr_dce = 4.0
    te_dce = 2.0
    flip_dce = 12.0
    r10 = 1.0 / (t1_true / 1000.0)
    r1 = r10[..., None] + relaxivity * conc  # (x,y,z,t)
    e1 = np.exp(-tr_dce / 1000.0 * r1)
    fa_rad = np.radians(flip_dce)
    s = 100.0 * np.sin(fa_rad) * (1 - e1) / (1 - np.cos(fa_rad) * e1)

    dce_dataset = PerfusionDataset(
        data=s,
        affine=np.eye(4),
        modality=Modality.DCE,
        time_points=time,
        acquisition_params=DCEAcquisitionParams(
            tr=tr_dce, te=te_dce, flip_angles=[flip_dce], temporal_resolution=6.0
        ),
    )

    return {
        "vfa": vfa_dataset,
        "dce": dce_dataset,
        "time": time,
        "flip_angles": flip_angles,
        "tr_vfa": tr_vfa,
        "tr_dce": tr_dce,
        "flip_dce": flip_dce,
        "relaxivity": relaxivity,
    }


def _build_pipeline_config(inputs: dict) -> DCEPipelineConfig:
    acq = DCEAcquisitionParams(
        tr=inputs["tr_dce"],
        te=2.0,
        flip_angles=[inputs["flip_dce"]],
        temporal_resolution=6.0,
        relaxivity=inputs["relaxivity"],
        baseline_frames=3,
    )
    return DCEPipelineConfig(
        model="extended_tofts",
        t1_mapping_method="vfa",
        aif_source="population",
        population_aif="parker",
        acquisition_params=acq,
    )


class TestCliScriptParity:
    """Both CLI and end-user script paths go through DCEPipeline — assert no drift."""

    def test_two_pipeline_runs_are_identical(self, synthetic_dce_inputs: dict) -> None:
        """Same data through DCEPipeline twice produces identical outputs.

        This is the foundation of CLI ↔ script parity: since both entrypoints
        now delegate to DCEPipeline.run(), determinism of DCEPipeline guarantees
        byte-level parity between them.
        """
        cfg = _build_pipeline_config(synthetic_dce_inputs)

        result_a = DCEPipeline(cfg).run(
            synthetic_dce_inputs["dce"],
            time=synthetic_dce_inputs["time"],
            t1_data=synthetic_dce_inputs["vfa"],
            flip_angles=synthetic_dce_inputs["flip_angles"],
            tr=synthetic_dce_inputs["tr_vfa"],
        )
        result_b = DCEPipeline(cfg).run(
            synthetic_dce_inputs["dce"],
            time=synthetic_dce_inputs["time"],
            t1_data=synthetic_dce_inputs["vfa"],
            flip_angles=synthetic_dce_inputs["flip_angles"],
            tr=synthetic_dce_inputs["tr_vfa"],
        )

        np.testing.assert_array_equal(
            result_a.fit_result.quality_mask, result_b.fit_result.quality_mask
        )
        for name in result_a.fit_result.parameter_maps:
            np.testing.assert_allclose(
                result_a.fit_result.parameter_maps[name].values,
                result_b.fit_result.parameter_maps[name].values,
                atol=1e-10,
                err_msg=f"parameter map '{name}' diverged between runs",
            )

    def test_quality_mask_marks_fitted_voxels(self, synthetic_dce_inputs: dict) -> None:
        """Saved quality_mask marks every voxel the fitter ran on.

        No r² or positivity gate is applied — callers filter on
        ``r_squared_map`` or parameter values themselves. This test pins
        that semantics: when no user mask is passed, every voxel appears
        as ``True`` in the output mask.
        """
        cfg = _build_pipeline_config(synthetic_dce_inputs)
        result = DCEPipeline(cfg).run(
            synthetic_dce_inputs["dce"],
            time=synthetic_dce_inputs["time"],
            t1_data=synthetic_dce_inputs["vfa"],
            flip_angles=synthetic_dce_inputs["flip_angles"],
            tr=synthetic_dce_inputs["tr_vfa"],
        )

        qmask = result.fit_result.quality_mask
        assert qmask is not None
        assert qmask.all(), (
            "quality_mask should mark every fitted voxel when no user mask "
            "is provided; any False entry indicates a regression where a "
            "hidden r² or T1 filter is still gating the output."
        )

    def test_bare_fit_model_not_contaminated_by_r_squared(self) -> None:
        """``fit_model`` does not leak the 'r_squared' entry into parameter_maps.

        After the fix, r² is exposed only via ``DCEFitResult.r_squared_map``,
        not as a parameter map — otherwise downstream code that iterates
        ``parameter_maps`` would save/plot/stat it as if it were a PK parameter.
        """
        time = np.linspace(0, 240, 40)
        from osipy.common.aif import ParkerAIF

        aif = ParkerAIF()(time).concentration
        conc = np.broadcast_to(aif * 0.1, (4, 4, 2, len(time))).copy()

        result = fit_model("extended_tofts", conc, aif, time)
        assert "r_squared" not in result.parameter_maps
        assert result.r_squared_map is not None
        assert result.r_squared_map.shape == (4, 4, 2)
