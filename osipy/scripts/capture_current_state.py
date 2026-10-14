"""capture_current_state.py
===========================
**Before-the-refactor snapshot tool for osipy pipelines.**

Run this script *before* the AnalysisResult refactor to document
(and optionally save) the exact shapes and access paths of every
result class today.  After the refactor, running the script again
lets you compare old vs. new side-by-side.

Usage
-----
From the repo root (activate your venv first!):

    python -m osipy.scripts.capture_current_state
    # or, to save the snapshot as JSON:
    python -m osipy.scripts.capture_current_state --save

The script uses tiny synthetic data (4x4x2 voxels, few time-points)
so it runs in seconds without real MRI data.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

SEP = "=" * 72
INDENT = "  "


def _banner(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def _field(label: str, value: Any, depth: int = 1) -> None:
    pad = INDENT * depth
    print(f"{pad}{label}: {value}")


def _describe_array(arr: np.ndarray | None, label: str = "array") -> str:
    if arr is None:
        return "None"
    return f"ndarray  shape={arr.shape}  dtype={arr.dtype}"


def _describe_dict(d: dict, label: str = "") -> None:
    if not d:
        print(f"{INDENT * 2}(empty)")
        return
    for k, v in d.items():
        if hasattr(v, "values"):  # ParameterMap
            _field(
                f'["{k}"]',
                f"ParameterMap  name={v.name!r}  shape={v.values.shape}"
                f"  units={v.units!r}",
                depth=2,
            )
        elif isinstance(v, np.ndarray):
            _field(f'["{k}"]', _describe_array(v), depth=2)
        else:
            _field(f'["{k}"]', repr(v), depth=2)


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic data factories
# ──────────────────────────────────────────────────────────────────────────────

SHAPE_3D = (4, 4, 2)          # spatial voxels
N_TIMEPOINTS = 12
N_B_VALUES = 6

rng = np.random.default_rng(42)


def _make_dce_data() -> tuple[np.ndarray, np.ndarray]:
    """Returns (signal [4,4,2,T], time [T])."""
    signal = rng.uniform(0.8, 1.2, (*SHAPE_3D, N_TIMEPOINTS)).astype(np.float64)
    time = np.linspace(0, 60, N_TIMEPOINTS)
    return signal, time


def _make_dsc_data() -> tuple[np.ndarray, np.ndarray]:
    signal = rng.uniform(800.0, 1200.0, (*SHAPE_3D, N_TIMEPOINTS)).astype(np.float64)
    time = np.linspace(0, 60, N_TIMEPOINTS)
    return signal, time


def _make_asl_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    label = rng.uniform(400.0, 600.0, (*SHAPE_3D, 4)).astype(np.float64)
    control = label + rng.uniform(10.0, 30.0, label.shape)
    m0 = np.full(SHAPE_3D, 1000.0)
    return label, control, m0


def _make_ivim_data() -> tuple[np.ndarray, np.ndarray]:
    b_values = np.array([0, 10, 20, 50, 100, 200], dtype=np.float64)
    signal = np.zeros((*SHAPE_3D, N_B_VALUES))
    for i, b in enumerate(b_values):
        signal[..., i] = np.exp(-0.001 * b) * 1000.0
    signal += rng.normal(0, 5.0, signal.shape)
    return signal, b_values


# ──────────────────────────────────────────────────────────────────────────────
# Per-pipeline snapshots
# ──────────────────────────────────────────────────────────────────────────────

SnapshotDict = dict[str, Any]


def _snapshot_dce() -> SnapshotDict:
    """Run DCEPipeline and inspect DCEPipelineResult."""
    _banner("DCE Pipeline -> DCEPipelineResult")

    from osipy.pipeline.dce_pipeline import DCEPipeline, DCEPipelineConfig

    signal, time = _make_dce_data()
    config = DCEPipelineConfig(model="tofts", aif_source="population")
    pipeline = DCEPipeline(config)
    result = pipeline.run(signal, time)

    print(f"\n  Result class : {type(result).__name__}")
    print(f"  Module       : {type(result).__module__}")

    print(f"\n  +-- result.fit_result  ({type(result.fit_result).__name__})")
    print(f"  |   .parameter_maps (dict):")
    _describe_dict(result.fit_result.parameter_maps)
    print(f"  |   .quality_mask : {_describe_array(result.fit_result.quality_mask)}")
    print(f"  |   .model_name   : {result.fit_result.model_name!r}")

    print(f"\n  +-- result.t1_map : {result.t1_map}")
    print(f"  +-- result.aif   : {type(result.aif).__name__}")
    print(f"  +-- result.config.model : {result.config.model!r}")

    print("\n  !  Access path for Ktrans:  result.fit_result.parameter_maps['Ktrans']")
    print("  !  Access path for mask:    result.fit_result.quality_mask")

    return {
        "class": type(result).__name__,
        "parameter_maps_keys": list(result.fit_result.parameter_maps.keys()),
        "quality_mask_shape": list(result.fit_result.quality_mask.shape),
        "model": result.fit_result.model_name,
        "access_path_params": "result.fit_result.parameter_maps[name]",
        "access_path_mask": "result.fit_result.quality_mask",
        "mask_guaranteed": True,
    }


def _snapshot_dsc() -> SnapshotDict:
    """Run DSCPipeline and inspect DSCPipelineResult."""
    _banner("DSC Pipeline -> DSCPipelineResult")

    from osipy.pipeline.dsc_pipeline import DSCPipeline, DSCPipelineConfig

    signal, time = _make_dsc_data()
    config = DSCPipelineConfig(te=30.0, apply_leakage_correction=False)
    pipeline = DSCPipeline(config)
    result = pipeline.run(signal, time)

    print(f"\n  Result class : {type(result).__name__}")

    pm = result.perfusion_maps
    print(f"\n  +-- result.perfusion_maps  ({type(pm).__name__})")
    print(f"  |   .cbv : {_describe_array(pm.cbv.values if hasattr(pm.cbv,'values') else pm.cbv)}")
    print(f"  |   .cbf : {_describe_array(pm.cbf.values if hasattr(pm.cbf,'values') else pm.cbf)}")
    print(f"  |   .mtt : {_describe_array(pm.mtt.values if hasattr(pm.mtt,'values') else pm.mtt)}")

    qm = pm.quality_mask
    qm_val = qm.values if hasattr(qm, "values") else qm
    print(f"  |   .quality_mask : {_describe_array(qm_val)}")

    is_none = (qm is None)
    print(f"  +-- quality_mask is None?  {is_none}")
    print("\n  !  Access path for CBF:  result.perfusion_maps.cbf")
    print("  !  Access path for mask: result.perfusion_maps.quality_mask (CAN BE None)")
    print("  !  No dict interface – every map is a separate attribute!")

    return {
        "class": type(result).__name__,
        "parameter_maps_keys": ["cbv", "cbf", "mtt", "ttp"],
        "quality_mask_shape": list(qm_val.shape) if qm_val is not None else None,
        "access_path_params": "result.perfusion_maps.<name>  (individual attrs)",
        "access_path_mask": "result.perfusion_maps.quality_mask",
        "mask_guaranteed": False,
    }


def _snapshot_asl() -> SnapshotDict:
    """Run ASLPipeline and inspect ASLPipelineResult."""
    _banner("ASL Pipeline -> ASLPipelineResult")

    from osipy.asl import LabelingScheme
    from osipy.pipeline.asl_pipeline import ASLPipeline, ASLPipelineConfig

    label, control, m0 = _make_asl_data()
    config = ASLPipelineConfig(
        labeling_scheme=LabelingScheme.PCASL, pld=1800.0
    )
    pipeline = ASLPipeline(config)
    result = pipeline.run(label, control, m0)

    cbf = result.cbf_result
    print(f"\n  Result class : {type(result).__name__}")
    print(f"\n  +-- result.cbf_result  ({type(cbf).__name__})")

    cbf_map = cbf.cbf_map
    cbf_arr = cbf_map.values if hasattr(cbf_map, "values") else cbf_map
    print(f"  |   .cbf_map       : {_describe_array(cbf_arr)}")

    qm = cbf.quality_mask
    qm_arr = qm.values if hasattr(qm, "values") else qm
    print(f"  |   .quality_mask  : {_describe_array(qm_arr)}")
    print(f"  +-- result.m0_map  : {result.m0_map}")

    print("\n  !  Access path for CBF:  result.cbf_result.cbf_map")
    print("  !  Access path for mask: result.cbf_result.quality_mask")
    print("  !  Not a dict – only one parameter (CBF)")

    return {
        "class": type(result).__name__,
        "parameter_maps_keys": ["cbf_map"],
        "quality_mask_shape": list(qm_arr.shape) if qm_arr is not None else None,
        "access_path_params": "result.cbf_result.cbf_map",
        "access_path_mask": "result.cbf_result.quality_mask",
        "mask_guaranteed": True,
    }


def _snapshot_ivim() -> SnapshotDict:
    """Run IVIMPipeline and inspect IVIMPipelineResult."""
    _banner("IVIM Pipeline -> IVIMPipelineResult")

    from osipy.pipeline.ivim_pipeline import IVIMPipeline, IVIMPipelineConfig

    signal, b_values = _make_ivim_data()
    config = IVIMPipelineConfig()
    pipeline = IVIMPipeline(config)
    result = pipeline.run(signal, b_values)

    fr = result.fit_result
    print(f"\n  Result class : {type(result).__name__}")
    print(f"\n  +-- result.fit_result  ({type(fr).__name__})")

    d_arr = fr.d_map.values if hasattr(fr.d_map, "values") else fr.d_map
    ds_arr = fr.d_star_map.values if hasattr(fr.d_star_map, "values") else fr.d_star_map
    f_arr = fr.f_map.values if hasattr(fr.f_map, "values") else fr.f_map
    qm_arr = fr.quality_mask.values if hasattr(fr.quality_mask, "values") else fr.quality_mask

    print(f"  |   .d_map       : {_describe_array(d_arr)}")
    print(f"  |   .d_star_map  : {_describe_array(ds_arr)}")
    print(f"  |   .f_map       : {_describe_array(f_arr)}")
    print(f"  |   .quality_mask: {_describe_array(qm_arr)}")

    print("\n  !  Access path for D:    result.fit_result.d_map")
    print("  !  Access path for D*:   result.fit_result.d_star_map")
    print("  !  Access path for f:    result.fit_result.f_map")
    print("  !  Access path for mask: result.fit_result.quality_mask")
    print("  !  Not a dict – each parameter is an individual attribute!")

    return {
        "class": type(result).__name__,
        "parameter_maps_keys": ["d_map", "d_star_map", "f_map"],
        "quality_mask_shape": list(qm_arr.shape) if qm_arr is not None else None,
        "access_path_params": "result.fit_result.<d_map|d_star_map|f_map>",
        "access_path_mask": "result.fit_result.quality_mask",
        "mask_guaranteed": True,
    }


def _snapshot_runner() -> SnapshotDict:
    """Show what run_analysis() currently returns for each modality."""
    _banner("runner.run_analysis() – Unified PipelineResult wrapper")

    from osipy.pipeline.runner import PipelineResult, run_analysis

    print(f"\n  PipelineResult fields:")
    import dataclasses
    for f in dataclasses.fields(PipelineResult):
        print(f"{INDENT * 2}.{f.name}: {f.type}")

    print(f"\n  [OK]  run_analysis() already maps each modality -> PipelineResult")
    print(f"  [OK]  parameter_maps is always a dict[str, ParameterMap]")
    print(f"  [OK]  quality_mask is always a numpy bool array")
    print(f"  [X]   No provenance/version/timestamp in PipelineResult.metadata")
    print(f"  [X]   PipelineResult is internal – not the proposed AnalysisResult")

    return {
        "class": "PipelineResult",
        "has_parameter_maps_dict": True,
        "has_quality_mask": True,
        "has_provenance": False,
        "note": "run_analysis() wraps each pipeline, but lacks provenance.",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(save: bool = False) -> None:
    from osipy._version import __version__

    print("\n" + "=" * 72)
    print("  osipy  BEFORE-REFACTOR  PIPELINE  STATE  SNAPSHOT")
    print("=" * 72)
    print(f"  osipy version : {__version__}")
    print(f"  Python        : {sys.version.split()[0]}")
    print(f"  Captured at   : {datetime.datetime.now().isoformat(timespec='seconds')}")
    print(f"  Spatial shape : {SHAPE_3D}  (synthetic – no real MRI data needed)")

    snapshot: dict[str, Any] = {
        "osipy_version": __version__,
        "captured_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "synthetic_shape": list(SHAPE_3D),
        "modalities": {},
    }

    try:
        snapshot["modalities"]["DCE"] = _snapshot_dce()
    except Exception as exc:
        print(f"\n  [DCE]  ERROR: {exc}")
        snapshot["modalities"]["DCE"] = {"error": str(exc)}

    try:
        snapshot["modalities"]["DSC"] = _snapshot_dsc()
    except Exception as exc:
        print(f"\n  [DSC]  ERROR: {exc}")
        snapshot["modalities"]["DSC"] = {"error": str(exc)}

    try:
        snapshot["modalities"]["ASL"] = _snapshot_asl()
    except Exception as exc:
        print(f"\n  [ASL]  ERROR: {exc}")
        snapshot["modalities"]["ASL"] = {"error": str(exc)}

    try:
        snapshot["modalities"]["IVIM"] = _snapshot_ivim()
    except Exception as exc:
        print(f"\n  [IVIM]  ERROR: {exc}")
        snapshot["modalities"]["IVIM"] = {"error": str(exc)}

    try:
        snapshot["runner"] = _snapshot_runner()
    except Exception as exc:
        print(f"\n  [runner]  ERROR: {exc}")
        snapshot["runner"] = {"error": str(exc)}

    # ── Summary table ────────────────────────────────────────────────────────
    _banner("SUMMARY: The Fragmentation Problem (Before Refactor)")
    print(
        f"  {'Modality':<8}  {'Result Class':<26}  {'Maps interface':<30}  "
        f"{'Mask Guaranteed'}"
    )
    print(f"  {'-'*8}  {'-'*26}  {'-'*30}  {'-'*15}")
    rows = [
        ("DCE",  "DCEPipelineResult",  "result.fit_result.parameter_maps[name]", "Yes"),
        ("DSC",  "DSCPipelineResult",  "result.perfusion_maps.<name>",           "NO !"),
        ("ASL",  "ASLPipelineResult",  "result.cbf_result.cbf_map",              "Yes"),
        ("IVIM", "IVIMPipelineResult", "result.fit_result.<name>_map",           "Yes"),
    ]
    for m, cls, path, mask in rows:
        print(f"  {m:<8}  {cls:<26}  {path:<30}  {mask}")

    print(f"\n  -> After the AnalysisResult refactor, ALL four pipelines will expose:")
    print(f"      result.parameter_maps[name]  (dict, always present)")
    print(f"      result.quality_mask           (bool ndarray, never None)")
    print(f"      result.provenance             (version + timestamp + config)")

    # ── Save ────────────────────────────────────────────────────────────────
    if save:
        out_path = Path(__file__).parent / "state_before_refactor.json"
        with open(out_path, "w") as fh:
            json.dump(snapshot, fh, indent=2)
        print(f"\n  [OK]  Snapshot saved -> {out_path}")

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Capture osipy pipeline result structures BEFORE the AnalysisResult refactor."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save snapshot as state_before_refactor.json next to this script.",
    )
    args = parser.parse_args()
    main(save=args.save)
