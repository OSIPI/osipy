"""Pipeline runner module.

This module provides a unified interface for running perfusion analysis
pipelines with automatic modality detection.
"""

from __future__ import annotations

import dataclasses
import datetime
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from osipy.common.exceptions import DataValidationError
from osipy.common.parameter_map import ParameterMap
from osipy.common.types import AnalysisResult, Modality

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class PipelineResult:
    """Generic pipeline result container (internal).

    This class is kept for backwards compatibility with code that calls
    individual ``*Pipeline.run()`` methods directly.  The public-facing
    unified result type returned by :func:`run_analysis` is
    :class:`~osipy.common.types.AnalysisResult`.

    Attributes
    ----------
    modality : Modality
        Analyzed modality.
    parameter_maps : dict[str, ParameterMap]
        Output parameter maps.
    quality_mask : NDArray[np.bool_]
        Quality mask for valid results.
    metadata : dict[str, Any]
        Additional metadata and statistics.
    """

    modality: Modality
    parameter_maps: dict[str, ParameterMap]
    quality_mask: "NDArray[np.bool_]"
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------

def _serialise_config(config: Any) -> dict[str, Any]:
    """Convert a pipeline config dataclass to a JSON-safe dict."""
    if config is None:
        return {}
    if dataclasses.is_dataclass(config):
        raw = dataclasses.asdict(config)
    elif hasattr(config, "__dict__"):
        raw = dict(vars(config))
    else:
        return {"repr": str(config)}

    # Make values JSON-safe (convert non-primitives to str)
    def _safe(v: Any) -> Any:
        if isinstance(v, (str, int, float, bool, type(None))):
            return v
        if isinstance(v, dict):
            return {k: _safe(vv) for k, vv in v.items()}
        if isinstance(v, (list, tuple)):
            return [_safe(i) for i in v]
        return str(v)

    return {k: _safe(v) for k, v in raw.items()}


def _make_provenance(
    modality: Modality,
    config: Any,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the provenance dict injected into every AnalysisResult."""
    from osipy._version import __version__

    prov: dict[str, Any] = {
        "osipy_version": __version__,
        "captured_at": datetime.datetime.now(datetime.UTC).isoformat(
            timespec="seconds"
        ),
        "modality": modality.value,
        "config": _serialise_config(config),
    }
    if extra_metadata:
        prov.update(extra_metadata)
    return prov


def _pipeline_result_to_analysis_result(
    pr: PipelineResult,
    provenance: dict[str, Any],
) -> AnalysisResult:
    """Wrap a PipelineResult in the public AnalysisResult contract."""
    return AnalysisResult(
        modality=pr.modality,
        parameter_maps=pr.parameter_maps,
        quality_mask=pr.quality_mask,
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_analysis(
    data: "NDArray[np.floating[Any]]",
    modality: Modality | str,
    **kwargs: Any,
) -> AnalysisResult:
    """Run perfusion analysis with specified modality.

    This is a unified entry point that dispatches to the appropriate
    pipeline based on the modality and returns a standardised
    :class:`~osipy.common.types.AnalysisResult` regardless of modality.

    Parameters
    ----------
    data : NDArray
        Input data appropriate for the modality.
    modality : Modality or str
        Analysis modality: 'dce', 'dsc', 'asl', or 'ivim'.
    **kwargs
        Additional arguments passed to the specific pipeline.
        See individual pipeline documentation for details.

    Returns
    -------
    AnalysisResult
        Standardised analysis result with:

        * ``parameter_maps`` – dict of named :class:`~osipy.common.parameter_map.ParameterMap`
        * ``quality_mask``   – boolean voxel-validity array (guaranteed non-None)
        * ``provenance``     – audit trail (version, timestamp, config)

    Examples
    --------
    >>> import numpy as np
    >>> from osipy.pipeline import run_analysis
    >>> from osipy.common.types import Modality
    >>>
    >>> # Run DCE analysis
    >>> result = run_analysis(
    ...     dce_data,
    ...     modality=Modality.DCE,
    ...     time=time_vector,
    ...     model='extended_tofts',
    ... )
    >>>
    >>> # Uniform interface – works for every modality
    >>> for name, pmap in result.parameter_maps.items():
    ...     print(name, pmap.values.shape)
    >>> result.quality_mask             # always a bool ndarray
    >>> result.provenance['osipy_version']  # reproducibility audit trail
    """
    # Normalise modality
    if isinstance(modality, str):
        modality_map = {
            "dce": Modality.DCE,
            "dsc": Modality.DSC,
            "asl": Modality.ASL,
            "ivim": Modality.IVIM,
        }
        modality = modality_map.get(modality.lower(), Modality.DCE)

    if modality == Modality.DCE:
        return _run_dce_analysis(data, **kwargs)
    elif modality == Modality.DSC:
        return _run_dsc_analysis(data, **kwargs)
    elif modality == Modality.ASL:
        return _run_asl_analysis(data, **kwargs)
    elif modality == Modality.IVIM:
        return _run_ivim_analysis(data, **kwargs)
    else:
        msg = f"Unsupported modality: {modality}"
        raise DataValidationError(msg)


# ---------------------------------------------------------------------------
# Private per-modality helpers
# ---------------------------------------------------------------------------

def _run_dce_analysis(
    data: "NDArray[np.floating[Any]]", **kwargs: Any
) -> AnalysisResult:
    """Run DCE-MRI analysis."""
    from osipy.pipeline.dce_pipeline import DCEPipeline, DCEPipelineConfig

    model = kwargs.pop("model", "extended_tofts")
    aif_source = kwargs.pop("aif_source", "population")

    config = DCEPipelineConfig(
        model=model,
        aif_source=aif_source,
    )

    pipeline = DCEPipeline(config)
    result = pipeline.run(data, **kwargs)

    pr = PipelineResult(
        modality=Modality.DCE,
        parameter_maps=result.fit_result.parameter_maps,
        quality_mask=result.fit_result.quality_mask,
        metadata={
            "model": result.fit_result.model_name,
            "fitting_stats": result.fit_result.fitting_stats,
        },
    )
    return _pipeline_result_to_analysis_result(
        pr, _make_provenance(Modality.DCE, config)
    )


def _run_dsc_analysis(
    data: "NDArray[np.floating[Any]]", **kwargs: Any
) -> AnalysisResult:
    """Run DSC-MRI analysis."""
    from osipy.pipeline.dsc_pipeline import DSCPipeline, DSCPipelineConfig

    te = kwargs.pop("te", 30.0)
    apply_leakage = kwargs.pop("apply_leakage_correction", True)

    config = DSCPipelineConfig(
        te=te,
        apply_leakage_correction=apply_leakage,
    )

    pipeline = DSCPipeline(config)
    result = pipeline.run(data, **kwargs)

    # Extract parameter maps into a uniform dict
    param_maps: dict[str, ParameterMap] = {
        "CBV": result.perfusion_maps.cbv,
        "CBF": result.perfusion_maps.cbf,
        "MTT": result.perfusion_maps.mtt,
    }
    if result.perfusion_maps.ttp is not None:
        param_maps["TTP"] = result.perfusion_maps.ttp

    # Guarantee quality mask is never None
    if result.perfusion_maps.quality_mask is not None:
        qm = result.perfusion_maps.quality_mask
        quality_mask = qm.values if hasattr(qm, "values") else qm
    else:
        # DSC had no mask — default to all-valid
        first_map = next(iter(param_maps.values()))
        arr = first_map.values if hasattr(first_map, "values") else first_map
        quality_mask = np.ones(arr.shape, dtype=bool)

    pr = PipelineResult(
        modality=Modality.DSC,
        parameter_maps=param_maps,
        quality_mask=quality_mask,
        metadata={},
    )
    return _pipeline_result_to_analysis_result(
        pr, _make_provenance(Modality.DSC, config)
    )


def _run_asl_analysis(
    data: "NDArray[np.floating[Any]]", **kwargs: Any
) -> AnalysisResult:
    """Run ASL analysis."""
    from osipy.asl import LabelingScheme
    from osipy.pipeline.asl_pipeline import ASLPipeline, ASLPipelineConfig

    labeling = kwargs.pop("labeling_scheme", LabelingScheme.PCASL)
    pld = kwargs.pop("pld", 1800.0)

    config = ASLPipelineConfig(
        labeling_scheme=labeling,
        pld=pld,
    )

    pipeline = ASLPipeline(config)

    if "control_data" in kwargs:
        result = pipeline.run(data, **kwargs)
    else:
        m0_data = kwargs.pop("m0_data", 1.0)
        result = pipeline.run_from_alternating(data, m0_data, **kwargs)

    cbf_map = result.cbf_result.cbf_map
    qm = result.cbf_result.quality_mask
    quality_mask = qm.values if hasattr(qm, "values") else qm

    pr = PipelineResult(
        modality=Modality.ASL,
        parameter_maps={"CBF": cbf_map},
        quality_mask=quality_mask,
        metadata={},
    )
    return _pipeline_result_to_analysis_result(
        pr, _make_provenance(Modality.ASL, config)
    )


def _run_ivim_analysis(
    data: "NDArray[np.floating[Any]]", **kwargs: Any
) -> AnalysisResult:
    """Run IVIM analysis."""
    from osipy.pipeline.ivim_pipeline import IVIMPipeline, IVIMPipelineConfig

    config = IVIMPipelineConfig()
    pipeline = IVIMPipeline(config)
    result = pipeline.run(data, **kwargs)

    fr = result.fit_result
    qm = fr.quality_mask
    quality_mask = qm.values if hasattr(qm, "values") else qm

    pr = PipelineResult(
        modality=Modality.IVIM,
        parameter_maps={
            "D": fr.d_map,
            "D*": fr.d_star_map,
            "f": fr.f_map,
        },
        quality_mask=quality_mask,
        metadata=fr.fitting_stats,
    )
    return _pipeline_result_to_analysis_result(
        pr, _make_provenance(Modality.IVIM, config)
    )
