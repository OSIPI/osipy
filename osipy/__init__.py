"""osipy: OSIPI-compliant MRI perfusion analysis library.

A comprehensive, modular Python library for MRI perfusion data analysis
serving researchers, clinicians, and medical imaging software developers.

GPU Acceleration
----------------
osipy supports optional GPU acceleration via CuPy. When CuPy is installed
and a CUDA-capable GPU is available, many operations are automatically
GPU-accelerated for improved performance on large datasets.

GPU acceleration is available for:
- DCE T1 mapping (VFA, Look-Locker)
- DCE pharmacokinetic model fitting (Tofts, Extended Tofts, Patlak, 2CXM)
- DSC signal processing and concentration conversion
- ASL CBF quantification
- IVIM bi-exponential fitting
- Convolution and deconvolution operations

Note: SVD-based deconvolution uses xp.linalg.svd but iterates per-voxel in Python loops, limiting GPU parallelism. Input arrays
remain on their original device but processing is sequential per voxel.

To check GPU availability::

    from osipy.common.backend import is_gpu_available, get_backend
    print(f"GPU available: {is_gpu_available()}")
    print(f"Current backend: {get_backend()}")

To force CPU execution even when GPU is available::

    from osipy.common.backend import set_backend, GPUConfig
    set_backend(GPUConfig(force_cpu=True))

Modules
-------
common
    Shared utilities for I/O, fitting, AIF, signal processing, and visualization.
dce
    Dynamic Contrast-Enhanced MRI analysis (T1 mapping, Tofts models).
dsc
    Dynamic Susceptibility Contrast MRI analysis (deconvolution, leakage correction).
asl
    Arterial Spin Labeling quantification (CBF, ATT estimation).
ivim
    Intravoxel Incoherent Motion analysis (diffusion-perfusion separation).
pipeline
    End-to-end processing pipelines for each modality.

References
----------
OSIPI (Open Science Initiative for Perfusion Imaging):
    https://www.osipi.org/
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from osipy._version import __version__, __version_info__

if TYPE_CHECKING:
    from osipy.asl import LabelingScheme, apply_m0_calibration, quantify_cbf
    from osipy.asl.quantification import MultiPLDParams, quantify_multi_pld
    from osipy.common.aif import (
        ArterialInputFunction,
        FritzHansenAIF,
        GeorgiouAIF,
        ParkerAIF,
        detect_aif,
        get_population_aif,
    )
    from osipy.common.backend import (
        GPU_AVAILABLE,
        GPUConfig,
        get_array_module,
        get_backend,
        is_gpu_available,
        set_backend,
        to_gpu,
        to_numpy,
    )
    from osipy.common.dataset import PerfusionDataset
    from osipy.common.io import export_bids, load_dicom, load_nifti, load_perfusion
    from osipy.common.parameter_map import ParameterMap
    from osipy.common.types import AIFType, AnalysisResult, FittingMethod, LabelingType, Modality
    from osipy.dce import (
        ExtendedToftsModel,
        PatlakModel,
        ToftsModel,
        TwoCompartmentModel,
        compute_t1_map,
        fit_model,
        get_model,
        list_models,
    )
    from osipy.dce import (
        signal_to_concentration as dce_signal_to_concentration,
    )
    from osipy.dsc import (
        compute_perfusion_maps,
        correct_leakage,
        get_deconvolver,
        list_deconvolvers,
        signal_to_delta_r2,
    )
    from osipy.ivim import (
        IVIMBiexponentialModel,
        IVIMFitParams,
        fit_ivim,
    )
    from osipy.pipeline import (
        ASLPipeline,
        DCEPipeline,
        DSCPipeline,
        IVIMPipeline,
        run_analysis,
    )

__all__ = [
    "GPU_AVAILABLE",
    "AIFType",
    "ASLPipeline",
    # AIF
    "ArterialInputFunction",
    # Result contract
    "AnalysisResult",
    # Pipelines
    "DCEPipeline",
    "DSCPipeline",
    "ExtendedToftsModel",
    "FittingMethod",
    "FritzHansenAIF",
    "GPUConfig",
    "GeorgiouAIF",
    "IVIMBiexponentialModel",
    "IVIMFitParams",
    "IVIMPipeline",
    "LabelingScheme",
    "LabelingType",
    "Modality",
    "MultiPLDParams",
    "ParameterMap",
    "ParkerAIF",
    "PatlakModel",
    # Common types
    "PerfusionDataset",
    "ToftsModel",
    "TwoCompartmentModel",
    # Version
    "__version__",
    "__version_info__",
    "apply_m0_calibration",
    "compute_perfusion_maps",
    # DCE
    "compute_t1_map",
    "correct_leakage",
    "dce_signal_to_concentration",
    "detect_aif",
    "export_bids",
    # IVIM
    "fit_ivim",
    "fit_model",
    # GPU/CPU backend
    "get_array_module",
    "get_backend",
    "get_deconvolver",
    "get_model",
    "get_population_aif",
    "is_gpu_available",
    "list_deconvolvers",
    "list_models",
    "load_dicom",
    # I/O
    "load_nifti",
    "load_perfusion",
    # ASL
    "quantify_cbf",
    "quantify_multi_pld",
    "run_analysis",
    "set_backend",
    # DSC
    "signal_to_delta_r2",
    "to_gpu",
    "to_numpy",
]

# Lazy import mapping: attribute name -> (module, name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # ASL
    "LabelingScheme": ("osipy.asl", "LabelingScheme"),
    "MultiPLDParams": ("osipy.asl.quantification", "MultiPLDParams"),
    "apply_m0_calibration": ("osipy.asl", "apply_m0_calibration"),
    "quantify_cbf": ("osipy.asl", "quantify_cbf"),
    "quantify_multi_pld": ("osipy.asl.quantification", "quantify_multi_pld"),
    # AIF
    "ArterialInputFunction": ("osipy.common.aif", "ArterialInputFunction"),
    "FritzHansenAIF": ("osipy.common.aif", "FritzHansenAIF"),
    "GeorgiouAIF": ("osipy.common.aif", "GeorgiouAIF"),
    "ParkerAIF": ("osipy.common.aif", "ParkerAIF"),
    "detect_aif": ("osipy.common.aif", "detect_aif"),
    "get_population_aif": ("osipy.common.aif", "get_population_aif"),
    # Backend
    "GPU_AVAILABLE": ("osipy.common.backend", "GPU_AVAILABLE"),
    "GPUConfig": ("osipy.common.backend", "GPUConfig"),
    "get_array_module": ("osipy.common.backend", "get_array_module"),
    "get_backend": ("osipy.common.backend", "get_backend"),
    "is_gpu_available": ("osipy.common.backend", "is_gpu_available"),
    "set_backend": ("osipy.common.backend", "set_backend"),
    "to_gpu": ("osipy.common.backend", "to_gpu"),
    "to_numpy": ("osipy.common.backend", "to_numpy"),
    # Dataset
    "PerfusionDataset": ("osipy.common.dataset", "PerfusionDataset"),
    # I/O
    "export_bids": ("osipy.common.io", "export_bids"),
    "load_dicom": ("osipy.common.io", "load_dicom"),
    "load_nifti": ("osipy.common.io", "load_nifti"),
    "load_perfusion": ("osipy.common.io", "load_perfusion"),
    # Parameter map
    "ParameterMap": ("osipy.common.parameter_map", "ParameterMap"),
    # Types
    "AIFType": ("osipy.common.types", "AIFType"),
    "AnalysisResult": ("osipy.common.types", "AnalysisResult"),
    "FittingMethod": ("osipy.common.types", "FittingMethod"),
    "LabelingType": ("osipy.common.types", "LabelingType"),
    "Modality": ("osipy.common.types", "Modality"),
    # DCE
    "ExtendedToftsModel": ("osipy.dce", "ExtendedToftsModel"),
    "PatlakModel": ("osipy.dce", "PatlakModel"),
    "ToftsModel": ("osipy.dce", "ToftsModel"),
    "TwoCompartmentModel": ("osipy.dce", "TwoCompartmentModel"),
    "compute_t1_map": ("osipy.dce", "compute_t1_map"),
    "fit_model": ("osipy.dce", "fit_model"),
    "get_model": ("osipy.dce", "get_model"),
    "list_models": ("osipy.dce", "list_models"),
    "dce_signal_to_concentration": ("osipy.dce", "signal_to_concentration"),
    # DSC
    "compute_perfusion_maps": ("osipy.dsc", "compute_perfusion_maps"),
    "correct_leakage": ("osipy.dsc", "correct_leakage"),
    "get_deconvolver": ("osipy.dsc", "get_deconvolver"),
    "list_deconvolvers": ("osipy.dsc", "list_deconvolvers"),
    "signal_to_delta_r2": ("osipy.dsc", "signal_to_delta_r2"),
    # IVIM
    "IVIMBiexponentialModel": ("osipy.ivim", "IVIMBiexponentialModel"),
    "IVIMFitParams": ("osipy.ivim", "IVIMFitParams"),
    "fit_ivim": ("osipy.ivim", "fit_ivim"),
    # Pipelines
    "ASLPipeline": ("osipy.pipeline", "ASLPipeline"),
    "DCEPipeline": ("osipy.pipeline", "DCEPipeline"),
    "DSCPipeline": ("osipy.pipeline", "DSCPipeline"),
    "IVIMPipeline": ("osipy.pipeline", "IVIMPipeline"),
    "run_analysis": ("osipy.pipeline", "run_analysis"),
}


def __getattr__(name: str) -> object:
    """Lazy import public API symbols on first access."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path)
        val = getattr(module, attr_name)
        # Cache on the module to avoid repeated lookups
        globals()[name] = val
        return val
    msg = f"module 'osipy' has no attribute {name!r}"
    raise AttributeError(msg)
