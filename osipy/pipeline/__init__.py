"""Unified perfusion analysis pipeline.

This module provides end-to-end processing pipelines for perfusion MRI analysis,
integrating data loading, preprocessing, model fitting, and results export.

Classes
-------
DCEPipeline
    End-to-end DCE-MRI analysis pipeline.
DSCPipeline
    End-to-end DSC-MRI analysis pipeline.
ASLPipeline
    End-to-end ASL analysis pipeline.
IVIMPipeline
    End-to-end IVIM analysis pipeline.

Functions
---------
run_analysis
    Run perfusion analysis with automatic modality detection.

References
----------
OSIPI (Open Science Initiative for Perfusion Imaging)
https://osipi.org/
"""

from osipy.pipeline.asl_pipeline import ASLPipeline, ASLPipelineConfig
from osipy.pipeline.dce_pipeline import DCEPipeline, DCEPipelineConfig
from osipy.pipeline.dsc_pipeline import DSCPipeline, DSCPipelineConfig
from osipy.pipeline.ivim_pipeline import IVIMPipeline, IVIMPipelineConfig
from osipy.pipeline.runner import AnalysisResult, PipelineResult, run_analysis

__all__ = [
    "ASLPipeline",
    "ASLPipelineConfig",
    "AnalysisResult",
    "DCEPipeline",
    "DCEPipelineConfig",
    "DSCPipeline",
    "DSCPipelineConfig",
    "IVIMPipeline",
    "IVIMPipelineConfig",
    "PipelineResult",
    "run_analysis",
]
