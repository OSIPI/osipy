"""ASL analysis pipeline.

This module provides an end-to-end ASL analysis pipeline,
integrating M0 calibration and CBF quantification following
the OSIPI ASL Lexicon conventions.

References
----------
.. [1] OSIPI ASL Lexicon, https://osipi.github.io/ASL-Lexicon/
.. [2] Suzuki Y et al. MRM 2024;91(4):1411-1421.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from osipy.asl import (
    ASLQuantificationParams,
    ASLQuantificationResult,
    LabelingScheme,
    M0CalibrationParams,
    apply_m0_calibration,
    quantify_cbf,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class ASLPipelineConfig:
    """Configuration for ASL pipeline.

    Attributes
    ----------
    labeling_scheme : LabelingScheme
        ASL labeling scheme.
    pld : float
        Post-labeling delay in milliseconds.
    label_duration : float
        Labeling duration in milliseconds (for pCASL/CASL).
    t1_blood : float
        Blood T1 in milliseconds.
    labeling_efficiency : float
        Labeling efficiency.
    m0_method : str
        M0 calibration method: 'single', 'voxelwise', or 'reference_region'.
    output_dir : Path | None
        Output directory for results.
    """

    labeling_scheme: LabelingScheme = LabelingScheme.PCASL
    pld: float = 1800.0
    label_duration: float = 1800.0
    t1_blood: float = 1650.0
    labeling_efficiency: float = 0.85
    m0_method: str = "single"
    output_dir: Path | None = None


@dataclass
class ASLPipelineResult:
    """Result of ASL pipeline.

    Attributes
    ----------
    cbf_result : ASLQuantificationResult
        CBF quantification results.
    m0_map : NDArray | None
        M0 values used.
    config : ASLPipelineConfig
        Pipeline configuration used.
    """

    cbf_result: ASLQuantificationResult
    m0_map: "NDArray[np.floating[Any]] | None"
    config: ASLPipelineConfig


class ASLPipeline:
    """End-to-end ASL analysis pipeline.

    This pipeline performs:
    1. ASL difference computation (label - control)
    2. M0 calibration
    3. CBF quantification

    Examples
    --------
    >>> from osipy.pipeline import ASLPipeline, ASLPipelineConfig
    >>> from osipy.asl import LabelingScheme
    >>> config = ASLPipelineConfig(
    ...     labeling_scheme=LabelingScheme.PCASL,
    ...     pld=1800.0,
    ... )
    >>> pipeline = ASLPipeline(config)
    >>> result = pipeline.run(label_images, control_images, m0_image)
    """

    def __init__(self, config: ASLPipelineConfig | None = None) -> None:
        """Initialize ASL pipeline.

        Parameters
        ----------
        config : ASLPipelineConfig | None
            Pipeline configuration.
        """
        self.config = config or ASLPipelineConfig()

    def run(
        self,
        label_data: "NDArray[np.floating[Any]]",
        control_data: "NDArray[np.floating[Any]]",
        m0_data: "NDArray[np.floating[Any]] | float",
        mask: "NDArray[np.bool_] | None" = None,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> ASLPipelineResult:
        """Run ASL analysis pipeline.

        Parameters
        ----------
        label_data : NDArray
            Label images, shape (...) or (..., n_averages).
        control_data : NDArray
            Control images, shape (...) or (..., n_averages).
        m0_data : NDArray or float
            M0 calibration image or single value.
        mask : NDArray, optional
            Brain mask.
        progress_callback : Callable, optional
            Callback for progress updates.

        Returns
        -------
        ASLPipelineResult
            Pipeline results.
        """
        # Step 1: Compute ASL difference (label - control)
        if progress_callback:
            progress_callback("Computing Difference", 0.0)

        # Average if multiple acquisitions
        if label_data.ndim > 3:
            label_mean = np.mean(label_data, axis=-1)
            control_mean = np.mean(control_data, axis=-1)
        else:
            label_mean = label_data
            control_mean = control_data

        # ASL difference: control - label (label has lower signal)
        delta_m = control_mean - label_mean

        if progress_callback:
            progress_callback("Computing Difference", 1.0)

        # Step 2: M0 calibration
        if progress_callback:
            progress_callback("M0 Calibration", 0.0)

        if isinstance(m0_data, np.ndarray):
            m0_params = M0CalibrationParams(method=self.config.m0_method)
            _, m0_corrected = apply_m0_calibration(delta_m, m0_data, m0_params, mask)
            m0_value = m0_corrected
        else:
            m0_value = m0_data
            m0_corrected = None

        if progress_callback:
            progress_callback("M0 Calibration", 1.0)

        # Step 3: CBF quantification
        if progress_callback:
            progress_callback("CBF Quantification", 0.0)

        quant_params = ASLQuantificationParams(
            labeling_scheme=self.config.labeling_scheme,
            pld=self.config.pld,
            label_duration=self.config.label_duration,
            t1_blood=self.config.t1_blood,
            labeling_efficiency=self.config.labeling_efficiency,
        )

        cbf_result = quantify_cbf(
            delta_m=delta_m,
            m0=m0_value,
            params=quant_params,
            mask=mask,
        )

        if progress_callback:
            progress_callback("CBF Quantification", 1.0)

        return ASLPipelineResult(
            cbf_result=cbf_result,
            m0_map=m0_corrected,
            config=self.config,
        )

    def run_from_alternating(
        self,
        asl_data: "NDArray[np.floating[Any]]",
        m0_data: "NDArray[np.floating[Any]] | float",
        label_control_order: str = "label_first",
        mask: "NDArray[np.bool_] | None" = None,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> ASLPipelineResult:
        """Run ASL pipeline from interleaved label/control data.

        Parameters
        ----------
        asl_data : NDArray
            Interleaved ASL data, shape (..., n_pairs*2).
        m0_data : NDArray or float
            M0 calibration data.
        label_control_order : str
            Order: 'label_first' or 'control_first'.
        mask : NDArray, optional
            Brain mask.
        progress_callback : Callable, optional
            Progress callback.

        Returns
        -------
        ASLPipelineResult
            Pipeline results.
        """
        # Separate label and control
        if label_control_order == "label_first":
            label_data = asl_data[..., 0::2]  # Even indices
            control_data = asl_data[..., 1::2]  # Odd indices
        else:
            control_data = asl_data[..., 0::2]
            label_data = asl_data[..., 1::2]

        return self.run(
            label_data=label_data,
            control_data=control_data,
            m0_data=m0_data,
            mask=mask,
            progress_callback=progress_callback,
        )
