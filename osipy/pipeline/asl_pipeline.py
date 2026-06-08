"""ASL analysis pipeline.

This module provides an end-to-end ASL analysis pipeline,
integrating M0 calibration and CBF quantification following
the OSIPI ASL Lexicon conventions.

References
----------
.. [1] OSIPI ASL Lexicon, https://osipi.github.io/ASL-Lexicon/
.. [2] Suzuki Y et al. MRM 2024;91(5):1743-1760.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from osipy.asl import (
    ASLQuantificationParams,
    ASLQuantificationResult,
    LabelingScheme,
    apply_m0_calibration,
    quantify_cbf,
)
from osipy.asl.config import (
    PairwiseDifferenceConfig,
    SingleM0Config,
    SinglePLDConfig,
    m0_params_from_config,
)
from osipy.common.config import MethodConfig

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class ASLPipelineConfig:
    """Configuration for ASL pipeline.

    Defaults match the ISMRM ASL consensus paper (Alsop et al.,
    MRM 2015;73:102, PMC4190138) for **healthy adults under 70 years**.
    For older adults or clinical-population studies, Alsop Table 1
    recommends PLD=2000 ms; override ``pld`` accordingly.

    Attributes
    ----------
    labeling_scheme : LabelingScheme
        ASL labeling scheme.
    pld : float
        Post-labeling delay in milliseconds. Default 1800 ms
        (Alsop 2015 healthy-adult value; use 2000 for clinical adults).
    label_duration : float
        Labeling duration in milliseconds (for pCASL/CASL).
        Default 1800 ms (Alsop 2015 recommendation).
    t1_blood : float
        Blood T1 in milliseconds. Default 1650 ms (Alsop 2015 at 3T).
    t1_tissue : float
        Tissue T1 in milliseconds. Default 1330 ms (gray matter at 3T).
    labeling_efficiency : float
        Labeling efficiency. Default 0.85 for PCASL (Alsop 2015).
    partition_coefficient : float
        Blood-brain partition coefficient (lambda). Default 0.9 mL/g.
    m0 : MethodConfig
        M0 calibration config (discriminated by ``method``:
        ``single``/``voxelwise``/``reference_region``), carrying its knobs
        (``t1_tissue``, ``tr_m0``, ``te_m0``, ``reference_region``).
    difference : MethodConfig
        Label/control difference config (discriminated by ``method``:
        ``pairwise``/``surround``/``mean``).
    quantification : MethodConfig
        Quantification mode config (discriminated by ``mode``:
        ``single_pld``/``multi_pld``); ``multi_pld`` carries the PLD schedule
        and the ATT model and routes to the multi-PLD path.
    output_dir : Path | None
        Output directory for results.
    """

    labeling_scheme: LabelingScheme = LabelingScheme.PCASL
    pld: float = 1800.0
    label_duration: float = 1800.0
    t1_blood: float = 1650.0
    t1_tissue: float = 1330.0
    labeling_efficiency: float = 0.85
    partition_coefficient: float = 0.9
    m0: MethodConfig = field(default_factory=SingleM0Config)
    difference: MethodConfig = field(default_factory=PairwiseDifferenceConfig)
    quantification: MethodConfig = field(default_factory=SinglePLDConfig)
    output_dir: Path | None = None


@dataclass
class ASLPipelineResult:
    """Result of ASL pipeline.

    Attributes
    ----------
    cbf_result : ASLQuantificationResult
        CBF quantification results (single-PLD path).
    m0_map : NDArray | None
        M0 values used.
    config : ASLPipelineConfig
        Pipeline configuration used.
    att_map : ParameterMap | None
        Arterial transit time (ATT) map in ms. Populated only when the
        ``multi_pld`` quantification mode is selected.
    """

    cbf_result: ASLQuantificationResult
    m0_map: "NDArray[np.floating[Any]] | None"
    config: ASLPipelineConfig
    att_map: Any = None


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
        is_multi_pld = self.config.quantification.mode == "multi_pld"  # type: ignore[attr-defined]

        # Step 1: Compute ASL difference (control - label).
        if progress_callback:
            progress_callback("Computing Difference", 0.0)

        if is_multi_pld:
            # Keep the PLD axis: one control-label difference per PLD volume.
            delta_m = control_data - label_data
            if mask is not None:
                delta_m = np.where(mask[..., np.newaxis], delta_m, 0)
        else:
            # Collapse to 3D via the configured difference method.
            delta_m = self._compute_difference(label_data, control_data, mask)

        if progress_callback:
            progress_callback("Computing Difference", 1.0)

        # Step 2: M0 calibration
        if progress_callback:
            progress_callback("M0 Calibration", 0.0)

        if isinstance(m0_data, np.ndarray):
            m0_params = m0_params_from_config(self.config.m0)
            # For multi-PLD, calibrate per PLD volume using a 3D M0 image.
            cal_target = delta_m[..., 0] if is_multi_pld else delta_m
            _, m0_corrected = apply_m0_calibration(cal_target, m0_data, m0_params, mask)
            m0_value = m0_corrected
        else:
            m0_value = m0_data
            m0_corrected = None

        if progress_callback:
            progress_callback("M0 Calibration", 1.0)

        # Multi-PLD path: route to Buxton CBF + ATT estimation.
        if is_multi_pld:
            return self._run_multi_pld(delta_m, m0_value, m0_corrected, mask)

        # Step 3: CBF quantification (single-PLD path)
        if progress_callback:
            progress_callback("CBF Quantification", 0.0)

        quant_params = ASLQuantificationParams(
            labeling_scheme=self.config.labeling_scheme,
            pld=self.config.pld,
            label_duration=self.config.label_duration,
            t1_blood=self.config.t1_blood,
            t1_tissue=self.config.t1_tissue,
            labeling_efficiency=self.config.labeling_efficiency,
            partition_coefficient=self.config.partition_coefficient,
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

    def _compute_difference(
        self,
        label_data: "NDArray[np.floating[Any]]",
        control_data: "NDArray[np.floating[Any]]",
        mask: "NDArray[np.bool_] | None",
    ) -> "NDArray[np.floating[Any]]":
        """Compute delta-M from separated label/control via the difference registry.

        Re-interleaves the label and control volumes into a single 4D stack with
        a matching aslcontext list, then dispatches to the configured difference
        method (``pairwise``/``surround``/``mean``) via
        :func:`compute_control_label_difference`. For 3D (single-pair) inputs the
        registry is bypassed with a direct ``control - label`` subtraction.
        """
        from osipy.asl import compute_control_label_difference

        method = self.config.difference.method  # type: ignore[attr-defined]

        # Single pair (no averaging dimension): direct subtraction.
        if label_data.ndim <= 3:
            delta_m = control_data - label_data
            if mask is not None:
                delta_m = np.where(mask, delta_m, 0)
            return delta_m

        # Interleave control/label into a 4D timeseries with aslcontext so the
        # registered difference method can operate on the raw volumes.
        n_pairs = min(label_data.shape[-1], control_data.shape[-1])
        spatial = label_data.shape[:-1]
        interleaved = np.empty((*spatial, 2 * n_pairs), dtype=float)
        interleaved[..., 0::2] = control_data[..., :n_pairs]
        interleaved[..., 1::2] = label_data[..., :n_pairs]
        context = ["control", "label"] * n_pairs
        return compute_control_label_difference(
            interleaved, context, method=method, mask=mask
        )

    def _run_multi_pld(
        self,
        delta_m: "NDArray[np.floating[Any]]",
        m0_value: "NDArray[np.floating[Any]] | float",
        m0_corrected: "NDArray[np.floating[Any]] | None",
        mask: "NDArray[np.bool_] | None",
    ) -> ASLPipelineResult:
        """Run the multi-PLD path (Buxton + ATT estimation).

        Builds :class:`MultiPLDParams` from the pipeline config and the selected
        ``multi_pld`` quantification config (PLD schedule), fits CBF + ATT via
        :func:`quantify_multi_pld`, and packs the result so ``cbf_result``
        carries CBF and ``att_map`` carries ATT.
        """
        from osipy.asl.quantification.multi_pld import (
            MultiPLDParams,
            quantify_multi_pld,
        )

        quant_cfg = self.config.quantification
        params = MultiPLDParams(
            labeling_scheme=self.config.labeling_scheme,
            plds=np.asarray(quant_cfg.plds, dtype=float),  # type: ignore[attr-defined]
            label_duration=self.config.label_duration,
            t1_blood=self.config.t1_blood,
            t1_tissue=self.config.t1_tissue,
            partition_coefficient=self.config.partition_coefficient,
            labeling_efficiency=self.config.labeling_efficiency,
        )
        result = quantify_multi_pld(
            delta_m=delta_m, m0=m0_value, params=params, mask=mask
        )
        cbf_result = ASLQuantificationResult(
            cbf_map=result.cbf_map,
            quality_mask=result.quality_mask,
            m0_used=m0_corrected,
            scaling_factor=6000.0,
        )
        return ASLPipelineResult(
            cbf_result=cbf_result,
            m0_map=m0_corrected,
            config=self.config,
            att_map=result.att_map,
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
