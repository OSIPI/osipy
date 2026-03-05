"""Metadata mapping to acquisition parameters.

This module provides mapping from various metadata sources
(BIDS sidecar, vendor metadata, DICOM) to osipy AcquisitionParams.

"""

import logging
from typing import Any

import numpy as np

from osipy.common.io.metadata.defaults import DEFAULT_IVIM_PARAMS, get_default_params
from osipy.common.io.metadata.prompter import ParameterPrompter
from osipy.common.io.metadata.validator import ParameterValidator
from osipy.common.io.vendors.base import VendorMetadata
from osipy.common.types import (
    AcquisitionParams,
    AnyAcquisitionParams,
    ASLAcquisitionParams,
    DCEAcquisitionParams,
    DSCAcquisitionParams,
    IVIMAcquisitionParams,
    LabelingType,
    Modality,
)

logger = logging.getLogger(__name__)


class MetadataMapper:
    """Maps metadata from various sources to AcquisitionParams.

    This class implements a priority chain for metadata extraction:
    1. BIDS sidecar JSON (highest priority)
    2. Vendor-specific DICOM tags
    3. Standard DICOM tags
    4. User prompts (if interactive)
    5. Default values (lowest priority)

    Examples
    --------
    >>> from osipy.common.types import Modality
    >>> mapper = MetadataMapper(Modality.ASL)
    >>> params = mapper.map_to_acquisition_params(
    ...     vendor_metadata=vendor_meta,
    ...     bids_sidecar=sidecar_json,
    ... )
    >>> print(params.labeling_type)
    LabelingType.PCASL
    """

    def __init__(
        self,
        modality: Modality,
        interactive: bool = True,
        use_defaults: bool = True,
    ):
        """Initialize the mapper.

        Parameters
        ----------
        modality : Modality
            Perfusion imaging modality.
        interactive : bool, default=True
            Whether to prompt for missing required parameters.
        use_defaults : bool, default=True
            Whether to use default values for missing parameters.
        """
        self.modality = modality
        self.interactive = interactive
        self.use_defaults = use_defaults
        self.validator = ParameterValidator(modality)
        self.prompter = ParameterPrompter(modality, interactive, use_defaults)

    def map_to_acquisition_params(
        self,
        vendor_metadata: VendorMetadata | None = None,
        bids_sidecar: dict[str, Any] | None = None,
        dicom_metadata: dict[str, Any] | None = None,
        user_overrides: dict[str, Any] | None = None,
    ) -> AnyAcquisitionParams:
        """Map metadata to modality-specific AcquisitionParams.

        Parameters
        ----------
        vendor_metadata : VendorMetadata | None
            Vendor-specific metadata from DICOM parsing.
        bids_sidecar : dict[str, Any] | None
            BIDS sidecar JSON content.
        dicom_metadata : dict[str, Any] | None
            Standard DICOM metadata.
        user_overrides : dict[str, Any] | None
            User-specified parameter overrides (highest priority).

        Returns
        -------
        AnyAcquisitionParams
            Modality-specific acquisition parameters.

        Raises
        ------
        MetadataError
            If required parameters are missing and interactive=False.
        """
        # Merge metadata with priority chain
        merged = self._merge_metadata(
            vendor_metadata, bids_sidecar, dicom_metadata, user_overrides
        )

        # Validate and prompt for missing required
        validation = self.validator.validate(merged)
        if not validation.is_valid:
            if self.interactive:
                merged = self.prompter.prompt_for_missing(
                    merged, validation.missing_required
                )
            elif self.use_defaults:
                merged = self._fill_defaults(merged, validation.missing_required)

        # Log warnings
        for warning in validation.warnings:
            logger.warning(warning)

        # Create modality-specific params
        return self._create_params(merged)

    def _merge_metadata(
        self,
        vendor_metadata: VendorMetadata | None,
        bids_sidecar: dict[str, Any] | None,
        dicom_metadata: dict[str, Any] | None,
        user_overrides: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Merge metadata from all sources with priority.

        Priority (highest to lowest):
        1. User overrides
        2. BIDS sidecar
        3. Vendor metadata
        4. Standard DICOM

        Parameters
        ----------
        vendor_metadata : VendorMetadata | None
            Vendor-specific metadata.
        bids_sidecar : dict | None
            BIDS sidecar JSON.
        dicom_metadata : dict | None
            Standard DICOM tags.
        user_overrides : dict | None
            User overrides.

        Returns
        -------
        dict[str, Any]
            Merged parameter dictionary.
        """
        merged: dict[str, Any] = {}

        # Layer 4: Standard DICOM (lowest priority)
        if dicom_metadata:
            merged.update(self._map_dicom_fields(dicom_metadata))

        # Layer 3: Vendor metadata
        if vendor_metadata:
            merged.update(self._map_vendor_fields(vendor_metadata))

        # Layer 2: BIDS sidecar
        if bids_sidecar:
            merged.update(self._map_bids_fields(bids_sidecar))

        # Layer 1: User overrides (highest priority)
        if user_overrides:
            for key, value in user_overrides.items():
                if value is not None:
                    merged[key] = value

        return merged

    def _map_dicom_fields(self, dicom: dict[str, Any]) -> dict[str, Any]:
        """Map standard DICOM fields to parameter names.

        Parameters
        ----------
        dicom : dict[str, Any]
            DICOM metadata dictionary.

        Returns
        -------
        dict[str, Any]
            Mapped parameters.
        """
        mapping = {
            "RepetitionTime": "tr",
            "EchoTime": "te",
            "InversionTime": "ti",
            "FlipAngle": "flip_angle",
            "MagneticFieldStrength": "field_strength",
        }

        result: dict[str, Any] = {}
        for dicom_key, param_key in mapping.items():
            if dicom_key in dicom and dicom[dicom_key] is not None:
                result[param_key] = dicom[dicom_key]

        return result

    def _map_vendor_fields(self, vendor: VendorMetadata) -> dict[str, Any]:
        """Map VendorMetadata to parameter dictionary.

        Parameters
        ----------
        vendor : VendorMetadata
            Vendor-specific metadata.

        Returns
        -------
        dict[str, Any]
            Mapped parameters.
        """
        result: dict[str, Any] = {}

        # Common fields
        if vendor.tr is not None:
            result["tr"] = vendor.tr
        if vendor.te is not None:
            result["te"] = vendor.te
        if vendor.ti is not None:
            result["ti"] = vendor.ti
        if vendor.flip_angle is not None:
            result["flip_angle"] = vendor.flip_angle
        if vendor.field_strength is not None:
            result["field_strength"] = vendor.field_strength
        if vendor.temporal_resolution is not None:
            result["temporal_resolution"] = vendor.temporal_resolution

        # ASL fields
        if vendor.labeling_type is not None:
            result["labeling_type"] = vendor.labeling_type
        if vendor.post_labeling_delay is not None:
            result["post_labeling_delay"] = vendor.post_labeling_delay
        if vendor.labeling_duration is not None:
            result["labeling_duration"] = vendor.labeling_duration
        if vendor.background_suppression is not None:
            result["background_suppression"] = vendor.background_suppression
        if vendor.bolus_cutoff_flag is not None:
            result["bolus_cutoff_flag"] = vendor.bolus_cutoff_flag
        if vendor.bolus_cutoff_delay is not None:
            result["bolus_cutoff_delay"] = vendor.bolus_cutoff_delay

        # Diffusion fields
        if vendor.b_values is not None:
            result["b_values"] = vendor.b_values

        return result

    def _map_bids_fields(self, bids: dict[str, Any]) -> dict[str, Any]:
        """Map BIDS sidecar fields to parameter names.

        Parameters
        ----------
        bids : dict[str, Any]
            BIDS sidecar JSON content.

        Returns
        -------
        dict[str, Any]
            Mapped parameters.

        Notes
        -----
        BIDS ASL fields reference:
        https://bids-specification.readthedocs.io/en/stable/modality-specific-files/magnetic-resonance-imaging-data.html#arterial-spin-labeling-perfusion-data
        """
        result: dict[str, Any] = {}

        # Common BIDS fields
        bids_mapping = {
            "RepetitionTimePreparation": "tr",
            "RepetitionTime": "tr",
            "EchoTime": "te",
            "InversionTime": "ti",
            "FlipAngle": "flip_angle",
            "MagneticFieldStrength": "field_strength",
        }

        for bids_key, param_key in bids_mapping.items():
            if bids_key in bids and bids[bids_key] is not None:
                value = bids[bids_key]
                # BIDS uses seconds for TR/TE, convert to ms
                if (
                    param_key in ["tr", "te", "ti"]
                    and isinstance(value, (int, float))
                    and value < 100
                ):
                    # BIDS TR is in seconds, convert to ms
                    value = value * 1000
                result[param_key] = value

        # ASL-specific BIDS fields
        asl_mapping = {
            "ArterialSpinLabelingType": "labeling_type",
            "PostLabelingDelay": "post_labeling_delay",
            "LabelingDuration": "labeling_duration",
            "BackgroundSuppression": "background_suppression",
            "BackgroundSuppressionPulseTime": "bs_pulse_times",
            "BolusCutOffFlag": "bolus_cutoff_flag",
            "BolusCutOffDelayTime": "bolus_cutoff_delay",
            "M0Type": "m0_type",
            "VascularCrushing": "vascular_crushing",
            "LabelingEfficiency": "labeling_efficiency",
        }

        for bids_key, param_key in asl_mapping.items():
            if bids_key in bids and bids[bids_key] is not None:
                value = bids[bids_key]
                # Handle PLD - BIDS uses seconds, convert to ms
                if param_key in ["post_labeling_delay", "labeling_duration"]:
                    if isinstance(value, (int, float)):
                        if value < 10:  # Likely in seconds
                            value = value * 1000
                    elif isinstance(value, list):
                        value = [v * 1000 if v < 10 else v for v in value]
                result[param_key] = value

        # IVIM b-values (from DWI BIDS extension)
        if "bValues" in bids:
            result["b_values"] = np.array(bids["bValues"])
        elif "DiffusionBValue" in bids:
            result["b_values"] = np.array([bids["DiffusionBValue"]])

        return result

    def _fill_defaults(
        self, params: dict[str, Any], missing: list[str]
    ) -> dict[str, Any]:
        """Fill missing parameters with defaults.

        Parameters
        ----------
        params : dict[str, Any]
            Current parameters.
        missing : list[str]
            Missing parameter names.

        Returns
        -------
        dict[str, Any]
            Updated parameters with defaults.
        """
        defaults = get_default_params(self.modality)
        result = params.copy()

        for param_name in missing:
            if param_name in defaults:
                result[param_name] = defaults[param_name]
                log_level = logging.DEBUG if param_name == "b_values" else logging.INFO
                logger.log(
                    log_level, f"Using default {param_name}={defaults[param_name]}"
                )

        return result

    def _create_params(self, params: dict[str, Any]) -> AnyAcquisitionParams:
        """Create modality-specific AcquisitionParams.

        Parameters
        ----------
        params : dict[str, Any]
            Merged parameter dictionary.

        Returns
        -------
        AnyAcquisitionParams
            Modality-specific acquisition parameters.
        """
        # Base parameters
        base_kwargs = {
            "tr": params.get("tr"),
            "te": params.get("te"),
            "field_strength": params.get("field_strength"),
        }

        if self.modality == Modality.DCE:
            return DCEAcquisitionParams(
                **base_kwargs,
                flip_angles=[params.get("flip_angle")]
                if params.get("flip_angle")
                else [],
                baseline_frames=params.get("baseline_frames", 5),
                temporal_resolution=params.get("temporal_resolution", 1.0),
                relaxivity=params.get("relaxivity", 4.5),
            )

        elif self.modality == Modality.DSC:
            return DSCAcquisitionParams(
                **base_kwargs,
                baseline_frames=params.get("baseline_frames", 10),
            )

        elif self.modality == Modality.ASL:
            # Parse labeling type
            labeling_type = self._parse_labeling_type(params.get("labeling_type"))

            return ASLAcquisitionParams(
                **base_kwargs,
                labeling_type=labeling_type,
                pld=params.get("post_labeling_delay", 1800.0),
                labeling_duration=params.get("labeling_duration", 1800.0),
                background_suppression=params.get("background_suppression", False),
                bs_efficiency=params.get("bs_efficiency", 1.0),
                m0_scale=params.get("m0_scale"),
            )

        elif self.modality == Modality.IVIM:
            b_values = params.get("b_values")
            if b_values is None:
                b_values = np.array(DEFAULT_IVIM_PARAMS["b_values"])
            elif not isinstance(b_values, np.ndarray):
                b_values = np.array(b_values)

            return IVIMAcquisitionParams(
                **base_kwargs,
                b_values=b_values,
            )

        else:
            return AcquisitionParams(**base_kwargs)

    def _parse_labeling_type(self, value: Any) -> LabelingType:
        """Parse labeling type to enum.

        Parameters
        ----------
        value : Any
            Labeling type value from metadata.

        Returns
        -------
        LabelingType
            Parsed labeling type enum.
        """
        if value is None:
            return LabelingType.PCASL

        if isinstance(value, LabelingType):
            return value

        value_str = str(value).upper()

        mapping = {
            "PCASL": LabelingType.PCASL,
            "PSEUDO-CONTINUOUS": LabelingType.PCASL,
            "PSEUDOCONTINUOUS": LabelingType.PCASL,
            "PASL": LabelingType.PASL_FAIR,
            "PULSED": LabelingType.PASL_FAIR,
            "FAIR": LabelingType.PASL_FAIR,
            "EPISTAR": LabelingType.PASL_EPISTAR,
            "PICORE": LabelingType.PASL_PICORE,
            "VSASL": LabelingType.VSASL,
            "VELOCITY-SELECTIVE": LabelingType.VSASL,
        }

        return mapping.get(value_str, LabelingType.PCASL)
