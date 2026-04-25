"""Philips-specific DICOM parsing for osipy.

This module extracts metadata from Philips MRI scanners,
including private tag parsing for acquisition parameters.

References
----------
Philips DICOM Conformance Statement
Private Tags: Groups 2001, 2005
"""

import contextlib
import logging
from typing import Any

import numpy as np

from osipy.common.io.vendors.base import VendorMetadata, VendorParser

logger = logging.getLogger(__name__)


class PhilipsParser(VendorParser):
    """Parser for Philips MRI DICOM files.

    Extracts metadata from Philips-specific private tags including
    sequence parameters, ASL info, and diffusion data.

    Notes
    -----
    Philips stores extended metadata in private tags. Common groups:
    - 2001: Philips Imaging DD 001 (main parameters)
    - 2005: Philips MR Imaging DD 005 (extended parameters)

    Philips ASL implementation varies significantly between product lines
    (e.g., 2D STAR vs 3D pCASL).

    Examples
    --------
    >>> from osipy.common.io.vendors.philips import PhilipsParser
    >>> parser = PhilipsParser()
    >>> if parser.can_parse(dcm):
    ...     metadata = parser.extract_metadata(dcm)
    """

    VENDOR_NAME = "Philips"

    # Private tag locations for Philips
    _DIFFUSION_B_VALUE = (0x2001, 0x1003)  # b-value
    _DIFFUSION_DIRECTION = (0x2001, 0x1004)  # Diffusion direction
    _DIFFUSION_B_FACTOR = (0x2005, 0x10B0)  # Alternative b-value location
    _SCALE_SLOPE = (0x2005, 0x100E)  # Scale slope for rescaling
    _SCALE_INTERCEPT = (0x2005, 0x100D)  # Scale intercept
    _STACK_SEQUENCE = (0x2001, 0x105F)  # Stack/Slice information
    _ASL_CONTEXT = (0x2005, 0x1429)  # ASL label/control indicator

    def can_parse(self, dcm: Any) -> bool:
        """Check if DICOM is from a Philips scanner.

        Parameters
        ----------
        dcm : pydicom.Dataset
            DICOM dataset to check.

        Returns
        -------
        bool
            True if Manufacturer tag contains 'PHILIPS'.
        """
        if not hasattr(dcm, "Manufacturer"):
            return False
        return "PHILIPS" in str(dcm.Manufacturer).upper()

    def extract_metadata(self, dcm: Any) -> VendorMetadata:
        """Extract Philips-specific metadata from DICOM.

        Parameters
        ----------
        dcm : pydicom.Dataset
            Philips DICOM dataset.

        Returns
        -------
        VendorMetadata
            Extracted metadata including Philips private tag info.
        """
        # Start with standard DICOM tags
        metadata = self._get_standard_metadata(dcm)

        # Extract diffusion parameters
        self._extract_diffusion_params(dcm, metadata)

        # Extract ASL parameters
        self._extract_asl_params(dcm, metadata)

        # Extract timing parameters
        self._extract_timing_params(dcm, metadata)

        # Extract sequence-specific info
        self._extract_sequence_info(dcm, metadata)

        return metadata

    def _extract_diffusion_params(self, dcm: Any, metadata: VendorMetadata) -> None:
        """Extract diffusion/IVIM parameters from Philips private tags.

        Parameters
        ----------
        dcm : pydicom.Dataset
            DICOM dataset.
        metadata : VendorMetadata
            Metadata object to populate.
        """
        # Try Philips private b-value tags
        b_value = self._safe_get_private_tag(dcm, 0x2001, 0x1003)
        if b_value is None:
            b_value = self._safe_get_private_tag(dcm, 0x2005, 0x10B0)

        if b_value is not None:
            with contextlib.suppress(ValueError, TypeError):
                metadata.b_values = np.array([float(b_value)])

        # Check standard DICOM tag as fallback
        if metadata.b_values is None and hasattr(dcm, "DiffusionBValue"):
            metadata.b_values = np.array([float(dcm.DiffusionBValue)])

        # Diffusion direction
        b_vec = self._safe_get_private_tag(dcm, 0x2001, 0x1004)
        if b_vec is not None:
            try:
                if hasattr(b_vec, "__len__") and len(b_vec) >= 3:
                    metadata.b_vectors = np.array([[float(b_vec[i]) for i in range(3)]])
            except (ValueError, TypeError, IndexError):
                pass

        # Standard tag fallback
        if metadata.b_vectors is None and hasattr(dcm, "DiffusionGradientOrientation"):
            try:
                vec = dcm.DiffusionGradientOrientation
                if len(vec) >= 3:
                    metadata.b_vectors = np.array([[float(vec[i]) for i in range(3)]])
            except (ValueError, TypeError, IndexError):
                pass

    def _extract_asl_params(self, dcm: Any, metadata: VendorMetadata) -> None:
        """Extract ASL parameters from Philips DICOM.

        Parameters
        ----------
        dcm : pydicom.Dataset
            DICOM dataset.
        metadata : VendorMetadata
            Metadata object to populate.
        """
        # Check sequence name and description for ASL indicators
        seq_name = metadata.sequence_name or ""
        series_desc = ""
        if hasattr(dcm, "SeriesDescription"):
            series_desc = str(dcm.SeriesDescription).lower()

        seq_name_lower = seq_name.lower()
        combined = seq_name_lower + " " + series_desc

        # Detect ASL labeling type
        # Philips uses different names: STAR (PASL), pCASL, etc.
        if "pcasl" in combined or "pseudo" in combined:
            metadata.labeling_type = "PCASL"
        elif "star" in combined or "pasl" in combined or "fair" in combined:
            metadata.labeling_type = "PASL"
        elif "asl" in combined:
            metadata.labeling_type = "PCASL"  # Default for modern Philips

        # Philips stores ASL parameters in private sequences
        # Post-labeling delay
        pld = self._safe_get_private_tag(dcm, 0x2005, 0x1060)
        if pld is not None:
            with contextlib.suppress(ValueError, TypeError):
                metadata.post_labeling_delay = float(pld)

        # Labeling duration
        label_dur = self._safe_get_private_tag(dcm, 0x2005, 0x1061)
        if label_dur is not None:
            with contextlib.suppress(ValueError, TypeError):
                metadata.labeling_duration = float(label_dur)

        # Try standard BIDS-compatible tags
        if metadata.post_labeling_delay is None and hasattr(dcm, "PostLabelingDelay"):
            metadata.post_labeling_delay = float(dcm.PostLabelingDelay)
        if metadata.labeling_duration is None and hasattr(dcm, "LabelingDuration"):
            metadata.labeling_duration = float(dcm.LabelingDuration)

        # Fallback: use TI for PASL
        if (
            metadata.post_labeling_delay is None
            and metadata.ti is not None
            and metadata.labeling_type in ["PASL", "STAR"]
        ):
            metadata.post_labeling_delay = metadata.ti

        # ASL context (label/control indicator)
        asl_context = self._safe_get_private_tag(dcm, 0x2005, 0x1429)
        if asl_context is not None:
            metadata.extra["asl_context"] = str(asl_context)

        # Background suppression - check for "BSUP" in scan options
        if hasattr(dcm, "ScanOptions"):
            scan_opts = str(dcm.ScanOptions).upper()
            if "BSUP" in scan_opts or "BS" in scan_opts:
                metadata.background_suppression = True

        # Philips-specific BS timing
        bs_time = self._safe_get_private_tag(dcm, 0x2005, 0x1062)
        if bs_time is not None:
            try:
                if hasattr(bs_time, "__iter__"):
                    metadata.bs_pulse_times = [float(t) for t in bs_time]
                else:
                    metadata.bs_pulse_times = [float(bs_time)]
                metadata.background_suppression = True
            except (ValueError, TypeError):
                pass

    def _extract_timing_params(self, dcm: Any, metadata: VendorMetadata) -> None:
        """Extract timing parameters from Philips DICOM.

        Parameters
        ----------
        dcm : pydicom.Dataset
            DICOM dataset.
        metadata : VendorMetadata
            Metadata object to populate.
        """
        # Slice timing from Philips private tag
        # Philips stores timing in the stack sequence
        stack_seq = self._safe_get_private_tag(dcm, 0x2001, 0x105F)
        if stack_seq is not None:
            # This is often a sequence - would need further parsing
            pass

        # Dynamic scan time
        dyn_time = self._safe_get_private_tag(dcm, 0x2001, 0x1008)
        if dyn_time is not None:
            with contextlib.suppress(ValueError, TypeError):
                metadata.temporal_resolution = float(dyn_time) / 1000.0

    def _extract_sequence_info(self, dcm: Any, metadata: VendorMetadata) -> None:
        """Extract sequence-specific information.

        Parameters
        ----------
        dcm : pydicom.Dataset
            DICOM dataset.
        metadata : VendorMetadata
            Metadata object to populate.
        """
        # Protocol name
        if hasattr(dcm, "ProtocolName"):
            metadata.extra["protocol_name"] = str(dcm.ProtocolName)

        # Series description
        if hasattr(dcm, "SeriesDescription"):
            metadata.extra["series_description"] = str(dcm.SeriesDescription)

        # Philips pulse sequence name
        pulse_seq = self._safe_get_private_tag(dcm, 0x2001, 0x100B)
        if pulse_seq is not None:
            metadata.extra["pulse_sequence_name"] = str(pulse_seq)

        # Scanning technique
        scan_tech = self._safe_get_private_tag(dcm, 0x2001, 0x1020)
        if scan_tech is not None:
            metadata.extra["scanning_technique"] = str(scan_tech)

        # Philips SENSE acceleration
        sense = self._safe_get_private_tag(dcm, 0x2005, 0x1009)
        if sense is not None:
            metadata.extra["sense_factor"] = str(sense)

        # Scale factors for proper rescaling (important for Philips).
        # Note: these values are *also* applied to pixel data during ingestion
        # in osipy.common.io.dicom._apply_pixel_scaling. They are retained in
        # metadata.extra for auditability.
        scale_slope = self._safe_get_private_tag(dcm, 0x2005, 0x100E)
        scale_intercept = self._safe_get_private_tag(dcm, 0x2005, 0x100D)
        if scale_slope is not None:
            with contextlib.suppress(ValueError, TypeError):
                metadata.extra["philips_scale_slope"] = float(scale_slope)
        if scale_intercept is not None:
            with contextlib.suppress(ValueError, TypeError):
                metadata.extra["philips_scale_intercept"] = float(scale_intercept)
