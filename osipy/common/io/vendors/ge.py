"""GE-specific DICOM parsing for osipy.

This module extracts metadata from GE MRI scanners,
including private tag parsing for acquisition parameters.

References
----------
GE DICOM Conformance Statement
Private Tags: Groups 0019, 0021, 0025, 0027, 0043
"""

import logging
from typing import Any

import numpy as np

from osipy.common.io.vendors.base import VendorMetadata, VendorParser

logger = logging.getLogger(__name__)


class GEParser(VendorParser):
    """Parser for GE MRI DICOM files.

    Extracts metadata from GE-specific private tags including
    sequence parameters and diffusion information.

    Notes
    -----
    GE stores extended metadata in private tags. Common groups:
    - 0019: Raw data parameters
    - 0021: Series parameters
    - 0025: Multi-phase parameters
    - 0027: Image parameters
    - 0043: Acquisition parameters, ASL info

    Examples
    --------
    >>> from osipy.common.io.vendors.ge import GEParser
    >>> parser = GEParser()
    >>> if parser.can_parse(dcm):
    ...     metadata = parser.extract_metadata(dcm)
    """

    VENDOR_NAME = "GE"

    # Private tag locations for GE
    _B_VALUE_TAG = (0x0043, 0x1039)  # b-value (often in element 4 of sequence)
    _USER_DATA_TAGS = (0x0043, 0x102A)  # User-defined parameters
    _PROTOCOL_DATA = (0x0025, 0x101B)  # Protocol data block

    def can_parse(self, dcm: Any) -> bool:
        """Check if DICOM is from a GE scanner.

        Parameters
        ----------
        dcm : pydicom.Dataset
            DICOM dataset to check.

        Returns
        -------
        bool
            True if Manufacturer tag contains 'GE'.
        """
        if not hasattr(dcm, "Manufacturer"):
            return False
        manufacturer = str(dcm.Manufacturer).upper()
        return "GE" in manufacturer or "GENERAL ELECTRIC" in manufacturer

    def extract_metadata(self, dcm: Any) -> VendorMetadata:
        """Extract GE-specific metadata from DICOM.

        Parameters
        ----------
        dcm : pydicom.Dataset
            GE DICOM dataset.

        Returns
        -------
        VendorMetadata
            Extracted metadata including GE private tag info.
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
        """Extract diffusion/IVIM parameters from GE private tags.

        Parameters
        ----------
        dcm : pydicom.Dataset
            DICOM dataset.
        metadata : VendorMetadata
            Metadata object to populate.
        """
        # GE stores b-value in (0043, 1039). It is a multi-valued element;
        # the b-value is in the first position and may be encoded
        # (e.g. b + 1e6 or b + 1e9 on some GE scanners — dcm2niix issue #149).
        # Values above 100000 s/mm^2 are non-physiological, so they are
        # decoded by taking modulo 100000 (matches dcm2niix behaviour).
        b_value = self._safe_get_private_tag(dcm, 0x0043, 0x1039)
        if b_value is not None:
            try:
                if hasattr(b_value, "__len__") and len(b_value) > 0:
                    raw = float(b_value[0])
                else:
                    raw = float(b_value)
                decoded = raw % 100000 if raw >= 100000 else raw
                if raw != decoded:
                    logger.debug(
                        "GE b-value appears encoded (raw=%s, decoded=%s); "
                        "applying modulo 100000 per dcm2niix convention.",
                        raw,
                        decoded,
                    )
                metadata.b_values = np.array([decoded])
            except (ValueError, TypeError, IndexError):
                pass

        # Check standard DICOM diffusion tags as fallback
        if metadata.b_values is None and hasattr(dcm, "DiffusionBValue"):
            metadata.b_values = np.array([float(dcm.DiffusionBValue)])

        # B-vectors from standard tag or GE private
        if hasattr(dcm, "DiffusionGradientOrientation"):
            try:
                vec = dcm.DiffusionGradientOrientation
                if len(vec) >= 3:
                    metadata.b_vectors = np.array([[float(vec[i]) for i in range(3)]])
            except (ValueError, TypeError, IndexError):
                pass

    def _extract_asl_params(self, dcm: Any, metadata: VendorMetadata) -> None:
        """Extract ASL parameters from GE DICOM.

        Parameters
        ----------
        dcm : pydicom.Dataset
            DICOM dataset.
        metadata : VendorMetadata
            Metadata object to populate.
        """
        # Check sequence name for ASL indicators
        seq_name = metadata.sequence_name or ""
        series_desc = ""
        if hasattr(dcm, "SeriesDescription"):
            series_desc = str(dcm.SeriesDescription).lower()

        seq_name_lower = seq_name.lower()
        combined = seq_name_lower + " " + series_desc

        # Detect ASL labeling type
        # GE 3D ASL is typically pCASL
        if "3dasl" in combined or "pcasl" in combined:
            metadata.labeling_type = "PCASL"
        elif "pasl" in combined or "fair" in combined:
            metadata.labeling_type = "PASL"
        elif "asl" in combined:
            metadata.labeling_type = "PCASL"  # GE default is typically pCASL

        # GE ASL parameters are often in private tags or require protocol parsing
        # Post-labeling delay from user-defined params (0043, 102A)
        user_data = self._safe_get_private_tag(dcm, 0x0043, 0x102A)
        if user_data is not None:
            self._parse_ge_user_data(user_data, metadata)

        # Try standard BIDS-compatible ASL tags
        if hasattr(dcm, "PostLabelingDelay"):
            metadata.post_labeling_delay = float(dcm.PostLabelingDelay)
        if hasattr(dcm, "LabelingDuration"):
            metadata.labeling_duration = float(dcm.LabelingDuration)

        # GE often uses inversion time for PLD in ASL sequences
        if (
            metadata.post_labeling_delay is None
            and metadata.ti is not None
            and metadata.labeling_type is not None
        ):
            metadata.post_labeling_delay = metadata.ti

        # Background suppression - GE typically applies this for 3D ASL
        if "3dasl" in combined:
            metadata.background_suppression = True

    def _parse_ge_user_data(self, user_data: Any, metadata: VendorMetadata) -> None:
        """Parse GE user-defined data block for ASL parameters.

        Parameters
        ----------
        user_data : Any
            User data from private tag.
        metadata : VendorMetadata
            Metadata object to populate.
        """
        try:
            # GE user data format varies by software version
            # Common structure is a list of floats with known positions
            if hasattr(user_data, "__len__") and len(user_data) > 10:
                # Position 7-8 often contain ASL timing info
                # This varies by GE product version
                pass  # Implementation depends on specific GE software version
        except (ValueError, TypeError, IndexError):
            pass

    def _extract_timing_params(self, dcm: Any, metadata: VendorMetadata) -> None:
        """Extract timing parameters from GE DICOM.

        Parameters
        ----------
        dcm : pydicom.Dataset
            DICOM dataset.
        metadata : VendorMetadata
            Metadata object to populate.
        """
        # Slice timing from private tag or standard location
        # GE stores this differently depending on sequence
        slice_time = self._safe_get_private_tag(dcm, 0x0021, 0x105E)
        if slice_time is not None:
            try:
                if hasattr(slice_time, "__iter__"):
                    metadata.slice_timing = [float(t) for t in slice_time]
            except (ValueError, TypeError):
                pass

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

        # GE internal pulse sequence name
        pulse_seq = self._safe_get_private_tag(dcm, 0x0019, 0x109C)
        if pulse_seq is not None:
            metadata.extra["pulse_sequence_name"] = str(pulse_seq)

        # Scanning options
        if hasattr(dcm, "ScanOptions"):
            metadata.extra["scan_options"] = str(dcm.ScanOptions)

        # GE-specific: ASSET acceleration
        asset = self._safe_get_private_tag(dcm, 0x0043, 0x1083)
        if asset is not None:
            metadata.extra["asset_r_factor"] = str(asset)
