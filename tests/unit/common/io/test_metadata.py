"""Tests for metadata mapping and validation.

Tests the metadata mapper, validator, prompter, and defaults.
"""

import numpy as np

from osipy.common.io.metadata.defaults import (
    get_default_params,
    get_default_value,
)
from osipy.common.io.metadata.mapper import MetadataMapper
from osipy.common.io.metadata.prompter import ParameterPrompter
from osipy.common.io.metadata.validator import (
    ParameterValidator,
    ValidationResult,
)
from osipy.common.io.vendors.base import VendorMetadata
from osipy.common.types import (
    ASLAcquisitionParams,
    DCEAcquisitionParams,
    IVIMAcquisitionParams,
    LabelingType,
    Modality,
)


class TestDefaults:
    """Tests for default parameter values."""

    def test_get_default_params_dce(self):
        """Test DCE defaults."""
        defaults = get_default_params(Modality.DCE)
        assert "tr" in defaults
        assert "flip_angle" in defaults
        assert defaults["relaxivity"] == 4.5

    def test_get_default_params_asl(self):
        """Test ASL defaults."""
        defaults = get_default_params(Modality.ASL)
        assert defaults["labeling_type"] == "PCASL"
        assert defaults["post_labeling_delay"] == 1800.0
        assert defaults["labeling_duration"] == 1800.0

    def test_get_default_params_ivim(self):
        """Test IVIM defaults."""
        defaults = get_default_params(Modality.IVIM)
        assert "b_values" in defaults
        assert 0 in defaults["b_values"]

    def test_get_default_value(self):
        """Test getting single default value."""
        pld = get_default_value(Modality.ASL, "post_labeling_delay")
        assert pld == 1800.0

        # Non-existent parameter
        result = get_default_value(Modality.ASL, "nonexistent")
        assert result is None


class TestValidator:
    """Tests for parameter validation."""

    def test_validate_asl_valid(self):
        """Test validation with valid ASL params."""
        validator = ParameterValidator(Modality.ASL)
        params = {
            "labeling_type": "PCASL",
            "post_labeling_delay": 1800.0,
        }
        result = validator.validate(params)
        assert result.is_valid
        assert len(result.missing_required) == 0

    def test_validate_asl_missing_required(self):
        """Test validation with missing required params."""
        validator = ParameterValidator(Modality.ASL)
        params = {"tr": 5000.0}  # Missing labeling_type and PLD
        result = validator.validate(params)
        assert not result.is_valid
        assert "labeling_type" in result.missing_required
        assert "post_labeling_delay" in result.missing_required

    def test_validate_dce_valid(self):
        """Test validation with valid DCE params."""
        validator = ParameterValidator(Modality.DCE)
        params = {
            "tr": 5.0,
            "flip_angle": 15.0,
        }
        result = validator.validate(params)
        assert result.is_valid

    def test_validate_ivim_valid(self):
        """Test validation with valid IVIM params."""
        validator = ParameterValidator(Modality.IVIM)
        params = {
            "b_values": [0, 50, 100, 200, 400, 800],
        }
        result = validator.validate(params)
        assert result.is_valid

    def test_validate_ivim_insufficient_b_values(self):
        """Test IVIM warning for insufficient b-values."""
        validator = ParameterValidator(Modality.IVIM)
        params = {
            "b_values": [0, 500],  # Only 2 b-values
        }
        result = validator.validate(params)
        assert result.is_valid  # Still valid, but with warning
        assert any("b-values" in w.lower() for w in result.warnings)

    def test_validate_range_warning(self):
        """Test out-of-range parameter warning."""
        validator = ParameterValidator(Modality.DCE)
        params = {
            "tr": 5.0,
            "flip_angle": 200.0,  # Out of typical range
        }
        result = validator.validate(params, check_ranges=True)
        assert "flip_angle" in result.out_of_range


class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_str_valid(self):
        """Test string representation for valid result."""
        result = ValidationResult(is_valid=True)
        assert "passed" in str(result).lower()

    def test_str_invalid(self):
        """Test string representation for invalid result."""
        result = ValidationResult(
            is_valid=False,
            missing_required=["tr", "flip_angle"],
        )
        result_str = str(result)
        assert "failed" in result_str.lower()
        assert "tr" in result_str


class TestPrompter:
    """Tests for interactive parameter prompting."""

    def test_non_interactive_returns_default(self):
        """Test non-interactive mode returns defaults."""
        prompter = ParameterPrompter(
            Modality.ASL,
            interactive=False,
            use_defaults=True,
        )
        value = prompter.prompt_for_value("post_labeling_delay")
        assert value == 1800.0  # Default ASL PLD

    def test_non_interactive_no_defaults(self):
        """Test non-interactive mode without defaults."""
        prompter = ParameterPrompter(
            Modality.ASL,
            interactive=False,
            use_defaults=False,
        )
        value = prompter.prompt_for_value("post_labeling_delay")
        assert value is None

    def test_parse_labeling_type(self):
        """Test labeling type parsing."""
        prompter = ParameterPrompter(Modality.ASL, interactive=False)

        assert prompter._parse_labeling_type("pcasl") == "PCASL"
        assert prompter._parse_labeling_type("PASL") == "PASL"
        assert prompter._parse_labeling_type("pseudo-continuous") == "PCASL"
        assert prompter._parse_labeling_type("FAIR") == "PASL"
        assert prompter._parse_labeling_type("vsasl") == "VSASL"

    def test_parse_b_values(self):
        """Test b-values parsing."""
        prompter = ParameterPrompter(Modality.IVIM, interactive=False)

        result = prompter._parse_b_values("0, 50, 100, 200")
        np.testing.assert_array_equal(result, [0, 50, 100, 200])

        result = prompter._parse_b_values("0;50;100;200")
        np.testing.assert_array_equal(result, [0, 50, 100, 200])

    def test_parse_bool(self):
        """Test boolean parsing."""
        prompter = ParameterPrompter(Modality.ASL, interactive=False)

        assert prompter._parse_bool("yes") is True
        assert prompter._parse_bool("true") is True
        assert prompter._parse_bool("1") is True
        assert prompter._parse_bool("no") is False
        assert prompter._parse_bool("false") is False

    def test_prompt_for_missing(self):
        """Test prompting for multiple missing params."""
        prompter = ParameterPrompter(
            Modality.ASL,
            interactive=False,
            use_defaults=True,
        )
        current = {"tr": 5000.0}
        missing = ["labeling_type", "post_labeling_delay"]

        updated = prompter.prompt_for_missing(current, missing)

        assert updated["tr"] == 5000.0
        assert updated["labeling_type"] == "PCASL"
        assert updated["post_labeling_delay"] == 1800.0


class TestMetadataMapper:
    """Tests for MetadataMapper."""

    def test_map_from_bids_sidecar(self):
        """Test mapping from BIDS sidecar JSON."""
        mapper = MetadataMapper(Modality.ASL, interactive=False)

        bids_sidecar = {
            "RepetitionTimePreparation": 4.5,  # seconds
            "ArterialSpinLabelingType": "PCASL",
            "PostLabelingDelay": 1.8,  # seconds - will be converted to ms
            "LabelingDuration": 1.8,
            "BackgroundSuppression": True,
        }

        params = mapper.map_to_acquisition_params(bids_sidecar=bids_sidecar)

        assert isinstance(params, ASLAcquisitionParams)
        assert params.labeling_type == LabelingType.PCASL
        assert params.pld == 1800.0  # Converted to ms
        assert params.labeling_duration == 1800.0
        assert params.background_suppression is True

    def test_map_from_vendor_metadata(self):
        """Test mapping from VendorMetadata."""
        mapper = MetadataMapper(Modality.ASL, interactive=False)

        vendor_meta = VendorMetadata(
            vendor="Siemens",
            tr=5000.0,
            te=12.0,
            labeling_type="PCASL",
            post_labeling_delay=1800.0,
            labeling_duration=1800.0,
        )

        params = mapper.map_to_acquisition_params(vendor_metadata=vendor_meta)

        assert isinstance(params, ASLAcquisitionParams)
        assert params.tr == 5000.0
        assert params.labeling_type == LabelingType.PCASL

    def test_priority_bids_over_vendor(self):
        """Test that BIDS takes priority over vendor metadata."""
        mapper = MetadataMapper(Modality.ASL, interactive=False)

        vendor_meta = VendorMetadata(
            vendor="Siemens",
            post_labeling_delay=1500.0,  # Vendor value
        )

        bids_sidecar = {
            "ArterialSpinLabelingType": "PCASL",
            "PostLabelingDelay": 2.0,  # BIDS value (seconds)
        }

        params = mapper.map_to_acquisition_params(
            vendor_metadata=vendor_meta,
            bids_sidecar=bids_sidecar,
        )

        # BIDS should take priority
        assert params.pld == 2000.0  # 2.0s = 2000ms

    def test_map_dce_params(self):
        """Test mapping DCE parameters."""
        mapper = MetadataMapper(Modality.DCE, interactive=False)

        bids_sidecar = {
            "RepetitionTime": 0.005,  # 5ms in seconds
            "FlipAngle": 15.0,
        }

        params = mapper.map_to_acquisition_params(bids_sidecar=bids_sidecar)

        assert isinstance(params, DCEAcquisitionParams)
        assert params.flip_angles == [15.0]

    def test_map_ivim_params(self):
        """Test mapping IVIM parameters."""
        mapper = MetadataMapper(Modality.IVIM, interactive=False)

        bids_sidecar = {
            "bValues": [0, 50, 100, 200, 400, 800],
        }

        params = mapper.map_to_acquisition_params(bids_sidecar=bids_sidecar)

        assert isinstance(params, IVIMAcquisitionParams)
        np.testing.assert_array_equal(params.b_values, [0, 50, 100, 200, 400, 800])

    def test_map_with_defaults(self):
        """Test that defaults are used when metadata missing."""
        mapper = MetadataMapper(
            Modality.ASL,
            interactive=False,
            use_defaults=True,
        )

        # Empty sidecar - should use defaults
        params = mapper.map_to_acquisition_params(bids_sidecar={})

        assert isinstance(params, ASLAcquisitionParams)
        assert params.labeling_type == LabelingType.PCASL  # Default
        assert params.pld == 1800.0  # Default

    def test_parse_labeling_type_enum(self):
        """Test labeling type parsing to enum."""
        mapper = MetadataMapper(Modality.ASL, interactive=False)

        assert mapper._parse_labeling_type("PCASL") == LabelingType.PCASL
        assert mapper._parse_labeling_type("PASL") == LabelingType.PASL_FAIR
        assert mapper._parse_labeling_type("FAIR") == LabelingType.PASL_FAIR
        assert mapper._parse_labeling_type("EPISTAR") == LabelingType.PASL_EPISTAR
        assert mapper._parse_labeling_type("VSASL") == LabelingType.VSASL
        assert mapper._parse_labeling_type(None) == LabelingType.PCASL

        # Already an enum
        assert mapper._parse_labeling_type(LabelingType.PCASL) == LabelingType.PCASL
