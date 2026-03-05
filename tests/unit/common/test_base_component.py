"""Tests for BaseComponent and the unified inheritance hierarchy."""

import pytest

from osipy.common.models.base import BaseComponent, BaseSignalModel


class TestBaseComponent:
    """Tests for BaseComponent ABC enforcement."""

    def test_cannot_instantiate_directly(self):
        """BaseComponent cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseComponent()  # type: ignore[abstract]

    def test_subclass_must_implement_name(self):
        """Subclass without name raises TypeError."""

        class BadComponent(BaseComponent):
            @property
            def reference(self) -> str:
                return "ref"

        with pytest.raises(TypeError):
            BadComponent()  # type: ignore[abstract]

    def test_subclass_must_implement_reference(self):
        """Subclass without reference raises TypeError."""

        class BadComponent(BaseComponent):
            @property
            def name(self) -> str:
                return "test"

        with pytest.raises(TypeError):
            BadComponent()  # type: ignore[abstract]

    def test_concrete_subclass_works(self):
        """Concrete subclass with both properties can be instantiated."""

        class GoodComponent(BaseComponent):
            @property
            def name(self) -> str:
                return "test"

            @property
            def reference(self) -> str:
                return "Test et al. 2024"

        comp = GoodComponent()
        assert comp.name == "test"
        assert comp.reference == "Test et al. 2024"


class TestBaseSignalModelInheritance:
    """Tests that BaseSignalModel inherits from BaseComponent."""

    def test_base_signal_model_is_component(self):
        assert issubclass(BaseSignalModel, BaseComponent)

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseSignalModel()  # type: ignore[abstract]


class TestDCEModelsAreComponents:
    """DCE models inherit from BaseComponent via BaseSignalModel."""

    def test_tofts_is_component(self):
        from osipy.dce.models.registry import get_model

        model = get_model("tofts")
        assert isinstance(model, BaseComponent)
        assert isinstance(model, BaseSignalModel)
        assert model.reference  # not empty

    def test_extended_tofts_is_component(self):
        from osipy.dce.models.registry import get_model

        model = get_model("extended_tofts")
        assert isinstance(model, BaseComponent)
        assert model.reference


class TestASLModelsAreComponents:
    """ASL models inherit from BaseComponent via BaseASLModel."""

    def test_pcasl_single_pld_is_component(self):
        from osipy.asl.quantification.registry import get_quantification_model

        model = get_quantification_model("pcasl_single_pld")
        assert isinstance(model, BaseComponent)
        assert isinstance(model, BaseSignalModel)
        assert model.reference
        assert hasattr(model, "labeling_type")
        assert model.labeling_type == "pcasl"

    def test_buxton_multi_pld_is_component(self):
        from osipy.asl.quantification.registry import get_quantification_model

        model = get_quantification_model("buxton_multi_pld")
        assert isinstance(model, BaseComponent)
        assert isinstance(model, BaseSignalModel)
        assert "CBF" in model.parameters
        assert "ATT" in model.parameters

    def test_backward_compat_att_registry(self):
        from osipy.asl.quantification.att_registry import get_att_model

        model = get_att_model("buxton")
        assert isinstance(model, BaseComponent)
        assert isinstance(model, BaseSignalModel)


class TestDSCComponentsAreComponents:
    """DSC components inherit from BaseComponent."""

    def test_deconvolvers_are_components(self):
        from osipy.dsc.deconvolution.registry import get_deconvolver

        for name in ["sSVD", "cSVD", "oSVD"]:
            deconv = get_deconvolver(name)
            assert isinstance(deconv, BaseComponent), f"{name} is not a BaseComponent"
            assert deconv.reference

    def test_leakage_correctors_are_components(self):
        from osipy.dsc.leakage.registry import get_leakage_corrector

        for name in ["bsw", "bidirectional"]:
            corrector = get_leakage_corrector(name)
            assert isinstance(corrector, BaseComponent), (
                f"{name} is not a BaseComponent"
            )
            assert corrector.reference

    def test_arrival_detectors_are_components(self):
        from osipy.dsc.arrival.registry import get_arrival_detector

        detector = get_arrival_detector("residue_peak")
        assert isinstance(detector, BaseComponent)
        assert detector.reference

    def test_dsc_convolution_model_is_signal_model(self):
        from osipy.dsc.deconvolution.signal_model import DSCConvolutionModel

        model = DSCConvolutionModel()
        assert isinstance(model, BaseComponent)
        assert isinstance(model, BaseSignalModel)
        assert model.parameters == ["CBF", "MTT", "Ta"]
        assert model.reference


class TestCalibrationComponentsAreComponents:
    """ASL calibration methods inherit from BaseComponent."""

    def test_m0_calibrations_are_components(self):
        from osipy.asl.calibration.registry import get_m0_calibration

        for name in ["voxelwise", "reference_region", "single"]:
            cal = get_m0_calibration(name)
            assert isinstance(cal, BaseComponent), f"{name} is not a BaseComponent"
            assert cal.reference


class TestAIFComponentsAreComponents:
    """AIF components inherit from BaseComponent."""

    def test_aif_detector_is_component(self):
        from osipy.common.aif.detection import MultiCriteriaAIFDetector

        detector = MultiCriteriaAIFDetector()
        assert isinstance(detector, BaseComponent)
        assert detector.reference
