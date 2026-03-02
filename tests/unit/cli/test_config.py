"""Unit tests for osipy CLI configuration and parser."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import pytest
import yaml
from pydantic import ValidationError

if TYPE_CHECKING:
    from pathlib import Path

from osipy.cli.config import (
    ASLPipelineYAML,
    BackendConfig,
    BayesianIVIMFittingConfig,
    DataConfig,
    DCEFittingConfig,
    DCEPipelineYAML,
    DSCPipelineYAML,
    IVIMFittingConfig,
    IVIMPipelineYAML,
    LoggingConfig,
    OutputConfig,
    PipelineConfig,
    dump_defaults,
    load_config,
)
from osipy.cli.main import create_parser

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_config(tmp_path: Path):
    """Create a helper to write YAML configs to temp files."""

    def _write(content: str) -> Path:
        p = tmp_path / "config.yaml"
        p.write_text(textwrap.dedent(content))
        return p

    return _write


# ---------------------------------------------------------------------------
# TestLoadConfig
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """Tests for load_config() YAML parsing and validation."""

    def test_load_minimal_dce_config(self, tmp_config) -> None:
        """Minimal valid DCE config requires only modality."""
        path = tmp_config("modality: dce\n")
        config = load_config(path)
        assert config.modality == "dce"

    def test_load_full_dce_config(self, tmp_config) -> None:
        """Full DCE config with all fields loads correctly."""
        path = tmp_config("""\
            modality: dce
            pipeline:
              model: extended_tofts
              t1_mapping_method: vfa
              aif_source: population
              population_aif: parker
              save_intermediate: true
              acquisition:
                tr: 5.0
                flip_angles: [2, 5, 10, 15]
                baseline_frames: 5
                relaxivity: 4.5
                t1_assumed: 1400.0
            data:
              format: nifti
              mask: mask.nii.gz
            output:
              format: nifti
            backend:
              force_cpu: true
            logging:
              level: DEBUG
        """)
        config = load_config(path)
        assert config.modality == "dce"
        assert config.data.mask == "mask.nii.gz"
        assert config.backend.force_cpu is True
        assert config.logging.level == "DEBUG"

    def test_load_dsc_config(self, tmp_config) -> None:
        """Valid DSC config loads successfully."""
        path = tmp_config("""\
            modality: dsc
            pipeline:
              te: 25.0
              deconvolution_method: oSVD
        """)
        config = load_config(path)
        assert config.modality == "dsc"

    def test_load_asl_config(self, tmp_config) -> None:
        """Valid ASL config loads successfully."""
        path = tmp_config("""\
            modality: asl
            pipeline:
              labeling_scheme: pcasl
              pld: 1800.0
        """)
        config = load_config(path)
        assert config.modality == "asl"

    def test_load_ivim_config(self, tmp_config) -> None:
        """Valid IVIM config loads successfully."""
        path = tmp_config("""\
            modality: ivim
            pipeline:
              fitting_method: segmented
              b_threshold: 200.0
        """)
        config = load_config(path)
        assert config.modality == "ivim"

    def test_load_nonexistent_file(self) -> None:
        """Missing config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_load_invalid_yaml(self, tmp_config) -> None:
        """Invalid YAML syntax raises an error."""
        path = tmp_config("modality: [invalid yaml\n")
        with pytest.raises(yaml.YAMLError):
            load_config(path)

    def test_load_empty_config(self, tmp_config) -> None:
        """Empty YAML file raises an error (not a mapping)."""
        path = tmp_config("")
        with pytest.raises((ValueError, ValidationError)):
            load_config(path)


# ---------------------------------------------------------------------------
# TestPipelineConfig
# ---------------------------------------------------------------------------


class TestPipelineConfig:
    """Tests for PipelineConfig validation."""

    def test_invalid_modality(self) -> None:
        """Unknown modality raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid modality"):
            PipelineConfig(modality="xxx")

    def test_valid_modalities(self) -> None:
        """All four modalities are accepted."""
        for mod in ("dce", "dsc", "asl", "ivim"):
            config = PipelineConfig(modality=mod)
            assert config.modality == mod

    def test_default_sections(self) -> None:
        """Default sub-configs are created when not provided."""
        config = PipelineConfig(modality="dce")
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.output, OutputConfig)
        assert isinstance(config.backend, BackendConfig)
        assert isinstance(config.logging, LoggingConfig)

    @pytest.mark.parametrize(
        "modality,cls",
        [
            ("dce", DCEPipelineYAML),
            ("dsc", DSCPipelineYAML),
            ("asl", ASLPipelineYAML),
            ("ivim", IVIMPipelineYAML),
        ],
    )
    def test_get_modality_config(self, modality: str, cls: type) -> None:
        """get_modality_config returns the correct class for each modality."""
        config = PipelineConfig(modality=modality)
        mc = config.get_modality_config()
        assert isinstance(mc, cls)


# ---------------------------------------------------------------------------
# TestDCEPipelineYAML
# ---------------------------------------------------------------------------


class TestDCEPipelineYAML:
    """Tests for DCE pipeline config validation."""

    def test_defaults(self) -> None:
        """Default DCE config values match expected."""
        cfg = DCEPipelineYAML()
        assert cfg.model == "extended_tofts"
        assert cfg.t1_mapping_method == "vfa"
        assert cfg.aif_source == "population"
        assert cfg.population_aif == "parker"
        assert cfg.save_intermediate is False

    def test_invalid_model(self) -> None:
        """Invalid model name raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid DCE model"):
            DCEPipelineYAML(model="nonexistent_model")

    def test_invalid_t1_method(self) -> None:
        """Invalid T1 mapping method raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid T1 mapping method"):
            DCEPipelineYAML(t1_mapping_method="invalid_method")

    def test_invalid_aif_source(self) -> None:
        """Invalid AIF source raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid AIF source"):
            DCEPipelineYAML(aif_source="invalid_source")

    def test_valid_models(self) -> None:
        """All registered DCE model names are accepted."""
        from osipy.dce import list_models

        for name in list_models():
            cfg = DCEPipelineYAML(model=name)
            assert cfg.model == name

    def test_acquisition_defaults(self) -> None:
        """Acquisition sub-model has correct defaults."""
        cfg = DCEPipelineYAML()
        assert cfg.acquisition.baseline_frames == 5
        assert cfg.acquisition.relaxivity == 4.5
        assert cfg.acquisition.tr is None
        assert cfg.acquisition.flip_angles is None
        assert cfg.acquisition.t1_assumed is None


# ---------------------------------------------------------------------------
# TestDSCPipelineYAML
# ---------------------------------------------------------------------------


class TestDSCPipelineYAML:
    """Tests for DSC pipeline config validation."""

    def test_defaults(self) -> None:
        """Default DSC config values match expected."""
        cfg = DSCPipelineYAML()
        assert cfg.te == 30.0
        assert cfg.deconvolution_method == "oSVD"
        assert cfg.apply_leakage_correction is True
        assert cfg.svd_threshold == 0.2
        assert cfg.baseline_frames == 10
        assert cfg.hematocrit_ratio == 0.73

    def test_invalid_deconvolution_method(self) -> None:
        """Invalid deconvolution method raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid deconvolution method"):
            DSCPipelineYAML(deconvolution_method="invalid_method")

    def test_valid_deconvolution_methods(self) -> None:
        """All registered deconvolution methods are accepted."""
        from osipy.dsc import list_deconvolvers

        for name in list_deconvolvers():
            cfg = DSCPipelineYAML(deconvolution_method=name)
            assert cfg.deconvolution_method == name


# ---------------------------------------------------------------------------
# TestASLPipelineYAML
# ---------------------------------------------------------------------------


class TestASLPipelineYAML:
    """Tests for ASL pipeline config validation."""

    def test_defaults(self) -> None:
        """Default ASL config values match expected."""
        cfg = ASLPipelineYAML()
        assert cfg.labeling_scheme == "pcasl"
        assert cfg.pld == 1800.0
        assert cfg.label_duration == 1800.0
        assert cfg.t1_blood == 1650.0
        assert cfg.labeling_efficiency == 0.85
        assert cfg.m0_method == "single"
        assert cfg.t1_tissue == 1330.0
        assert cfg.partition_coefficient == 0.9
        assert cfg.difference_method == "pairwise"
        assert cfg.label_control_order == "label_first"

    def test_invalid_labeling_scheme(self) -> None:
        """Invalid labeling scheme raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid labeling scheme"):
            ASLPipelineYAML(labeling_scheme="invalid")

    def test_invalid_m0_method(self) -> None:
        """Invalid M0 method raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid M0 method"):
            ASLPipelineYAML(m0_method="invalid")

    def test_invalid_label_control_order(self) -> None:
        """Invalid label/control order raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid label/control order"):
            ASLPipelineYAML(label_control_order="invalid")

    def test_valid_labeling_schemes(self) -> None:
        """All valid labeling schemes are accepted."""
        for scheme in ("pasl", "casl", "pcasl"):
            cfg = ASLPipelineYAML(labeling_scheme=scheme)
            assert cfg.labeling_scheme == scheme


# ---------------------------------------------------------------------------
# TestIVIMPipelineYAML
# ---------------------------------------------------------------------------


class TestIVIMPipelineYAML:
    """Tests for IVIM pipeline config validation."""

    def test_defaults(self) -> None:
        """Default IVIM config values match expected."""
        cfg = IVIMPipelineYAML()
        assert cfg.fitting_method == "segmented"
        assert cfg.b_threshold == 200.0
        assert cfg.normalize_signal is True
        assert cfg.fitting.max_iterations == 500
        assert cfg.fitting.tolerance == 1e-6

    def test_invalid_fitting_method(self) -> None:
        """Invalid fitting method raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid fitting method"):
            IVIMPipelineYAML(fitting_method="invalid")

    def test_valid_fitting_methods(self) -> None:
        """All valid fitting methods are accepted."""
        for method in ("segmented", "full", "bayesian"):
            cfg = IVIMPipelineYAML(fitting_method=method)
            assert cfg.fitting_method == method


# ---------------------------------------------------------------------------
# TestDCEFittingConfig
# ---------------------------------------------------------------------------


class TestDCEFittingConfig:
    """Tests for DCE fitting configuration."""

    def test_defaults(self) -> None:
        """Default DCE fitting config values match expected."""
        cfg = DCEFittingConfig()
        assert cfg.fitter == "lm"
        assert cfg.max_iterations == 100
        assert cfg.tolerance == 1e-6
        assert cfg.r2_threshold == 0.5
        assert cfg.bounds is None
        assert cfg.initial_guess is None

    def test_invalid_fitter(self) -> None:
        """Invalid fitter name raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid fitter"):
            DCEFittingConfig(fitter="nonexistent_fitter")

    def test_valid_fitters(self) -> None:
        """All registered fitters are accepted."""
        from osipy.common.fitting.registry import list_fitters

        for name in list_fitters():
            cfg = DCEFittingConfig(fitter=name)
            assert cfg.fitter == name

    def test_bounds_override(self) -> None:
        """Bounds override parses correctly."""
        cfg = DCEFittingConfig(bounds={"Ktrans": [0.0, 3.0], "ve": [0.01, 0.8]})
        assert cfg.bounds["Ktrans"] == [0.0, 3.0]
        assert cfg.bounds["ve"] == [0.01, 0.8]

    def test_bounds_validation_wrong_length(self) -> None:
        """Bounds with != 2 elements raises ValidationError."""
        with pytest.raises(ValidationError, match="must be"):
            DCEFittingConfig(bounds={"Ktrans": [0.0, 1.0, 2.0]})

    def test_bounds_validation_lower_gt_upper(self) -> None:
        """Lower bound > upper bound raises ValidationError."""
        with pytest.raises(ValidationError, match="Lower bound > upper bound"):
            DCEFittingConfig(bounds={"Ktrans": [5.0, 1.0]})

    def test_initial_guess_override(self) -> None:
        """Initial guess override parses correctly."""
        cfg = DCEFittingConfig(initial_guess={"Ktrans": 0.05, "ve": 0.3})
        assert cfg.initial_guess["Ktrans"] == 0.05
        assert cfg.initial_guess["ve"] == 0.3

    def test_dce_yaml_includes_fitting(self) -> None:
        """DCEPipelineYAML includes fitting sub-config with defaults."""
        cfg = DCEPipelineYAML()
        assert isinstance(cfg.fitting, DCEFittingConfig)
        assert cfg.fitting.fitter == "lm"

    def test_dce_yaml_with_fitting_section(self) -> None:
        """DCE YAML with explicit fitting section loads correctly."""
        cfg = DCEPipelineYAML(
            fitting={"fitter": "lm", "max_iterations": 200}
        )
        assert cfg.fitting.fitter == "lm"
        assert cfg.fitting.max_iterations == 200
        assert cfg.fitting.tolerance == 1e-6  # default preserved

    def test_dce_config_from_yaml(self, tmp_config) -> None:
        """Full DCE config with fitting section loads from YAML."""
        path = tmp_config("""\
            modality: dce
            pipeline:
              model: tofts
              fitting:
                fitter: lm
                max_iterations: 200
                tolerance: 1.0e-8
                r2_threshold: 0.6
                bounds:
                  Ktrans: [0.0, 3.0]
                  ve: [0.01, 0.8]
                initial_guess:
                  Ktrans: 0.05
                  ve: 0.3
        """)
        config = load_config(path)
        mc = config.get_modality_config()
        assert mc.fitting.fitter == "lm"
        assert mc.fitting.max_iterations == 200
        assert mc.fitting.tolerance == 1e-8
        assert mc.fitting.r2_threshold == 0.6
        assert mc.fitting.bounds == {"Ktrans": [0.0, 3.0], "ve": [0.01, 0.8]}
        assert mc.fitting.initial_guess == {"Ktrans": 0.05, "ve": 0.3}


# ---------------------------------------------------------------------------
# TestIVIMFittingConfig
# ---------------------------------------------------------------------------


class TestIVIMFittingConfig:
    """Tests for IVIM fitting configuration."""

    def test_defaults(self) -> None:
        """Default IVIM fitting config values match expected."""
        cfg = IVIMFittingConfig()
        assert cfg.max_iterations == 500
        assert cfg.tolerance == 1e-6
        assert cfg.bounds is None
        assert cfg.initial_guess is None
        assert isinstance(cfg.bayesian, BayesianIVIMFittingConfig)

    def test_bounds_override(self) -> None:
        """IVIM bounds override parses correctly."""
        cfg = IVIMFittingConfig(
            bounds={"D": [1e-4, 3e-3], "D_star": [5e-3, 0.05], "f": [0.0, 0.5]}
        )
        assert cfg.bounds["D"] == [1e-4, 3e-3]
        assert cfg.bounds["f"] == [0.0, 0.5]

    def test_bounds_validation_wrong_length(self) -> None:
        """Bounds with != 2 elements raises ValidationError."""
        with pytest.raises(ValidationError, match="must be"):
            IVIMFittingConfig(bounds={"D": [1e-4]})

    def test_initial_guess_override(self) -> None:
        """IVIM initial guess override parses correctly."""
        cfg = IVIMFittingConfig(initial_guess={"D": 0.8e-3, "D_star": 0.02, "f": 0.15})
        assert cfg.initial_guess["D"] == 0.8e-3

    def test_bayesian_defaults(self) -> None:
        """Bayesian sub-config has expected defaults."""
        cfg = BayesianIVIMFittingConfig()
        assert cfg.prior_scale == 1.5
        assert cfg.noise_std is None
        assert cfg.compute_uncertainty is True

    def test_bayesian_custom_priors(self) -> None:
        """Custom Bayesian parameters parse correctly."""
        cfg = IVIMFittingConfig(
            bayesian={"prior_scale": 2.0, "compute_uncertainty": False}
        )
        assert cfg.bayesian.prior_scale == 2.0
        assert cfg.bayesian.compute_uncertainty is False
        assert cfg.bayesian.noise_std is None  # default preserved

    def test_ivim_yaml_includes_fitting(self) -> None:
        """IVIMPipelineYAML includes fitting sub-config with defaults."""
        cfg = IVIMPipelineYAML()
        assert isinstance(cfg.fitting, IVIMFittingConfig)
        assert cfg.fitting.max_iterations == 500

    def test_ivim_config_from_yaml(self, tmp_config) -> None:
        """Full IVIM config with fitting section loads from YAML."""
        path = tmp_config("""\
            modality: ivim
            pipeline:
              fitting_method: bayesian
              b_threshold: 150.0
              fitting:
                max_iterations: 1000
                tolerance: 1.0e-8
                bounds:
                  D: [1.0e-4, 3.0e-3]
                  D_star: [5.0e-3, 0.05]
                  f: [0.0, 0.5]
                initial_guess:
                  D: 0.8e-3
                  D_star: 0.02
                  f: 0.15
                bayesian:
                  prior_scale: 2.0
                  compute_uncertainty: false
        """)
        config = load_config(path)
        mc = config.get_modality_config()
        assert mc.fitting_method == "bayesian"
        assert mc.fitting.max_iterations == 1000
        assert mc.fitting.tolerance == 1e-8
        assert mc.fitting.bounds["D"] == [1e-4, 3e-3]
        assert mc.fitting.initial_guess["D"] == 0.8e-3
        assert mc.fitting.bayesian.prior_scale == 2.0
        assert mc.fitting.bayesian.compute_uncertainty is False

    def test_fitting_section_optional(self, tmp_config) -> None:
        """IVIM config without fitting section uses defaults."""
        path = tmp_config("""\
            modality: ivim
            pipeline:
              fitting_method: segmented
              b_threshold: 200.0
        """)
        config = load_config(path)
        mc = config.get_modality_config()
        assert mc.fitting.max_iterations == 500
        assert mc.fitting.tolerance == 1e-6
        assert mc.fitting.bounds is None


# ---------------------------------------------------------------------------
# TestDumpDefaults
# ---------------------------------------------------------------------------


class TestDumpDefaults:
    """Tests for dump_defaults() template generation."""

    def test_dump_invalid(self) -> None:
        """Invalid modality raises ValueError."""
        with pytest.raises(ValueError, match="Unknown modality"):
            dump_defaults("invalid")

    def test_dump_defaults_are_valid(self, tmp_path: Path) -> None:
        """Round-trip: dump_defaults output can be loaded and validated."""
        for modality in ("dce", "dsc", "asl", "ivim"):
            template = dump_defaults(modality)
            config_path = tmp_path / f"{modality}_config.yaml"
            config_path.write_text(template)
            config = load_config(config_path)
            assert config.modality == modality
            # Should not raise - validates pipeline section too
            config.get_modality_config()


# ---------------------------------------------------------------------------
# TestCLIParser
# ---------------------------------------------------------------------------


class TestCLIParser:
    """Tests for the argparse CLI parser."""

    def test_version_flag(self) -> None:
        """Parser handles --version without error."""
        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0

    def test_dump_defaults_flag(self) -> None:
        """Parser handles --dump-defaults with valid modality."""
        parser = create_parser()
        args = parser.parse_args(["--dump-defaults", "dce"])
        assert args.dump_defaults == "dce"

    def test_validate_flag(self) -> None:
        """Parser handles --validate with a file path."""
        parser = create_parser()
        args = parser.parse_args(["--validate", "config.yaml"])
        assert args.validate == "config.yaml"

    def test_verbose_flag(self) -> None:
        """Parser handles -v / --verbose flag."""
        parser = create_parser()
        args = parser.parse_args(["-v", "config.yaml", "/data"])
        assert args.verbose is True

        args_long = parser.parse_args(["--verbose", "config.yaml", "/data"])
        assert args_long.verbose is True

    def test_run_args(self) -> None:
        """Parser handles config and data_path positional args."""
        parser = create_parser()
        args = parser.parse_args(["config.yaml", "/path/to/data"])
        assert args.config == "config.yaml"
        assert args.data_path == "/path/to/data"

    def test_output_override(self) -> None:
        """Parser handles --output / -o flag."""
        parser = create_parser()
        args = parser.parse_args(["--output", "/out", "config.yaml", "/data"])
        assert args.output == "/out"

        args_short = parser.parse_args(["-o", "/out", "config.yaml", "/data"])
        assert args_short.output == "/out"
