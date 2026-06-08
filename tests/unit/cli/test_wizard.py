"""Unit tests for osipy CLI interactive configuration wizard."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import yaml

if TYPE_CHECKING:
    from pathlib import Path

from osipy.cli.wizard import (
    _check_tty,
    _collect_asl_config,
    _collect_data_config,
    _collect_dce_config,
    _collect_dsc_config,
    _collect_ivim_config,
    _format_yaml_value,
    _generate_yaml,
    _prompt_choice,
    _prompt_path,
    _prompt_value,
    _prompt_yes_no,
    _validate_config,
    run_wizard,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_input_fn(responses: list[str]):
    """Return a side_effect function that yields *responses* in order."""
    it = iter(responses)

    def _input(_prompt: str = "") -> str:
        return next(it)

    return _input


# ---------------------------------------------------------------------------
# TTY detection
# ---------------------------------------------------------------------------


class TestCheckTTY:
    """Tests for _check_tty()."""

    def test_exits_when_not_tty(self) -> None:
        """Non-interactive stdin triggers SystemExit."""
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            with pytest.raises(SystemExit, match="1"):
                _check_tty()

    def test_passes_when_tty(self) -> None:
        """Interactive stdin does not raise."""
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            _check_tty()  # should not raise


# ---------------------------------------------------------------------------
# Prompt utilities
# ---------------------------------------------------------------------------


class TestPromptChoice:
    """Tests for _prompt_choice()."""

    def test_select_by_number(self) -> None:
        """Selecting an option by its 1-based number works."""
        with patch("builtins.input", side_effect=_make_input_fn(["2"])):
            result = _prompt_choice("Pick:", ["a", "b", "c"])
        assert result == "b"

    def test_select_by_name(self) -> None:
        """Selecting an option by name works."""
        with patch("builtins.input", side_effect=_make_input_fn(["oSVD"])):
            result = _prompt_choice("Pick:", ["sSVD", "cSVD", "oSVD"])
        assert result == "oSVD"

    def test_select_by_name_case_insensitive(self) -> None:
        """Name matching is case-insensitive."""
        with patch("builtins.input", side_effect=_make_input_fn(["OSVD"])):
            result = _prompt_choice("Pick:", ["sSVD", "cSVD", "oSVD"])
        assert result == "oSVD"

    def test_default_on_enter(self) -> None:
        """Pressing Enter selects the default."""
        with patch("builtins.input", side_effect=_make_input_fn([""])):
            result = _prompt_choice("Pick:", ["a", "b"], default="b")
        assert result == "b"

    def test_retry_on_invalid_then_valid(self) -> None:
        """Invalid input retries, then accepts valid input."""
        with patch("builtins.input", side_effect=_make_input_fn(["99", "1"])):
            result = _prompt_choice("Pick:", ["x", "y"])
        assert result == "x"

    def test_eof_raises_system_exit(self) -> None:
        """EOFError during input triggers SystemExit."""
        with (
            patch("builtins.input", side_effect=EOFError),
            pytest.raises(SystemExit),
        ):
            _prompt_choice("Pick:", ["a"])

    def test_keyboard_interrupt_raises_system_exit(self) -> None:
        """KeyboardInterrupt during input triggers SystemExit."""
        with (
            patch("builtins.input", side_effect=KeyboardInterrupt),
            pytest.raises(SystemExit),
        ):
            _prompt_choice("Pick:", ["a"])


class TestPromptValue:
    """Tests for _prompt_value()."""

    def test_returns_string(self) -> None:
        """String values are returned as-is."""
        with patch("builtins.input", side_effect=_make_input_fn(["hello"])):
            result = _prompt_value("Val")
        assert result == "hello"

    def test_returns_float(self) -> None:
        """Float type coercion works."""
        with patch("builtins.input", side_effect=_make_input_fn(["3.14"])):
            result = _prompt_value("Val", expected_type=float)
        assert result == pytest.approx(3.14)

    def test_returns_int(self) -> None:
        """Int type coercion works."""
        with patch("builtins.input", side_effect=_make_input_fn(["42"])):
            result = _prompt_value("Val", expected_type=int)
        assert result == 42

    def test_default_on_enter(self) -> None:
        """Pressing Enter returns the default value."""
        with patch("builtins.input", side_effect=_make_input_fn([""])):
            result = _prompt_value("Val", default=99, expected_type=int)
        assert result == 99

    def test_none_on_optional_skip(self) -> None:
        """Pressing Enter on an optional field returns None."""
        with patch("builtins.input", side_effect=_make_input_fn([""])):
            result = _prompt_value("Val")
        assert result is None

    def test_retry_on_bad_type(self) -> None:
        """Invalid type retries, then accepts valid input."""
        with patch("builtins.input", side_effect=_make_input_fn(["abc", "10"])):
            result = _prompt_value("Val", expected_type=int)
        assert result == 10

    def test_bool_yes(self) -> None:
        """Bool parsing accepts 'yes'."""
        with patch("builtins.input", side_effect=_make_input_fn(["yes"])):
            result = _prompt_value("Val", expected_type=bool)
        assert result is True

    def test_bool_no(self) -> None:
        """Bool parsing accepts 'no'."""
        with patch("builtins.input", side_effect=_make_input_fn(["no"])):
            result = _prompt_value("Val", expected_type=bool)
        assert result is False


class TestPromptYesNo:
    """Tests for _prompt_yes_no()."""

    def test_yes(self) -> None:
        """'y' returns True."""
        with patch("builtins.input", side_effect=_make_input_fn(["y"])):
            assert _prompt_yes_no("Ok?") is True

    def test_no(self) -> None:
        """'n' returns False."""
        with patch("builtins.input", side_effect=_make_input_fn(["n"])):
            assert _prompt_yes_no("Ok?") is False

    def test_default_true(self) -> None:
        """Empty input with default=True returns True."""
        with patch("builtins.input", side_effect=_make_input_fn([""])):
            assert _prompt_yes_no("Ok?", default=True) is True

    def test_default_false(self) -> None:
        """Empty input with default=False returns False."""
        with patch("builtins.input", side_effect=_make_input_fn([""])):
            assert _prompt_yes_no("Ok?", default=False) is False


class TestPromptPath:
    """Tests for _prompt_path()."""

    def test_returns_path(self) -> None:
        """Entered path is returned as a string."""
        with patch("builtins.input", side_effect=_make_input_fn(["/tmp/file.nii"])):
            result = _prompt_path("Path")
        assert result == "/tmp/file.nii"

    def test_skip_returns_none(self) -> None:
        """Pressing Enter with no default returns None."""
        with patch("builtins.input", side_effect=_make_input_fn([""])):
            result = _prompt_path("Path")
        assert result is None

    def test_default_on_enter(self) -> None:
        """Pressing Enter returns the default path."""
        with patch("builtins.input", side_effect=_make_input_fn([""])):
            result = _prompt_path("Path", default="/default.yaml")
        assert result == "/default.yaml"


# ---------------------------------------------------------------------------
# YAML generation
# ---------------------------------------------------------------------------


class TestFormatYamlValue:
    """Tests for _format_yaml_value()."""

    def test_bool_true(self) -> None:
        """True formats as 'true'."""
        assert _format_yaml_value(True) == "true"

    def test_bool_false(self) -> None:
        """False formats as 'false'."""
        assert _format_yaml_value(False) == "false"

    def test_list(self) -> None:
        """Lists format as YAML inline arrays."""
        assert _format_yaml_value([1, 2, 3]) == "[1, 2, 3]"

    def test_string(self) -> None:
        """Normal strings are unquoted."""
        assert _format_yaml_value("oSVD") == "oSVD"

    def test_reserved_string_quoted(self) -> None:
        """YAML-reserved strings are quoted."""
        assert _format_yaml_value("true") == '"true"'
        assert _format_yaml_value("null") == '"null"'

    def test_float(self) -> None:
        """Floats format as decimal strings."""
        assert _format_yaml_value(30.0) == "30.0"

    def test_small_float_scientific(self) -> None:
        """Small floats use scientific notation."""
        result = _format_yaml_value(0.001)
        assert "e" in result.lower()

    def test_int(self) -> None:
        """Integers format as plain strings."""
        assert _format_yaml_value(10) == "10"


class TestGenerateYaml:
    """Tests for _generate_yaml()."""

    def test_generates_valid_yaml(self) -> None:
        """Generated string parses as valid YAML."""
        yaml_str = _generate_yaml("dce", {"model": "tofts"}, {"format": "auto"})
        parsed = yaml.safe_load(yaml_str)
        assert parsed["modality"] == "dce"
        assert parsed["pipeline"]["model"] == "tofts"
        assert parsed["data"]["format"] == "auto"

    def test_includes_output_backend_logging(self) -> None:
        """Generated YAML includes default output, backend, logging sections."""
        yaml_str = _generate_yaml("dsc", {"te": 30.0}, {})
        parsed = yaml.safe_load(yaml_str)
        assert parsed["output"]["format"] == "nifti"
        assert parsed["backend"]["force_cpu"] is False
        assert parsed["logging"]["level"] == "INFO"

    def test_inline_comments_present(self) -> None:
        """Known keys get inline comments."""
        yaml_str = _generate_yaml("dce", {"model": "tofts"}, {})
        assert "# pharmacokinetic model" in yaml_str

    def test_nested_dict_renders_as_subsection(self) -> None:
        """Nested dicts (e.g. acquisition) render as YAML sub-sections."""
        pipeline = {
            "model": "tofts",
            "acquisition": {"baseline_frames": 5, "relaxivity": 4.5},
        }
        yaml_str = _generate_yaml("dce", pipeline, {"format": "auto"})
        parsed = yaml.safe_load(yaml_str)
        assert parsed["pipeline"]["acquisition"]["baseline_frames"] == 5
        assert parsed["pipeline"]["acquisition"]["relaxivity"] == 4.5

    def test_nested_t1_assumed(self) -> None:
        """Nested t1_assumed in acquisition is valid YAML."""
        pipeline = {
            "model": "extended_tofts",
            "aif_source": "population",
            "population_aif": "parker",
            "acquisition": {
                "t1_assumed": 1400.0,
                "baseline_frames": 5,
                "relaxivity": 4.5,
            },
        }
        yaml_str = _generate_yaml("dce", pipeline, {"format": "auto"})
        parsed = yaml.safe_load(yaml_str)
        assert parsed["pipeline"]["acquisition"]["t1_assumed"] == 1400.0


class TestValidateConfig:
    """Tests for _validate_config()."""

    def test_valid_dce_with_assumed_t1(self) -> None:
        """DCE config with t1_assumed (no T1 data) passes validation."""
        yaml_str = _generate_yaml(
            "dce",
            {
                "model": "extended_tofts",
                "aif_source": "population",
                "population_aif": "parker",
                "acquisition": {
                    "t1_assumed": 1400.0,
                    "baseline_frames": 5,
                    "relaxivity": 4.5,
                },
            },
            {"format": "auto"},
        )
        _validate_config(yaml_str)  # should not raise

    def test_invalid_raises(self) -> None:
        """Invalid model name causes validation failure."""
        from pydantic import ValidationError

        yaml_str = _generate_yaml(
            "dce",
            {"model": "nonexistent_model"},
            {"format": "auto"},
        )
        with pytest.raises(ValidationError):
            _validate_config(yaml_str)


# ---------------------------------------------------------------------------
# Per-modality collectors
# ---------------------------------------------------------------------------


class TestCollectDataConfig:
    """Tests for _collect_data_config()."""

    def test_auto_format_no_mask(self) -> None:
        """Minimal data config with auto format and no optional fields."""
        inputs = _make_input_fn(
            [
                "",  # format: default auto
                "",  # mask: skip
            ]
        )
        with patch("builtins.input", side_effect=inputs):
            cfg = _collect_data_config("dsc", {})
        assert cfg["format"] == "auto"
        assert "mask" not in cfg

    def test_bids_format_collects_subject(self) -> None:
        """BIDS format prompts for subject ID."""
        inputs = _make_input_fn(
            [
                "4",  # format: bids (4th option)
                "",  # mask: skip
                "01",  # subject
                "",  # session: skip
            ]
        )
        with patch("builtins.input", side_effect=inputs):
            cfg = _collect_data_config("dsc", {})
        assert cfg["format"] == "bids"
        assert cfg["subject"] == "01"

    def test_dce_data_with_t1_data(self) -> None:
        """DCE data config prompts for T1 map when user has T1 data."""
        pipeline = {"aif_source": "population", "acquisition": {}}
        inputs = _make_input_fn(
            [
                "",  # format: auto
                "",  # mask: skip
                "t1.nii.gz",  # T1 map (shown because no t1_assumed)
            ]
        )
        with patch("builtins.input", side_effect=inputs):
            cfg = _collect_data_config("dce", pipeline)
        assert cfg["t1_map"] == "t1.nii.gz"
        assert "aif_file" not in cfg  # not prompted (aif_source != manual)

    def test_dce_data_without_t1_data(self) -> None:
        """DCE data config skips T1 map when t1_assumed is set."""
        pipeline = {"aif_source": "population", "acquisition": {"t1_assumed": 1400.0}}
        inputs = _make_input_fn(
            [
                "",  # format: auto
                "",  # mask: skip
                # no T1 map prompt (t1_assumed present)
                # no AIF file prompt (aif_source != manual)
            ]
        )
        with patch("builtins.input", side_effect=inputs):
            cfg = _collect_data_config("dce", pipeline)
        assert "t1_map" not in cfg

    def test_dce_data_manual_aif(self) -> None:
        """DCE data config prompts for AIF file when aif_source is manual."""
        pipeline = {"aif_source": "manual", "acquisition": {"t1_assumed": 1400.0}}
        inputs = _make_input_fn(
            [
                "",  # format: auto
                "",  # mask: skip
                "my_aif.txt",  # AIF file (shown because aif_source=manual)
            ]
        )
        with patch("builtins.input", side_effect=inputs):
            cfg = _collect_data_config("dce", pipeline)
        assert cfg["aif_file"] == "my_aif.txt"

    def test_asl_no_m0_shows_note(self, capsys) -> None:
        """ASL data config prints a note when M0 image is skipped."""
        inputs = _make_input_fn(
            [
                "",  # format: auto
                "",  # mask: skip
                "",  # M0 image: skip
            ]
        )
        with patch("builtins.input", side_effect=inputs):
            _collect_data_config("asl", {})
        captured = capsys.readouterr()
        assert "relative units" in captured.out

    def test_ivim_inline_bvalues(self) -> None:
        """IVIM data config collects inline b-values."""
        inputs = _make_input_fn(
            [
                "",  # format: auto
                "",  # mask: skip
                "inline",  # b-values source
                "0,10,50,200,400,800",  # b-values
            ]
        )
        with patch("builtins.input", side_effect=inputs):
            cfg = _collect_data_config("ivim", {})
        assert cfg["b_values"] == [0.0, 10.0, 50.0, 200.0, 400.0, 800.0]


class TestCollectDCEConfig:
    """Tests for _collect_dce_config()."""

    def test_defaults_with_t1_data(self) -> None:
        """Defaults with T1 data: model, has_t1=yes, t1_method, aif, pop_aif, baseline, relaxivity."""
        inputs = _make_input_fn(
            [
                "",  # model: extended_tofts
                "",  # has T1 data: yes (default)
                "",  # T1 method: vfa
                "",  # AIF source: population
                "",  # population AIF: parker
                "",  # baseline frames: 5
                "",  # relaxivity: 4.5
            ]
        )
        with patch("builtins.input", side_effect=inputs):
            cfg = _collect_dce_config()
        assert cfg["model"] == "extended_tofts"
        assert cfg["t1_mapping_method"] == "vfa"
        assert cfg["aif_source"] == "population"
        assert cfg["population_aif"] == "parker"
        assert cfg["acquisition"]["baseline_frames"] == 5
        assert cfg["acquisition"]["relaxivity"] == 4.5
        assert "t1_assumed" not in cfg["acquisition"]

    def test_no_t1_data_uses_assumed(self) -> None:
        """No T1 data: prompts for t1_assumed, skips T1 method."""
        inputs = _make_input_fn(
            [
                "",  # model: extended_tofts
                "n",  # has T1 data: no
                "",  # t1_assumed: 1400.0
                "",  # AIF source: population
                "",  # population AIF: parker
                "",  # baseline frames: 5
                "",  # relaxivity: 4.5
            ]
        )
        with patch("builtins.input", side_effect=inputs):
            cfg = _collect_dce_config()
        assert cfg["acquisition"]["t1_assumed"] == 1400.0
        assert "t1_mapping_method" not in cfg

    def test_detect_aif_skips_population(self) -> None:
        """Choosing detect AIF does not prompt for population AIF."""
        inputs = _make_input_fn(
            [
                "",  # model
                "",  # has T1 data: yes
                "",  # T1 method
                "detect",  # AIF source
                "",  # baseline frames
                "",  # relaxivity
            ]
        )
        with patch("builtins.input", side_effect=inputs):
            cfg = _collect_dce_config()
        assert cfg["aif_source"] == "detect"
        assert "population_aif" not in cfg


class TestCollectDSCConfig:
    """Tests for _collect_dsc_config()."""

    def test_defaults(self) -> None:
        """Pressing Enter for all prompts returns defaults."""
        # Prompts: te, baseline_frames, hematocrit_ratio, leakage, method,
        # then oSVD's two params (oscillation_index, default_threshold).
        inputs = _make_input_fn([""] * 7)
        with patch("builtins.input", side_effect=inputs):
            cfg = _collect_dsc_config()
        assert cfg["te"] == 30.0
        assert cfg["apply_leakage_correction"] is True
        assert cfg["deconvolution"]["method"] == "oSVD"
        assert cfg["deconvolution"]["oscillation_index"] == 0.035
        assert cfg["deconvolution"]["default_threshold"] == 0.2


class TestCollectASLConfig:
    """Tests for _collect_asl_config()."""

    def test_defaults(self) -> None:
        """Pressing Enter for all prompts returns defaults."""
        inputs = _make_input_fn([""] * 10)
        with patch("builtins.input", side_effect=inputs):
            cfg = _collect_asl_config()
        assert cfg["labeling_scheme"] == "pcasl"
        assert cfg["pld"] == 1800.0
        assert cfg["m0_method"] == "single"
        assert cfg["difference_method"] == "pairwise"
        assert cfg["label_control_order"] == "label_first"


class TestCollectIVIMConfig:
    """Tests for _collect_ivim_config()."""

    def test_defaults(self) -> None:
        """Pressing Enter for all prompts returns defaults."""
        inputs = _make_input_fn(["", "", ""])
        with patch("builtins.input", side_effect=inputs):
            cfg = _collect_ivim_config()
        assert cfg["fitting_method"] == "segmented"
        assert cfg["b_threshold"] == 200.0
        assert cfg["normalize_signal"] is True


# ---------------------------------------------------------------------------
# All-modalities round-trip validation
# ---------------------------------------------------------------------------


class TestAllModalitiesValidation:
    """Verify that default wizard values for each modality produce valid configs."""

    @pytest.mark.parametrize("modality", ["dce", "dsc", "asl", "ivim"])
    def test_defaults_round_trip(self, modality: str) -> None:
        """Default values for each modality produce valid YAML configs."""
        # Map of default pipeline configs matching what defaults produce
        defaults = {
            "dce": {
                "model": "extended_tofts",
                "t1_mapping_method": "vfa",
                "aif_source": "population",
                "population_aif": "parker",
                "acquisition": {"baseline_frames": 5, "relaxivity": 4.5},
            },
            "dsc": {
                "deconvolution_method": "oSVD",
                "te": 30.0,
                "apply_leakage_correction": True,
                "svd_threshold": 0.2,
                "baseline_frames": 10,
                "hematocrit_ratio": 0.73,
            },
            "asl": {
                "labeling_scheme": "pcasl",
                "pld": 1800.0,
                "label_duration": 1800.0,
                "t1_blood": 1650.0,
                "labeling_efficiency": 0.85,
                "m0_method": "single",
                "t1_tissue": 1330.0,
                "partition_coefficient": 0.9,
                "difference_method": "pairwise",
                "label_control_order": "label_first",
            },
            "ivim": {
                "fitting_method": "segmented",
                "b_threshold": 200.0,
                "normalize_signal": True,
            },
        }
        yaml_str = _generate_yaml(modality, defaults[modality], {"format": "auto"})
        parsed = yaml.safe_load(yaml_str)
        assert parsed["modality"] == modality
        _validate_config(yaml_str)


# ---------------------------------------------------------------------------
# CLI parser integration
# ---------------------------------------------------------------------------


class TestCLIHelpMeFlag:
    """Tests for the --help-me-pls CLI flag."""

    def test_parser_recognizes_help_me_pls(self) -> None:
        """Parser recognizes --help-me-pls without positional args."""
        from osipy.cli.main import create_parser

        parser = create_parser()
        args = parser.parse_args(["--help-me-pls"])
        assert args.help_me_pls is True
        assert args.config is None
        assert args.data_path is None

    def test_help_me_pls_false_by_default(self) -> None:
        """--help-me-pls is False when not provided."""
        from osipy.cli.main import create_parser

        parser = create_parser()
        args = parser.parse_args(["--dump-defaults", "dce"])
        assert args.help_me_pls is False


# ---------------------------------------------------------------------------
# run_wizard end-to-end
# ---------------------------------------------------------------------------


class TestRunWizard:
    """End-to-end tests for run_wizard()."""

    def test_full_dce_wizard(self, tmp_path: Path) -> None:
        """Walk through full DCE wizard and verify output file."""
        out_file = tmp_path / "test_config.yaml"
        inputs = _make_input_fn(
            [
                "1",  # modality: dce
                # -- pipeline (collected first) --
                "",  # model: extended_tofts
                "",  # has T1 data: yes
                "",  # T1 method: vfa
                "",  # AIF source: population
                "",  # population AIF: parker
                "",  # baseline frames: 5
                "",  # relaxivity: 4.5
                # -- data (collected second) --
                "",  # format: auto
                "",  # mask: skip
                "",  # T1 map: skip (shown because has T1 data)
                # -- output --
                str(out_file),  # output path
            ]
        )
        with (
            patch("builtins.input", side_effect=inputs),
            patch("sys.stdin") as mock_stdin,
        ):
            mock_stdin.isatty.return_value = True
            run_wizard()

        assert out_file.exists()
        parsed = yaml.safe_load(out_file.read_text())
        assert parsed["modality"] == "dce"
        assert parsed["pipeline"]["model"] == "extended_tofts"
        assert parsed["pipeline"]["acquisition"]["baseline_frames"] == 5

    def test_full_ivim_wizard(self, tmp_path: Path) -> None:
        """Walk through full IVIM wizard and verify output file."""
        out_file = tmp_path / "ivim_config.yaml"
        inputs = _make_input_fn(
            [
                "4",  # modality: ivim
                # -- pipeline (first) --
                "",  # fitting method: segmented
                "",  # b threshold: 200.0
                "",  # normalize: yes
                # -- data (second) --
                "",  # format: auto
                "",  # mask: skip
                "",  # b-values source: auto
                # -- output --
                str(out_file),  # output path
            ]
        )
        with (
            patch("builtins.input", side_effect=inputs),
            patch("sys.stdin") as mock_stdin,
        ):
            mock_stdin.isatty.return_value = True
            run_wizard()

        assert out_file.exists()
        parsed = yaml.safe_load(out_file.read_text())
        assert parsed["modality"] == "ivim"

    def test_overwrite_declined(self, tmp_path: Path) -> None:
        """User declining overwrite does not modify existing file."""
        out_file = tmp_path / "existing.yaml"
        out_file.write_text("original content")
        inputs = _make_input_fn(
            [
                "3",  # modality: asl
                # -- pipeline (first) --
                "",  # labeling scheme: default
                "",  # pld: default
                "",  # label duration: default
                "",  # t1 blood: default
                "",  # labeling efficiency: default
                "",  # m0 method: default
                "",  # t1 tissue: default
                "",  # partition coeff: default
                "",  # difference method: default
                "",  # label/control order: default
                # -- data (second) --
                "",  # data format: auto
                "",  # mask: skip
                "",  # m0 calibration image: skip
                # -- output --
                str(out_file),  # output path (exists)
                "n",  # decline overwrite
            ]
        )
        with (
            patch("builtins.input", side_effect=inputs),
            patch("sys.stdin") as mock_stdin,
        ):
            mock_stdin.isatty.return_value = True
            run_wizard()

        assert out_file.read_text() == "original content"

    def test_non_tty_exits(self) -> None:
        """Non-interactive terminal exits immediately."""
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            with pytest.raises(SystemExit):
                run_wizard()
