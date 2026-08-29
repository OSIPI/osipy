"""CLI entry point for osipy pipeline runner.

Provides the ``osipy`` command for running MRI perfusion analysis
pipelines from YAML configuration files.

Examples
--------
Run a pipeline::

    osipy config.yaml /path/to/data

Validate a config file::

    osipy --validate config.yaml

Print a default YAML template::

    osipy --dump-defaults dce
"""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger(__name__)

VALID_MODALITIES = ("dce", "dsc", "asl", "ivim")


def _get_version() -> str:
    """Read version without importing the full osipy package.

    Falls back to a placeholder instead of raising when osipy isn't
    installed with discoverable package metadata (e.g. a source checkout
    run without ``pip install -e .``). This is called eagerly while
    building the parser, so a raise here would take down the entire CLI,
    not just ``--version``.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("osipy")
    except PackageNotFoundError:
        return "(version unknown)"


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the osipy CLI.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="osipy",
        description="Run osipy MRI perfusion analysis pipelines from YAML config.",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"osipy {_get_version()}",
    )

    parser.add_argument(
        "--validate",
        metavar="CONFIG",
        help="Validate a YAML config file and exit.",
    )

    parser.add_argument(
        "--dump-defaults",
        metavar="MODALITY",
        choices=VALID_MODALITIES,
        help="Print a default YAML template for the given modality and exit.",
    )

    parser.add_argument(
        "--help-me-pls",
        action="store_true",
        help="Interactive wizard to create a YAML config file.",
    )

    parser.add_argument(
        "config",
        nargs="?",
        help="Path to YAML config file.",
    )

    parser.add_argument(
        "data_path",
        nargs="?",
        help="Path to input data directory or file.",
    )

    parser.add_argument(
        "--output",
        "-o",
        metavar="DIR",
        help="Override the output directory.",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )

    return parser


def main() -> None:
    """Entry point for the osipy CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # --dump-defaults: print template and exit
    if args.dump_defaults is not None:
        from osipy.cli.config import dump_defaults

        print(dump_defaults(args.dump_defaults))
        return

    # --help-me-pls: interactive wizard
    if args.help_me_pls:
        from osipy.cli.wizard import run_wizard

        run_wizard()
        return

    # --validate: validate config and exit
    if args.validate is not None:
        try:
            from osipy.cli.config import load_config

            load_config(args.validate)
        except Exception as exc:
            print(f"Config validation failed: {exc}", file=sys.stderr)
            sys.exit(1)
        else:
            print("Config valid.")
        return

    # Normal run mode: require both config and data_path
    if args.config is None or args.data_path is None:
        parser.error("the following arguments are required: config, data_path")

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        from osipy.cli.config import load_config
        from osipy.cli.runner import run_pipeline

        config = load_config(args.config)
        run_pipeline(config, args.data_path, output_dir=args.output)
    except Exception as exc:
        logger.debug("Pipeline failed", exc_info=True)
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
