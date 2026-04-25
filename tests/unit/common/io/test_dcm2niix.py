"""Unit tests for the dcm2niix converter wrapper.

These tests focus on the binary-resolution behavior added when dcm2niix
was promoted from an optional system dependency to a PyPI wheel.
"""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


class TestDcm2niixBinaryResolution:
    """Resolution priority: explicit arg > PyPI wheel > PATH."""

    def test_explicit_path_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from osipy.common.io.converters.dcm2niix import Dcm2niixConverter

        # Ensure shutil.which would return a different value.
        monkeypatch.setattr(
            "osipy.common.io.converters.dcm2niix.shutil.which",
            lambda _name: "/usr/bin/dcm2niix",
        )

        converter = Dcm2niixConverter(dcm2niix_path="/custom/dcm2niix")

        assert converter.dcm2niix_path == "/custom/dcm2niix"

    def test_pypi_wheel_binary_preferred_over_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When `dcm2niix` is importable, its `.bin` overrides `shutil.which`."""
        from osipy.common.io.converters import dcm2niix as converter_mod

        fake_pkg = types.ModuleType("dcm2niix")
        fake_pkg.bin = "/opt/wheel/bin/dcm2niix"  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "dcm2niix", fake_pkg)

        # Make PATH fallback return something different so we can tell them apart.
        monkeypatch.setattr(
            converter_mod.shutil, "which", lambda _name: "/usr/bin/dcm2niix"
        )

        converter = converter_mod.Dcm2niixConverter()

        assert converter.dcm2niix_path == "/opt/wheel/bin/dcm2niix"

    def test_falls_back_to_path_when_wheel_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from osipy.common.io.converters import dcm2niix as converter_mod

        # Simulate the wheel not being installed.
        monkeypatch.setitem(sys.modules, "dcm2niix", None)

        monkeypatch.setattr(
            converter_mod.shutil, "which", lambda _name: "/usr/bin/dcm2niix"
        )

        converter = converter_mod.Dcm2niixConverter()

        assert converter.dcm2niix_path == "/usr/bin/dcm2niix"
