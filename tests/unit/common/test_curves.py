"""Unit tests for curve visualization.

Tests for osipy/common/visualization/curves.py.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest


def _make_time_conc_basic():
    """Basic time-concentration curve args."""
    time = np.linspace(0, 60, 30)
    concentration = np.sin(time / 10) * 10 + 5
    return {"time": time, "concentration": concentration}


def _make_time_conc_with_aif():
    """Time-concentration curve with AIF overlay."""
    time = np.linspace(0, 60, 30)
    tissue_conc = np.sin(time / 10) * 10 + 5
    aif = np.exp(-((time - 15) ** 2) / 50) * 50
    return {"time": time, "concentration": tissue_conc, "aif": aif}


def _make_residue():
    """Residue function args."""
    time = np.linspace(0, 30, 60)
    residue = np.exp(-time / 5)
    return {"time": time, "residue": residue}


def _make_aif():
    """Basic AIF args."""
    time = np.linspace(0, 60, 120)
    aif = np.exp(-((time - 15) ** 2) / 50) * 50
    return {"time": time, "aif": aif}


def _make_signal():
    """Basic signal time course args."""
    time = np.linspace(0, 60, 60)
    signal = 1000 * np.exp(-time / 100) + np.random.randn(60) * 10
    return {"time": time, "signal": signal}


@pytest.mark.parametrize(
    "import_name, kwargs_factory",
    [
        pytest.param(
            "plot_time_concentration_curve",
            _make_time_conc_basic,
            id="time_concentration_basic",
        ),
        pytest.param(
            "plot_time_concentration_curve",
            _make_time_conc_with_aif,
            id="time_concentration_with_aif",
        ),
        pytest.param(
            "plot_residue_function",
            _make_residue,
            id="residue_function",
        ),
        pytest.param(
            "plot_aif",
            _make_aif,
            id="aif_basic",
        ),
        pytest.param(
            "plot_signal_timecourse",
            _make_signal,
            id="signal_timecourse",
        ),
    ],
)
def test_curve_plotting(import_name: str, kwargs_factory) -> None:
    """Smoke test for curve visualization functions."""
    from osipy.common.visualization import curves

    plot_fn = getattr(curves, import_name)
    fig = plot_fn(**kwargs_factory())
    assert fig is not None
