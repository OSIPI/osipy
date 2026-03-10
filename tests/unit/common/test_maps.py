"""Unit tests for map visualization.

Tests for osipy/common/visualization/maps.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from osipy.common.parameter_map import ParameterMap


def _make_single_map(name="Ktrans", symbol="K^{trans}", units="min-1"):
    """Create a single ParameterMap for testing."""
    values = np.random.rand(64, 64, 20)
    return ParameterMap(
        name=name,
        symbol=symbol,
        units=units,
        values=values,
        affine=np.eye(4),
    )


def _make_map_with_mask():
    """ParameterMap with quality mask, calling plot_parameter_map with show_mask."""
    values = np.random.rand(64, 64, 20)
    mask = np.zeros((64, 64, 20), dtype=bool)
    mask[20:40, 20:40, :] = True
    param_map = ParameterMap(
        name="Ktrans",
        symbol="K^{trans}",
        units="min-1",
        values=values,
        affine=np.eye(4),
        quality_mask=mask,
    )
    return param_map


def _call_plot_parameter_map_basic():
    """Basic plot_parameter_map call."""
    from osipy.common.visualization.maps import plot_parameter_map

    return plot_parameter_map(_make_single_map())


def _call_plot_parameter_map_with_mask():
    """plot_parameter_map with mask overlay."""
    from osipy.common.visualization.maps import plot_parameter_map

    return plot_parameter_map(_make_map_with_mask(), show_mask=True)


def _call_plot_parameter_comparison():
    """plot_parameter_comparison with three maps."""
    from osipy.common.visualization.maps import plot_parameter_comparison

    maps = [
        _make_single_map("Ktrans", "K^{trans}", "min-1"),
        _make_single_map("ve", "v_e", ""),
        _make_single_map("vp", "v_p", ""),
    ]
    return plot_parameter_comparison(maps, slice_idx=10)


def _call_plot_parameter_comparison_two():
    """plot_parameter_comparison between two maps."""
    from osipy.common.visualization.maps import plot_parameter_comparison

    values1 = np.random.rand(64, 64, 20) * 0.3
    values2 = values1 + np.random.randn(64, 64, 20) * 0.05
    map1 = ParameterMap(
        name="Ktrans_computed",
        symbol="K^{trans}",
        units="min-1",
        values=values1,
        affine=np.eye(4),
    )
    map2 = ParameterMap(
        name="Ktrans_reference",
        symbol="K^{trans}_{ref}",
        units="min-1",
        values=values2,
        affine=np.eye(4),
    )
    return plot_parameter_comparison(map1, map2, slice_idx=10)


@pytest.mark.parametrize(
    "call_factory",
    [
        pytest.param(_call_plot_parameter_map_basic, id="parameter_map_basic"),
        pytest.param(_call_plot_parameter_map_with_mask, id="parameter_map_with_mask"),
        pytest.param(_call_plot_parameter_comparison, id="parameter_comparison"),
        pytest.param(
            _call_plot_parameter_comparison_two, id="parameter_comparison_two"
        ),
    ],
)
def test_map_visualization(call_factory) -> None:
    """Smoke test for map visualization functions."""
    fig = call_factory()
    assert fig is not None
