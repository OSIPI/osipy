"""Tests for ParameterMap failure_reasons and quality_mask validation."""

from __future__ import annotations

import numpy as np
import pytest

from osipy.common.parameter_map import ParameterMap


def _make_basic_map():
    """Create a minimal ParameterMap for testing."""
    return ParameterMap(
        name="Ktrans",
        symbol="K^{trans}",
        units="min-1",
        values=np.ones((3, 3, 3)),
        affine=np.eye(4),
    )


class TestFailureReasonsDefault:
    """failure_reasons should be None by default."""

    def test_failure_reasons_is_none_by_default(self):
        param_map = _make_basic_map()
        assert param_map.failure_reasons is None

    def test_failure_reasons_can_be_set(self):
        values = np.ones((3, 3, 3))
        reasons = np.full((3, 3, 3), "", dtype=object)
        reasons[0, 0, 0] = "out_of_bounds"
        param_map = ParameterMap(
            name="Ktrans",
            symbol="K^{trans}",
            units="min-1",
            values=values,
            affine=np.eye(4),
            quality_mask=np.ones((3, 3, 3), dtype=bool),
            failure_reasons=reasons,
        )
        assert param_map.failure_reasons is not None
        assert param_map.failure_reasons[0, 0, 0] == "out_of_bounds"

    def test_passing_voxels_have_empty_failure_reason(self):
        values = np.ones((3, 3, 3))
        reasons = np.full((3, 3, 3), "", dtype=object)
        param_map = ParameterMap(
            name="Ktrans",
            symbol="K^{trans}",
            units="min-1",
            values=values,
            affine=np.eye(4),
            quality_mask=np.ones((3, 3, 3), dtype=bool),
            failure_reasons=reasons,
        )
        assert (param_map.failure_reasons == "").all()


class TestQualityMaskShape:
    """quality_mask shape must match values shape."""

    def test_matching_shapes_accepted(self):
        values = np.ones((3, 3, 3))
        mask = np.ones((3, 3, 3), dtype=bool)
        param_map = ParameterMap(
            name="Ktrans",
            symbol="K^{trans}",
            units="min-1",
            values=values,
            affine=np.eye(4),
            quality_mask=mask,
        )
        assert param_map.quality_mask.shape == param_map.values.shape

    def test_mismatched_shapes_raise_error(self):
        values = np.ones((3, 3, 3))
        wrong_mask = np.ones((2, 2, 2), dtype=bool)
        with pytest.raises((ValueError, Exception)):
            ParameterMap(
                name="Ktrans",
                symbol="K^{trans}",
                units="min-1",
                values=values,
                affine=np.eye(4),
                quality_mask=wrong_mask,
            )
