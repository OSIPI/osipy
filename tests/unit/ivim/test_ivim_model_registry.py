"""Tests for IVIM model registry."""

import pytest

from osipy.common.exceptions import DataValidationError
from osipy.ivim.models import IVIMModel, get_ivim_model, list_models


class TestIVIMModelRegistry:
    @pytest.mark.parametrize("name", list_models())
    def test_get_model(self, name):
        assert isinstance(get_ivim_model(name), IVIMModel)

    def test_unknown_raises(self):
        with pytest.raises(DataValidationError, match="Unknown IVIM model"):
            get_ivim_model("unknown")

    def test_list_contains_builtins(self):
        result = list_models()
        assert "biexponential" in result
        assert "simplified" in result

    def test_list_sorted(self):
        result = list_models()
        assert result == sorted(result)
