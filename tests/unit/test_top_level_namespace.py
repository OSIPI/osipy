"""Tests for the top-level ``osipy`` namespace."""

import pytest

import osipy
from osipy import asl, dce, dsc

PREFIXED_NAMES = {
    "asl_quantify_cbf": asl.quantify_cbf,
    "dce_fit_model": dce.fit_model,
    "dce_get_model": dce.get_model,
    "dce_list_models": dce.list_models,
    "dsc_compute_perfusion_maps": dsc.compute_perfusion_maps,
}

DEPRECATED_NAMES = {
    "compute_perfusion_maps": "dsc_compute_perfusion_maps",
    "fit_model": "dce_fit_model",
    "get_model": "dce_get_model",
    "list_models": "dce_list_models",
    "quantify_cbf": "asl_quantify_cbf",
}


class TestTopLevelNamespace:
    @pytest.mark.parametrize(("name", "expected"), PREFIXED_NAMES.items())
    def test_prefixed_name_resolves(self, name, expected):
        assert getattr(osipy, name) is expected

    @pytest.mark.parametrize("name", PREFIXED_NAMES)
    def test_prefixed_name_in_all(self, name):
        assert name in osipy.__all__

    @pytest.mark.parametrize(("old", "new"), DEPRECATED_NAMES.items())
    def test_deprecated_name_warns_and_resolves(self, old, new):
        with pytest.warns(DeprecationWarning, match=f"osipy.{new}"):
            value = getattr(osipy, old)
        assert value is getattr(osipy, new)

    @pytest.mark.parametrize("name", DEPRECATED_NAMES)
    def test_deprecated_name_not_in_all(self, name):
        assert name not in osipy.__all__

    def test_unknown_attribute_raises(self):
        with pytest.raises(AttributeError):
            osipy.does_not_exist  # noqa: B018
