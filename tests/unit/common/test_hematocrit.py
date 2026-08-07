import numpy as np
import pytest

from osipy.common.aif.base import ArterialInputFunction
from osipy.common.aif.hematocrit import (
    DEFAULT_HEMATOCRIT,
    correct_hematocrit,
)
from osipy.common.exceptions import DataValidationError
from osipy.common.types import AIFType


class TestCorrectHematocrit:
    def test_default_hematocrit_value(self):
        assert DEFAULT_HEMATOCRIT == 0.45

    def test_basic_correction(self):
        blood = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        plasma = correct_hematocrit(blood, hematocrit=0.45)

        expected = blood / (1.0 - 0.45)
        np.testing.assert_allclose(plasma, expected)

    def test_default_hematocrit_parameter(self):
        blood = np.array([1.0, 2.0, 3.0])
        plasma = correct_hematocrit(blood)

        expected = blood / (1.0 - DEFAULT_HEMATOCRIT)
        np.testing.assert_allclose(plasma, expected)

    def test_custom_hematocrit(self):
        blood = np.array([1.0, 2.0, 3.0])
        plasma_neonatal = correct_hematocrit(blood, hematocrit=0.60)
        expected = blood / (1.0 - 0.60)
        np.testing.assert_allclose(plasma_neonatal, expected)
        plasma_anemia = correct_hematocrit(blood, hematocrit=0.30)
        expected = blood / (1.0 - 0.30)
        np.testing.assert_allclose(plasma_anemia, expected)

    def test_higher_hematocrit_gives_higher_plasma(self):
        blood = np.array([1.0, 2.0, 3.0])
        plasma_low = correct_hematocrit(blood, hematocrit=0.30)
        plasma_high = correct_hematocrit(blood, hematocrit=0.60)
        assert np.all(plasma_high > plasma_low)

    def test_preserves_shape_1d(self):
        blood = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        plasma = correct_hematocrit(blood)
        assert plasma.shape == blood.shape

    def test_preserves_shape_2d(self):
        blood = np.random.rand(10, 5)
        plasma = correct_hematocrit(blood)
        assert plasma.shape == blood.shape

    def test_preserves_shape_3d(self):
        blood = np.random.rand(4, 4, 10)
        plasma = correct_hematocrit(blood)
        assert plasma.shape == blood.shape

    def test_zero_concentration_unchanged(self):
        blood = np.zeros(5)
        plasma = correct_hematocrit(blood)
        np.testing.assert_array_equal(plasma, 0.0)

    def test_plasma_always_greater_than_blood(self):
        blood = np.array([0.5, 1.0, 2.0, 5.0])
        plasma = correct_hematocrit(blood, hematocrit=0.45)
        assert np.all(plasma >= blood)


class TestCorrectHematocritWithArterialInputFunction:
    def _make_aif(self, n_time=5):
        time = np.linspace(0, 300, n_time)
        concentration = np.array([0.0, 1.0, 2.0, 1.5, 0.5])[:n_time]
        return ArterialInputFunction(
            time=time,
            concentration=concentration,
            aif_type=AIFType.POPULATION,
            population_model="test",
            reference="Test AIF",
        )

    def test_returns_arterial_input_function(self):
        aif = self._make_aif()
        corrected = correct_hematocrit(aif, hematocrit=0.45)
        assert isinstance(corrected, ArterialInputFunction)

    def test_corrects_concentration(self):
        aif = self._make_aif()
        corrected = correct_hematocrit(aif, hematocrit=0.45)

        expected = aif.concentration / (1.0 - 0.45)
        np.testing.assert_allclose(corrected.concentration, expected)

    def test_preserves_time(self):
        aif = self._make_aif()
        corrected = correct_hematocrit(aif, hematocrit=0.45)

        np.testing.assert_array_equal(corrected.time, aif.time)

    def test_preserves_metadata(self):
        aif = self._make_aif()
        corrected = correct_hematocrit(aif, hematocrit=0.45)

        assert corrected.aif_type == aif.aif_type
        assert corrected.population_model == aif.population_model
        assert corrected.reference == aif.reference

    def test_original_unchanged(self):
        aif = self._make_aif()
        original_conc = aif.concentration.copy()
        correct_hematocrit(aif, hematocrit=0.45)

        np.testing.assert_array_equal(aif.concentration, original_conc)

    def test_default_hematocrit(self):
        aif = self._make_aif()
        corrected = correct_hematocrit(aif)

        expected = aif.concentration / (1.0 - DEFAULT_HEMATOCRIT)
        np.testing.assert_allclose(corrected.concentration, expected)


class TestHematocritValidation:
    def test_hematocrit_zero_raises(self):
        blood = np.array([1.0, 2.0])
        with pytest.raises(DataValidationError, match="between 0 and 1"):
            correct_hematocrit(blood, hematocrit=0.0)

    def test_hematocrit_one_raises(self):
        blood = np.array([1.0, 2.0])
        with pytest.raises(DataValidationError, match="between 0 and 1"):
            correct_hematocrit(blood, hematocrit=1.0)

    def test_negative_hematocrit_raises(self):
        blood = np.array([1.0, 2.0])
        with pytest.raises(DataValidationError, match="between 0 and 1"):
            correct_hematocrit(blood, hematocrit=-0.1)

    def test_hematocrit_above_one_raises(self):
        blood = np.array([1.0, 2.0])
        with pytest.raises(DataValidationError, match="between 0 and 1"):
            correct_hematocrit(blood, hematocrit=1.5)

    def test_non_numeric_hematocrit_raises(self):
        blood = np.array([1.0, 2.0])
        with pytest.raises(DataValidationError, match="must be a number"):
            correct_hematocrit(blood, hematocrit="0.45")

    def test_edge_hematocrit_near_zero(self):
        blood = np.array([1.0, 2.0])
        plasma = correct_hematocrit(blood, hematocrit=0.01)
        np.testing.assert_allclose(plasma, blood / 0.99, rtol=1e-10)

    def test_edge_hematocrit_near_one(self):
        blood = np.array([1.0, 2.0])
        plasma = correct_hematocrit(blood, hematocrit=0.99)
        np.testing.assert_allclose(plasma, blood * 100.0, rtol=1e-10)

    def test_validation_with_arterial_input_function(self):
        aif = ArterialInputFunction(
            time=np.linspace(0, 300, 5),
            concentration=np.array([0.0, 1.0, 2.0, 1.5, 0.5]),
            aif_type=AIFType.POPULATION,
        )
        with pytest.raises(DataValidationError, match="between 0 and 1"):
            correct_hematocrit(aif, hematocrit=0.0)


class TestHematocritWithDtypes:
    def test_float32(self):
        blood = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        plasma = correct_hematocrit(blood)
        assert plasma.dtype == np.float32

    def test_float64(self):
        blood = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        plasma = correct_hematocrit(blood)
        assert plasma.dtype == np.float64

    def test_integer_hematocrit(self):
        blood = np.array([1.0, 2.0])
        with pytest.raises(DataValidationError):
            correct_hematocrit(blood, hematocrit=0)


class TestBackwardCompatibility:
    def test_no_hematocrit_no_change(self):
        assert DEFAULT_HEMATOCRIT == 0.45
        assert callable(correct_hematocrit)
