from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from osipy.common.aif.base import ArterialInputFunction
from osipy.common.backend.array_module import get_array_module
from osipy.common.exceptions import DataValidationError

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


DEFAULT_HEMATOCRIT: float = 0.45


@overload
def correct_hematocrit(
    aif: ArterialInputFunction,
    hematocrit: float = ...,
) -> ArterialInputFunction: ...


@overload
def correct_hematocrit(
    aif: NDArray[np.floating[Any]],
    hematocrit: float = ...,
) -> NDArray[np.floating[Any]]: ...


def correct_hematocrit(
    aif: ArterialInputFunction | NDArray[np.floating[Any]],
    hematocrit: float = DEFAULT_HEMATOCRIT,
) -> ArterialInputFunction | NDArray[np.floating[Any]]:
    """Scale an AIF from whole-blood to plasma concentration: Cp = Cb / (1 - Hct)."""
    _validate_hematocrit(hematocrit)

    if isinstance(aif, ArterialInputFunction):
        new_conc = _scale_to_plasma(aif.concentration, hematocrit)
        return ArterialInputFunction(
            time=aif.time,
            concentration=new_conc,
            aif_type=aif.aif_type,
            population_model=aif.population_model,
            model_parameters=aif.model_parameters,
            source_roi=aif.source_roi,
            extraction_method=aif.extraction_method,
            reference=aif.reference,
        )

    return _scale_to_plasma(aif, hematocrit)


def _scale_to_plasma(
    concentration: NDArray[np.floating[Any]],
    hematocrit: float,
) -> NDArray[np.floating[Any]]:
    xp = get_array_module(concentration)
    return xp.asarray(concentration / (1.0 - hematocrit))


def _validate_hematocrit(hematocrit: float) -> None:
    if not isinstance(hematocrit, int | float):
        msg = f"hematocrit must be a number, got {type(hematocrit).__name__}"
        raise DataValidationError(msg)

    if not 0.0 < hematocrit < 1.0:
        msg = f"hematocrit must be between 0 and 1 (exclusive), got {hematocrit}"
        raise DataValidationError(msg)
