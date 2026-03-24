"""Hematocrit correction for arterial input functions.

In DCE-MRI, the AIF is often measured (or modeled) as whole-blood
concentration, but pharmacokinetic models like Tofts actually work
with plasma concentration.  Since contrast agent doesn't enter red
blood cells, we need to divide by the plasma fraction (1 - Hct) to
get the right numbers.

Skipping this step means Ktrans ends up roughly 40-45 % too low for
a typical adult hematocrit of ~0.45.  The math is straightforward:

    Cp(t) = Cb(t) / (1 - Hct)

References
----------
.. [1] Tofts PS (1997). Modeling tracer kinetics in dynamic
   Gd-DTPA MR imaging. J Magn Reson Imaging 7(1):91-101.
.. [2] Parker GJM et al. (2006). Experimentally-derived functional
   form for a population-averaged AIF. Magn Reson Med 56(5):993-1000.
.. [3] Sourbron SP, Buckley DL (2013). Tracer kinetic modelling
   in MRI: estimating perfusion and capillary permeability. Phys Med Biol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from osipy.common.aif.base import ArterialInputFunction
from osipy.common.backend.array_module import get_array_module
from osipy.common.exceptions import DataValidationError

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

# Typical adult hematocrit — used when the caller doesn't specify one.
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
    """Scale an AIF from whole-blood to plasma concentration.

    Gadolinium-based contrast agents stay in the plasma — they don't
    cross into red blood cells.  So when a pharmacokinetic model
    expects *plasma* concentration but the AIF is in whole-blood
    units, we need to bump it up by 1/(1-Hct).  For a normal adult
    Hct of 0.45, that's roughly a 1.8x multiplier.

    You can hand this function either a plain NumPy/CuPy array or an
    ``ArterialInputFunction`` dataclass.  If you pass in the latter,
    you'll get back a brand-new ``ArterialInputFunction`` with the
    corrected concentrations (everything else stays the same).

    Parameters
    ----------
    aif : ArterialInputFunction or NDArray[np.floating]
        The whole-blood AIF — either an ``ArterialInputFunction``
        instance or a raw concentration array (any shape works).
    hematocrit : float, optional
        Red blood cell volume fraction, must sit strictly between
        0 and 1.  Defaults to 0.45 (a reasonable adult value).

    Returns
    -------
    ArterialInputFunction or NDArray[np.floating]
        The corrected plasma AIF.  You get back the same type you
        put in — an ``ArterialInputFunction`` gives you a new one
        with updated concentrations, while a plain array gives you
        a plain array of the same shape.

    Raises
    ------
    DataValidationError
        If ``hematocrit`` falls outside the open interval (0, 1).

    Examples
    --------
    Quick sanity check with a raw array:

    >>> import numpy as np
    >>> from osipy.common.aif.hematocrit import correct_hematocrit
    >>> cb = np.array([0.0, 1.0, 2.0, 1.5, 0.5])
    >>> cp = correct_hematocrit(cb, hematocrit=0.45)
    >>> np.allclose(cp, cb / 0.55)
    True

    Works the same way with an ArterialInputFunction object:

    >>> from osipy.common.aif.base import ArterialInputFunction
    >>> from osipy.common.types import AIFType
    >>> aif_obj = ArterialInputFunction(
    ...     time=np.linspace(0, 300, 5),
    ...     concentration=np.array([0.0, 1.0, 2.0, 1.5, 0.5]),
    ...     aif_type=AIFType.POPULATION,
    ... )
    >>> corrected = correct_hematocrit(aif_obj, hematocrit=0.45)
    >>> isinstance(corrected, ArterialInputFunction)
    True

    References
    ----------
    .. [1] Tofts PS (1997). J Magn Reson Imaging 7(1):91-101.
    .. [2] Sourbron SP, Buckley DL (2013). Phys Med Biol.
    """
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
    """Divide by (1 - Hct) to go from blood to plasma concentration.

    Parameters
    ----------
    concentration : NDArray[np.floating]
        Blood-level concentration values.
    hematocrit : float
        Already-validated hematocrit fraction.

    Returns
    -------
    NDArray[np.floating]
        Plasma-level concentration values.
    """
    xp = get_array_module(concentration)
    return xp.asarray(concentration / (1.0 - hematocrit))


def _validate_hematocrit(hematocrit: float) -> None:
    """Make sure hematocrit is a sensible number between 0 and 1.

    Parameters
    ----------
    hematocrit : float
        The value to check.

    Raises
    ------
    DataValidationError
        When the value is non-numeric or outside (0, 1).
    """
    if not isinstance(hematocrit, int | float):
        msg = f"hematocrit must be a number, got {type(hematocrit).__name__}"
        raise DataValidationError(msg)

    if not 0.0 < hematocrit < 1.0:
        msg = f"hematocrit must be between 0 and 1 (exclusive), got {hematocrit}"
        raise DataValidationError(msg)
