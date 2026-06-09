"""Registry-driven config models for IVIM selection points.

Each IVIM selection point — the fitting *strategy* (segmented / full /
Bayesian) and the signal *model* (bi-exponential / simplified) — declares
its tunable knobs as a :class:`~osipy.common.config.MethodConfig`. These
compose into discriminated unions for the CLI config (so selecting a
method/model surfaces exactly that option's parameters and rejects
cross-method keys), and the matching registries build the live component
from a validated config instance.

Mirrors :mod:`osipy.asl.config` and :mod:`osipy.dce.config`. Adding
``@register_ivim_fitter`` / ``@register_ivim_model`` plus an entry here
automatically surfaces the new option (and its knobs) as a CLI toggle that
both validates input *and* builds the component — an option can never be
"collected but silently ignored".

Notes
-----
* The **fitting strategy** is discriminated by ``method``. Shared knobs
  (``max_iterations``, ``tolerance``, ``bounds``, ``initial_guess``,
  ``normalize_signal``) live on every strategy; ``b_threshold`` appears
  only on the methods that use it (segmented + Bayesian — the *full*
  strategy fits all b-values jointly and sets the threshold to 0
  internally), and the Bayesian prior knobs (``prior_scale``,
  ``noise_std``, ``compute_uncertainty``) appear only for Bayesian.
* The **signal model** is discriminated by ``model``. The bi-exponential
  model carries no constructor knobs; the simplified model exposes its
  ``b_threshold`` (above which the perfusion term is treated as negligible).

References
----------
.. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
.. [2] Le Bihan D et al. (1988). Radiology 168(2):497-505.
.. [3] Dickie BR et al. MRM 2024;91(5):1761-1773. doi:10.1002/mrm.29840
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator

from osipy.common.config import MethodConfig

# Import the models package so the built-in models register themselves
# (``register_ivim_model("biexponential"/"simplified")``) before we
# snapshot the registry below.
from osipy.ivim import models as _ivim_models  # noqa: F401
from osipy.ivim.models.registry import IVIM_MODEL_REGISTRY

# ---------------------------------------------------------------------------
# Shared fitting knobs (validators reused across strategy configs)
# ---------------------------------------------------------------------------


def _validate_bounds_pairs(
    v: dict[str, list[float]] | None,
) -> dict[str, list[float]] | None:
    """Validate parameter bounds are ``[lower, upper]`` pairs with lower<=upper."""
    if v is None:
        return v
    for name, pair in v.items():
        if len(pair) != 2:
            msg = f"Bounds for '{name}' must be [lower, upper], got {pair}"
            raise ValueError(msg)
        if pair[0] > pair[1]:
            msg = f"Lower bound > upper bound for '{name}': {pair}"
            raise ValueError(msg)
    return v


class _IVIMFittingBase(MethodConfig):
    """Shared knobs for every IVIM fitting strategy.

    The discriminator (``method``) and any method-specific knobs are added
    by the concrete subclasses below.
    """

    max_iterations: int = Field(default=500, gt=0)
    tolerance: float = Field(default=1e-6, gt=0.0)
    bounds: dict[str, list[float]] | None = Field(
        default=None,
        description="override model defaults (omit to use model defaults)",
        json_schema_extra={
            "yaml_example": (
                "S0: [0.0, 1.0e+10]        # signal units\n"
                "D: [1.0e-4, 5.0e-3]       # mm^2/s\n"
                "D_star: [2.0e-3, 0.1]     # mm^2/s\n"
                "f: [0.0, 0.7]             # dimensionless"
            )
        },
    )
    initial_guess: dict[str, float] | None = Field(
        default=None,
        description="override data-driven initial estimates",
        json_schema_extra={
            "yaml_example": (
                "D: 1.0e-3                 # mm^2/s\n"
                "D_star: 0.01              # mm^2/s\n"
                "f: 0.1                    # dimensionless"
            )
        },
    )

    @field_validator("bounds")
    @classmethod
    def _check_bounds(
        cls, v: dict[str, list[float]] | None
    ) -> dict[str, list[float]] | None:
        return _validate_bounds_pairs(v)


# ---------------------------------------------------------------------------
# Fitting strategy selection (@register_ivim_fitter)
#
# Discriminated by ``method``. ``b_threshold`` appears only on the strategies
# that use it; the Bayesian prior knobs appear only for Bayesian.
# ---------------------------------------------------------------------------


class SegmentedFittingConfig(_IVIMFittingBase):
    """Two-step (segmented) IVIM fitting."""

    method: Literal["segmented"] = "segmented"
    b_threshold: float = Field(
        default=200.0,
        gt=0.0,
        description="s/mm^2, threshold separating D and D* regimes",
    )


class FullFittingConfig(_IVIMFittingBase):
    """Full bi-exponential IVIM fitting (all b-values jointly).

    The full strategy fits every b-value at once and uses no segmentation
    threshold, so ``b_threshold`` is intentionally absent.
    """

    method: Literal["full"] = "full"


class BayesianFittingConfig(_IVIMFittingBase):
    """Two-stage Bayesian MAP IVIM fitting with empirical priors."""

    method: Literal["bayesian"] = "bayesian"
    b_threshold: float = Field(
        default=200.0,
        gt=0.0,
        description="s/mm^2, threshold separating D and D* regimes",
    )
    prior_scale: float = Field(
        default=1.5, gt=0.0, description="scale of the empirical priors"
    )
    noise_std: float | None = Field(
        default=None,
        description="measurement noise std (estimated from data when omitted)",
        examples=[0.01],
    )
    compute_uncertainty: bool = Field(
        default=True, description="propagate posterior uncertainty"
    )


#: name -> config model (source for the discriminated union)
IVIM_FITTING_CONFIGS: dict[str, type[MethodConfig]] = {
    "segmented": SegmentedFittingConfig,
    "full": FullFittingConfig,
    "bayesian": BayesianFittingConfig,
}


# ---------------------------------------------------------------------------
# Signal model selection (@register_ivim_model)
#
# Discriminated by ``model``. The bi-exponential model takes no knobs; the
# simplified model exposes its perfusion-cutoff ``b_threshold``.
# ---------------------------------------------------------------------------

MODEL_DISCRIMINATOR = "model"


class BiexponentialModelConfig(MethodConfig):
    """IVIM bi-exponential signal model: S0, D, D*, f."""

    model: Literal["biexponential"] = "biexponential"


class SimplifiedModelConfig(MethodConfig):
    """IVIM simplified signal model: S0, D, f (assumes D* >> D)."""

    model: Literal["simplified"] = "simplified"
    b_threshold: float = Field(
        default=200.0,
        gt=0.0,
        description="s/mm^2, b-value above which the perfusion term is negligible",
    )


#: name -> config model (source for the discriminated union)
IVIM_MODEL_CONFIGS: dict[str, type[MethodConfig]] = {
    "biexponential": BiexponentialModelConfig,
    "simplified": SimplifiedModelConfig,
}

#: name -> model class (constructed from config; see ``IVIM_MODEL_REGISTRY``)
IVIM_SIGNAL_MODEL_REGISTRY: dict[str, Any] = dict(IVIM_MODEL_REGISTRY)


__all__ = [
    "IVIM_FITTING_CONFIGS",
    "IVIM_MODEL_CONFIGS",
    "IVIM_SIGNAL_MODEL_REGISTRY",
    "MODEL_DISCRIMINATOR",
    "BayesianFittingConfig",
    "BiexponentialModelConfig",
    "FullFittingConfig",
    "SegmentedFittingConfig",
    "SimplifiedModelConfig",
]
