"""Registry-driven config models for DCE selection points.

Each DCE selection point — pharmacokinetic model, T1 mapping method,
signal-to-concentration model, and population AIF — declares its tunable
knobs as a :class:`~osipy.common.config.MethodConfig`. These compose into
discriminated unions for the CLI config (so selecting a method surfaces
exactly that method's parameters and rejects cross-method keys), and the
matching ``*_REGISTRY`` maps build the live component from a validated
config instance via :func:`~osipy.common.config.construct_from_config`.

Mirrors :mod:`osipy.dsc.deconvolution.config`. Adding ``@register_model``
(or ``@register_t1_method`` / ``@register_concentration_model`` /
``@register_aif``) plus an entry here automatically surfaces the new option
(and its knobs) as a CLI toggle that both validates input and builds the
component — an option can never be "collected but silently ignored".

References
----------
.. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
.. [2] Dickie BR et al. MRM 2024;91(5):1761-1773. doi:10.1002/mrm.29840
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, create_model

from osipy.common.aif.population import AIF_REGISTRY, list_aifs
from osipy.common.config import MethodConfig
from osipy.dce.models.registry import MODEL_REGISTRY

# ---------------------------------------------------------------------------
# Pharmacokinetic model selection (@register_model)
#
# The registered PK models take no constructor arguments, so each config is a
# single-field MethodConfig carrying only its ``method`` discriminator. The
# registry maps the name to the model class (constructed with no kwargs).
# ---------------------------------------------------------------------------


class ToftsConfig(MethodConfig):
    """Standard Tofts model (OSIPI: M.IC1.004)."""

    method: Literal["tofts"] = "tofts"


class ExtendedToftsConfig(MethodConfig):
    """Extended Tofts model with plasma term (OSIPI: M.IC1.005)."""

    method: Literal["extended_tofts"] = "extended_tofts"


class PatlakConfig(MethodConfig):
    """Patlak model (OSIPI: M.IC1.006)."""

    method: Literal["patlak"] = "patlak"


class TwoCXMConfig(MethodConfig):
    """Two-compartment exchange model (OSIPI: M.IC1.009)."""

    method: Literal["2cxm"] = "2cxm"


class TwoCUMConfig(MethodConfig):
    """Two-compartment uptake model (2CUM)."""

    method: Literal["2cum"] = "2cum"


#: name -> config model (source for the discriminated union)
DCE_MODEL_CONFIGS: dict[str, type[MethodConfig]] = {
    "tofts": ToftsConfig,
    "extended_tofts": ExtendedToftsConfig,
    "patlak": PatlakConfig,
    "2cxm": TwoCXMConfig,
    "2cum": TwoCUMConfig,
}

#: name -> model class (constructed from config; see ``MODEL_REGISTRY``)
DCE_MODEL_REGISTRY: dict[str, Any] = dict(MODEL_REGISTRY)


# ---------------------------------------------------------------------------
# T1 mapping method selection (@register_t1_method)
#
# VFA exposes a real ``fit_method`` knob (linear vs nonlinear refinement);
# Look-Locker has no extra knobs. Modeled as a discriminated union so the
# nonlinear-VFA path is a first-class, selectable option.
# ---------------------------------------------------------------------------


class VFAConfig(MethodConfig):
    """Variable Flip Angle T1 mapping (OSIPI: P.NR2.002)."""

    method: Literal["vfa"] = "vfa"
    fit_method: Literal["linear", "nonlinear"] = Field(
        "linear",
        description="VFA fit: linear (fast) or nonlinear (LM refinement)",
    )


class LookLockerConfig(MethodConfig):
    """Look-Locker inversion-recovery T1 mapping (OSIPI: P.NR2.004)."""

    method: Literal["look_locker"] = "look_locker"


#: name -> config model (source for the discriminated union)
T1_METHOD_CONFIGS: dict[str, type[MethodConfig]] = {
    "vfa": VFAConfig,
    "look_locker": LookLockerConfig,
}


# ---------------------------------------------------------------------------
# Signal-to-concentration model selection (@register_concentration_model)
#
# Both SPGR and linear conversions take no extra knobs (the relaxivity / TR /
# flip angle come from the acquisition block), so each is a single-field
# MethodConfig. Exposing both makes the linear path a selectable option.
# ---------------------------------------------------------------------------


class SPGRConcentrationConfig(MethodConfig):
    """Spoiled gradient-echo signal-to-concentration model (OSIPI: P.SC1.001)."""

    method: Literal["spgr"] = "spgr"


class LinearConcentrationConfig(MethodConfig):
    """Linear signal-to-concentration approximation (small enhancement)."""

    method: Literal["linear"] = "linear"


#: name -> config model (source for the discriminated union)
CONCENTRATION_CONFIGS: dict[str, type[MethodConfig]] = {
    "spgr": SPGRConcentrationConfig,
    "linear": LinearConcentrationConfig,
}


# ---------------------------------------------------------------------------
# Population AIF selection (@register_aif)
#
# Population AIFs take no constructor arguments (defaults from each model's
# published parameters), so their configs are single-field MethodConfigs
# generated directly from the registry — adding ``@register_aif("foo")``
# surfaces ``foo`` as a selectable population AIF automatically. The
# discriminator field is ``name`` (an AIF is identified by its name, not a
# "method").
# ---------------------------------------------------------------------------

AIF_DISCRIMINATOR = "name"


def _make_aif_configs() -> dict[str, type[MethodConfig]]:
    """Generate single-field AIF config models from the AIF registry.

    Each canonical AIF name (aliases excluded by ``list_aifs()``) becomes a
    :class:`MethodConfig` subclass whose ``name`` discriminator literal equals
    that name. Generated dynamically so new ``@register_aif`` entries surface
    without editing this module.
    """
    configs: dict[str, type[MethodConfig]] = {}
    for aif_name in list_aifs():
        model = create_model(
            f"{aif_name.title().replace('_', '')}AIFConfig",
            __base__=MethodConfig,
            **{AIF_DISCRIMINATOR: (Literal[aif_name], aif_name)},  # type: ignore[call-overload]
        )
        model.__doc__ = f"Population AIF: {aif_name}."
        configs[aif_name] = model
    return configs


#: name -> config model (source for the discriminated union)
POPULATION_AIF_CONFIGS: dict[str, type[MethodConfig]] = _make_aif_configs()

#: name -> AIF class (constructed from config; canonical names only)
POPULATION_AIF_REGISTRY: dict[str, Any] = {
    name: AIF_REGISTRY[name] for name in list_aifs()
}


__all__ = [
    "AIF_DISCRIMINATOR",
    "CONCENTRATION_CONFIGS",
    "DCE_MODEL_CONFIGS",
    "DCE_MODEL_REGISTRY",
    "POPULATION_AIF_CONFIGS",
    "POPULATION_AIF_REGISTRY",
    "T1_METHOD_CONFIGS",
    "ExtendedToftsConfig",
    "LinearConcentrationConfig",
    "LookLockerConfig",
    "PatlakConfig",
    "SPGRConcentrationConfig",
    "ToftsConfig",
    "TwoCUMConfig",
    "TwoCXMConfig",
    "VFAConfig",
]
