"""Registry-driven config models for ASL selection points.

Each ASL selection point — M0 calibration method, label/control difference
method, and the CBF quantification *mode* (single-PLD vs multi-PLD) — declares
its tunable knobs as a :class:`~osipy.common.config.MethodConfig`. These compose
into discriminated unions for the CLI config (so selecting a method/mode
surfaces exactly that option's parameters and rejects cross-method keys).

Mirrors :mod:`osipy.dsc.deconvolution.config` and :mod:`osipy.dce.config`.

Notes
-----
* **M0 calibration** strategy classes are stateless (their tunable knobs flow
  into :class:`~osipy.asl.calibration.m0.M0CalibrationParams`, which is passed
  to ``calibrate()``), so the config carries the knobs but the live component is
  selected by name via :class:`M0CalibrationParams.method`. ``M0_REGISTRY``
  therefore maps to the strategy classes, and the helper
  :func:`m0_params_from_config` builds the params dataclass from a config.
* **Difference methods** are plain functions selected by name (no extra knobs),
  so each config is a single-field selection.
* The **quantification mode** discriminates single-PLD vs multi-PLD analysis;
  single-PLD takes no extra knobs (timing comes from the pipeline block), while
  multi-PLD adds the PLD schedule and the ATT model.

References
----------
.. [1] OSIPI ASL Lexicon, https://osipi.github.io/ASL-Lexicon/
.. [2] Suzuki Y et al. MRM 2024;91(5):1743-1760. doi:10.1002/mrm.29815
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from osipy.asl.calibration.registry import M0_CALIBRATION_REGISTRY
from osipy.asl.quantification.att_registry import ATT_MODEL_REGISTRY
from osipy.common.config import MethodConfig

# ---------------------------------------------------------------------------
# M0 calibration method selection (@register_m0_calibration)
#
# Discriminated by ``method``. The knobs map to ``M0CalibrationParams`` fields;
# ``reference_region`` exposes the extra ``reference_region`` knob.
# ---------------------------------------------------------------------------


class SingleM0Config(MethodConfig):
    """Single (mean) M0 calibration (OSIPI ASL Lexicon)."""

    method: Literal["single"] = "single"
    t1_tissue: float = Field(
        1330.0, gt=0.0, description="ms, T1 of calibration tissue (M0 recovery)"
    )
    tr_m0: float = Field(6000.0, gt=0.0, description="ms, TR of the M0 acquisition")
    te_m0: float = Field(13.0, ge=0.0, description="ms, TE of the M0 acquisition")


class VoxelwiseM0Config(MethodConfig):
    """Voxel-by-voxel M0 calibration (OSIPI ASL Lexicon)."""

    method: Literal["voxelwise"] = "voxelwise"
    t1_tissue: float = Field(
        1330.0, gt=0.0, description="ms, T1 of calibration tissue (M0 recovery)"
    )
    tr_m0: float = Field(6000.0, gt=0.0, description="ms, TR of the M0 acquisition")
    te_m0: float = Field(13.0, ge=0.0, description="ms, TE of the M0 acquisition")


class ReferenceRegionM0Config(MethodConfig):
    """Reference-region M0 calibration (OSIPI ASL Lexicon)."""

    method: Literal["reference_region"] = "reference_region"
    reference_region: Literal["csf", "white_matter", "custom"] = Field(
        "csf", description="reference tissue for the single M0 value"
    )
    t1_tissue: float = Field(
        1330.0, gt=0.0, description="ms, T1 of calibration tissue (M0 recovery)"
    )
    tr_m0: float = Field(6000.0, gt=0.0, description="ms, TR of the M0 acquisition")
    te_m0: float = Field(13.0, ge=0.0, description="ms, TE of the M0 acquisition")


#: name -> config model (source for the discriminated union)
M0_CONFIGS: dict[str, type[MethodConfig]] = {
    "single": SingleM0Config,
    "voxelwise": VoxelwiseM0Config,
    "reference_region": ReferenceRegionM0Config,
}

#: name -> calibration strategy class (selected by ``M0CalibrationParams.method``)
M0_REGISTRY: dict[str, type] = dict(M0_CALIBRATION_REGISTRY)


def m0_params_from_config(config: MethodConfig) -> object:
    """Build an :class:`M0CalibrationParams` from a validated M0 config.

    The config's ``method`` discriminator plus its knobs map directly onto the
    :class:`~osipy.asl.calibration.m0.M0CalibrationParams` dataclass fields.
    """
    from osipy.asl.calibration.m0 import M0CalibrationParams

    kwargs = config.model_dump()
    return M0CalibrationParams(**kwargs)


# ---------------------------------------------------------------------------
# Label/control difference method selection (@register_difference_method)
#
# Difference methods are plain functions with no extra knobs, so each config is
# a single-field selection. Exposing them as a union makes the difference method
# a first-class, validated, wired option (previously collected but ignored).
# ---------------------------------------------------------------------------


class PairwiseDifferenceConfig(MethodConfig):
    """Pair-wise control-label subtraction, then average."""

    method: Literal["pairwise"] = "pairwise"


class SurroundDifferenceConfig(MethodConfig):
    """Surround subtraction (average adjacent controls per label)."""

    method: Literal["surround"] = "surround"


class MeanDifferenceConfig(MethodConfig):
    """Mean subtraction (average controls and labels separately)."""

    method: Literal["mean"] = "mean"


#: name -> config model (source for the discriminated union)
DIFFERENCE_CONFIGS: dict[str, type[MethodConfig]] = {
    "pairwise": PairwiseDifferenceConfig,
    "surround": SurroundDifferenceConfig,
    "mean": MeanDifferenceConfig,
}


# ---------------------------------------------------------------------------
# Quantification mode selection (single-PLD vs multi-PLD)
#
# Discriminated by ``mode``. Single-PLD takes no extra knobs (timing comes from
# the pipeline block). Multi-PLD (Buxton + ATT estimation) adds the PLD schedule
# and the ATT model, routing the pipeline to the multi-PLD path.
# ---------------------------------------------------------------------------

QUANT_DISCRIMINATOR = "mode"


class SinglePLDConfig(MethodConfig):
    """Single-PLD CBF quantification (one delay per voxel)."""

    mode: Literal["single_pld"] = "single_pld"


class MultiPLDConfig(MethodConfig):
    """Multi-PLD CBF + ATT estimation via the Buxton general kinetic model."""

    mode: Literal["multi_pld"] = "multi_pld"
    plds: list[float] = Field(
        default_factory=lambda: [500.0, 1000.0, 1500.0, 2000.0, 2500.0],
        description="ms, post-labeling delay schedule (one volume per PLD)",
    )
    att_model: str = Field("buxton", description="ATT estimation model (registry name)")


#: name -> config model (source for the discriminated union)
QUANTIFICATION_CONFIGS: dict[str, type[MethodConfig]] = {
    "single_pld": SinglePLDConfig,
    "multi_pld": MultiPLDConfig,
}

#: name -> ATT model class (constructed from config; see ``ATT_MODEL_REGISTRY``)
ATT_REGISTRY: dict[str, type] = dict(ATT_MODEL_REGISTRY)


__all__ = [
    "ATT_REGISTRY",
    "DIFFERENCE_CONFIGS",
    "M0_CONFIGS",
    "M0_REGISTRY",
    "QUANTIFICATION_CONFIGS",
    "QUANT_DISCRIMINATOR",
    "MeanDifferenceConfig",
    "MultiPLDConfig",
    "PairwiseDifferenceConfig",
    "ReferenceRegionM0Config",
    "SingleM0Config",
    "SinglePLDConfig",
    "SurroundDifferenceConfig",
    "VoxelwiseM0Config",
    "m0_params_from_config",
]
