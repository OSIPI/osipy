"""Registry-driven configuration.

The CLI/YAML configuration is *generated* from the component registries rather
than hand-written: each registered component declares its own pydantic config
model (a ``MethodConfig`` carrying a ``method`` discriminator literal plus its
tunable knobs), and the helpers here compose those models into discriminated
unions for the CLI config and construct the component straight from a validated
config instance.

This makes the registry the single source of truth: adding ``@register_x("foo")``
with a config model automatically surfaces ``foo`` (and its parameters) as a CLI
toggle, and the same schema both validates input *and* builds the component — so
an option can never be "collected but silently ignored".

Example
-------
>>> from osipy.common.config import method_union, construct_from_config
>>> Union = method_union(DECONVOLVER_CONFIGS)          # discriminated union type
>>> deconvolver = construct_from_config(REGISTRY, cfg)  # cfg.method -> instance
"""

from __future__ import annotations

import functools
import operator
from typing import Annotated, Any

from pydantic import BaseModel, Field

DISCRIMINATOR = "method"


class MethodConfig(BaseModel):
    """Base class for a component's configuration model.

    Subclasses set the discriminator field to a ``Literal`` of their registry
    name, then list their tunable knobs as ordinary pydantic fields::

        class OSVDConfig(MethodConfig):
            method: Literal["oSVD"] = "oSVD"
            oscillation_index: float = Field(0.035, gt=0.0)

    Unknown keys are rejected (``extra="forbid"``) so typos in YAML surface as
    validation errors rather than being silently dropped.
    """

    model_config = {"extra": "forbid"}


def method_union(
    configs: dict[str, type[BaseModel]],
    discriminator: str = DISCRIMINATOR,
) -> Any:
    """Build a discriminated-union type from a mapping of name -> config model.

    Parameters
    ----------
    configs : dict[str, type[BaseModel]]
        Mapping of registry name to its pydantic config model. Each model must
        carry a ``Literal`` *discriminator* field equal to its key.
    discriminator : str
        Name of the discriminator field (default ``"method"``).

    Returns
    -------
    type
        A single model (if only one) or
        ``Annotated[Union[...], Field(discriminator=...)]`` suitable as a
        pydantic field type.
    """
    models = [configs[name] for name in sorted(configs)]
    if not models:
        msg = "Cannot build a config union from an empty registry."
        raise ValueError(msg)
    if len(models) == 1:
        return models[0]
    union = functools.reduce(operator.or_, models)  # A | B | C
    return Annotated[union, Field(discriminator=discriminator)]


def construct_from_config(
    registry: dict[str, Any],
    config: BaseModel,
    discriminator: str = DISCRIMINATOR,
) -> Any:
    """Instantiate the registered component selected by *config*.

    ``config.<discriminator>`` selects the registry entry; the remaining fields
    are passed as constructor keyword arguments.
    """
    name = getattr(config, discriminator)
    if name not in registry:
        from osipy.common.exceptions import DataValidationError

        valid = ", ".join(sorted(registry))
        msg = f"Unknown '{name}'. Valid: {valid}"
        raise DataValidationError(msg)
    kwargs = config.model_dump(exclude={discriminator})
    return registry[name](**kwargs)


def config_params(
    config: BaseModel, discriminator: str = DISCRIMINATOR
) -> dict[str, Any]:
    """Return *config*'s fields excluding the discriminator (the construct kwargs)."""
    return config.model_dump(exclude={discriminator})


__all__ = [
    "DISCRIMINATOR",
    "MethodConfig",
    "config_params",
    "construct_from_config",
    "method_union",
]
