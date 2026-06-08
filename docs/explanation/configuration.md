# Registry-Driven Configuration

osipy's YAML/CLI configuration is **generated from the component registries**,
not hand-written. Each pipeline-component selection — the DCE pharmacokinetic
model, the DSC deconvolution method, the ASL M0 calibration, the IVIM fitting
strategy, and so on — is a nested, validated block whose available options and
parameters come directly from what is registered. This page explains why that
design exists and how it works.

## The Problem It Solves

Earlier, each pipeline option was a flat string in the config, with the
component's parameters as separate sibling keys disconnected from the
selection. This had two recurring failure modes:

- **Collected but ignored.** A knob could be parsed into the config object yet
  never reach the component that needed it, so changing it silently did nothing.
- **Mismatched parameters.** A threshold meant for one method could be set while
  a different method was selected, with no error — the value was simply unused.

The registry-driven config closes this gap: the same schema that *validates*
your input is also what *constructs* the component, so an option can never be
silently dropped or applied to the wrong method.

## How It Works

### Each component declares a `MethodConfig`

Every selectable component ships a small pydantic model
(`osipy.common.config.MethodConfig`) that carries a discriminator field — a
`Literal` equal to the component's registry name — plus exactly that
component's tunable knobs:

```python
class OSVDConfig(MethodConfig):
    method: Literal["oSVD"] = "oSVD"
    oscillation_index: float = Field(0.035, gt=0.0)
    default_threshold: float = Field(0.2, gt=0.0, lt=1.0)
```

`MethodConfig` sets `extra="forbid"`, so a typo or a knob that belongs to a
different method raises a validation error instead of being quietly ignored.

### The discriminator selects the parameters you see

In YAML, you select a component by its discriminator and then only the knobs
for *that* component are valid:

```yaml
deconvolution:
  method: oSVD            # oSVD | sSVD | cSVD
  oscillation_index: 0.035
  default_threshold: 0.2
```

Switch the method and the surfaced knobs change with it — `sSVD` and `cSVD`
expose a single `threshold`, while `oSVD` exposes `oscillation_index` and
`default_threshold`. The discriminator is `method` for most components,
`mode` for the ASL quantification block (single-PLD vs multi-PLD), `model`
for the IVIM signal model, and `name` for the population AIF.

### The CLI config is generated from `registry × schema`

The per-component config models are composed into discriminated unions
(`method_union()`), which form the modality config models used by the CLI.
The `--dump-defaults` templates and the interactive wizard (`--help-me-pls`)
are produced from these same models, so the documented defaults always match
the code.

### The same schema validates *and* builds the component

When a config is loaded, the discriminator picks the registry entry and the
remaining fields become that component's constructor arguments
(`construct_from_config()`):

```python
deconvolver = construct_from_config(DECONVOLVER_REGISTRY, cfg)  # cfg.method -> instance
```

Because validation and construction share one schema, every accepted knob is
guaranteed to reach the live component.

## Consequences for Contributors

Adding a new method is the registry pattern plus one config model:

1. Register the component, e.g. `@register_deconvolver("mymethod")` (see
   [Extension Points](architecture.md#extension-points)).
2. Give it a `MethodConfig` subclass listing its discriminator and knobs, and
   add it to the modality's `*_CONFIGS` mapping.

That's it — the new method automatically appears as a selectable option in the
CLI config, the `--dump-defaults` template, and the interactive wizard, with
its parameters validated and wired through. No hand-editing of the config
schema, runner, or wizard is required.

## Per-Modality Shape

The nested blocks per modality are:

| Modality | Nested component blocks |
|----------|-------------------------|
| DCE | `model.method`, `t1_mapping_method.method` (+ `fit_method`), `concentration.method`, `population_aif.name` |
| DSC | `deconvolution.method` (+ method-specific thresholds) |
| ASL | `m0.method`, `difference.method`, `quantification.mode` (single-PLD or multi-PLD + ATT) |
| IVIM | `fitting.method` (segmented / full / bayesian), `model.model` (biexponential / simplified) |

Physiological and acquisition parameters that are not method-specific (such as
the ASL labeling timing, the DSC echo time, or IVIM `normalize_signal`) stay
as flat keys in the `pipeline` block. See
[How to Run a Pipeline from YAML](../how-to/run-pipeline-cli.md) for complete,
runnable examples, and generate an authoritative template at any time with:

```bash
osipy --dump-defaults dce   # or dsc, asl, ivim
```

## See Also

- [Architecture Overview](architecture.md) — the registry pattern and the full
  extension-point table
- [How to Run a Pipeline from YAML](../how-to/run-pipeline-cli.md) — task-oriented
  config recipes
