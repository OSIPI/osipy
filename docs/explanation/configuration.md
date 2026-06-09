# Registry-Driven Configuration

osipy's YAML/CLI configuration is **generated from the component registries**.
Each pipeline-component selection — the DCE pharmacokinetic model, the DSC
deconvolution method, the ASL M0 calibration, the IVIM fitting strategy, and so
on — is a nested, validated block whose options and parameters come directly
from what is registered.

## How it works

### Each component declares a `MethodConfig`

Every selectable component ships a small pydantic model
(`osipy.common.config.MethodConfig`) with a discriminator field — a `Literal`
equal to the component's registry name — plus that component's tunable knobs:

```python
class OSVDConfig(MethodConfig):
    method: Literal["oSVD"] = "oSVD"
    oscillation_index: float = Field(0.035, gt=0.0)
    default_threshold: float = Field(0.2, gt=0.0, lt=1.0)
```

`MethodConfig` sets `extra="forbid"`, so a typo or a knob belonging to a
different method raises a validation error instead of being silently ignored.

### The discriminator selects the parameters you see

In YAML you pick a component by its discriminator, and only that component's
knobs are valid:

```yaml
deconvolution:
  method: oSVD            # oSVD | sSVD | cSVD
  oscillation_index: 0.035
  default_threshold: 0.2
```

Switch the method and the surfaced knobs change with it (`sSVD`/`cSVD` expose a
single `threshold`; `oSVD` exposes `oscillation_index` and `default_threshold`).
The discriminator is `method` for most components, `mode` for the ASL
quantification block, `model` for the IVIM signal model, and `name` for the
population AIF.

### The config is generated from `registry × schema`

The per-component models are composed into discriminated unions
(`method_union()`) that form the modality config models. The `--dump-defaults`
templates and the `--help-me-pls` wizard are produced from these same models —
including the `# A | B | C` option comments, which are derived from the union
members (the registry keys), not hand-written.

### The same schema validates *and* builds the component

On load, the discriminator picks the registry entry and the remaining fields
become its constructor arguments (`construct_from_config()`):

```python
deconvolver = construct_from_config(DECONVOLVER_REGISTRY, cfg)  # cfg.method -> instance
```

Validation and construction share one schema, so every accepted knob is
guaranteed to reach the live component.

### Adding a method

1. Register the component, e.g. `@register_deconvolver("mymethod")` (see
   [Extension Points](architecture.md#extension-points)).
2. Give it a `MethodConfig` subclass and add it to the modality's `*_CONFIGS`
   mapping.

It then appears automatically as a selectable option in the config,
`--dump-defaults`, and the wizard — no hand-editing of the schema, runner, or
wizard.

## Per-modality shape

| Modality | Nested component blocks |
|----------|-------------------------|
| DCE | `model.method`, `t1_mapping_method.method` (+ `fit_method`), `concentration.method`, `population_aif.name` |
| DSC | `deconvolution.method` (+ method-specific thresholds) |
| ASL | `m0.method`, `difference.method`, `quantification.mode` (single-PLD or multi-PLD + ATT) |
| IVIM | `fitting.method` (segmented / full / bayesian), `model.model` (biexponential / simplified) |

Non-method-specific parameters (ASL labeling timing, DSC echo time, IVIM
`normalize_signal`, …) stay as flat keys in the `pipeline` block. Generate an
authoritative template with `osipy --dump-defaults dce` (or `dsc`/`asl`/`ivim`);
see [How to Run a Pipeline from YAML](../how-to/run-pipeline-cli.md) for runnable
examples.
