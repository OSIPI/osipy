# How to Run a Pipeline from YAML (CLI)

Run osipy pipelines from the command line using YAML configuration files.

## Create a Config File

### Interactive wizard

The wizard walks you through every setting interactively and writes a
validated config file.  It adapts its questions based on your earlier
answers -- for example, it only asks for a T1 mapping method if you have
T1-weighted data, and only prompts for an AIF file path when you choose
manual AIF:

```bash
osipy --help-me-pls
```

### Template

Alternatively, dump a fully-commented template and edit it by hand:

```bash
osipy --dump-defaults dce > config.yaml
```

Available modalities: `dce`, `dsc`, `asl`, `ivim`.

## Run a Pipeline

Point the CLI at your config and data:

```bash
osipy config.yaml /path/to/data.nii.gz
```

Override the output directory:

```bash
osipy config.yaml /path/to/data.nii.gz -o results/
```

Enable verbose logging:

```bash
osipy config.yaml /path/to/data.nii.gz -v
```

## Validate a Config

Check a config file for errors without running the pipeline:

```bash
osipy --validate config.yaml
```

## YAML Config Structure

Every config has these top-level sections:

| Section    | Purpose                                        |
|------------|------------------------------------------------|
| `modality` | Which pipeline to run (`dce`, `dsc`, `asl`, `ivim`) |
| `pipeline` | Modality-specific parameters                    |
| `data`     | Input data paths and format settings            |
| `output`   | Output format and directory                     |
| `backend`  | GPU/CPU settings                                |
| `logging`  | Log level configuration                         |

### Common Fields

**data**:

- `format` -- Input format: `auto`, `nifti`, `dicom`, or `bids`
- `mask` -- Path to brain/tissue mask
- `t1_map` -- Pre-computed T1 map (DCE)
- `aif_file` -- Measured AIF file
- `m0_data` -- M0 calibration image (ASL)
- `b_values` -- b-value list (IVIM)
- `b_values_file` -- Path to b-values file (IVIM)

**output**:

- `format` -- Output format (`nifti`)

**backend**:

- `force_cpu` -- Force CPU execution even if GPU is available (`true`/`false`)

**logging**:

- `level` -- Log verbosity: `INFO`, `DEBUG`, or `WARNING`

## Modality Examples

### DCE-MRI

With T1 mapping from VFA data:

```yaml
modality: dce
pipeline:
  model:
    method: extended_tofts
  t1_mapping_method:
    method: vfa
    fit_method: linear        # linear (fast) or nonlinear (LM refinement)
  concentration:
    method: spgr              # spgr or linear
  aif_source: population
  population_aif:
    name: parker
  acquisition:
    tr: 5.0
    flip_angles: [2, 5, 10, 15]
    baseline_frames: 5
    relaxivity: 4.5
data:
  mask: brain_mask.nii.gz
output:
  format: nifti
```

Each component selection is a nested block discriminated by a key
(`method` for the PK model, T1 method, and concentration model; `name` for
the population AIF). Selecting a method surfaces exactly that method's
knobs — for example, `t1_mapping_method.fit_method` only exists for `vfa`.

Without T1 data (assumed T1):

```yaml
modality: dce
pipeline:
  model:
    method: extended_tofts
  aif_source: population
  population_aif:
    name: parker
  acquisition:
    t1_assumed: 1400.0
    baseline_frames: 5
    relaxivity: 4.5
data:
  mask: brain_mask.nii.gz
output:
  format: nifti
```

### DSC-MRI

```yaml
modality: dsc
pipeline:
  te: 30.0
  baseline_frames: 10
  apply_leakage_correction: true
  deconvolution:
    method: oSVD            # oSVD | sSVD | cSVD
    oscillation_index: 0.035
    default_threshold: 0.2
data:
  mask: brain_mask.nii.gz
output:
  format: nifti
```

`deconvolution` is discriminated by `method`. `oSVD` exposes
`oscillation_index` and `default_threshold`; `sSVD` and `cSVD` instead take a
single `threshold`, e.g. `deconvolution: {method: sSVD, threshold: 0.2}`.

### ASL

```yaml
modality: asl
pipeline:
  labeling_scheme: pcasl
  pld: 1800.0
  label_duration: 1800.0
  t1_blood: 1650.0
  labeling_efficiency: 0.85
  m0:
    method: single          # single | voxelwise | reference_region
  difference:
    method: pairwise        # pairwise | surround | mean
  quantification:
    mode: single_pld        # single_pld | multi_pld
data:
  mask: brain_mask.nii.gz
  m0_data: m0.nii.gz
output:
  format: nifti
```

For multi-PLD acquisitions, switch the quantification mode to estimate both
CBF and arterial transit time (ATT) via the Buxton general kinetic model:

```yaml
  quantification:
    mode: multi_pld
    plds: [500.0, 1000.0, 1500.0, 2000.0, 2500.0]  # ms, one volume per PLD
    att_model: buxton
```

### IVIM

```yaml
modality: ivim
pipeline:
  fitting:
    method: segmented        # segmented | full | bayesian
    b_threshold: 200.0       # only for segmented / bayesian
  model:
    model: biexponential     # biexponential | simplified
  normalize_signal: true
data:
  mask: brain_mask.nii.gz
  b_values: [0, 10, 20, 50, 100, 200, 400, 800]
output:
  format: nifti
```

`fitting` is discriminated by `method`: `segmented` and `bayesian` expose
`b_threshold`, while `full` fits all b-values jointly (no threshold).
`bayesian` adds `prior_scale`, `noise_std`, and `compute_uncertainty`.
The `model` block selects the signal model — `simplified` exposes its own
`b_threshold` above which the perfusion term is treated as negligible.
`bounds` and `initial_guess` overrides live under `fitting`.

## See Also

- [YAML Configuration Reference](../reference/cli-config.md) — Complete field-by-field reference (auto-generated)
- [How to Run a Complete Pipeline](run-complete-pipeline.md) — Python API for programmatic control
- [Architecture](../explanation/architecture.md) — Design context and pipeline internals
