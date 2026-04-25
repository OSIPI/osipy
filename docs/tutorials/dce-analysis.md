# DCE-MRI Analysis Tutorial

DCE-MRI workflow: data loading, T1 mapping, signal-to-concentration conversion, model fitting, and BIDS export.

## Prerequisites

- Completed [Getting Started](getting-started.md) tutorial
- DCE-MRI data with known acquisition parameters
- VFA data for T1 mapping (or pre-computed T1 map)

**Using the CLI?** Generate a config with `osipy --dump-defaults dce > config.yaml`, edit it, then run `osipy config.yaml data.nii.gz`. See [How to Run Pipeline from YAML](../how-to/run-pipeline-cli.md). The tutorial below covers the Python API for step-by-step control.

## Background

DCE-MRI measures tissue perfusion by tracking the passage of a gadolinium-based contrast agent. The analysis pipeline involves:

1. **T1 Mapping**: Estimate pre-contrast T1 relaxation times
2. **Signal Conversion**: Convert MRI signal to contrast agent concentration
3. **Model Fitting**: Fit pharmacokinetic models to extract parameters

Key parameters include:

| Parameter | Symbol | Units | Description |
|-----------|--------|-------|-------------|
| Transfer constant | Ktrans | min⁻¹ | Rate of contrast transfer from plasma to EES |
| EES volume fraction | ve | fraction | Extravascular extracellular space volume |
| Plasma volume fraction | vp | fraction | Blood plasma volume in tissue |

Symbols and units follow the OSIPI CAPLEX consensus lexicon (Dickie BR
et al. MRM 2024;91(5):1761-1773, [doi:10.1002/mrm.29840](https://doi.org/10.1002/mrm.29840),
PMID 37831600). CAPLEX lists ve/vp in mL/100mL; osipy uses the
numerically-equivalent fraction convention.

For more theory, see [Understanding Pharmacokinetic Models](../explanation/pharmacokinetic-models.md).

## Step 1: Load Your Data

```python
import numpy as np
import osipy

# Load DCE-MRI data (returns PerfusionDataset)
dce_dataset = osipy.load_nifti("dce_4d.nii.gz")

# Load VFA images for T1 mapping
# These should be acquired at different flip angles
vfa_data = osipy.load_nifti("vfa_images.nii.gz")

# Define acquisition parameters
flip_angles = np.array([2, 5, 10, 15])  # degrees
tr = 5.0  # Repetition time in ms

# DCE timing
n_timepoints = dce_dataset.shape[-1]
temporal_resolution = 3.5  # seconds between frames
time = np.arange(n_timepoints) * temporal_resolution

print(f"DCE data shape: {dce_dataset.shape}")
print(f"VFA data shape: {vfa_data.shape}")
print(f"Time points: {n_timepoints}, spanning {time[-1]:.1f} seconds")
```

!!! info "Data Organization"

    DCE-MRI data should be a 4D array with shape `(x, y, z, time)`.
    VFA data should be a 4D array with shape `(x, y, z, flip_angles)`.

## Step 2: Compute T1 Map

Accurate T1 mapping is essential for converting signal to concentration:

```python
# Set acquisition parameters on VFA dataset so compute_t1_map has what it needs
from osipy.dce import DCEAcquisitionParams
vfa_data.acquisition_params = DCEAcquisitionParams(
    tr=tr, flip_angles=flip_angles
)

# Compute T1 map using Variable Flip Angle method
t1_result = osipy.compute_t1_map(vfa_data, method="vfa")

# The result is a T1MappingResult with t1_map (ParameterMap) and quality_mask
t1_values = t1_result.t1_map.values
t1_quality = t1_result.quality_mask

print(f"T1 map shape: {t1_values.shape}")
print(f"T1 range: {t1_values[t1_quality > 0].min():.0f} - {t1_values[t1_quality > 0].max():.0f} ms")
```

!!! tip "T1 Quality Check"

    Typical tissue T1 values at 3T:

    - White matter: ~832 ms (Wansapura 1999, PMID 10232510)
    - Gray matter: ~1331 ms (Wansapura 1999, PMID 10232510)
    - Blood: ~1664 ms arterial at Hct 0.42 (Lu 2004, PMID 15334591);
      ISMRM ASL consensus uses 1650 ms (Alsop 2015, PMC4190138)
    - CSF: ~4000+ ms (Rooney et al., MRM 2007;57:308, PMID 17260370)

    If your values are outside these ranges, check your flip angles and TR.

## Step 3: Create a Tissue Mask

```python
# Simple threshold-based mask
# Use the mean DCE signal to identify tissue (.data accesses the raw array)
mean_signal = dce_dataset.data.mean(axis=-1)
signal_threshold = mean_signal.max() * 0.1  # 10% of max

# Combine with T1 quality
mask = (mean_signal > signal_threshold) & (t1_quality > 0)

# Optionally: exclude CSF (very high T1)
mask = mask & (t1_values < 3000)

print(f"Tissue voxels: {mask.sum()} / {mask.size}")
```

## Step 4: Convert Signal to Concentration

```python
from osipy.dce import DCEAcquisitionParams

# Define acquisition parameters
acq_params = DCEAcquisitionParams(
    tr=5.0,            # ms
    te=2.0,            # ms
    flip_angles=[15],  # degrees (DCE flip angle)
    baseline_frames=5,
    relaxivity=4.5,    # mM^-1 s^-1 — Gd-DTPA in water at 3T
                       # (Noebauer-Huhmann 2005, PMID 16462135: R1_3T_water=4.50).
                       # In vivo plasma r1 at 3T is lower; override for
                       # agent-specific in-vivo accuracy.
)

# Convert signal to concentration
# dce_signal_to_concentration wraps signal_to_concentration from osipy.dce
concentration = osipy.dce_signal_to_concentration(
    signal=dce_dataset.data,
    t1_map=t1_result.t1_map,
    acquisition_params=acq_params,
)

print(f"Concentration shape: {concentration.shape}")

# Check concentration range
valid_conc = concentration[mask]
print(f"Concentration range: {valid_conc.min():.3f} - {valid_conc.max():.3f} mM")
```

!!! warning "Negative Concentrations"

    If you see negative concentrations, check:

    1. Baseline selection (should be pre-contrast)
    2. T1 map accuracy
    3. Flip angle and TR values

## Step 5: Select an Arterial Input Function

The AIF describes contrast agent concentration in blood plasma. You have several options:

### Option A: Population-based AIF (Recommended for beginners)

```python
# Parker AIF - widely used population average
# Derived from 23 abdominal cancer patients at 1.5T receiving gadodiamide
# (Parker 2006, PMID 17036301). Use with caution at 3T, in brain/head-and-neck,
# or with different contrast agents — consider a measured or study-matched AIF
# when those conditions differ.
aif = osipy.ParkerAIF()(time)

# Alternative: Georgiou AIF (different shape)
# aif = osipy.GeorgiouAIF()(time)

print(f"AIF peak: {aif.concentration.max():.2f} mM at {time[aif.concentration.argmax()]:.1f} s")
```

### Option B: Measured AIF

```python
# If you have manually selected an arterial ROI:
# aif_roi_indices = [...]  # Indices of arterial voxels
# aif_signal = dce_dataset.data[aif_roi_indices].mean(axis=0)

# Convert to concentration using the same acquisition parameters
# aif_conc = osipy.dce_signal_to_concentration(
#     signal=aif_signal,
#     t1_map=None,  # Pass None if no T1 map for AIF voxels
#     acquisition_params=acq_params,
#     t1_blood=1600,  # Blood T1 at 3T in ms
# )

# Create AIF object
# aif = osipy.ArterialInputFunction(time=time, concentration=aif_conc)
```

See [How to Use a Custom AIF](../how-to/use-custom-aif.md) for detailed instructions.

## Step 6: Fit the Pharmacokinetic Model

```python
# Fit Extended Tofts model
result = osipy.fit_model(
    "extended_tofts",
    concentration=concentration,
    aif=aif,
    time=time,
    mask=mask
)

# Extract parameter maps (result is a DCEFitResult object)
ktrans = result.parameter_maps["Ktrans"].values
ve = result.parameter_maps["ve"].values
vp = result.parameter_maps["vp"].values
r_squared = result.r_squared_map
quality_mask = result.quality_mask

# Statistics on valid voxels
valid = quality_mask
print(f"\nFitting Results (valid voxels: {valid.sum()}):")
print(f"  Ktrans: {ktrans[valid].mean():.4f} ± {ktrans[valid].std():.4f} min⁻¹")
print(f"  ve:     {ve[valid].mean():.4f} ± {ve[valid].std():.4f}")
print(f"  vp:     {vp[valid].mean():.4f} ± {vp[valid].std():.4f}")
print(f"  R²:     {r_squared[valid].mean():.4f} ± {r_squared[valid].std():.4f}")
```

See [Understanding Pharmacokinetic Models](../explanation/pharmacokinetic-models.md) for model comparison and selection guidance.

### Compare with Standard Tofts

```python
# Compare with Standard Tofts
result_standard = osipy.fit_model(
    "tofts",
    concentration=concentration,
    aif=aif,
    time=time,
    mask=mask
)

print(f"\nStandard Tofts R²: {result_standard.r_squared_map[valid].mean():.4f}")
print(f"Extended Tofts R²: {r_squared[valid].mean():.4f}")
```

## Step 7: Visualize Results

### Parameter Maps

```python
import matplotlib.pyplot as plt

# Select a representative slice
slice_idx = dce_dataset.shape[2] // 2

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Row 1: Parameter maps
im0 = axes[0, 0].imshow(ktrans[:, :, slice_idx], cmap='hot', vmin=0, vmax=0.5)
axes[0, 0].set_title('Ktrans (min⁻¹)')
plt.colorbar(im0, ax=axes[0, 0])

im1 = axes[0, 1].imshow(ve[:, :, slice_idx], cmap='viridis', vmin=0, vmax=1)
axes[0, 1].set_title('ve')
plt.colorbar(im1, ax=axes[0, 1])

im2 = axes[0, 2].imshow(vp[:, :, slice_idx], cmap='plasma', vmin=0, vmax=0.2)
axes[0, 2].set_title('vp')
plt.colorbar(im2, ax=axes[0, 2])

# Row 2: Quality metrics
im3 = axes[1, 0].imshow(r_squared[:, :, slice_idx], cmap='gray', vmin=0, vmax=1)
axes[1, 0].set_title('R²')
plt.colorbar(im3, ax=axes[1, 0])

im4 = axes[1, 1].imshow(quality_mask[:, :, slice_idx], cmap='binary')
axes[1, 1].set_title('Quality Mask')
plt.colorbar(im4, ax=axes[1, 1])

# Anatomy reference
im5 = axes[1, 2].imshow(mean_signal[:, :, slice_idx], cmap='gray')
axes[1, 2].set_title('Mean Signal')
plt.colorbar(im5, ax=axes[1, 2])

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.savefig('dce_parameter_maps.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Visualize Time Curves

```python
# Plot concentration time curves for sample voxels
fig, ax = plt.subplots(figsize=(10, 5))

# AIF
ax.plot(time, aif.concentration, 'r-', linewidth=2, label='AIF')

# Sample tissue voxels
sample_indices = np.where(valid)
n_samples = min(5, len(sample_indices[0]))
for i in range(n_samples):
    x, y, z = sample_indices[0][i*10], sample_indices[1][i*10], sample_indices[2][i*10]
    ax.plot(time, concentration[x, y, z, :], alpha=0.5, label=f'Voxel {i+1}')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Concentration (mM)')
ax.set_title('DCE-MRI Concentration Time Curves')
ax.legend()
ax.grid(True, alpha=0.3)

plt.savefig('dce_time_curves.png', dpi=150)
plt.show()
```

## Step 8: Export Results

Save your results in BIDS-compliant format:

!!! warning "Experimental Feature"

    BIDS derivative export is partially implemented and may not produce fully compliant output for all use cases.

### Export to BIDS derivatives

```python
# Export to BIDS derivatives
# export_bids signature: (parameter_maps, output_dir, subject_id, session_id, metadata)
# parameter_maps must be dict[str, ParameterMap]
osipy.export_bids(
    parameter_maps=result.parameter_maps,
    output_dir="derivatives/osipy",
    subject_id="01",
    session_id="01",
)

print("Results exported to derivatives/osipy/")
```

The output structure will be:

```text
derivatives/osipy/
└── sub-01/
    └── ses-01/
        └── perf/
            ├── sub-01_ses-01_Ktrans.nii.gz
            ├── sub-01_ses-01_ve.nii.gz
            ├── sub-01_ses-01_vp.nii.gz
            ├── sub-01_ses-01_R2.nii.gz
            ├── sub-01_ses-01_quality-mask.nii.gz
            └── sub-01_ses-01_dce.json  # Provenance metadata
```

## Complete Example

### Pipeline

```python
from pathlib import Path

import numpy as np

import osipy
from osipy.common.io import save_nifti
from osipy.dce import DCEAcquisitionParams

DATA_DIR = Path("/path/to/dce_dicom_dir")
OUT = Path("osipy_output")

series = osipy.discover_dicom(DATA_DIR)
perf = osipy.load_dicom_series(
    next(s for s in series if s.role_hint == "dynamic"),
    modality=osipy.Modality.DCE,
)
vfa = osipy.load_dicom_series(
    [s for s in series if s.role_hint == "vfa"],
    modality=osipy.Modality.DCE,
)

config = osipy.DCEPipelineConfig(
    acquisition_params=DCEAcquisitionParams(
        tr=perf.acquisition_params.tr,
        te=perf.acquisition_params.te,
        flip_angles=[perf.acquisition_params.flip_angle],
        baseline_frames=5,
        relaxivity=4.5,
    ),
)
result = osipy.DCEPipeline(config).run(
    perf,
    time=perf.time_points,
    t1_data=vfa,
    flip_angles=vfa.acquisition_params.flip_angles,
    tr=vfa.acquisition_params.tr,
)

OUT.mkdir(exist_ok=True)
for name, pmap in result.fit_result.parameter_maps.items():
    save_nifti(
        pmap, OUT / f"{name.lower().replace('*', '_star')}.nii.gz", affine=perf.affine
    )
save_nifti(
    result.fit_result.quality_mask.astype(np.uint8),
    OUT / "quality_mask.nii.gz",
    affine=perf.affine,
)
```

### Stepwise

```python
from pathlib import Path

import numpy as np

import osipy
from osipy.common.io import save_nifti
from osipy.dce import DCEAcquisitionParams
from osipy.dce.concentration import signal_to_concentration
from osipy.dce.t1_mapping import compute_t1_vfa

DATA_DIR = Path("/path/to/dce_dicom_dir")
OUT = Path("osipy_output_stepwise")

# 1. Load dynamic + VFA series.
series = osipy.discover_dicom(DATA_DIR)
perf = osipy.load_dicom_series(
    next(s for s in series if s.role_hint == "dynamic"),
    modality=osipy.Modality.DCE,
)
vfa = osipy.load_dicom_series(
    [s for s in series if s.role_hint == "vfa"],
    modality=osipy.Modality.DCE,
)

# 2. VFA T1 mapping (linear SPGR solve).
t1_result = compute_t1_vfa(
    signal=vfa.data,
    flip_angles=vfa.acquisition_params.flip_angles,
    tr=vfa.acquisition_params.tr,
    method="linear",
)
t1_map = t1_result.t1_map

# 3. Signal → concentration (SPGR equation).
acq = DCEAcquisitionParams(
    tr=perf.acquisition_params.tr,
    te=perf.acquisition_params.te,
    flip_angles=[perf.acquisition_params.flip_angle],
    baseline_frames=5,
    relaxivity=4.5,
)
concentration = signal_to_concentration(
    signal=perf.data,
    t1_map=t1_map,
    acquisition_params=acq,
    method="spgr",
)

# 4. Population AIF: build the model, then sample it at the timepoints.
aif_model = osipy.ParkerAIF()
aif = aif_model(perf.time_points)

# 5. Extended Tofts fit.
fit_result = osipy.fit_model(
    model_name="extended_tofts",
    concentration=concentration,
    aif=aif,
    time=perf.time_points,
)

# 6. Save.
OUT.mkdir(exist_ok=True)
for name, pmap in fit_result.parameter_maps.items():
    save_nifti(
        pmap, OUT / f"{name.lower().replace('*', '_star')}.nii.gz", affine=perf.affine
    )
save_nifti(
    fit_result.quality_mask.astype(np.uint8),
    OUT / "quality_mask.nii.gz",
    affine=perf.affine,
)
```

## Next Steps

- [How to Use a Custom AIF](../how-to/use-custom-aif.md) for measured AIF workflows
- [How to Compare Multiple DCE Models](../how-to/fit-multiple-models.md) for model selection
- [Understanding Pharmacokinetic Models](../explanation/pharmacokinetic-models.md) for theory
- [DSC-MRI Tutorial](dsc-analysis.md) for brain perfusion analysis

## Troubleshooting

### Low R² Values

- Check T1 map quality (typical R² > 0.9 for VFA fitting)
- Verify AIF timing matches bolus arrival — try `fit_delay=True` to estimate arterial delay automatically
- Consider simpler model (Standard Tofts)

### Unrealistic Ktrans Values

- Ktrans > 1 min⁻¹ often indicates fitting failure
- Check time units (osipy expects seconds)
- Verify relaxivity value for your contrast agent

### Memory Issues

- Use masking to reduce voxel count
- Process in chunks for very large datasets
- Consider GPU acceleration for large-scale fitting
