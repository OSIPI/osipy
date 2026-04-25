# How to Run a Complete Pipeline (Python API)

For most users, the CLI is the simplest way to run a pipeline — see [How to Run Pipeline from YAML](run-pipeline-cli.md). This guide covers the Python API for users who need programmatic control over pipeline steps.

## Pipeline Overview

osipy provides pipeline classes for each modality:

- `DCEPipeline` - Complete DCE-MRI analysis
- `DSCPipeline` - Complete DSC-MRI analysis
- `ASLPipeline` - Complete ASL analysis
- `IVIMPipeline` - Complete IVIM analysis

## Quick Start: Unified Interface

Use `run_analysis()` for automatic modality detection:

```python
import osipy

# Automatic pipeline selection
# data should be an NDArray or PerfusionDataset, not a file path string.
# Load data first with osipy.load_nifti() or similar I/O functions.
data = osipy.load_nifti("path/to/data.nii.gz")
result = osipy.run_analysis(
    data=data,
    modality="dce",  # or "dsc", "asl", "ivim"
)
```

## DCE Pipeline

```python
from pathlib import Path

import numpy as np

import osipy
from osipy.common.io import save_nifti
from osipy.dce import DCEAcquisitionParams

DATA_DIR = Path("/path/to/dce_dicom_dir")
OUT = Path("osipy_output")

# 1. Discover + load DICOM (VFA for T1 mapping, dynamic for DCE).
series = osipy.discover_dicom(DATA_DIR)
perf = osipy.load_dicom_series(
    next(s for s in series if s.role_hint == "dynamic"),
    modality=osipy.Modality.DCE,
)
vfa = osipy.load_dicom_series(
    [s for s in series if s.role_hint == "vfa"],
    modality=osipy.Modality.DCE,
)

# 2. Configure + run the pipeline.
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

# 3. Save parameter maps + quality mask.
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

Pass a pre-computed T1 map via `t1_map=` instead of `t1_data=` to skip VFA fitting.

### Arterial Delay Estimation

For data where the bolus arrives at different times across tissue, use `fit_delay=True` in direct `fit_model()` calls:

```python
result = osipy.fit_model(
    "extended_tofts", concentration, aif, time,
    fit_delay=True  # Estimates per-voxel delay
)

# Delay map in seconds
delay = result.parameter_maps["delay"].values
```

### DCE Pipeline Steps

1. Load DCE and variable flip angle (VFA) data
2. Compute T1 map (VFA method)
3. Create tissue mask
4. Convert signal to concentration
5. Set up arterial input function (AIF)
6. Fit pharmacokinetic model
7. Generate quality metrics
8. Export BIDS derivatives

## DSC Pipeline

Complete DSC-MRI analysis:

```python
import osipy
from osipy.pipeline import DSCPipeline, DSCPipelineConfig

# Create pipeline with config object
config = DSCPipelineConfig(
    deconvolution_method="oSVD",
    apply_leakage_correction=True,
)

pipeline = DSCPipeline(config)

# Run
result = pipeline.run(dsc_data, time)

# Results wrap DSCPerfusionMaps in DSCPipelineResult
maps = result.perfusion_maps
print(f"CBV mean: {maps.cbv.values[brain_mask].mean():.2f} ml/100g")
print(f"CBF mean: {maps.cbf.values[brain_mask].mean():.1f} ml/100g/min")
```

### DSC Pipeline Steps

1. Load DSC data
2. Create brain mask
3. Identify baseline and bolus
4. Convert signal to ΔR2*
5. Detect/select AIF
6. Perform SVD deconvolution
7. Calculate CBV from AUC
8. Apply leakage correction (optional)
9. Normalize to white matter (optional)
10. Export results

## ASL Pipeline

Complete ASL CBF quantification:

```python
import osipy
from osipy.asl import LabelingScheme
from osipy.pipeline import ASLPipeline, ASLPipelineConfig

# Create pipeline with config object
config = ASLPipelineConfig(
    labeling_scheme=LabelingScheme.PCASL,
    # Defaults match ISMRM consensus for healthy adults <70y
    # (Alsop et al. 2015, doi:10.1002/mrm.25197, PMC4190138, Table 1).
    # For older adults or clinical populations, use pld=2000.0.
    label_duration=1800.0,          # ms
    pld=1800.0,                     # ms
    t1_blood=1650.0,                # ms at 3T (Lu 2004 PMID 15334591)
    labeling_efficiency=0.85,       # pCASL (Alsop Table 3)
)

pipeline = ASLPipeline(config)

# Run with separate label, control, and M0 data
result = pipeline.run(label_data, control_data, m0_data)

cbf_map = result.cbf_result.cbf_map.values
quality = result.cbf_result.quality_mask
print(f"CBF mean: {cbf_map[quality].mean():.1f} ml/100g/min")
```

### ASL Pipeline Steps

1. Load ASL and M0 data
2. Create brain mask
3. Compute perfusion difference (ΔM)
4. Apply M0 calibration
5. Quantify CBF
6. Export results

## IVIM Pipeline

Complete IVIM analysis:

```python
import osipy
from osipy.ivim import FittingMethod
from osipy.pipeline import IVIMPipeline, IVIMPipelineConfig

# Create pipeline with config object
config = IVIMPipelineConfig(
    fitting_method=FittingMethod.SEGMENTED,
)

pipeline = IVIMPipeline(config)

# Run
result = pipeline.run(dwi_data, b_values)

fit = result.fit_result
D = fit.d_map.values
f = fit.f_map.values
valid = fit.quality_mask > 0
print(f"D mean: {D[valid].mean()*1e3:.3f} x10^-3 mm²/s")
print(f"f mean: {f[valid].mean():.3f}")
```

### IVIM Pipeline Steps

1. Load DWI data
2. Create tissue mask
3. Fit bi-exponential model
4. Apply quality control
5. Export D, D*, f maps

## Batch Processing

Process multiple subjects:

```python
import osipy
from pathlib import Path

# Find all subjects
data_dir = Path("bids_dataset/")
subjects = [d.name for d in data_dir.glob("sub-*")]

from osipy.pipeline import DCEPipeline, DCEPipelineConfig

config = DCEPipelineConfig(
    model="extended_tofts",
    aif_source="population",
    population_aif="parker",
)
pipeline = DCEPipeline(config)

results = {}
for subject in subjects:
    print(f"Processing {subject}...")

    # Load subject data
    data = osipy.load_nifti(data_dir / subject / "perf" / f"{subject}_dce.nii.gz")
    time = np.arange(data.shape[-1]) * 3.5

    try:
        results[subject] = pipeline.run(data, time)
        valid = results[subject].quality_mask
        print(f"  Success: {valid.sum()} valid voxels")
    except Exception as e:
        print(f"  Failed: {e}")
        results[subject] = None

# Summary
successful = sum(1 for r in results.values() if r is not None)
print(f"\nProcessed {successful}/{len(subjects)} subjects successfully")
```

## Pipeline Output

All pipelines generate:

1. **Parameter maps** (NIfTI format)
2. **Quality mask** (NIfTI format)
3. **Provenance JSON** (analysis parameters)
4. **Summary statistics** (CSV)

Access and save pipeline outputs:

```python
# Access all outputs (DCEPipelineResult wraps DCEFitResult)
result = pipeline.run(dce_data, time)
fit = result.fit_result

# Parameter maps (each is a ParameterMap with .values attribute)
ktrans = fit.parameter_maps["Ktrans"].values
ve = fit.parameter_maps["ve"].values
vp = fit.parameter_maps["vp"].values
quality = fit.quality_mask

# Save custom summary
import pandas as pd

summary = pd.DataFrame({
    'Ktrans_mean': [ktrans[quality].mean()],
    'Ktrans_std': [ktrans[quality].std()],
    'valid_voxels': [quality.sum()],
})
summary.to_csv('summary.csv', index=False)
```

## Pipeline Customization

Override default steps:

```python
class CustomDCEPipeline(osipy.DCEPipeline):
    def create_mask(self):
        """Custom masking logic."""
        # Call parent's method
        mask = super().create_mask()

        # Add custom filtering
        mask = mask & (self.t1_map > 500) & (self.t1_map < 3000)

        return mask

    def select_aif(self):
        """Use custom AIF."""
        # Load pre-computed AIF
        aif_data = np.load("my_aif.npy")
        return osipy.ArterialInputFunction(self.time, aif_data)
```

## See Also

- [DCE-MRI Tutorial](../tutorials/dce-analysis.md)
- [How to Export BIDS Results](export-bids.md)
- [How to Compare Multiple Models](fit-multiple-models.md)
