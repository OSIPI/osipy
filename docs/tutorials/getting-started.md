# Getting Started with osipy

Install osipy, verify your setup, and run your first perfusion analysis.

## Prerequisites

- Python 3.12 or later
- Basic familiarity with the command line (for CLI usage) or Python and NumPy (for the Python API)

## Step 1: Install osipy

Install with pip:

```bash
pip install osipy
```

Or install with uv:

```bash
uv add osipy
```

!!! tip "GPU Acceleration"

    For GPU support with CuPy, install the GPU extras:

    ```bash
    pip install osipy[gpu]
    ```

    GPU acceleration can significantly speed up fitting on large datasets.

## Step 2: Verify Your Installation

```bash
osipy --version
```

Or from Python:

```python
import osipy
print(f"osipy version: {osipy.__version__}")
print(f"GPU available: {osipy.is_gpu_available()}")
```

## Step 3: Run Your First Analysis (CLI)

### Option A: Interactive wizard (recommended for first-time users)

The wizard walks you through selecting a modality, configuring models and
methods, and pointing to your data.  It writes a validated YAML config file:

```bash
osipy --help-me-pls
```

You will be prompted for:

1. **Modality** -- DCE, DSC, ASL, or IVIM
2. **Pipeline settings** -- models, AIF source, fitting method, etc.
3. **Data paths** -- input files, masks, and optional data (T1 maps, M0 scans, b-values)

The wizard adapts its questions to your choices.  For example, if you select
DCE but do not have T1-weighted data, it will ask for an assumed T1 value
instead of a T1 mapping method.

Once the wizard finishes, run the generated config:

```bash
osipy config.yaml /path/to/dce_data.nii.gz
```

### Option B: Start from a template

If you already know which settings you need, generate a commented template
and edit it directly:

```bash
osipy --dump-defaults dce > config.yaml
# Edit config.yaml to set your model, AIF, and other options, then run:
osipy config.yaml /path/to/dce_data.nii.gz
```

This produces parameter maps (Ktrans, ve, vp), a quality mask, and provenance metadata in the output directory. See [How to Run Pipeline from YAML](../how-to/run-pipeline-cli.md) for the full config reference.

## Step 4: Using the Python API

For programmatic control — custom AIFs, batch processing, visualization — use the Python API directly.

### Create Sample Data

We'll use synthetic DCE-MRI data here; in practice, load from NIfTI or DICOM.

```python
import numpy as np
import osipy

# Create time points (60 time points over 5 minutes)
n_timepoints = 60
time = np.linspace(0, 300, n_timepoints)  # seconds

# Create a small 3D volume (8x8x4 voxels)
shape = (8, 8, 4, n_timepoints)

# Generate a population AIF (Parker model)
aif = osipy.ParkerAIF()(time)

# Create synthetic concentration curves
# Ground truth parameters
ktrans_true = 0.1   # min^-1
ve_true = 0.2       # fraction
vp_true = 0.02      # fraction

# Generate concentration using Extended Tofts model
# This simulates what real tissue would produce
np.random.seed(42)
concentration = np.zeros(shape)

for i in range(shape[0]):
    for j in range(shape[1]):
        for k in range(shape[2]):
            # Add some variation across voxels
            ktrans = ktrans_true * (0.8 + 0.4 * np.random.rand())
            ve = ve_true * (0.8 + 0.4 * np.random.rand())
            vp = vp_true * (0.8 + 0.4 * np.random.rand())
            # Simple Extended Tofts forward model
            # (In practice, use osipy's model prediction)
            cp = aif.concentration
            kep = ktrans / ve
            time_min = time / 60  # Convert to minutes
            # Convolution integral (simplified)
            dt = time_min[1] - time_min[0]
            impulse = ktrans * np.exp(-kep * time_min)
            ct = np.convolve(cp, impulse)[:n_timepoints] * dt + vp * cp
            # Add noise
            noise = np.random.normal(0, 0.005, n_timepoints)
            concentration[i, j, k, :] = ct + noise

print(f"Created synthetic data with shape: {concentration.shape}")
```

### Fit the Model

```python
# Fit Extended Tofts model
result = osipy.fit_model(
    "extended_tofts",
    concentration=concentration,
    aif=aif,
    time=time,
)

print(f"Fitting complete.")
print(f"Model: {result.model_name}")
print(f"Parameters: {list(result.parameter_maps.keys())}")
```

The result is a `DCEFitResult` object containing parameter maps:

```text
Fitting complete!
Model: extended_tofts
Parameters: ['Ktrans', 've', 'vp']
```

### Examine the Results

```python
# Extract parameter maps from DCEFitResult
ktrans_map = result.parameter_maps["Ktrans"].values
ve_map = result.parameter_maps["ve"].values
vp_map = result.parameter_maps["vp"].values
r_squared = result.r_squared_map
quality_mask = result.quality_mask

# Calculate statistics on valid voxels
valid = quality_mask

print(f"\nParameter Statistics (valid voxels only):")
print(f"Ktrans: mean={ktrans_map[valid].mean():.4f}, std={ktrans_map[valid].std():.4f}")
print(f"ve:     mean={ve_map[valid].mean():.4f}, std={ve_map[valid].std():.4f}")
print(f"vp:     mean={vp_map[valid].mean():.4f}, std={vp_map[valid].std():.4f}")
print(f"R²:     mean={r_squared[valid].mean():.4f}")
print(f"\nValid voxels: {valid.sum()} / {quality_mask.size}")
```

Output:

```text
Parameter Statistics (valid voxels only):
Ktrans: mean=0.0987, std=0.0198
ve:     mean=0.1985, std=0.0412
vp:     mean=0.0203, std=0.0051
R²:     mean=0.9823

Valid voxels: 256 / 256
```

The fitted values are close to ground truth (Ktrans=0.1, ve=0.2, vp=0.02).

### Understanding the Quality Mask

The `quality_mask` is crucial for reliable results:

```python
# The quality mask indicates which voxels have reliable fits
# - Values > 0: Valid fits (R² above threshold)
# - Values == 0: Failed fits or poor quality

print(f"Quality mask unique values: {np.unique(quality_mask)}")

# By default, R² threshold is 0.5
# You can check R² directly:
good_fits = r_squared > 0.5
print(f"Voxels with R² > 0.5: {good_fits.sum()}")
```

!!! warning "Always Use Quality Masks"

    When analyzing parameter maps, always apply the quality mask to exclude unreliable voxels. Invalid voxels may contain NaN or zero values that will skew your statistics.

### Visualize Results

```python
import matplotlib.pyplot as plt

# Select a slice
slice_idx = 2

fig, axes = plt.subplots(1, 4, figsize=(14, 3))

# Ktrans
im0 = axes[0].imshow(ktrans_map[:, :, slice_idx], cmap='hot', vmin=0, vmax=0.2)
axes[0].set_title('Ktrans (min⁻¹)')
plt.colorbar(im0, ax=axes[0])

# ve
im1 = axes[1].imshow(ve_map[:, :, slice_idx], cmap='viridis', vmin=0, vmax=0.5)
axes[1].set_title('ve (fraction)')
plt.colorbar(im1, ax=axes[1])

# vp
im2 = axes[2].imshow(vp_map[:, :, slice_idx], cmap='plasma', vmin=0, vmax=0.1)
axes[2].set_title('vp (fraction)')
plt.colorbar(im2, ax=axes[2])

# R²
im3 = axes[3].imshow(r_squared[:, :, slice_idx], cmap='gray', vmin=0, vmax=1)
axes[3].set_title('R²')
plt.colorbar(im3, ax=axes[3])

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.savefig('dce_results.png', dpi=150)
plt.show()
```

### Working with Real Data

In practice, you'll load data from files:

```python
# Load NIfTI data (returns PerfusionDataset)
dataset = osipy.load_nifti("path/to/dce_data.nii.gz")

# Access the data array
concentration = dataset.data
print(f"Data shape: {concentration.shape}")

# Load from BIDS dataset
# from osipy.common.io import load_bids
# dataset = load_bids("path/to/bids/dataset", subject="01")
```

See [How to Load Perfusion Data](../how-to/load-perfusion-data.md) for detailed loading instructions.

## Next Steps

- [DCE-MRI Analysis Tutorial](dce-analysis.md) -- complete workflow with real data
- [ASL Tutorial](asl-analysis.md) -- CBF quantification
- [IVIM Tutorial](ivim-analysis.md) -- diffusion-perfusion separation
- [DSC Tutorial](dsc-analysis.md) -- perfusion maps from DSC-MRI

## Troubleshooting

### Import Error

If you see `ModuleNotFoundError: No module named 'osipy'`:

1. Ensure you've activated the correct virtual environment
2. Reinstall: `pip install --force-reinstall osipy`

### GPU Not Detected

If `is_gpu_available()` returns `False` but you have a GPU:

1. Install CuPy: `pip install cupy-cuda12x` (match your CUDA version)
2. Verify CUDA is installed: `nvidia-smi`

### Poor Fitting Results

If R² values are consistently low:

1. Check your time points match the acquisition
2. Verify the AIF is appropriate for your data
3. Consider using a simpler model (Standard Tofts instead of Extended)
