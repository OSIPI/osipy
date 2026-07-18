# How to Use a Custom AIF

Provide your own measured arterial input function for DCE or DSC analysis.

## Via CLI (YAML Config)

Point to your AIF file in the config:

```yaml
modality: dce
pipeline:
  model:
    method: extended_tofts
  aif_source: manual
data:
  aif_file: /path/to/aif.npy  # NumPy array with shape (n_timepoints,)
```

The AIF file should contain concentration values in mM, sampled at the same time points as your data.

## Via Python API

### Create AIF from Measured Data

Create an `ArterialInputFunction` from your measured values:

```python
import numpy as np
import osipy
from osipy.common.types import AIFType

# Your measured AIF data
time = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])  # seconds
concentration = np.array([0, 0.1, 2.5, 5.0, 3.2, 1.8, 1.0, 0.6, 0.4, 0.3])  # mM

# Create AIF object
aif = osipy.ArterialInputFunction(
    time=time,
    concentration=concentration,
    aif_type=AIFType.MEASURED,
)

print(f"AIF peak: {aif.peak_concentration:.2f} mM at {aif.peak_time:.1f} s")
```

## Extract AIF from DCE Data

Select arterial voxels and create an AIF:

```python
import numpy as np
import osipy
from osipy.common.types import AIFType

# Load DCE concentration data
concentration = np.load("concentration_4d.npy")  # shape: (x, y, z, time)
time = np.linspace(0, 300, concentration.shape[-1])

# Method 1: Manual ROI selection
# Define arterial voxel coordinates (e.g., from ROI drawing)
aif_coords = [(32, 45, 12), (33, 45, 12), (32, 46, 12)]

# Extract and average
aif_curves = [concentration[x, y, z, :] for x, y, z in aif_coords]
aif_mean = np.mean(aif_curves, axis=0)

# Create AIF
aif = osipy.ArterialInputFunction(time=time, concentration=aif_mean, aif_type=AIFType.MEASURED)
```

## Automatic AIF Detection

Use osipy's automatic detection:

```python
import osipy

# Automatic AIF detection for DCE
# detect_aif takes a PerfusionDataset, not raw arrays
aif_result = osipy.detect_aif(
    dataset=dce_dataset,         # PerfusionDataset from load_nifti()
    roi_mask=brain_mask,
    method="cluster",            # or "threshold"
)

# Access the detected AIF (AIFDetectionResult object)
aif = aif_result.aif             # ArterialInputFunction object
aif_mask = aif_result.voxel_mask # Which voxels were selected

print(f"Selected {aif_mask.sum()} arterial voxels")
```

## Interpolate AIF to Match Data

When AIF time points don't match your data:

```python
import numpy as np
import osipy
from osipy.common.types import AIFType

# Original AIF at different time points
aif_time = np.array([0, 10, 20, 30, 40])
aif_conc = np.array([0, 3.0, 5.0, 2.0, 1.0])

# Your data time points
data_time = np.linspace(0, 40, 100)

# Interpolate using numpy
aif_interp = np.interp(data_time, aif_time, aif_conc)

# Create interpolated AIF
aif = osipy.ArterialInputFunction(time=data_time, concentration=aif_interp, aif_type=AIFType.MEASURED)
```

## AIF Quality Checks

Verify your AIF is suitable:

```python
import numpy as np

def check_aif_quality(aif):
    """Check AIF quality metrics."""
    time = aif.time
    conc = aif.concentration
    # Check for baseline before bolus
    baseline = conc[:5].mean()
    if baseline > 0.1:
        print("Warning: High baseline concentration")
    # Check peak is well-defined
    peak_idx = np.argmax(conc)
    peak_time = time[peak_idx]
    peak_conc = conc[peak_idx]
    if peak_idx < 3:
        print("Warning: Peak too early, may miss bolus arrival")
    if peak_idx > len(conc) - 5:
        print("Warning: Peak too late, may be truncated")
    # Check signal-to-noise
    tail_std = conc[-10:].std()
    snr = peak_conc / tail_std if tail_std > 0 else np.inf
    if snr < 10:
        print(f"Warning: Low SNR ({snr:.1f})")
    print(f"Peak: {peak_conc:.2f} mM at {peak_time:.1f} s")
    print(f"Estimated SNR: {snr:.1f}")

check_aif_quality(aif)
```

## AIF Correction

Apply partial volume or dispersion correction:

```python
from osipy.common.types import AIFType

def correct_aif_partial_volume(aif, pv_fraction):
    """Correct AIF for partial volume effects.

    Parameters
    ----------
    aif : ArterialInputFunction
        Original AIF
    pv_fraction : float
        Estimated arterial volume fraction (0-1)
    """
    corrected_conc = aif.concentration / pv_fraction
    return osipy.ArterialInputFunction(
        time=aif.time,
        concentration=corrected_conc,
        aif_type=AIFType.MEASURED,
    )

# Apply 30% partial volume correction
aif_corrected = correct_aif_partial_volume(aif, pv_fraction=0.3)
```

## Use AIF in Fitting

Pass the AIF to fitting functions:

```python
# DCE fitting with custom AIF
result = osipy.fit_model(
    "extended_tofts",
    concentration=concentration,
    aif=aif,  # Your custom AIF
    time=time,
)

# DSC deconvolution with custom AIF
perfusion_maps = osipy.compute_perfusion_maps(
    concentration=delta_r2,
    aif=aif.concentration,  # Pass the concentration array
    time=time,
    mask=brain_mask
)
```

## Visualize AIF

Plot your AIF for verification:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))

# Plot AIF
ax.plot(aif.time, aif.concentration, 'r-', linewidth=2, label='Custom AIF')

# Add reference population AIF for comparison
parker = osipy.ParkerAIF()(aif.time)
ax.plot(aif.time, parker.concentration, 'b--', label='Parker AIF (reference)')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Concentration (mM)')
ax.set_title('Arterial Input Function Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

plt.savefig('aif_comparison.png', dpi=150)
plt.show()
```

## Common Issues

### Time Unit Mismatch

Ensure time units are consistent (osipy expects seconds):

```python
# If your AIF is in minutes:
aif_time_seconds = aif_time_minutes * 60

aif = osipy.ArterialInputFunction(
    time=aif_time_seconds,
    concentration=aif_conc,
    aif_type=AIFType.MEASURED,
)
```

### Negative Concentrations

Clip negative values (can arise from noise):

```python
aif_conc_clipped = np.maximum(aif_conc, 0)

aif = osipy.ArterialInputFunction(
    time=time,
    concentration=aif_conc_clipped,
    aif_type=AIFType.MEASURED,
)
```

### Late Bolus Arrival

Shift AIF using the shift_aif utility:

```python
from osipy.common.aif import shift_aif

# Shift AIF concentration by a known delay (in seconds)
shifted_conc = shift_aif(aif_conc, time, delay=20.0, xp=np)

aif = osipy.ArterialInputFunction(
    time=time,
    concentration=shifted_conc,
    aif_type=AIFType.MEASURED,
)
```

Alternatively, use `fit_delay=True` in `fit_model()` to estimate the delay automatically per voxel:

```python
result = osipy.fit_model(
    "extended_tofts", concentration, aif, time,
    fit_delay=True  # Estimates per-voxel arterial delay
)
delay_map = result.parameter_maps["delay"].values  # in seconds
```

## See Also

- [How to Choose Population AIF](choose-population-aif.md)
- [DCE-MRI Tutorial](../tutorials/dce-analysis.md)
- [DSC-MRI Tutorial](../tutorials/dsc-analysis.md)
