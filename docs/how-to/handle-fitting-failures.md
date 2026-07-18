# How to Handle Fitting Failures

Diagnose and resolve common fitting problems in perfusion analysis.

## CLI Troubleshooting

If a CLI pipeline run produces poor results:

```bash
# Validate your config first
osipy --validate config.yaml

# Run with verbose logging to see per-step diagnostics
osipy config.yaml data.nii.gz -v
```

Common config issues: wrong `population_aif`, incorrect `flip_angles` or `tr`, missing `mask`. Check the verbose output for warnings about low R² or high failure rates.

## Identify Fitting Failures

Check the quality mask and R² values:

```python
import numpy as np
import osipy

# After fitting
result = osipy.fit_model("extended_tofts", concentration, aif, time)

# Check quality
quality_mask = result.quality_mask
r_squared = result.r_squared_map

total_voxels = quality_mask.size
valid_voxels = (quality_mask > 0).sum()
failed_voxels = total_voxels - valid_voxels

print(f"Total voxels: {total_voxels}")
print(f"Valid fits: {valid_voxels} ({100*valid_voxels/total_voxels:.1f}%)")
print(f"Failed fits: {failed_voxels} ({100*failed_voxels/total_voxels:.1f}%)")

# R² distribution
valid_r2 = r_squared[quality_mask > 0]
print(f"\nR² statistics (valid voxels):")
print(f"  Mean: {valid_r2.mean():.4f}")
print(f"  Min:  {valid_r2.min():.4f}")
print(f"  <0.5: {(valid_r2 < 0.5).sum()} voxels")
```

## Common Causes and Solutions

### Poor Signal Quality

**Symptoms**: Low R², many failed voxels, noisy parameter maps

Diagnose by checking the signal-to-noise ratio:

```python
# Check signal-to-noise ratio
baseline = concentration[..., :5].mean(axis=-1)
noise = concentration[..., :5].std(axis=-1)
snr = baseline / (noise + 1e-10)

print(f"SNR range: {snr.min():.1f} - {snr.max():.1f}")
print(f"Low SNR voxels (<10): {(snr < 10).sum()}")
```

Apply a stricter mask to exclude low-SNR voxels:

```python
# 1. Apply stricter mask
high_snr_mask = snr > 10
result = osipy.fit_model("extended_tofts", concentration, aif, time, mask=high_snr_mask)

# 2. Spatial smoothing (before fitting)
# Note: osipy does not currently include a spatial smoothing function.
# Consider using an external library for spatial smoothing before fitting,
# or use osipy's temporal_filter for temporal smoothing:
# from osipy.common.signal.filtering import temporal_filter
```

### Incorrect Time Units

**Symptoms**: Ktrans clustering at bounds (0 or 5.0), all fits fail

Check the time array:

```python
print(f"Time array: {time[:5]} ... {time[-5:]}")
print(f"Expected: seconds from injection")
```

Convert to seconds if needed:

```python
# Convert from minutes to seconds if needed
time_seconds = time * 60

# Or from milliseconds
time_seconds = time / 1000
```

### AIF (Arterial Input Function) Timing Mismatch

**Symptoms**: Systematically low R², concentration curves don't match AIF shape

Plot the AIF alongside a tissue curve to check alignment:

```python
import matplotlib.pyplot as plt

# Compare AIF and tissue timing
fig, ax = plt.subplots()
ax.plot(time, aif.concentration, 'r-', label='AIF')

# Sample tissue curve
sample_curve = concentration[32, 32, 16, :]
ax.plot(time, sample_curve / sample_curve.max() * aif.concentration.max(),
        'b-', label='Tissue (scaled)')

ax.set_xlabel('Time (s)')
ax.legend()
plt.show()
```

Option 1 -- automatic delay fitting (recommended):

```python
# Let osipy estimate per-voxel arterial delay automatically
result = osipy.fit_model(
    "extended_tofts", concentration, aif, time,
    fit_delay=True  # Adds a "delay" parameter to the fit
)

# Check estimated delays
delay = result.parameter_maps["delay"].values
print(f"Estimated delay range: {delay[result.quality_mask > 0].min():.1f} - "
      f"{delay[result.quality_mask > 0].max():.1f} s")
```

Option 2 -- manual AIF shift with the `shift_aif` utility:

```python
from osipy.common.aif import shift_aif
import numpy as np

# Shift AIF by a known delay
shifted_conc = shift_aif(aif.concentration, time, delay=5.0, xp=np)
aif_shifted = osipy.ArterialInputFunction(time=time, concentration=shifted_conc)
result = osipy.fit_model("extended_tofts", concentration, aif_shifted, time)
```

### Inappropriate Model

**Symptoms**: Good fits in some regions, poor in others

Compare models to check whether the chosen model is appropriate:

```python
# Compare models
result_standard = osipy.fit_model("tofts", concentration, aif, time)
result_extended = osipy.fit_model("extended_tofts", concentration, aif, time)

# Check R² improvement
r2_diff = result_extended.r_squared_map - result_standard.r_squared_map
print(f"R² improvement with Extended Tofts: {r2_diff.mean():.4f}")
```

**Solution**: Try different models (see [Compare Multiple Models](fit-multiple-models.md))

### Boundary Violations

**Symptoms**: Parameters at bounds (Ktrans=0 or max), warnings about convergence

Check whether parameters cluster at their bounds:

```python
ktrans = result.parameter_maps['Ktrans'].values
valid = quality_mask > 0

# Check for boundary clustering
at_lower = (ktrans[valid] < 0.001).sum()
at_upper = (ktrans[valid] > 0.9).sum()

print(f"At lower bound: {at_lower}")
print(f"At upper bound: {at_upper}")
```

Adjust bounds if physiologically justified:

```python
# Adjust bounds if physiologically justified
result = osipy.fit_model(
    "extended_tofts", concentration, aif, time,
    bounds_override={
        'Ktrans': (0.0001, 2.0),  # Wider range
        've': (0.01, 0.99),
        'vp': (0.001, 0.5)
    }
)
```

### Negative Concentrations

**Symptoms**: Fitting errors, NaN results

Check for negative values:

```python
negative_voxels = (concentration < 0).any(axis=-1).sum()
print(f"Voxels with negative concentration: {negative_voxels}")
```

Clip or correct them:

```python
# Clip negative values
concentration_clipped = np.maximum(concentration, 0)

# Or improve baseline correction
baseline = concentration[..., :n_baseline].mean(axis=-1, keepdims=True)
concentration_corrected = concentration - baseline
concentration_corrected = np.maximum(concentration_corrected, 0)
```

## Visualize Failures

Plot failed vs successful fits:

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_fitting_failures(result, concentration, time, aif, slice_idx=None):
    """Visualize fitting success and failures."""
    if slice_idx is None:
        slice_idx = result.quality_mask.shape[2] // 2
    quality = result.quality_mask[:, :, slice_idx]
    r_squared = result.r_squared_map[:, :, slice_idx]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    # Success/failure map
    axes[0, 0].imshow(quality, cmap='RdYlGn')
    axes[0, 0].set_title(f'Quality Mask\n(Green=valid)')
    # R² map
    im = axes[0, 1].imshow(r_squared, cmap='viridis', vmin=0, vmax=1)
    axes[0, 1].set_title('R² Map')
    plt.colorbar(im, ax=axes[0, 1])
    # R² histogram
    valid_r2 = result.r_squared_map[result.quality_mask > 0]
    axes[0, 2].hist(valid_r2, bins=50, color='steelblue')
    axes[0, 2].axvline(0.5, color='r', linestyle='--', label='Threshold')
    axes[0, 2].set_xlabel('R²')
    axes[0, 2].set_title('R² Distribution')
    axes[0, 2].legend()
    # Sample successful fit
    success_idx = np.where(quality > 0)
    if len(success_idx[0]) > 0:
        x, y = success_idx[0][0], success_idx[1][0]
        axes[1, 0].plot(time, concentration[x, y, slice_idx, :], 'ko', label='Data')
        axes[1, 0].set_title(f'Successful Fit\nR²={r_squared[x,y]:.3f}')
        axes[1, 0].legend()
    # Sample failed fit
    fail_idx = np.where(quality == 0)
    if len(fail_idx[0]) > 0:
        x, y = fail_idx[0][0], fail_idx[1][0]
        axes[1, 1].plot(time, concentration[x, y, slice_idx, :], 'ko', label='Data')
        axes[1, 1].set_title(f'Failed Fit\nR²={r_squared[x,y]:.3f}')
        axes[1, 1].legend()
    # Failure reasons
    axes[1, 2].axis('off')
    n_total = quality.size
    n_valid = (quality > 0).sum()
    n_low_r2 = (r_squared < 0.5).sum() - (quality == 0).sum()
    text = f"""Fitting Summary:
    Total voxels: {n_total}
    Valid fits: {n_valid} ({100*n_valid/n_total:.1f}%)
    Low R² (<0.5): {n_low_r2}
    """
    axes[1, 2].text(0.1, 0.5, text, fontsize=12, family='monospace')
    plt.tight_layout()
    return fig

fig = visualize_fitting_failures(result, concentration, time, aif)
plt.savefig('fitting_diagnostics.png', dpi=150)
```

## Retry Strategies

### Progressive Simplification

Start with a complex model and fall back to a simpler one for failed voxels:

```python
# Start with complex model, fall back to simpler
def fit_with_fallback(concentration, aif, time, mask):
    # Try Extended Tofts first
    result = osipy.fit_model("extended_tofts", concentration, aif, time, mask=mask)
    # Find failed voxels
    failed = (result.quality_mask == 0) & mask
    if failed.sum() > 0:
        print(f"Retrying {failed.sum()} voxels with Standard Tofts")
        # Retry with simpler model
        result_simple = osipy.fit_model("tofts", concentration, aif, time,
                                        mask=failed)
        # Merge results
        # Note: Merging DCEFitResult objects requires manual attribute updates.
        # This is a simplified example—in practice, you would need to update
        # result.parameter_maps, result.quality_mask, and result.r_squared_map.
    return result

result = fit_with_fallback(concentration, aif, time, mask)
```

### Different Bounds Overrides

Try multiple bound configurations and keep the best result:

```python
# Multiple bound configurations
def fit_with_multiple_bounds(concentration, aif, time):
    best_result = None
    best_r2 = -np.inf
    # Try different bounds_override configurations
    bounds_configs = [
        # Default-like tight bounds
        {'Ktrans': (0.0001, 1.0), 've': (0.01, 0.99), 'vp': (0.001, 0.1)},
        # Wider bounds for high-permeability tissue
        {'Ktrans': (0.0001, 2.0), 've': (0.01, 0.99), 'vp': (0.001, 0.5)},
        # Narrow bounds for low-permeability tissue
        {'Ktrans': (0.0001, 0.1), 've': (0.1, 0.8), 'vp': (0.001, 0.05)},
    ]
    for bounds in bounds_configs:
        result = osipy.fit_model("extended_tofts", concentration, aif, time,
                                bounds_override=bounds)
        mean_r2 = result.r_squared_map[result.quality_mask > 0].mean()
        if mean_r2 > best_r2:
            best_r2 = mean_r2
            best_result = result
    return best_result
```

## Prevention Tips

1. **Check data quality first**: SNR, motion artifacts, baseline
2. **Verify timing**: Time units, bolus arrival, AIF alignment
3. **Start simple**: Fit Standard Tofts before Extended
4. **Use masks**: Exclude non-tissue voxels
5. **Monitor progress**: Check R² during development

## See Also

- [DCE-MRI Tutorial](../tutorials/dce-analysis.md)
- [How to Compare Multiple Models](fit-multiple-models.md)
- [Understanding Pharmacokinetic Models](../explanation/pharmacokinetic-models.md)
