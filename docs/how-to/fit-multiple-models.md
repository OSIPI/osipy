# How to Compare Multiple DCE Models

Fit and compare different pharmacokinetic models to determine which best describes your data.

## Via CLI

Run separate configs with different models and compare outputs:

```bash
# Generate a base config
osipy --dump-defaults dce > config_tofts.yaml
# Edit config_tofts.yaml: set model.method: tofts, then copy and change the name
osipy config_tofts.yaml data.nii.gz -o results/tofts/
osipy config_etofts.yaml data.nii.gz -o results/extended_tofts/
osipy config_patlak.yaml data.nii.gz -o results/patlak/
```

The only field that changes between configs is `pipeline.model.method`
(`tofts`, `extended_tofts`, `patlak`, `2cum`, or `2cxm`). Compare the output
parameter maps and R² maps across runs.

## Available Models

osipy provides five pharmacokinetic models:

| Model | Parameters | Use Case |
|-------|------------|----------|
| Standard Tofts | Ktrans, ve | Simple exchange, low vp |
| Extended Tofts | Ktrans, ve, vp | General use, tumors |
| Patlak | Ktrans, vp | Unidirectional influx |
| 2CUM | Fp, PS, vp | Unidirectional uptake, flow/permeability separation |
| 2CXM | Fp, PS, ve, vp | High temporal resolution, full exchange |

## Fit All Models (Python API)

Fit each model to the same data:

```python
import osipy
import numpy as np

# Your data
# concentration: 4D array (x, y, z, time)
# aif: ArterialInputFunction
# time: 1D array of time points

# Fit all models
results = {}

# Standard Tofts
results['tofts'] = osipy.fit_model(
    "tofts", concentration, aif, time
)

# Extended Tofts
results['extended'] = osipy.fit_model(
    "extended_tofts", concentration, aif, time
)

# Patlak
results['patlak'] = osipy.fit_model(
    "patlak", concentration, aif, time
)

# Two-compartment uptake model
results['2cum'] = osipy.fit_model(
    "2cum", concentration, aif, time
)

# Two-compartment exchange model
results['2cxm'] = osipy.fit_model(
    "2cxm", concentration, aif, time
)
```

## Compare Model Fit Quality

Use R² to compare goodness of fit:

```python
# Create combined quality mask (valid in all models)
combined_mask = np.ones_like(results['tofts'].quality_mask, dtype=bool)
for name, result in results.items():
    combined_mask &= (result.quality_mask > 0)

print(f"Voxels valid in all models: {combined_mask.sum()}")

# Compare R² values
print("\nModel Comparison (R²):")
for name, result in results.items():
    r2_values = result.r_squared_map[combined_mask]
    print(f"  {name:12s}: {r2_values.mean():.4f} ± {r2_values.std():.4f}")
```

## Statistical Model Comparison

Use information criteria for model selection:

```python
import numpy as np

def compute_aic(r_squared, n_params, n_points):
    """Compute Akaike Information Criterion.

    Lower AIC = better model (balancing fit quality and complexity)
    """
    # Approximate residual sum of squares from R²
    # RSS = (1 - R²) * TSS, assume TSS = 1 for comparison
    rss = 1 - r_squared
    # AIC = n * log(RSS/n) + 2k
    # Simplified for comparison: -n * log(R²) + 2k
    aic = -n_points * np.log(np.maximum(r_squared, 1e-10)) + 2 * n_params
    return aic

def compute_bic(r_squared, n_params, n_points):
    """Compute Bayesian Information Criterion."""
    bic = -n_points * np.log(np.maximum(r_squared, 1e-10)) + n_params * np.log(n_points)
    return bic

# Model parameters
model_params = {
    'tofts': 2,      # Ktrans, ve
    'extended': 3,   # Ktrans, ve, vp
    'patlak': 2,     # Ktrans, vp
    '2cum': 3,       # Fp, PS, vp
    '2cxm': 4,       # Fp, PS, ve, vp
}

n_timepoints = concentration.shape[-1]

# Compute AIC/BIC for each voxel
print("\nInformation Criteria (lower = better):")
for name, result in results.items():
    aic = compute_aic(result.r_squared_map[combined_mask],
                     model_params[name], n_timepoints)
    bic = compute_bic(result.r_squared_map[combined_mask],
                     model_params[name], n_timepoints)
    print(f"  {name:12s}: AIC={aic.mean():.1f}, BIC={bic.mean():.1f}")
```

## Voxelwise Model Selection

Select the best model for each voxel:

```python
def select_best_model(results, mask, criterion='aic'):
    """Select best model per voxel based on information criterion."""
    model_names = list(results.keys())
    n_models = len(model_names)
    # Initialize criterion arrays
    shape = results[model_names[0]].r_squared_map.shape
    criteria = np.zeros((*shape, n_models))
    n_points = 60  # number of time points
    for i, name in enumerate(model_names):
        r2 = results[name].r_squared_map
        n_params = model_params[name]
        if criterion == 'aic':
            criteria[..., i] = compute_aic(r2, n_params, n_points)
        else:
            criteria[..., i] = compute_bic(r2, n_params, n_points)
    # Select model with lowest criterion
    best_model_idx = np.argmin(criteria, axis=-1)
    # Create selection map
    model_selection = np.zeros(shape, dtype=int)
    model_selection[mask] = best_model_idx[mask]
    return model_selection, model_names

selection, names = select_best_model(results, combined_mask)

# Count selections
print("\nModel Selection Results:")
for i, name in enumerate(names):
    count = (selection[combined_mask] == i).sum()
    pct = 100 * count / combined_mask.sum()
    print(f"  {name:12s}: {count:6d} voxels ({pct:.1f}%)")
```

## Visualize Model Comparison

Plot R² differences:

```python
import matplotlib.pyplot as plt

slice_idx = concentration.shape[2] // 2

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# R² for each model
for ax, (name, result) in zip(axes.flat, results.items()):
    r2 = result.r_squared_map[:, :, slice_idx]
    im = ax.imshow(r2, cmap='viridis', vmin=0.5, vmax=1)
    ax.set_title(f'{name}\nR² = {r2[combined_mask[:,:,slice_idx]].mean():.3f}')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()
```

## Compare Fitted Parameters

Compare parameter estimates between models:

```python
import matplotlib.pyplot as plt

# Compare Ktrans between models
fig, ax = plt.subplots(figsize=(8, 6))

ktrans_tofts = results['tofts'].parameter_maps['Ktrans'].values[combined_mask]
ktrans_ext = results['extended'].parameter_maps['Ktrans'].values[combined_mask]

ax.scatter(ktrans_tofts, ktrans_ext, alpha=0.3, s=1)
ax.plot([0, 0.5], [0, 0.5], 'r--', label='Identity')
ax.set_xlabel('Ktrans (Standard Tofts)')
ax.set_ylabel('Ktrans (Extended Tofts)')
ax.set_title('Ktrans Comparison')
ax.legend()
ax.set_xlim(0, 0.5)
ax.set_ylim(0, 0.5)

plt.savefig('ktrans_comparison.png', dpi=150)
plt.show()

# Correlation
correlation = np.corrcoef(ktrans_tofts, ktrans_ext)[0, 1]
print(f"Ktrans correlation (Tofts vs Extended): {correlation:.4f}")
```

## Example: Tumor Analysis

Analyze parameter estimates across models within a tumor ROI:

```python
# For tumor ROI analysis
tumor_mask = load_tumor_mask()  # Your tumor segmentation

print("Tumor ROI Analysis:")
for name, result in results.items():
    ktrans = result.parameter_maps['Ktrans'].values[tumor_mask & combined_mask]
    r2 = result.r_squared_map[tumor_mask & combined_mask]
    print(f"\n{name}:")
    print(f"  Ktrans: {ktrans.mean():.4f} ± {ktrans.std():.4f}")
    print(f"  R²:     {r2.mean():.4f}")
    print(f"  Valid:  {len(ktrans)} voxels")
```

## See Also

- [Understanding Pharmacokinetic Models](../explanation/pharmacokinetic-models.md)
- [DCE-MRI Tutorial](../tutorials/dce-analysis.md)
- [How to Handle Fitting Failures](handle-fitting-failures.md)
