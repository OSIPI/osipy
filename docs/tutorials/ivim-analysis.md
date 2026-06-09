# IVIM Analysis Tutorial

Separate diffusion and perfusion components from multi-b-value DWI data using bi-exponential IVIM fitting.

## Prerequisites

- Completed [Getting Started](getting-started.md) tutorial
- Multi-b-value DWI data (minimum 4 b-values, ideally 8+)
- Understanding of basic diffusion MRI concepts

**Using the CLI?** Generate a config with `osipy --dump-defaults ivim > config.yaml`, edit it, then run `osipy config.yaml data.nii.gz`. The fitting strategy (`fitting.method`: segmented, full, or bayesian) and the signal model (`model.model`: biexponential or simplified) are selectable config blocks; the Bayesian strategy exposes `prior_scale`/`noise_std`/`compute_uncertainty`, and `bounds`/`initial_guess` overrides live under `fitting`. See [How to Run Pipeline from YAML](../how-to/run-pipeline-cli.md). The tutorial below covers the Python API for step-by-step control.

## Background

IVIM analysis separates two signal decay components in DWI:

1. **Tissue diffusion (D)**: True water diffusion in tissue (~0.7-1.5 × 10⁻³ mm²/s)
2. **Pseudo-diffusion (D*)**: Fast signal decay from blood microcirculation (~5-20 × 10⁻³ mm²/s)

The bi-exponential signal model is:

$$
S(b) = S_0 \left[ f \cdot e^{-b \cdot D^*} + (1-f) \cdot e^{-b \cdot D} \right]
$$

Where:

| Parameter | Symbol | Units | Typical Range |
|-----------|--------|-------|---------------|
| Diffusion coefficient | D | mm²/s | 0.5-2.0 × 10⁻³ |
| Pseudo-diffusion | D* | mm²/s | 5-100 × 10⁻³ |
| Perfusion fraction | f | fraction | 0.05-0.30 |

For theory details, see [Understanding IVIM](../explanation/ivim-theory.md).

## Step 1: Load Multi-b-value Data

!!! example "Load multi-b-value DWI data"

    ```python
    import numpy as np
    import osipy

    # Load 4D DWI data (returns PerfusionDataset)
    dwi_dataset = osipy.load_nifti("dwi_multi_b.nii.gz")

    # Define b-values (must match acquisition order)
    b_values = np.array([0, 10, 20, 50, 100, 200, 400, 800])  # s/mm²

    print(f"DWI data shape: {dwi_dataset.shape}")
    print(f"Number of b-values: {len(b_values)}")
    print(f"B-values: {b_values}")

    # Verify dimensions match
    assert dwi_dataset.shape[-1] == len(b_values), "B-value count must match last dimension"
    ```

!!! info "B-value Recommendations"

    For reliable IVIM fitting:

    - **Minimum**: 4 b-values (but results may be unstable)
    - **Recommended**: 8+ b-values
    - **Low b-values** (0-100): Capture perfusion effects
    - **High b-values** (200-800): Capture diffusion

    Example optimal sampling: 0, 10, 20, 40, 80, 110, 140, 200, 400, 800 s/mm²

## Step 2: Create a Brain/Tissue Mask

!!! example "Create a tissue mask"

    ```python
    # Use b=0 image for masking (access .data for raw array)
    b0_image = dwi_dataset.data[..., 0]

    # Simple threshold mask
    threshold = np.percentile(b0_image[b0_image > 0], 5)
    mask = b0_image > threshold

    # Optional: exclude very bright (CSF) voxels
    csf_threshold = np.percentile(b0_image[mask], 98)
    mask = mask & (b0_image < csf_threshold)

    print(f"Tissue voxels: {mask.sum()} / {mask.size}")
    ```

## Step 3: Examine Signal Decay

!!! example "Plot signal decay to verify IVIM behavior"

    ```python
    import matplotlib.pyplot as plt

    # Select a sample voxel with good signal
    sample_indices = np.where(mask)
    center_idx = len(sample_indices[0]) // 2
    x, y, z = sample_indices[0][center_idx], sample_indices[1][center_idx], sample_indices[2][center_idx]

    # Extract signal at all b-values
    signal = dwi_dataset.data[x, y, z, :]

    # Normalize to b=0
    s0 = signal[0]
    signal_norm = signal / s0

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Linear scale
    axes[0].plot(b_values, signal_norm, 'ko-', markersize=8)
    axes[0].set_xlabel('b-value (s/mm²)')
    axes[0].set_ylabel('S/S₀')
    axes[0].set_title('Signal Decay (Linear)')
    axes[0].grid(True, alpha=0.3)

    # Log scale - should show bi-exponential behavior
    axes[1].semilogy(b_values, signal_norm, 'ko-', markersize=8)
    axes[1].set_xlabel('b-value (s/mm²)')
    axes[1].set_ylabel('S/S₀ (log scale)')
    axes[1].set_title('Signal Decay (Log) - Look for curvature at low b')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ivim_signal_decay.png', dpi=150)
    plt.show()
    ```

!!! tip "Interpreting the Log Plot"

    In the log plot:

    - **Linear decay** (straight line): Pure diffusion, no IVIM effect
    - **Curved at low b-values**: IVIM effect present (perfusion component)

    The curvature at low b-values indicates the perfusion fraction.

## Step 4: Fit the IVIM Model

!!! example "Fit the IVIM bi-exponential model"

    ```python
    from osipy.ivim import FittingMethod

    # Fit IVIM bi-exponential model (takes raw signal array, not PerfusionDataset)
    result = osipy.fit_ivim(
        signal=dwi_dataset.data,
        b_values=b_values,
        mask=mask,
        method=FittingMethod.SEGMENTED,  # SEGMENTED or FULL
    )

    # result is an IVIMFitResult object with ParameterMap attributes
    # Access .values on each ParameterMap to get the numpy arrays
    D = result.d_map.values           # Tissue diffusion
    D_star = result.d_star_map.values # Pseudo-diffusion
    f = result.f_map.values           # Perfusion fraction
    s0 = result.s0_map.values         # Signal at b=0
    r_squared = result.r_squared
    quality_mask = result.quality_mask

    # Statistics on valid voxels
    valid = quality_mask > 0
    print(f"\nIVIM Results (valid voxels: {valid.sum()}):")
    print(f"  D:     {D[valid].mean()*1e3:.3f} ± {D[valid].std()*1e3:.3f} × 10⁻³ mm²/s")
    print(f"  D*:    {D_star[valid].mean()*1e3:.1f} ± {D_star[valid].std()*1e3:.1f} × 10⁻³ mm²/s")
    print(f"  f:     {f[valid].mean():.3f} ± {f[valid].std():.3f}")
    print(f"  R²:    {r_squared[valid].mean():.4f}")
    ```

### Fitting Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `segmented` | Two-step: D from high b, then D*/f | Noisy data, faster |
| `full` | Simultaneous 4-parameter fit | High SNR data |

!!! example "Compare segmented and full fitting methods"

    ```python
    # Compare fitting methods
    result_full = osipy.fit_ivim(
        signal=dwi_dataset.data,
        b_values=b_values,
        mask=mask,
        method=FittingMethod.FULL,
    )

    print(f"\nMethod Comparison (R²):")
    print(f"  Segmented: {result.r_squared[valid].mean():.4f}")
    print(f"  Full:      {result_full.r_squared[valid].mean():.4f}")
    ```

## Step 5: Understanding Parameter Constraints

!!! example "Customize parameter constraints"

    IVIM fitting uses physiologically motivated constraints:

    ```python
    from osipy.ivim import IVIMFitParams

    # Default parameter bounds (can be customized)
    # D:     [0.1, 5.0] × 10⁻³ mm²/s
    # D*:    [2.0, 100.0] × 10⁻³ mm²/s
    # f:     [0.0, 0.7]
    # S0:    [0, ∞]

    # Critical constraint: D* > D (always enforced)
    # This ensures D* represents faster pseudo-diffusion

    # Custom bounds for specific applications
    custom_params = IVIMFitParams(
        method=FittingMethod.SEGMENTED,
        bounds={
            "D": (0.0005, 0.003),    # Tighter D range
            "D_star": (0.01, 0.05),  # Tighter D* range
            "f": (0.0, 0.5),         # Lower max f
        },
    )

    custom_result = osipy.fit_ivim(
        signal=dwi_dataset.data,
        b_values=b_values,
        mask=mask,
        params=custom_params,
    )
    ```

!!! warning "The D* > D Constraint"

    osipy enforces D* > D to maintain physical meaning:

    - D* (pseudo-diffusion from blood flow) should be faster than D
    - Violations typically indicate fitting failure
    - These voxels are excluded from the quality mask

## Step 6: Visualize Parameter Maps

!!! example "Plot IVIM parameter maps"

    ```python
    # Select representative slice
    slice_idx = dwi_dataset.shape[2] // 2

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Row 1: IVIM parameters
    im0 = axes[0, 0].imshow(D[:, :, slice_idx] * 1e3, cmap='viridis', vmin=0, vmax=2)
    axes[0, 0].set_title('D (×10⁻³ mm²/s)')
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(D_star[:, :, slice_idx] * 1e3, cmap='plasma', vmin=0, vmax=50)
    axes[0, 1].set_title('D* (×10⁻³ mm²/s)')
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].imshow(f[:, :, slice_idx], cmap='hot', vmin=0, vmax=0.3)
    axes[0, 2].set_title('f (perfusion fraction)')
    plt.colorbar(im2, ax=axes[0, 2])

    # Row 2: Quality metrics
    im3 = axes[1, 0].imshow(r_squared[:, :, slice_idx], cmap='gray', vmin=0.5, vmax=1)
    axes[1, 0].set_title('R²')
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(quality_mask[:, :, slice_idx], cmap='binary')
    axes[1, 1].set_title('Quality Mask')
    plt.colorbar(im4, ax=axes[1, 1])

    im5 = axes[1, 2].imshow(b0_image[:, :, slice_idx], cmap='gray')
    axes[1, 2].set_title('b=0 Reference')
    plt.colorbar(im5, ax=axes[1, 2])

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('ivim_parameter_maps.png', dpi=300, bbox_inches='tight')
    plt.show()
    ```

### Fitted vs Measured Signal

!!! example "Compare fitted model to measured data"

    ```python
    # Compare fitted model to measured data for sample voxels
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()

    # Select 6 random valid voxels
    valid_indices = np.where(valid)
    np.random.seed(42)
    sample_idx = np.random.choice(len(valid_indices[0]), 6, replace=False)

    b_dense = np.linspace(0, b_values.max(), 100)

    for i, idx in enumerate(sample_idx):
        x, y, z = valid_indices[0][idx], valid_indices[1][idx], valid_indices[2][idx]

        # Measured signal
        signal_measured = dwi_dataset.data[x, y, z, :]
        s0_voxel = s0[x, y, z]
        signal_norm = signal_measured / s0_voxel

        # Fitted model
        D_voxel = D[x, y, z]
        D_star_voxel = D_star[x, y, z]
        f_voxel = f[x, y, z]

        signal_fit = f_voxel * np.exp(-b_dense * D_star_voxel) + (1 - f_voxel) * np.exp(-b_dense * D_voxel)

        # Plot
        axes[i].semilogy(b_values, signal_norm, 'ko', markersize=8, label='Measured')
        axes[i].semilogy(b_dense, signal_fit, 'r-', linewidth=2, label='Fitted')
        axes[i].set_xlabel('b-value (s/mm²)')
        axes[i].set_ylabel('S/S₀')
        axes[i].set_title(f'Voxel ({x},{y},{z})\nD={D_voxel*1e3:.2f}, D*={D_star_voxel*1e3:.1f}, f={f_voxel:.2f}')
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(0.1, 1.1)

    plt.tight_layout()
    plt.savefig('ivim_fitting_quality.png', dpi=150)
    plt.show()
    ```

## Step 7: Parameter Histograms

!!! example "Plot parameter distributions"

    ```python
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # D histogram
    axes[0].hist(D[valid] * 1e3, bins=50, color='steelblue', edgecolor='white')
    axes[0].axvline(D[valid].mean() * 1e3, color='red', linestyle='--')
    axes[0].set_xlabel('D (×10⁻³ mm²/s)')
    axes[0].set_ylabel('Voxel Count')
    axes[0].set_title(f'Diffusion Coefficient\nMean: {D[valid].mean()*1e3:.3f}')

    # D* histogram
    axes[1].hist(D_star[valid] * 1e3, bins=50, color='purple', edgecolor='white', range=(0, 100))
    axes[1].axvline(D_star[valid].mean() * 1e3, color='red', linestyle='--')
    axes[1].set_xlabel('D* (×10⁻³ mm²/s)')
    axes[1].set_ylabel('Voxel Count')
    axes[1].set_title(f'Pseudo-diffusion\nMean: {D_star[valid].mean()*1e3:.1f}')

    # f histogram
    axes[2].hist(f[valid], bins=50, color='orange', edgecolor='white')
    axes[2].axvline(f[valid].mean(), color='red', linestyle='--')
    axes[2].set_xlabel('f (fraction)')
    axes[2].set_ylabel('Voxel Count')
    axes[2].set_title(f'Perfusion Fraction\nMean: {f[valid].mean():.3f}')

    plt.tight_layout()
    plt.savefig('ivim_histograms.png', dpi=150)
    plt.show()
    ```

## Step 8: Export Results

Save results in BIDS format:

!!! warning "Experimental Feature"

    BIDS derivative export is partially implemented and may not produce fully compliant output for all use cases.

!!! example "Export to BIDS derivatives"

    ```python
    # Export to BIDS derivatives
    # export_bids expects dict[str, ParameterMap] as first argument
    osipy.export_bids(
        parameter_maps={
            "D": result.d_map,
            "D_star": result.d_star_map,
            "f": result.f_map,
            "S0": result.s0_map,
        },
        output_dir="derivatives/osipy",
        subject_id="01",
        session_id="01",
        metadata={
            "b_values": b_values.tolist(),
            "fitting_method": "segmented",
        },
    )

    print("Results exported to derivatives/osipy/")
    ```

## Complete Example

!!! example "Complete IVIM analysis workflow"

    ```python
    import numpy as np
    import osipy
    from osipy.ivim import FittingMethod

    # 1. Load data (returns PerfusionDataset)
    dwi_dataset = osipy.load_nifti("dwi_multi_b.nii.gz")
    b_values = np.array([0, 10, 20, 50, 100, 200, 400, 800])

    # 2. Create mask (use .data for raw array)
    b0_image = dwi_dataset.data[..., 0]
    mask = b0_image > np.percentile(b0_image[b0_image > 0], 5)

    # 3. Fit IVIM model
    result = osipy.fit_ivim(
        signal=dwi_dataset.data,
        b_values=b_values,
        mask=mask,
        method=FittingMethod.SEGMENTED,
    )

    # 4. Extract and report results (IVIMFitResult has ParameterMap attributes)
    D = result.d_map.values
    D_star = result.d_star_map.values
    f = result.f_map.values
    valid = result.quality_mask > 0

    print(f"D mean: {D[valid].mean()*1e3:.3f} × 10⁻³ mm²/s")
    print(f"D* mean: {D_star[valid].mean()*1e3:.1f} × 10⁻³ mm²/s")
    print(f"f mean: {f[valid].mean():.3f}")

    # 5. Export (expects dict[str, ParameterMap])
    osipy.export_bids(
        {"D": result.d_map, "D_star": result.d_star_map, "f": result.f_map},
        "derivatives/osipy", "01", "01",
    )
    ```

## Next Steps

- [Understanding IVIM](../explanation/ivim-theory.md) for deeper theory
- [How to Handle Fitting Failures](../how-to/handle-fitting-failures.md) for troubleshooting
- [DCE-MRI Tutorial](dce-analysis.md) for contrast-based perfusion

## Troubleshooting

### High D* Values (>100 × 10⁻³)

- May indicate fitting instability
- Check if low b-values (0-100) show clear curvature
- Consider using tighter D* bounds

### Negative or Zero f Values

- Check signal quality at low b-values
- Verify b=0 image has adequate SNR
- May indicate absence of perfusion effect

### Poor R² (<0.9)

- Expected for noisy data or low perfusion
- Consider averaging multiple acquisitions
- Verify b-values are correctly specified

### D* ≈ D (Constraint Violations)

- Indicates weak or absent perfusion signal
- May occur in white matter (low f)
- These voxels are typically excluded from analysis
