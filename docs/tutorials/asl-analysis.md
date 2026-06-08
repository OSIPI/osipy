# ASL CBF Quantification Tutorial

Quantify Cerebral Blood Flow (CBF) from Arterial Spin Labeling (ASL) MRI data: label/control differencing, M0 calibration, single- and multi-PLD quantification.

## Prerequisites

- Completed [Getting Started](getting-started.md) tutorial
- ASL data with known acquisition parameters
- M0 calibration scan (recommended)

**Using the CLI?** Generate a config with `osipy --dump-defaults asl > config.yaml`, edit it, then run `osipy config.yaml data.nii.gz`. The M0 calibration, label/control difference, and quantification mode are each selectable config blocks — set `quantification.mode: multi_pld` to estimate both CBF and arterial transit time (ATT) from a multi-PLD acquisition. See [How to Run Pipeline from YAML](../how-to/run-pipeline-cli.md). The tutorial below covers the Python API for step-by-step control.

## Background

Arterial Spin Labeling is a non-contrast MRI technique that magnetically labels arterial blood water as an endogenous tracer. The CBF is calculated from the difference between labeled and control images.

Key parameters:

| Parameter | Symbol | Units | Description |
|-----------|--------|-------|-------------|
| Cerebral Blood Flow | CBF | ml/100g/min | Tissue perfusion rate |
| Arterial Transit Time | ATT | ms | Time for blood to reach tissue |
| Labeling duration | τ | seconds | Duration of labeling pulse |
| Post-labeling delay | PLD | seconds | Wait time after labeling |

For theory details, see [Understanding ASL Physics](../explanation/asl-physics.md).

## Step 1: Load ASL Data

!!! example "Load ASL and M0 calibration data"

    ```python
    import numpy as np
    import osipy

    # Load 4D ASL data (alternating label/control)
    asl_dataset = osipy.load_nifti("asl_data.nii.gz")

    # Load M0 calibration image
    m0_dataset = osipy.load_nifti("m0_calibration.nii.gz")

    print(f"ASL data shape: {asl_dataset.shape}")  # (x, y, z, volumes)
    print(f"M0 data shape: {m0_dataset.shape}")    # (x, y, z) or (x, y, z, 1)

    # Determine number of label/control pairs
    # Access .data for the raw numpy array
    n_volumes = asl_dataset.shape[-1]
    n_pairs = n_volumes // 2
    print(f"Label/control pairs: {n_pairs}")
    ```

!!! info "Data Ordering"

    ASL data is typically organized as:

    - **Control-Label**: Volumes alternate starting with control (C, L, C, L, ...)
    - **Label-Control**: Volumes alternate starting with label (L, C, L, C, ...)

    Check your scanner's convention in the DICOM headers or BIDS sidecar.

## Step 2: Compute Difference Images

!!! example "Compute label-control difference images"

    ```python
    # Separate label and control images
    # Assuming control-label ordering (even=control, odd=label)
    control_idx = np.arange(0, n_volumes, 2)
    label_idx = np.arange(1, n_volumes, 2)

    control_images = asl_dataset.data[..., control_idx]
    label_images = asl_dataset.data[..., label_idx]

    # Compute mean difference
    delta_m = np.mean(control_images - label_images, axis=-1)

    # Also compute mean control for M0 estimation if needed
    mean_control = np.mean(control_images, axis=-1)

    print(f"ΔM shape: {delta_m.shape}")
    print(f"ΔM range: {delta_m.min():.1f} to {delta_m.max():.1f}")
    ```

!!! warning "Sign Convention"

    The difference should be **Control - Label** (positive ΔM indicates perfusion).
    If your ΔM values are predominantly negative, swap the order.

## Step 3: Configure Labeling Scheme

!!! example "Configure pCASL acquisition parameters"

    ```python
    from osipy.asl import ASLQuantificationParams, LabelingScheme

    # Configure pCASL (pseudo-continuous ASL) quantification parameters
    quant_params = ASLQuantificationParams(
        labeling_scheme=LabelingScheme.PCASL,
        label_duration=1800.0,        # ms
        pld=1800.0,                   # ms
        labeling_efficiency=0.85,
        t1_blood=1650.0,              # ms at 3T
        partition_coefficient=0.9,    # λ (ml/g)
    )

    print(f"Labeling type: {quant_params.labeling_scheme.value}")
    print(f"τ = {quant_params.label_duration} ms")
    print(f"PLD = {quant_params.pld} ms")
    ```

### Labeling Type Parameters

The three main labeling types are pulsed ASL (PASL), continuous ASL (CASL), and pseudo-continuous ASL (pCASL):

| Parameter | PASL | CASL | pCASL |
|-----------|------|------|-------|
| labeling_duration | TI₁ | τ | τ |
| labeling_efficiency | 0.95-0.98 | 0.68-0.73 | 0.80-0.90 |
| Typical PLD | 0.8-1.2 s | 1.5-2.0 s | 1.5-2.0 s |

## Step 4: Perform M0 Calibration

!!! example "Apply M0 calibration"

    The M0 image provides the equilibrium magnetization needed for absolute CBF quantification:

    ```python
    from osipy.asl import M0CalibrationParams

    # Apply M0 calibration
    # M0 should have long TR (>5s) for full relaxation
    m0_params = M0CalibrationParams(
        method="voxelwise",
        t1_tissue=1330.0,       # T1 of gray matter (ms)
        tr_m0=6000.0,           # TR of M0 scan (ms)
    )

    # apply_m0_calibration takes (asl_data, m0_image, params) and returns a tuple
    calibrated_data, m0_corrected = osipy.apply_m0_calibration(
        asl_data=delta_m,
        m0_image=m0_dataset.data,
        params=m0_params,
    )

    # Create brain mask from corrected M0
    m0_threshold = np.percentile(m0_corrected[m0_corrected > 0], 10)
    brain_mask = m0_corrected > m0_threshold

    print(f"M0 range: {m0_corrected[brain_mask].min():.0f} - {m0_corrected[brain_mask].max():.0f}")
    print(f"Brain voxels: {brain_mask.sum()}")
    ```

!!! tip "No M0 Scan?"

    If you don't have a dedicated M0 scan, you can use the mean control image
    as a proxy, but this reduces quantification accuracy:

    ```python
    m0_proxy = mean_control
    ```

## Step 5: Quantify CBF

!!! example "Quantify CBF using the ASL kinetic model"

    ```python
    # Quantify CBF
    # quantify_cbf returns an ASLQuantificationResult object
    cbf_result = osipy.quantify_cbf(
        delta_m=delta_m,
        m0=m0_corrected,
        params=quant_params,
        mask=brain_mask,
    )

    # Access the CBF map (a ParameterMap with .values attribute)
    cbf_map = cbf_result.cbf_map.values

    # Check CBF values
    valid_cbf = cbf_map[brain_mask]
    print(f"\nCBF Statistics (ml/100g/min):")
    print(f"  Mean: {valid_cbf.mean():.1f}")
    print(f"  Std:  {valid_cbf.std():.1f}")
    print(f"  Range: {valid_cbf.min():.1f} - {valid_cbf.max():.1f}")
    ```

### Expected CBF Values

| Tissue | CBF (ml/100g/min) |
|--------|-------------------|
| Gray matter | 50-80 |
| White matter | 20-30 |
| Whole brain average | 40-60 |
| Tumor (enhancing) | 50-150+ |

!!! warning "Physiological Plausibility"

    CBF values outside 0-200 ml/100g/min typically indicate:

    - Incorrect labeling parameters
    - M0 calibration issues
    - Motion artifacts
    - Partial volume effects

## Step 6: Multi-PLD Analysis (Optional)

!!! example "Estimate CBF and ATT from multi-PLD data"

    ```python
    # For multi-PLD data
    plds = np.array([500, 1000, 1500, 2000, 2500, 3000])  # milliseconds

    # Load multi-PLD data (shape: x, y, z, n_plds)
    # Each PLD should have averaged label/control pairs
    multi_pld_data = osipy.load_nifti("asl_multi_pld.nii.gz")

    # Quantify CBF and ATT
    from osipy.asl.quantification import MultiPLDParams, quantify_multi_pld

    multi_pld_params = MultiPLDParams(
        plds=plds,
        labeling_scheme=LabelingScheme.PCASL,
        labeling_efficiency=0.85,
        label_duration=1800.0,  # ms
    )

    result = quantify_multi_pld(
        delta_m=multi_pld_data.data,
        m0=m0_corrected,
        params=multi_pld_params,
        mask=brain_mask,
    )

    # result is a MultiPLDResult object with ParameterMap attributes
    cbf_map = result.cbf_map.values
    att_map = result.att_map.values
    r_squared = result.r_squared

    print(f"\nMulti-PLD Results:")
    print(f"  CBF mean: {cbf_map[brain_mask].mean():.1f} ml/100g/min")
    print(f"  ATT mean: {att_map[brain_mask].mean():.2f} ms")
    print(f"  R² mean:  {r_squared[brain_mask].mean():.3f}")
    ```

### Multi-PLD Benefits

- More accurate CBF quantification
- Arterial transit time maps
- Reduced sensitivity to timing assumptions
- Better partial volume correction

## Step 7: Visualize Results

!!! example "Visualize CBF maps and quality metrics"

    ```python
    import matplotlib.pyplot as plt

    # Select axial slice
    slice_idx = delta_m.shape[2] // 2

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Row 1: Input data
    im0 = axes[0, 0].imshow(mean_control[:, :, slice_idx], cmap='gray')
    axes[0, 0].set_title('Mean Control')
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(delta_m[:, :, slice_idx], cmap='bwr', vmin=-50, vmax=50)
    axes[0, 1].set_title('ΔM (a.u.)')
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].imshow(m0_corrected[:, :, slice_idx], cmap='gray')
    axes[0, 2].set_title('M0')
    plt.colorbar(im2, ax=axes[0, 2])

    # Row 2: Results
    im3 = axes[1, 0].imshow(cbf_map[:, :, slice_idx], cmap='hot', vmin=0, vmax=100)
    axes[1, 0].set_title('CBF (ml/100g/min)')
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(brain_mask[:, :, slice_idx], cmap='binary')
    axes[1, 1].set_title('Brain Mask')
    plt.colorbar(im4, ax=axes[1, 1])

    # Histogram
    axes[1, 2].hist(valid_cbf, bins=50, color='steelblue', edgecolor='white')
    axes[1, 2].axvline(valid_cbf.mean(), color='red', linestyle='--', label=f'Mean: {valid_cbf.mean():.1f}')
    axes[1, 2].set_xlabel('CBF (ml/100g/min)')
    axes[1, 2].set_ylabel('Voxel Count')
    axes[1, 2].set_title('CBF Distribution')
    axes[1, 2].legend()

    for ax in axes.flat[:5]:
        ax.axis('off')
    axes[1, 2].axis('on')

    plt.tight_layout()
    plt.savefig('asl_cbf_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    ```

### Multi-slice Montage

!!! example "Create a CBF montage across slices"

    ```python
    # Create CBF montage
    n_slices = cbf_map.shape[2]
    n_cols = 6
    n_rows = (n_slices + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))
    axes = axes.flatten()

    for i in range(n_slices):
        axes[i].imshow(cbf_map[:, :, i], cmap='hot', vmin=0, vmax=100)
        axes[i].axis('off')
        axes[i].set_title(f'z={i}', fontsize=8)

    # Hide unused axes
    for i in range(n_slices, len(axes)):
        axes[i].axis('off')

    fig.suptitle('CBF Maps (ml/100g/min)', y=1.02)
    plt.tight_layout()
    plt.savefig('asl_cbf_montage.png', dpi=200)
    plt.show()
    ```

## Step 8: Export Results

Save results in BIDS format:

!!! warning "Experimental Feature"

    BIDS derivative export is partially implemented and may not produce fully compliant output for all use cases.

!!! example "Export to BIDS derivatives"

    ```python
    # Export to BIDS derivatives
    # export_bids signature: (parameter_maps, output_dir, subject_id, session_id, metadata)
    osipy.export_bids(
        parameter_maps={"CBF": cbf_result.cbf_map},
        output_dir="derivatives/osipy",
        subject_id="01",
        session_id="01",
        metadata={
            "labeling_type": quant_params.labeling_scheme.value,
            "label_duration": quant_params.label_duration,
            "pld": quant_params.pld,
            "labeling_efficiency": quant_params.labeling_efficiency,
        },
    )

    print("Results exported to derivatives/osipy/")
    ```

## Complete Example

!!! example "Complete ASL CBF quantification workflow"

    ```python
    import numpy as np
    import osipy
    from osipy.asl import ASLQuantificationParams, LabelingScheme, M0CalibrationParams

    # 1. Load data (returns PerfusionDataset)
    asl_dataset = osipy.load_nifti("asl_data.nii.gz")
    m0_dataset = osipy.load_nifti("m0_calibration.nii.gz")

    # 2. Compute difference images (use .data for raw array)
    n_volumes = asl_dataset.shape[-1]
    control_images = asl_dataset.data[..., 0::2]
    label_images = asl_dataset.data[..., 1::2]
    delta_m = np.mean(control_images - label_images, axis=-1)

    # 3. Configure quantification parameters
    quant_params = ASLQuantificationParams(
        labeling_scheme=LabelingScheme.PCASL,
        label_duration=1800.0,      # ms
        pld=1800.0,                 # ms
        labeling_efficiency=0.85,
        t1_blood=1650.0,            # ms at 3T
        partition_coefficient=0.9,
    )

    # 4. M0 calibration
    calibrated_data, m0_corrected = osipy.apply_m0_calibration(delta_m, m0_dataset.data)
    brain_mask = m0_corrected > np.percentile(m0_corrected[m0_corrected > 0], 10)

    # 5. Quantify CBF
    cbf_result = osipy.quantify_cbf(delta_m, m0_corrected, quant_params, mask=brain_mask)
    cbf_map = cbf_result.cbf_map.values

    # 6. Export (export_bids expects dict[str, ParameterMap])
    osipy.export_bids({"CBF": cbf_result.cbf_map}, "derivatives/osipy", "01", "01")

    print(f"CBF mean: {cbf_map[brain_mask].mean():.1f} ml/100g/min")
    ```

## Next Steps

- [How to Choose Population AIF](../how-to/choose-population-aif.md) for DCE comparison
- [Understanding ASL Physics](../explanation/asl-physics.md) for deeper theory
- [IVIM Tutorial](ivim-analysis.md) for another non-contrast technique

## Troubleshooting

### Negative or Zero CBF

- Check label/control ordering (try swapping)
- Verify M0 values are positive
- Check brain mask coverage

### CBF Too High (>200 ml/100g/min)

- Verify labeling efficiency (may be lower than expected)
- Check M0 TR (should be >5s for full relaxation)
- Look for motion artifacts

### Patchy CBF Maps

- Increase averaging (more label/control pairs)
- Check for susceptibility artifacts near sinuses
- Consider partial volume correction
