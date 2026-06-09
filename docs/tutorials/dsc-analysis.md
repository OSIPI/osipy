# DSC-MRI Perfusion Analysis Tutorial

Generate CBV, CBF, and MTT maps from Dynamic Susceptibility Contrast (DSC) MRI using SVD deconvolution, with optional leakage correction.

## Prerequisites

- Completed [Getting Started](getting-started.md) tutorial
- DSC-MRI data with known acquisition parameters
- Understanding of basic perfusion MRI concepts

**Using the CLI?** Generate a config with `osipy --dump-defaults dsc > config.yaml`, edit it, then run `osipy config.yaml data.nii.gz`. The deconvolution method is a selectable config block (`deconvolution.method`: oSVD, sSVD, or cSVD) that surfaces its own parameters — `oSVD` takes `oscillation_index`/`default_threshold`, while `sSVD`/`cSVD` take a single `threshold`. See [How to Run Pipeline from YAML](../how-to/run-pipeline-cli.md). The tutorial below covers the Python API for step-by-step control.

## Background

DSC-MRI tracks the first pass of a gadolinium bolus through brain tissue. The T2* signal drop is related to contrast agent concentration, and deconvolution with the arterial input function yields the tissue residue function, from which perfusion parameters are calculated.

Key parameters:

| Parameter | Symbol | Units | Description |
|-----------|--------|-------|-------------|
| Cerebral Blood Volume | CBV | ml/100g | Blood volume in tissue |
| Cerebral Blood Flow | CBF | ml/100g/min | Blood flow rate |
| Mean Transit Time | MTT | seconds | Average transit time |
| Time to Peak | TTP | seconds | Time to signal minimum |
| Tmax | Tmax | seconds | Time to residue function maximum |

For theory, see [Understanding DSC Deconvolution](../explanation/dsc-deconvolution.md).

## Step 1: Load DSC-MRI Data

!!! example "Load DSC-MRI data"

    ```python
    import numpy as np
    import osipy

    # Load 4D DSC-MRI data (returns PerfusionDataset)
    dsc_dataset = osipy.load_nifti("dsc_4d.nii.gz")

    # Define timing parameters
    n_timepoints = dsc_dataset.shape[-1]
    tr = 1.5  # Repetition time in seconds
    te = 30.0  # Echo time in milliseconds

    # Create time array
    time = np.arange(n_timepoints) * tr

    print(f"DSC data shape: {dsc_dataset.shape}")
    print(f"Timepoints: {n_timepoints}")
    print(f"Duration: {time[-1]:.1f} seconds")
    ```

!!! info "Typical DSC Acquisition"

    - **TR** (repetition time): 1-2 seconds
    - **TE** (echo time): 25-40 ms (for T2* weighting)
    - **Duration**: 60-120 seconds
    - **Timepoints**: 40-80 frames

## Step 2: Create Brain Mask

!!! example "Create a brain mask"

    ```python
    # Use mean signal for masking (access .data for raw array operations)
    mean_signal = dsc_dataset.data.mean(axis=-1)

    # Threshold-based mask
    threshold = np.percentile(mean_signal[mean_signal > 0], 10)
    brain_mask = mean_signal > threshold

    # Optional: exclude ventricles (high signal variance during bolus)
    signal_std = dsc_dataset.data.std(axis=-1)
    csf_mask = signal_std > np.percentile(signal_std[brain_mask], 95)
    brain_mask = brain_mask & ~csf_mask

    print(f"Brain voxels: {brain_mask.sum()}")
    ```

## Step 3: Identify Baseline and Bolus

!!! example "Identify baseline and bolus arrival"

    ```python
    # Calculate global signal time course
    global_signal = dsc_dataset.data[brain_mask].mean(axis=0)

    # Find bolus arrival (signal drop)
    baseline_signal = global_signal[:10].mean()
    signal_drop = (baseline_signal - global_signal) / baseline_signal

    # Bolus arrival when signal drops > 5%
    bolus_arrival = np.where(signal_drop > 0.05)[0][0]

    # Signal minimum (peak contrast)
    signal_min_idx = np.argmin(global_signal)

    print(f"Baseline frames: 0-{bolus_arrival-1}")
    print(f"Bolus arrival: frame {bolus_arrival} ({time[bolus_arrival]:.1f} s)")
    print(f"Peak contrast: frame {signal_min_idx} ({time[signal_min_idx]:.1f} s)")

    # Visualize
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, global_signal, 'b-', linewidth=2)
    ax.axvline(time[bolus_arrival], color='r', linestyle='--', label='Bolus arrival')
    ax.axvspan(0, time[bolus_arrival], alpha=0.2, color='green', label='Baseline')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Signal (a.u.)')
    ax.set_title('Global DSC-MRI Signal Time Course')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('dsc_signal_timecourse.png', dpi=150)
    plt.show()
    ```

## Step 4: Convert Signal to ΔR2*

!!! example "Convert signal to delta R2*"

    ```python
    # Convert signal to ΔR2* (proportional to contrast concentration)
    delta_r2 = osipy.signal_to_delta_r2(
        signal=dsc_dataset.data,
        te=te,
        baseline_end=bolus_arrival,  # Last baseline frame
    )

    print(f"ΔR2* shape: {delta_r2.shape}")
    print(f"ΔR2* range: {delta_r2[brain_mask].min():.2f} to {delta_r2[brain_mask].max():.2f} s⁻¹")
    ```

!!! note "ΔR2* Relationship"

    ΔR2* is proportional to contrast agent concentration:

    $$
    \Delta R_2^* = -\frac{1}{TE} \ln\left(\frac{S(t)}{S_0}\right) \propto C(t)
    $$

    where S₀ is the baseline signal.

## Step 5: Select Arterial Input Function

The AIF is critical for accurate perfusion quantification:

### Option A: Automatic AIF Detection

!!! example "Detect AIF from high-signal voxels"

    ```python
    # Automatic AIF detection requires a PerfusionDataset object
    # For simplicity, manually extract AIF from high-signal voxels
    # (osipy.detect_aif expects a PerfusionDataset, not raw arrays)

    # Select voxels with largest signal drop (arterial candidates)
    peak_drop = delta_r2.max(axis=-1)
    aif_threshold = np.percentile(peak_drop[brain_mask], 95)
    aif_mask = brain_mask & (peak_drop > aif_threshold)

    # Average across selected voxels to get AIF curve
    aif_curve = delta_r2[aif_mask].mean(axis=0)

    print(f"AIF voxels selected: {aif_mask.sum()}")
    print(f"AIF peak: {aif_curve.max():.2f} s⁻¹ at {time[aif_curve.argmax()]:.1f} s")
    ```

### Option B: Manual ROI Selection

!!! example "Use a manually defined arterial ROI"

    ```python
    # If you have manually defined an arterial ROI
    # aif_indices = [...]  # Your selected voxel indices
    # aif_curve = delta_r2[aif_indices].mean(axis=0)

    # Create ArterialInputFunction object
    # aif = osipy.ArterialInputFunction(time=time, concentration=aif_curve)
    ```

### Visualize AIF

!!! example "Visualize the AIF"

    ```python
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # AIF time course
    axes[0].plot(time, aif_curve, 'r-', linewidth=2)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('ΔR2* (s⁻¹)')
    axes[0].set_title('Arterial Input Function')
    axes[0].grid(True, alpha=0.3)

    # AIF location
    slice_idx = dsc_dataset.shape[2] // 2
    axes[1].imshow(mean_signal[:, :, slice_idx], cmap='gray')
    axes[1].imshow(aif_mask[:, :, slice_idx], cmap='Reds', alpha=0.5)
    axes[1].set_title('AIF Voxel Location')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('dsc_aif.png', dpi=150)
    plt.show()
    ```

## Step 6: SVD Deconvolution

!!! example "Perform oSVD deconvolution"

    ```python
    # Perform oSVD deconvolution (recommended)
    deconvolver = osipy.get_deconvolver("oSVD")
    deconv_result = deconvolver.deconvolve(
        concentration=delta_r2,
        aif=aif_curve,
        time=time,
        mask=brain_mask,
    )

    # Extract residue function and perfusion parameters
    # deconv_result is a DeconvolutionResult object
    residue = deconv_result.residue_function
    cbf_raw = deconv_result.cbf
    mtt = deconv_result.mtt
    delay = deconv_result.delay

    print(f"Deconvolution complete")
    print(f"CBF range: {cbf_raw[brain_mask].min():.1f} to {cbf_raw[brain_mask].max():.1f}")
    ```

### Deconvolution Methods

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| sSVD | Standard SVD | Simple | Noise sensitive |
| cSVD | Circular SVD | Delay-insensitive | May underestimate |
| oSVD | Oscillation-index SVD | Robust, accurate | More complex |

## Step 7: Calculate CBV

!!! example "Calculate CBV from the concentration integral"

    ```python
    # CBV from area under the curve
    # Integration using trapezoidal rule
    from osipy.common.backend import get_array_module
    xp = get_array_module(delta_r2)

    # Time step
    dt = time[1] - time[0]

    # Integrate tissue and AIF curves
    # CBV = k * integral(C_tissue) / integral(C_aif)
    tissue_integral = xp.trapezoid(delta_r2, dx=dt, axis=-1)
    aif_integral = xp.trapezoid(aif_curve, dx=dt)

    # CBV in arbitrary units (need density correction for absolute values)
    cbv = tissue_integral / aif_integral

    # Apply brain mask
    cbv = cbv * brain_mask

    print(f"CBV range: {cbv[brain_mask].min():.3f} to {cbv[brain_mask].max():.3f}")
    ```

## Step 8: Compute Perfusion Maps

!!! example "Compute perfusion maps"

    ```python
    # Use the unified perfusion maps function
    perfusion_maps = osipy.compute_perfusion_maps(
        delta_r2=delta_r2,
        aif=aif_curve,
        time=time,
        mask=brain_mask,
    )

    # Extract all maps (perfusion_maps is a DSCPerfusionMaps object)
    # Each attribute is a ParameterMap — use .values to get the numpy array
    cbv = perfusion_maps.cbv.values       # ml/100g
    cbf = perfusion_maps.cbf.values       # ml/100g/min
    mtt = perfusion_maps.mtt.values       # seconds

    # ttp and tmax are optional (may be None)
    ttp = perfusion_maps.ttp.values if perfusion_maps.ttp is not None else None
    tmax = perfusion_maps.tmax.values if perfusion_maps.tmax is not None else None

    # Statistics
    print(f"\nPerfusion Map Statistics (brain voxels):")
    print(f"  CBV: {cbv[brain_mask].mean():.2f} ± {cbv[brain_mask].std():.2f} ml/100g")
    print(f"  CBF: {cbf[brain_mask].mean():.1f} ± {cbf[brain_mask].std():.1f} ml/100g/min")
    print(f"  MTT: {mtt[brain_mask].mean():.2f} ± {mtt[brain_mask].std():.2f} s")
    ```

## Step 9: Leakage Correction (Optional)

For tumors or lesions with BBB breakdown, apply leakage correction:

!!! example "Apply BSW leakage correction"

    ```python
    # Check if leakage correction is needed
    # High T1 enhancement during contrast passage indicates leakage

    # Apply Boxerman-Schmainda-Weisskoff (BSW) correction
    # correct_leakage returns a LeakageCorrectionResult with corrected ΔR2*
    leakage_result = osipy.correct_leakage(
        delta_r2=delta_r2,
        aif=aif_curve,
        time=time,
        mask=brain_mask,
    )

    # Re-compute perfusion maps with corrected ΔR2*
    corrected_maps = osipy.compute_perfusion_maps(
        delta_r2=leakage_result.corrected_delta_r2,
        aif=aif_curve,
        time=time,
        mask=brain_mask,
    )

    cbv_corrected = corrected_maps.cbv.values
    cbf_corrected = corrected_maps.cbf.values

    print(f"\nLeakage-corrected CBV: {cbv_corrected[brain_mask].mean():.2f} ml/100g")
    ```

!!! warning "When to Use Leakage Correction"

    Apply leakage correction when:

    - Studying tumors (especially high-grade gliomas)
    - Signal increases during bolus passage (T1 effect)
    - CBV appears artificially reduced

## Step 10: Normalize to White Matter (rCBV)

!!! example "Normalize CBV to white matter (rCBV)"

    ```python
    # Define white matter ROI (or use automatic segmentation)
    # wm_mask = ...  # Your white matter mask

    # For demonstration, use high-CBV threshold as proxy
    wm_proxy = (cbv > np.percentile(cbv[brain_mask], 30)) & \
               (cbv < np.percentile(cbv[brain_mask], 50))

    # Calculate mean white matter CBV
    wm_cbv_mean = cbv[wm_proxy].mean()

    # Relative CBV
    rcbv = cbv / wm_cbv_mean

    print(f"White matter CBV (reference): {wm_cbv_mean:.2f}")
    print(f"rCBV range: {rcbv[brain_mask].min():.2f} to {rcbv[brain_mask].max():.2f}")
    ```

## Step 11: Visualize the Results

!!! example "Plot perfusion parameter maps"

    ```python
    # Select representative slice
    slice_idx = dsc_dataset.shape[2] // 2

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Row 1: Main perfusion parameters
    im0 = axes[0, 0].imshow(cbv[:, :, slice_idx], cmap='hot', vmin=0, vmax=10)
    axes[0, 0].set_title('CBV (ml/100g)')
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(cbf[:, :, slice_idx], cmap='jet', vmin=0, vmax=100)
    axes[0, 1].set_title('CBF (ml/100g/min)')
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].imshow(mtt[:, :, slice_idx], cmap='viridis', vmin=0, vmax=10)
    axes[0, 2].set_title('MTT (s)')
    plt.colorbar(im2, ax=axes[0, 2])

    # Row 2: Timing and anatomy
    if ttp is not None:
        im3 = axes[1, 0].imshow(ttp[:, :, slice_idx], cmap='plasma', vmin=0, vmax=30)
        axes[1, 0].set_title('TTP (s)')
        plt.colorbar(im3, ax=axes[1, 0])

    if tmax is not None:
        im4 = axes[1, 1].imshow(tmax[:, :, slice_idx], cmap='inferno', vmin=0, vmax=10)
        axes[1, 1].set_title('Tmax (s)')
        plt.colorbar(im4, ax=axes[1, 1])

    im5 = axes[1, 2].imshow(mean_signal[:, :, slice_idx], cmap='gray')
    axes[1, 2].set_title('Mean Signal (Anatomy)')
    plt.colorbar(im5, ax=axes[1, 2])

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('dsc_perfusion_maps.png', dpi=300, bbox_inches='tight')
    plt.show()
    ```

### Time Course Comparison

!!! example "Compare AIF and tissue time courses"

    ```python
    # Compare tissue and AIF time courses
    fig, ax = plt.subplots(figsize=(10, 5))

    # AIF
    ax.plot(time, aif_curve / aif_curve.max(), 'r-', linewidth=2, label='AIF (normalized)')

    # Sample tissue curves
    sample_indices = np.where(brain_mask)
    for i in range(3):
        idx = i * len(sample_indices[0]) // 4
        x, y, z = sample_indices[0][idx], sample_indices[1][idx], sample_indices[2][idx]
        tissue_curve = delta_r2[x, y, z, :]
        ax.plot(time, tissue_curve / tissue_curve.max(), alpha=0.7, label=f'Tissue {i+1}')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normalized ΔR2*')
    ax.set_title('AIF vs Tissue Concentration Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig('dsc_time_courses.png', dpi=150)
    plt.show()
    ```

## Step 12: Export Results

Save results in BIDS format:

!!! warning "Experimental Feature"

    BIDS derivative export is partially implemented and may not produce fully compliant output for all use cases.

!!! example "Export to BIDS derivatives"

    ```python
    # Export to BIDS derivatives
    # export_bids expects dict[str, ParameterMap] as first argument
    maps_to_export = {
        "CBV": perfusion_maps.cbv,
        "CBF": perfusion_maps.cbf,
        "MTT": perfusion_maps.mtt,
    }
    if perfusion_maps.ttp is not None:
        maps_to_export["TTP"] = perfusion_maps.ttp
    if perfusion_maps.tmax is not None:
        maps_to_export["Tmax"] = perfusion_maps.tmax

    osipy.export_bids(
        parameter_maps=maps_to_export,
        output_dir="derivatives/osipy",
        subject_id="01",
        session_id="01",
        metadata={
            "deconvolution_method": "oSVD",
            "te": te,
            "tr": tr,
        },
    )

    print("Results exported to derivatives/osipy/")
    ```

## Complete Example

!!! example "Complete DSC-MRI workflow"

    ```python
    import numpy as np
    import osipy

    # 1. Load data (returns PerfusionDataset)
    dsc_dataset = osipy.load_nifti("dsc_4d.nii.gz")
    tr, te = 1.5, 30.0  # TR in seconds, TE in milliseconds
    time = np.arange(dsc_dataset.shape[-1]) * tr

    # 2. Create mask
    mean_signal = dsc_dataset.data.mean(axis=-1)
    brain_mask = mean_signal > np.percentile(mean_signal[mean_signal > 0], 10)

    # 3. Identify baseline
    global_signal = dsc_dataset.data[brain_mask].mean(axis=0)
    baseline_end = np.where(
        (global_signal[:10].mean() - global_signal) / global_signal[:10].mean() > 0.05
    )[0][0]

    # 4. Convert to ΔR2*
    delta_r2 = osipy.signal_to_delta_r2(dsc_dataset.data, te, baseline_end=baseline_end)

    # 5. Extract AIF (manual selection from high-signal voxels)
    peak_drop = delta_r2.max(axis=-1)
    aif_mask = brain_mask & (peak_drop > np.percentile(peak_drop[brain_mask], 95))
    aif_curve = delta_r2[aif_mask].mean(axis=0)

    # 6. Compute perfusion maps
    perfusion_maps = osipy.compute_perfusion_maps(delta_r2, aif_curve, time, brain_mask)

    # 7. Export (expects dict[str, ParameterMap])
    osipy.export_bids(
        {"CBV": perfusion_maps.cbv, "CBF": perfusion_maps.cbf, "MTT": perfusion_maps.mtt},
        "derivatives/osipy", "01", "01",
    )

    print(f"CBV mean: {perfusion_maps.cbv.values[brain_mask].mean():.2f} ml/100g")
    print(f"CBF mean: {perfusion_maps.cbf.values[brain_mask].mean():.1f} ml/100g/min")
    ```

## Next Steps

- [Understanding DSC Deconvolution](../explanation/dsc-deconvolution.md) for theory
- [How to Use a Custom AIF](../how-to/use-custom-aif.md) for measured AIF
- [DCE-MRI Tutorial](dce-analysis.md) for pharmacokinetic analysis

## Troubleshooting

### Negative CBV/CBF Values

- Check AIF selection (should have sharp peak)
- Verify baseline period is pre-contrast
- Look for motion artifacts

### Noisy MTT Maps

- Increase SVD threshold (e.g., 0.15-0.2)
- Use oSVD instead of sSVD
- Apply spatial smoothing

### Leakage Effects

- Signal increases during bolus = T1 shortening (leakage)
- Apply BSW correction for accurate CBV
- Consider dual-echo acquisition

### Arterial Delay

- Use cSVD or oSVD (delay-insensitive)
- Check Tmax maps for regional delays
- Consider bolus timing correction
