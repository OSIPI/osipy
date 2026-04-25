# OSIPI Standards Compliance

osipy aligns with the Open Science Initiative for Perfusion Imaging (OSIPI) standards for parameter naming, units, and validation.

## What is OSIPI?

[OSIPI](https://osipi.ismrm.org/) is an ISMRM initiative that aims to:

- Develop consensus-based standards for perfusion imaging
- Create publicly available software tools
- Share code and data for validation
- Promote reproducibility in perfusion research

## OSIPI Task Forces

OSIPI is organized into task forces focusing on different aspects:

| Task Force | Focus | osipy Relevance |
|------------|-------|-----------------|
| TF 1.1 | ASL lexicon | Parameter naming |
| TF 1.2 | DSC/DCE lexicon | Parameter naming |
| TF 2.1 | DCE software inventory | Validated against |
| TF 2.3 | ASL software inventory | Validated against |
| TF 2.4 | IVIM/DWI software | Validated against |
| TF 4.1 | DCE DRO | Validation data |
| TF 6.1 | ASL DRO | Validation data |

## CAPLEX Naming Convention

### What is CAPLEX?

CAPLEX (Contrast Agent based Perfusion Lexicon) standardizes parameter names:

!!! example "CAPLEX parameter naming in osipy"

    ```python
    # osipy uses CAPLEX names
    result = osipy.fit_model("extended_tofts", ...)

    # Standard names:
    ktrans = result.parameter_maps["Ktrans"]  # Not "Ktr", "k_trans", "ktrans"
    ve = result.parameter_maps["ve"]          # Not "EES", "v_e", "Ve"
    vp = result.parameter_maps["vp"]          # Not "plasma_fraction", "v_p"
    ```

### Standard Parameters

| CAPLEX Name | Description | Units |
|-------------|-------------|-------|
| Ktrans | Volume transfer constant | min⁻¹ |
| ve | EES volume fraction | fraction [0, 1] (CAPLEX: mL/100mL; numerically 100×) |
| vp | Plasma volume fraction | fraction [0, 1] (CAPLEX: mL/100mL; numerically 100×) |
| kep | Rate constant (EES to plasma) | min⁻¹ |
| Fp | Plasma flow | mL/min/100mL |
| PS | Permeability-surface area | mL/min/100mL |
| CBV | Cerebral blood volume | mL/100g |
| CBF | Cerebral blood flow | mL/100g/min |
| MTT | Mean transit time | s |

### osipy Parameter Classes

!!! example "Create a ParameterMap with OSIPI-compliant metadata"

    ```python
    from osipy.common.parameter_map import ParameterMap

    # ParameterMap includes OSIPI-compliant metadata
    ktrans_map = ParameterMap(
        name="Ktrans",          # CAPLEX name
        symbol="Ktrans",        # ASCII symbol
        units="1/min",          # Standard units
        values=ktrans_array,
        affine=np.eye(4),       # NIfTI affine
        quality_mask=quality_mask,
    )
    ```

## Digital Reference Objects (DROs)

### What are DROs?

DROs are synthetic datasets with known ground truth parameters:

- Generated from mathematical models
- Include realistic noise and artifacts
- Enable quantitative validation

### OSIPI DCE DRO

osipy is validated against the [OSIPI DCE-MRI DRO](https://osf.io/u7a6f/):

!!! example "Validate osipy against OSIPI DCE DRO"

    ```python
    # Validation workflow
    from osipy.common.validation import load_dro, validate_against_dro

    # Load DRO ground truth parameters
    dro_data = load_dro("path/to/dro/")
    # dro_data.parameters contains ground truth: {"Ktrans": ..., "ve": ..., "vp": ...}

    # Fit with osipy (concentration, aif, time come from your imaging data)
    result = osipy.fit_model("extended_tofts", concentration, aif, time)

    # Compare computed maps to DRO ground truth
    validation = validate_against_dro(
        computed=result.parameter_maps,
        reference=dro_data,
    )

    print(validation.summary())
    ```

### Validation Metrics

osipy reports standard validation metrics:

| Metric | Description |
|--------|-------------|
| Bias | Mean(estimated - true) |
| RMSE | Root mean squared error |
| CCC | Concordance correlation coefficient |
| %CV | Coefficient of variation |

## Unit Conventions

### Time Units

OSIPI specifies time units for each context:

| Context | Unit | Example |
|---------|------|---------|
| Time arrays (all modalities) | seconds | `time = np.linspace(0, 300, 60)` |
| DCE models (internal) | minutes | Ktrans in min⁻¹ |
| ASL (PLD, τ, T1) | milliseconds | `pld=1800.0`, `label_duration=1800.0` |
| DSC (TE, TR in params) | milliseconds | `te=30.0`, `tr=1500.0` |

osipy handles conversions automatically:

!!! example "Automatic time unit conversion"

    ```python
    # User provides seconds
    result = osipy.fit_model("extended_tofts", conc, aif, time_in_seconds)

    # Ktrans is returned in min⁻¹ (OSIPI standard)
    print(f"Ktrans: {result.parameter_maps['Ktrans'].values.mean():.4f} min⁻¹")
    ```

### Concentration Units

| Technique | Unit |
|-----------|------|
| DCE (tissue) | mM (millimolar) |
| DSC (ΔR2*) | s⁻¹ |
| ASL (ΔM) | arbitrary units |

## Quality Control Standards

### R² Threshold

osipy uses R² = 0.5 as the default quality threshold:

!!! example "R-squared quality threshold"

    ```python
    # Voxels with R² < 0.5 are flagged in quality mask
    quality_mask = r_squared >= 0.5
    ```

This follows OSIPI recommendations for excluding unreliable fits.

### Quality Mask Convention

!!! example "Filter parameters using quality mask"

    ```python
    # quality_mask values:
    # > 0: Valid fit, reliable parameter
    # = 0: Failed fit, parameter should not be used

    # Always filter with quality mask
    valid_ktrans = ktrans[quality_mask > 0]
    ```

## BIDS Compliance

### BIDS Derivatives

osipy outputs follow [BIDS derivatives](https://bids-specification.readthedocs.io/en/stable/derivatives/introduction.html):

!!! example "BIDS derivatives directory structure"

    ```text
    derivatives/osipy/
    ├── dataset_description.json
    └── sub-01/
        └── ses-01/
            └── perf/
                ├── sub-01_ses-01_desc-Ktrans_parameter.nii.gz
                ├── sub-01_ses-01_desc-ve_parameter.nii.gz
                ├── sub-01_ses-01_desc-qualitymask_mask.nii.gz
                └── sub-01_ses-01_desc-dce_provenance.json
    ```

### Provenance JSON

!!! example "Provenance JSON metadata format"

    ```json
    {
      "software": "osipy",
      "version": "0.1.0",
      "model": "extended_tofts",
      "aif_type": "parker",
      "parameters": {
        "r2_threshold": 0.5,
        "relaxivity": 4.5
      },
      "timestamp": "2024-01-15T10:30:00Z"
    }
    ```

## Interoperability

### Compatible with OSIPI Software

Results from osipy can be compared with:

- ROCKETSHIP (MATLAB)
- TOPPCAT (Python)
- NordicICE (commercial)
- Olea Sphere (commercial)

### Data Exchange

Standard file formats ensure interoperability:

| Format | Use |
|--------|-----|
| NIfTI | Parameter maps |
| JSON | Metadata |
| BIDS | Dataset organization |

## OSIPI CodeCollection Compliance

osipy is validated against the [OSIPI DCE-DSC-MRI CodeCollection](https://github.com/OSIPI/DCE-DSC-MRI_CodeCollection), the community benchmark for DCE/DSC-MRI implementations. Reference test data and tolerances are committed in `tests/fixtures/osipi_codecollection/` for automated cross-implementation testing.

### OSIPI Tolerances

Validation uses OSIPI-agreed tolerances per parameter:

| Parameter | Absolute Tolerance | Relative Tolerance |
|-----------|-------------------|-------------------|
| Ktrans | 0.005 | 0.1 |
| ve | 0.05 | 0.0 |
| vp | 0.025 | 0.0 |
| Fp | 5.0 | 0.1 |
| PS | 0.005 | 0.1 |
| delay | 1.0 s | 0.0 |
| CBF (DSC) | 15.0 | 0.1 |

These tolerances are defined in `osipy.common.validation.comparison.DEFAULT_TOLERANCES`.

## Validation Reports

`validate_against_dro()` returns a `ValidationReport` dataclass with comparison metrics:

!!! example "Inspect validation report results"

    ```python
    from osipy.common.validation import validate_against_dro

    # Compare computed results against DRO ground truth
    report = validate_against_dro(
        computed=result.parameter_maps,
        reference=dro_data,
    )

    # Inspect results
    print(f"Overall pass: {report.overall_pass}")
    for param, rate in report.pass_rate.items():
        print(f"  {param} pass rate: {rate:.1%}")
    ```

### Export as JSON

`ValidationReport` supports structured export for CI/CD integration and record-keeping:

!!! example "Export validation report to JSON"

    ```python
    # As a dictionary
    report_dict = report.to_dict()

    # As a JSON string
    json_str = report.to_json()

    # Write directly to file
    report.to_json("validation_report.json")
    ```

## OSIPI Resources

### Useful Links

- [OSIPI Website](https://osipi.ismrm.org/)
- [CAPLEX Lexicon](https://osipi.github.io/OSIPI_CAPLEX/)
- [DCE-MRI DRO (OSF)](https://osf.io/u7a6f/)
- ASL DRO (OSF) — link pending verification
- [OSIPI GitHub](https://github.com/OSIPI)

### Task Force Reports

- [TF 1.2: DCE/DSC Lexicon (Dickie et al.)](https://doi.org/10.1002/mrm.29840)
- [TF 2.1: DCE Software Inventory (van Houdt et al.)](https://doi.org/10.1002/mrm.29826)
- [TF 4.1: DCE DRO Description (Shalom et al.)](https://doi.org/10.1002/mrm.29909)

## Contributing to OSIPI

osipy contributes to the OSIPI ecosystem through:

1. **Code contributions**: Sharing implementations
2. **Validation data**: Testing against DROs
3. **Documentation**: Standards compliance docs
4. **Benchmarks**: Performance comparisons

## See Also

- [Architecture Overview](architecture.md)
- [DCE-MRI Tutorial](../tutorials/dce-analysis.md)
- [How to Export BIDS Results](../how-to/export-bids.md)
