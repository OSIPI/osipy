# How to Export Results in BIDS Format

Export analysis results in Brain Imaging Data Structure (BIDS) derivatives format.

!!! warning "Experimental Feature"

    BIDS derivative export is partially implemented and may not produce fully compliant output for all use cases.

## Basic Export

!!! example "Export parameter maps to BIDS derivatives"

    ```python
    import osipy

    # After fitting
    result = osipy.fit_model("extended_tofts", concentration, aif, time)

    # Export to BIDS
    # export_bids signature: (parameter_maps, output_dir, subject_id, session_id, metadata)
    osipy.export_bids(
        parameter_maps=result.parameter_maps,
        output_dir="derivatives/osipy",
        subject_id="01",
        session_id="01",
    )
    ```

## Output Structure

!!! example "BIDS-compliant output directory structure"

    ```text
    derivatives/osipy/
    ├── dataset_description.json
    └── sub-01/
        └── ses-01/
            └── perf/
                ├── sub-01_ses-01_desc-Ktrans_parameter.nii.gz
                ├── sub-01_ses-01_desc-ve_parameter.nii.gz
                ├── sub-01_ses-01_desc-vp_parameter.nii.gz
                ├── sub-01_ses-01_desc-R2_parameter.nii.gz
                ├── sub-01_ses-01_desc-qualitymask_mask.nii.gz
                └── sub-01_ses-01_desc-dce_provenance.json
    ```

## Include Spatial Information

!!! example "Preserve spatial alignment with source data"

    ```python
    # Load source data for affine
    source = osipy.load_nifti("source_data.nii.gz")

    # Export with metadata
    # Note: spatial alignment is preserved via each ParameterMap's affine attribute
    osipy.export_bids(
        parameter_maps=result.parameter_maps,
        output_dir="derivatives/osipy",
        subject_id="01",
        session_id="baseline",
    )
    ```

## Export Specific Parameters

!!! example "Choose which parameters to export"

    ```python
    # Export only specific maps
    osipy.export_bids(
        parameter_maps={
            "Ktrans": result.parameter_maps["Ktrans"],
            "ve": result.parameter_maps["ve"],
            # Exclude vp
        },
        output_dir="derivatives/osipy",
        subject_id="01",
        session_id="01",
    )
    ```

## Add Metadata

!!! example "Include analysis provenance metadata"

    ```python
    osipy.export_bids(
        parameter_maps=result.parameter_maps,
        output_dir="derivatives/osipy",
        subject_id="01",
        session_id="01",
        metadata={
            "model": "extended_tofts",
            "aif_type": "parker",
            "fitting_method": "lm",
            "r2_threshold": 0.5,
            "software": "osipy",
            "version": osipy.__version__,
        },
    )
    ```

## Export Different Modalities

### DCE-MRI

!!! example "Export DCE-MRI parameter maps"

    ```python
    osipy.export_bids(
        parameter_maps={
            "Ktrans": ktrans_map,
            "ve": ve_map,
            "vp": vp_map,
        },
        output_dir="derivatives/osipy",
        subject_id="01",
    )
    ```

### DSC-MRI

!!! example "Export DSC-MRI perfusion maps"

    ```python
    osipy.export_bids(
        parameter_maps={
            "CBV": perfusion_maps.cbv,
            "CBF": perfusion_maps.cbf,
            "MTT": perfusion_maps.mtt,
        },
        output_dir="derivatives/osipy",
        subject_id="01",
    )
    ```

### ASL

!!! example "Export ASL CBF map with labeling metadata"

    ```python
    osipy.export_bids(
        parameter_maps={"CBF": cbf_result.cbf_map},
        output_dir="derivatives/osipy",
        subject_id="01",
        metadata={
            "labeling_type": "pcasl",
            "post_labeling_delay": 1.8,
            "labeling_duration": 1.8,
        },
    )
    ```

### IVIM

!!! example "Export IVIM diffusion and perfusion maps"

    ```python
    osipy.export_bids(
        parameter_maps={
            "D": result.d_map,
            "D_star": result.d_star_map,
            "f": result.f_map,
        },
        output_dir="derivatives/osipy",
        subject_id="01",
        metadata={
            "b_values": b_values.tolist(),
            "fitting_method": "segmented",
        },
    )
    ```

## Batch Export

!!! example "Export results for multiple subjects"

    ```python
    subjects = ['01', '02', '03']

    for subj in subjects:
        # Load and process
        data = osipy.load_nifti(f"sub-{subj}/perf/dce.nii.gz")
        result = osipy.fit_model("extended_tofts", data.data, aif, time)

        # Export
        osipy.export_bids(
            parameter_maps=result.parameter_maps,
            output_dir="derivatives/osipy",
            subject_id=subj,
        )
    ```

## Create Dataset Description

!!! example "Generate the required dataset_description.json"

    ```python
    import json

    description = {
        "Name": "osipy DCE-MRI Analysis",
        "BIDSVersion": "1.8.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "osipy",
                "Version": osipy.__version__,
                "CodeURL": "https://github.com/OSIPI/osipy"
            }
        ],
        "SourceDatasets": [
            {
                "DOI": "doi:xxxxx",  # If applicable
                "URL": "path/to/source"
            }
        ]
    }

    with open("derivatives/osipy/dataset_description.json", "w") as f:
        json.dump(description, f, indent=2)
    ```

## See Also

- [How to Load Perfusion Data](load-perfusion-data.md)
- [BIDS Derivatives Specification](https://bids-specification.readthedocs.io/en/stable/derivatives/introduction.html)
