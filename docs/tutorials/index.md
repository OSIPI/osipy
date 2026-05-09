# Tutorials

End-to-end analysis workflows for each imaging modality.

Most users should start with the CLI — see [How to Run Pipeline from YAML](../how-to/run-pipeline-cli.md). The tutorials below cover the Python API for users who need programmatic control over individual analysis steps.

## Beginner

- [Getting Started](getting-started.md) — Install osipy, verify your setup, and run your first analysis.

## By Modality (Python API)

- [DCE-MRI Analysis](dce-analysis.md) — T1 mapping, signal-to-concentration conversion, and pharmacokinetic model fitting with the Extended Tofts model.
- [ASL CBF Quantification](asl-analysis.md) — Cerebral blood flow from pCASL data with M0 calibration.
- [IVIM Analysis](ivim-analysis.md) — Separate diffusion and perfusion components using bi-exponential fitting of multi-b-value DWI data.
- [DSC-MRI Analysis](dsc-analysis.md) — CBV, CBF, and MTT from DSC-MRI using SVD deconvolution with optional leakage correction.

## Jupyter Notebook Tutorials

Hands-on Jupyter notebooks from conferences and training events live under `tutorials/notebooks/`. They use kebab-case filenames (e.g. `ismrm2026-dce-tutorial.ipynb`) and are rendered with their cached outputs — the docs build does not re-execute them, so notebooks may reference local data paths from the original session.

- [ISMRM 2026 — DCE-MRI Walkthrough](notebooks/ismrm2026-dce-tutorial.ipynb) — End-to-end DCE pipeline: VFA T1 mapping, signal-to-concentration conversion, AIF selection, and Extended Tofts fitting.
