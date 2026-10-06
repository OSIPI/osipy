---
hide:
  - navigation
---

<div class="hero" markdown>

# osipy

Unified OSIPI-compliant MRI perfusion analysis.
{ .hero-tagline }

[Get Started](tutorials/getting-started.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/OSIPI/osipy){ .md-button }

</div>

!!! danger "GSoC Contributors — Read This First"
    **Do not open issues or pull requests before reaching out.**
    Join the [OSIPI Slack workspace](https://osipi.slack.com/) and introduce yourself in the project channel.
    Discuss your proposed contribution with a maintainer **before** writing any code or filing issues.
    Issues and PRs containing AI-generated content (ChatGPT, Copilot, etc.) will be closed without review.
    We value quality contributions — come talk to us first so we can help you succeed.

!!! warning "Dev Release (v0.1.0)"
    osipy is under active development and has **not** been validated for clinical, research, or any other use.
    Do not use in professional imaging pipelines — testing and validation are still in progress.
    APIs may change between releases.

<div class="grid cards" markdown>

-   :material-check-decagram:{ .lg .middle } **OSIPI Standards**

    ---

    All naming, units, and symbols follow [CAPLEX](https://osipi.github.io/OSIPI_CAPLEX/) and the [ASL Lexicon](https://osipi.github.io/ASL-Lexicon/).

-   :material-chip:{ .lg .middle } **CPU & GPU**

    ---

    Array-module abstraction (`xp`) runs identical code on NumPy or CuPy without changes.

-   :material-puzzle-outline:{ .lg .middle } **Registry-Driven**

    ---

    Add a model or method with one file and one decorator. No wiring code needed.

-   :material-speedometer:{ .lg .middle } **Fast**

    ---

    Vectorized Levenberg-Marquardt fitting and batch processing — fast on both CPU and GPU.

</div>

## Modalities

<div class="modality-grid" markdown>

<div class="modality-card dce" markdown>

### DCE-MRI

Pharmacokinetic modeling: Tofts, Extended Tofts, Patlak, 2CXM, 2CUM. T1 mapping, signal-to-concentration conversion, population and patient-specific AIFs.

[Tutorial :octicons-arrow-right-24:](tutorials/dce-analysis.md)

</div>

<div class="modality-card dsc" markdown>

### DSC-MRI

SVD-based deconvolution (sSVD, cSVD, oSVD, Tikhonov). Bolus arrival detection, leakage correction, concentration-time curve processing.

[Tutorial :octicons-arrow-right-24:](tutorials/dsc-analysis.md)

</div>

<div class="modality-card asl" markdown>

### ASL

CBF quantification for pCASL and PASL. Single and multi-PLD support, M0 calibration, arterial transit time estimation.

[Tutorial :octicons-arrow-right-24:](tutorials/asl-analysis.md)

</div>

<div class="modality-card ivim" markdown>

### IVIM

Bi-exponential diffusion fitting: segmented, full, and Bayesian strategies for D, D*, and perfusion fraction.

[Tutorial :octicons-arrow-right-24:](tutorials/ivim-analysis.md)

</div>

</div>

## Quick Start

=== "CLI"

    ```bash
    # Interactive wizard — walks you through every option
    osipy --help-me-pls

    # Or start from a default template
    osipy --dump-defaults dce > config.yaml

    # Then run
    osipy config.yaml /path/to/dce_data.nii.gz
    ```

=== "Python API"

    ```python
    import numpy as np
    import osipy

    # Load DCE-MRI data
    dataset = osipy.load_nifti("dce_data.nii.gz")

    # Time array (seconds)
    time_array = np.arange(dataset.shape[-1]) * 3.5

    # Population AIF
    aif = osipy.ParkerAIF()(time_array)

    # Fit Extended Tofts model
    result = osipy.dce_fit_model(
        "extended_tofts",
        concentration=dataset.data,
        aif=aif,
        time=time_array,
    )

    # Parameter maps
    ktrans = result.parameter_maps["Ktrans"].values
    ve = result.parameter_maps["ve"].values
    vp = result.parameter_maps["vp"].values
    ```

## Installation

=== "pip"

    ```bash
    pip install osipy
    ```

=== "uv"

    ```bash
    uv add osipy
    ```

=== "GPU support"

    ```bash
    pip install osipy[gpu]
    ```

=== "Development"

    ```bash
    git clone https://github.com/OSIPI/osipy.git
    cd osipy
    pip install -e ".[dev]"
    ```
