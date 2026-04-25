# osipy

OSIPI-compliant MRI perfusion analysis library.

> [!CAUTION]
> ### GSoC Contributors — Read This First
>
> **Do not open issues or pull requests before reaching out.** Join the [OSIPI Slack workspace](https://osipi.slack.com/) and introduce yourself in the project channel. Discuss your proposed contribution with a maintainer **before** writing any code or filing issues.
>
> **Issues and PRs containing AI-generated content (ChatGPT, Copilot, etc.) will be closed without review.** We want to see *your* understanding of the problem, not a language model's. Come talk to us first so we can help you make a meaningful contribution.

> [!WARNING]
> osipy is under active development and has **not** been validated for clinical, research, or any other use. Do not use in professional imaging pipelines. APIs may change between releases.

A Python library implementing [OSIPI](https://www.osipi.org/) standards for MRI perfusion data analysis across four modalities: DCE-MRI, DSC-MRI, ASL, and IVIM. Supports multi-vendor DICOM, NIfTI, and BIDS data formats with optional GPU acceleration.

## Features

- **DCE-MRI** — T1 mapping (VFA, Look-Locker), signal-to-concentration conversion, pharmacokinetic modeling (Tofts, Extended Tofts, Patlak, 2CXM, 2CUM), arterial delay fitting
- **DSC-MRI** — Delta-R2* conversion, SVD deconvolution (sSVD/cSVD/oSVD/Tikhonov), leakage correction, CBV/CBF/MTT/Tmax maps
- **ASL** — PCASL/PASL/CASL labeling schemes, M0 calibration, single and multi-PLD CBF quantification, ATT estimation
- **IVIM** — Bi-exponential diffusion-perfusion fitting with segmented, simultaneous, and Bayesian approaches
- **Standards** — Parameter naming, units, and symbols follow [CAPLEX](https://osipi.github.io/OSIPI_CAPLEX/) (DCE/DSC) and the [ASL Lexicon](https://osipi.github.io/ASL-Lexicon/)
- **Extensible** — Registry-driven architecture: add models, fitters, or methods with a single decorator
- **GPU-ready** — Transparent CPU/GPU portability via array module abstraction (NumPy/CuPy)
- **Multi-vendor** — DICOM metadata extraction for GE, Siemens, and Philips scanners

## Installation

```bash
pip install osipy
```

For GPU acceleration (CUDA 12.x):

```bash
pip install osipy[gpu]
```

**Requires Python 3.12+.**

## Quick Start

### Programmatic API

```python
from osipy.pipeline import run_analysis

# Unified entry point for any modality
result = run_analysis(data, modality="dce", model="extended_tofts", aif=aif, time=time)

# Access parameter maps
ktrans = result.parameter_maps["Ktrans"]
print(ktrans.statistics())
```

### Modality-Specific Usage

**DCE-MRI:**

```python
from osipy.dce import fit_model, compute_t1_map, list_models

# Available models
print(list_models())  # ['tofts', 'extended_tofts', 'patlak', '2cxm', '2cum']

# T1 mapping
t1_map = compute_t1_map(signal, flip_angles, tr, method="vfa")

# Pharmacokinetic fitting
result = fit_model("extended_tofts", concentration=conc, aif=aif, time=time, mask=mask)
```

**DSC-MRI:**

```python
from osipy.dsc import compute_perfusion_maps, get_deconvolver

maps = compute_perfusion_maps(delta_r2, aif, time, mask=mask, deconvolution_method="oSVD")
# maps.cbv, maps.cbf, maps.mtt, maps.ttp, maps.tmax
```

**ASL:**

```python
from osipy.asl import quantify_cbf

cbf = quantify_cbf(
    difference_image, labeling_scheme="pcasl",
    pld=1800, label_duration=1800, m0=m0_data,
)
```

**IVIM:**

```python
from osipy.ivim import fit_ivim

result = fit_ivim(dwi_data, b_values, fitting_method="segmented", b_threshold=200)
# result.d_map, result.d_star_map, result.f_map
```

### CLI

```bash
# Run a pipeline from a YAML config
osipy config.yaml /path/to/data --output results/

# Print a default config template
osipy --dump-defaults dce

# Validate a config file
osipy --validate config.yaml

# Interactive setup wizard
osipy --help-me-pls
```

### Data Loading

```python
from osipy.common.io import discover_dicom, load_dicom_series, load_nifti

# NIfTI (auto-reads an adjacent BIDS sidecar when present)
dataset = load_nifti("/path/to/dce.nii.gz", modality="dce")

# DICOM: observe first, then load.
series = discover_dicom("/path/to/dicom_dir")
dataset = load_dicom_series(
    next(s for s in series if s.role_hint == "dynamic"),
    modality="dce",
)
```

## Extending osipy

All components use a registry pattern. Adding a custom model is one file, one decorator:

```python
from osipy.dce.models.registry import register_model
from osipy.dce.models.base import BasePerfusionModel

@register_model("my_model")
class MyModel(BasePerfusionModel):
    ...

# Now usable everywhere:
result = fit_model("my_model", concentration=conc, aif=aif, time=time)
```

Registries exist for models, fitters, deconvolvers, AIF detectors, leakage correctors, T1 methods, convolution methods, and more. See the [extension points documentation](docs/CODEBASE_MAP.md) for the full list.

## Population AIFs

```python
from osipy.common.aif import get_population_aif, list_aifs

print(list_aifs())  # ['parker', 'fritz_hansen', 'georgiou', 'mcgrath']
aif = get_population_aif("parker")
```

## GPU Acceleration

```python
from osipy.common.backend import is_gpu_available, set_backend, GPUConfig

if is_gpu_available():
    set_backend(GPUConfig(force_cpu=False))
# All numerical code automatically uses GPU when available
```

## Development

```bash
pytest                    # Run all tests
pytest tests/unit/        # Unit tests only
pytest tests/integration/ # Integration tests
ruff check .              # Lint
mypy osipy                # Type check
mkdocs serve              # Local docs at http://127.0.0.1:8000
```

## License

MIT License

## References

- [OSIPI (Open Science Initiative for Perfusion Imaging)](https://www.osipi.org/)
- Dickie BR et al. CAPLEX: a standardized reporting framework for perfusion MRI. *MRM* 2024;91(5):1761-1773. [doi:10.1002/mrm.29840](https://doi.org/10.1002/mrm.29840)
- Suzuki Y et al. OSIPI ASL Lexicon. *MRM* 2024;91(5):1743-1760. [doi:10.1002/mrm.29815](https://doi.org/10.1002/mrm.29815)
