# osipy — Product Requirements Document

**Version**: 0.0.1
**Date**: 2026-03-02
**Status**: Active
**License**: MIT

---

## 1. Executive Summary

osipy is an OSIPI-compliant MRI perfusion imaging analysis library that handles GPU/CPU-agnostic quantification across four modalities: DCE-MRI, DSC-MRI, ASL, and IVIM. The library handles data loading, signal processing, pharmacokinetic model fitting, and BIDS-compliant export using a custom numerical stack with optional CuPy GPU acceleration.

osipy is a single library that covers the perfusion imaging analysis workflow from raw data to quantitative parameter maps, following OSIPI standards and producing reproducible results across platforms. It provides consistent APIs, standardized terminology, and validated implementations for all four perfusion modalities — unlike the current situation where you'd need separate tools for each.

The MVP goal is a working library that processes DCE-MRI, DSC-MRI, ASL, and IVIM data end-to-end with OSIPI-validated accuracy, GPU/CPU transparency, and enough documentation that a new user can run an analysis without outside help.

## 2. Mission

**Mission Statement**: Build a validated, standards-compliant Python library for MRI perfusion analysis with optional GPU acceleration. Focus areas: correctness against published references, reproducible results, and usable API design.

### Core Principles

1. **Scientific Rigor** — All algorithms reference peer-reviewed literature or OSIPI consensus guidelines. Implementations are validated against OSIPI Digital Reference Objects (DROs) where available.

2. **Standardized Terminology** — OSIPI CAPLEX lexicon for DCE/DSC parameters (Ktrans, ve, vp, kep, CBV, CBF, MTT), OSIPI ASL lexicon for ASL parameters (CBF, ATT), and established IVIM terminology (D, D*, f).

3. **Modularity & Extensibility** — Each perfusion modality is an independent module with shared infrastructure in `common/`. All modalities use a consistent `@register_*` / `get_*()` / `list_*()` registry pattern. Contributing a new model or processing method requires a single file with a decorator — no core modifications. 17+ extension points span all modalities, including: DCE pharmacokinetic models (`@register_model`), IVIM signal models (`@register_ivim_model`), DSC deconvolution methods (`@register_deconvolver`), DSC leakage correctors (`@register_leakage_corrector`), DSC arrival detectors (`@register_arrival_detector`), DSC normalizers (`@register_normalizer`), ASL quantification models (`@register_quantification_model`), ASL ATT models (`@register_att_model`), ASL difference methods (`@register_difference_method`), M0 calibration methods (`@register_m0_calibration`), population AIFs (`@register_aif`), fitters (`@register_fitter`), T1 mapping methods (`@register_t1_method`), concentration models (`@register_concentration_model`), IVIM fitting strategies (`@register_ivim_fitter`), AIF detectors (`@register_aif_detector`), and convolution methods (`@register_convolution`).

4. **Hardware Abstraction** — All numerical code uses the `xp = get_array_module()` pattern for CPU/GPU transparency. GPU acceleration via CuPy is optional; the library functions fully on CPU when CuPy is not installed.

5. **No scipy** — All numerical operations use `xp`-compatible code (vectorized Levenberg-Marquardt, FFT convolution, `xp.linalg`) to maintain GPU compatibility.

## 3. Target Users

### Primary Personas

**Clinical Researcher** (P1)
- Runs perfusion analyses on patient datasets for oncology, neurology, or other clinical research
- Comfortable with Python scripting but not a software engineer
- Needs validated, reproducible results with OSIPI-standard parameter naming
- Problem: Existing tools produce inconsistent results across software packages (28-78% accuracy scores per OSIPI-DCE challenge)

**MRI Physicist** (P2)
- Develops and validates new acquisition protocols
- Deep understanding of MRI physics and quantification models
- Needs access to multiple models, detailed fitting diagnostics, and multi-PLD/multi-b-value support
- Problem: No single library covers all four perfusion modalities with consistent APIs

**Software Developer / Pipeline Engineer** (P3)
- Integrates perfusion analysis into automated imaging pipelines
- Needs programmatic APIs, BIDS-compliant I/O, and headless operation
- Problem: Existing tools require manual intervention and produce non-standard output formats

**Research Team Lead** (P4)
- Processes large multi-center studies (hundreds of subjects)
- Needs GPU acceleration to reduce processing time from hours to minutes
- Problem: CPU-only tools are impractical for population-scale datasets

## 4. MVP Scope

### In Scope — Core Functionality

- ✅ DCE-MRI: T1 mapping (VFA, Look-Locker), signal-to-concentration, 5 PK models (Tofts, Extended Tofts, Patlak, 2CXM, 2CUM)
- ✅ DSC-MRI: Signal-to-deltaR2*, 3 SVD deconvolution variants (sSVD, cSVD, oSVD), Tikhonov regularization, BSW and bidirectional leakage correction, bolus arrival detection, CBV/CBF/MTT maps
- ✅ ASL: pCASL/PASL/CASL quantification, multi-PLD with ATT estimation, M0 calibration
- ✅ IVIM: Segmented, simultaneous, and Bayesian bi-exponential fitting
- ✅ AIF: 5 population models (Parker, Georgiou, Fritz-Hansen, Weinmann, McGrath) + automatic detection
- ✅ I/O: NIfTI, DICOM, and BIDS loading; BIDS derivative export
- ✅ Fitting: Custom vectorized Levenberg-Marquardt optimizer
- ✅ Convolution: Piecewise-linear, exponential (Flouri et al.), FFT, and matrix-based deconvolution
- ✅ CLI: `osipy` command for running pipelines from YAML configuration files with Pydantic v2 validation

### In Scope — Technical

- ✅ GPU/CPU agnostic code via `xp = get_array_module()` pattern
- ✅ Python 3.12+ with type hints throughout
- ✅ Comprehensive test suite with synthetic DRO fixtures
- ✅ Documentation following Diataxis framework (tutorials, how-to, reference, explanation)
- ✅ Custom exception hierarchy (OsipyError, DataValidationError, FittingError, MetadataError, AIFError, IOError, ValidationError, MissingParametersError)
- ✅ Quality masks for all fitting outputs (R² > 0.5 threshold)
- ✅ End-to-end pipeline wrappers per modality
- ✅ CLI entry point (`osipy`) with YAML config validation, template generation, and verbose logging

### Out of Scope — Deferred to Future Phases

- ❌ Preprocessing (motion correction, skull stripping, registration, distortion correction) — delegate to FSL/ANTs/SPM
- ❌ Model selection framework (AIC/BIC for choosing between DCE models)
- ❌ Hematocrit correction for AIF
- ❌ PSR/SR clinical metrics for DSC
- ❌ Tri-exponential or kurtosis-IVIM models
- ❌ Partial volume correction
- ❌ Bayesian spatial priors
- ❌ Deep learning fitting (IVIM-NET, adds PyTorch dependency)
- ❌ SNR estimation and outlier detection
- ❌ Spatial regularization
- ❌ Monte Carlo uncertainty tools
- ❌ Interactive visualization (Plotly)

## 5. User Stories

### US-1: DCE-MRI Analysis Workflow (P1 — MVP)

> As a clinical researcher, I want to load a DCE-MRI dataset, compute T1 maps, fit pharmacokinetic models, and export quantitative parameter maps, so that I can compare tumor perfusion across patients using validated, OSIPI-standard parameters.

**Acceptance Criteria**:
1. Given a 4D NIfTI DCE-MRI dataset with multiple flip angles, when the user runs T1 mapping followed by Tofts model fitting, then Ktrans, ve, and vp maps are produced within OSIPI-published tolerance ranges.
2. Given a dataset without a measured AIF, when the user selects a Parker population AIF, then the system applies the published model and produces valid parameter estimates.
3. Given fitting fails for some voxels, when the computation completes, then those voxels are flagged in an output quality mask.

### US-2: DSC-MRI Cerebral Perfusion (P2)

> As a neuroradiologist, I want to generate leakage-corrected rCBV maps from DSC-MRI data normalized to contralateral white matter, so that I can accurately assess brain tumor perfusion.

**Acceptance Criteria**:
1. Given 4D DSC-MRI data, when signal-to-concentration conversion and deconvolution are run, then rCBV, CBF, and MTT maps are produced.
2. Given DSC data from a tumor with contrast leakage, when BSW correction is applied, then corrected rCBV more accurately reflects true blood volume.

### US-3: ASL CBF Quantification (P3)

> As an MRI physicist, I want to process multi-PLD pCASL data to generate CBF maps in physiological units with arterial transit time estimates, so that I can validate new ASL acquisition protocols.

**Acceptance Criteria**:
1. Given multi-PLD pCASL data with M0 calibration, when ASL quantification is run, then CBF maps in ml/100g/min match OSIPI reference values.
2. Given multi-PLD data with ATT estimation enabled, then voxel-wise ATT maps are produced alongside CBF.

### US-4: IVIM Diffusion-Perfusion Analysis (P4)

> As a researcher, I want to separate true diffusion from pseudo-diffusion in multi-b-value data using validated fitting methods, so that I can quantify tissue microperfusion without contrast agents.

**Acceptance Criteria**:
1. Given multi-b-value diffusion data, when segmented IVIM fitting is run, then D, D*, and f parameter maps are produced with physiological bounds enforced.
2. Given a choice of Bayesian fitting, then uncertainty estimates are provided for each parameter.

### US-5: Automated Pipeline Integration (P5)

> As a pipeline engineer, I want to integrate osipy into an automated workflow that processes incoming studies and exports BIDS-compliant results, so that large-scale studies can run without manual intervention.

**Acceptance Criteria**:
1. Given DICOM input, when the pipeline runs automatically, then data is loaded, processed, and results exported as BIDS derivatives.
2. Given parallel processing, then results are identical to serial processing (deterministic behavior).

### US-6: GPU-Accelerated Processing (P2)

> As a research team lead, I want GPU acceleration to reduce per-subject processing time by 10x+ while maintaining identical numerical results, so that population-scale studies are practical.

**Acceptance Criteria**:
1. Given a GPU workstation with CuPy, when DCE analysis runs, then fitting completes at least 10x faster than single-threaded CPU.
2. Given the same input data, then GPU and CPU results are numerically identical within floating-point tolerance (relative error < 1e-6).
3. Given CuPy is not installed, then the library loads and operates fully on CPU without errors.

### US-7: Command-Line Pipeline Execution (P3)

> As a pipeline engineer, I want to run osipy analyses from the command line using YAML configuration files, so that I can integrate perfusion analysis into automated workflows without writing Python scripts.

**Acceptance Criteria**:
1. Given a YAML config file specifying modality and pipeline parameters, when `osipy config.yaml /path/to/data` is executed, then the correct modality pipeline runs and results are saved to the output directory.
2. Given `osipy --dump-defaults dce`, then a commented YAML template for DCE pipeline configuration is printed to stdout.
3. Given `osipy --validate config.yaml`, then the config file is validated against Pydantic models and errors are reported without running the pipeline.

## 6. Core Architecture & Patterns

### High-Level Architecture

```
Input Files (NIfTI/DICOM/BIDS)
        |
        v
   PerfusionDataset (data container + metadata)
        |
        +-- AcquisitionParams (modality-specific)
        +-- ArterialInputFunction (population or detected)
        |
        v
   Model Fitting (vectorized LM, Bayesian)
        |
        v
   FittingResult (per-voxel) --> ParameterMap (aggregated)
        |
        +-- ValidationReport (optional DRO comparison)
        v
   Output Files (BIDS derivatives)
```

### Directory Structure

```
osipy/
├── common/             # Shared infrastructure (~58 files)
│   ├── backend/        # GPU/CPU abstraction (get_array_module, to_gpu, to_numpy)
│   ├── aif/            # Arterial Input Function (Parker, Georgiou, FritzHansen, detection)
│   ├── fitting/        # Model fitting (vectorized LM, Bayesian, parallel)
│   ├── convolution/    # Convolution/deconvolution (piecewise-linear, exp, FFT, matrix)
│   ├── io/             # File I/O (NIfTI, DICOM, BIDS, vendor parsers)
│   ├── signal/         # Signal processing utilities
│   ├── validation/     # DRO comparison tools
│   ├── visualization/  # Plotting (matplotlib)
│   ├── dataset.py      # PerfusionDataset container
│   ├── parameter_map.py # ParameterMap container
│   ├── types.py        # Enums (Modality, LabelingType, FittingMethod, AIFType)
│   ├── exceptions.py   # Custom exceptions
│   └── acquisition.py  # AcquisitionParams per modality
├── dce/                # DCE-MRI (~13 files)
│   ├── t1_mapping/     # VFA and Look-Locker T1 estimation
│   ├── concentration/  # Signal-to-concentration conversion
│   ├── models/         # PK models (Tofts, Extended Tofts, Patlak, 2CXM)
│   └── fitting.py      # High-level fitting API
├── dsc/                # DSC-MRI (~10 files)
│   ├── concentration/  # Signal-to-deltaR2* conversion
│   ├── deconvolution/  # SVD variants (sSVD, cSVD, oSVD)
│   ├── leakage/        # BSW leakage correction
│   ├── parameters/     # Perfusion maps (CBV, CBF, MTT)
│   └── normalization.py
├── asl/                # ASL (~8 files)
│   ├── labeling/       # Labeling schemes (PASL, CASL, pCASL)
│   ├── calibration/    # M0 calibration
│   └── quantification/ # CBF quantification, multi-PLD
├── ivim/               # IVIM (~6 files)
│   ├── models/         # Bi-exponential model
│   └── fitting/        # Segmented, full, and Bayesian estimators
├── cli/                # CLI entry point
│   ├── main.py         # argparse CLI (osipy)
│   ├── config.py       # Pydantic v2 config models, load_config(), dump_defaults()
│   └── runner.py       # Pipeline orchestration from config
└── pipeline/           # End-to-end workflows (~6 files)
    ├── dce_pipeline.py
    ├── dsc_pipeline.py
    ├── asl_pipeline.py
    ├── ivim_pipeline.py
    └── runner.py       # Unified run_analysis() dispatcher
```

### Key Design Patterns

**Array Module Pattern (GPU/CPU Abstraction)**:
```python
from osipy.common.backend.array_module import get_array_module

def my_numerical_function(data, aif):
    xp = get_array_module(data, aif)  # Returns numpy or cupy
    result = xp.exp(-data / time)     # Same code, CPU or GPU
    return result
```

**Component Hierarchy**: `BaseComponent(ABC)` is the root for all registered components, providing `name` and `reference` properties. `BaseSignalModel(BaseComponent)` adds `parameters`, `parameter_units`, and `get_bounds()` for all signal models. Modality-specific ABCs inherit from these: `BasePerfusionModel[P]` (DCE), `BaseASLModel` (ASL), `IVIMModel` (IVIM), `DSCConvolutionModel` (DSC). Non-model ABCs (`BaseDeconvolver`, `BaseLeakageCorrector`, `BaseArrivalDetector`, `BaseM0Calibration`, `BaseAIFDetector`, `BaseAIF`, `BaseFitter`) also inherit from `BaseComponent`.

**FittableModel Protocol**: All four modalities use a shared `LevenbergMarquardtFitter` via the `FittableModel` protocol. Each modality wraps its signal model in a binding adapter (`BoundDCEModel`, `BoundIVIMModel`, `BoundASLModel`, `BoundGammaVariateModel`, `BoundDSCModel`) that fixes independent variables so the fitter only sees free parameters. `BaseBoundModel` provides shared fixed-parameter logic.

**Vectorized Fitting**: Custom Levenberg-Marquardt optimizer fits N voxels simultaneously using batched xp operations — no Python loops over voxels.

**Convolution Strategy**: Piecewise-linear convolution with analytic integration (following dcmri library) for pharmacokinetic models on non-uniform time grids. FFT convolution retained as optimization for large uniform datasets.

## 7. Features

### Data Loading

| Feature | Description |
|---------|-------------|
| NIfTI loading | Load 3D/4D NIfTI files via nibabel |
| DICOM loading | Load DICOM series with vendor-specific parsers (GE, Siemens, Philips) |
| BIDS loading | Load BIDS-formatted datasets (hand-rolled parser over nibabel + JSON sidecars) |
| BIDS export | Export parameter maps as BIDS derivatives (NIfTI + sidecar JSON + dataset_description.json) |

### DCE-MRI

| Feature | Description |
|---------|-------------|
| T1 mapping (VFA) | Variable Flip Angle T1 estimation |
| T1 mapping (Look-Locker) | Look-Locker T1 estimation |
| Signal-to-concentration | Convert signal intensity to contrast agent concentration |
| Standard Tofts model | Ktrans, ve fitting |
| Extended Tofts model | Ktrans, ve, vp fitting |
| Patlak model | Ktrans, vp fitting (linear graphical analysis) |
| Two-Compartment Exchange | Fp, PS, ve, vp fitting (2CXM) |
| Two-Compartment Uptake | Ktrans, Kep, ve fitting (2CUM) |
| Arterial delay fitting | Optional delay parameter via `fit_model(..., fit_delay=True)` |

**Additional Expert Requirements — DCE-MRI**

| Requirement | Priority | Rationale |
|-------------|----------|-----------|
| | | |

### DSC-MRI

| Feature | Description |
|---------|-------------|
| Signal-to-deltaR2* | Convert T2*-weighted signal to concentration proxy |
| SVD deconvolution | Standard SVD with truncation threshold |
| cSVD deconvolution | Block-circulant SVD (delay-insensitive) |
| oSVD deconvolution | Oscillation-index regularized SVD |
| Tikhonov regularization | Tikhonov-regularized deconvolution |
| BSW leakage correction | Boxerman-Schmainda-Weisskoff T1/T2* leakage correction |
| Bidirectional leakage correction | Bidirectional T1/T2* leakage correction |
| Bolus arrival detection | Automatic bolus arrival time detection |
| White matter normalization | Normalize perfusion maps to contralateral white matter |
| Perfusion maps | rCBV, CBF, MTT, Tmax, TTP computation |

**Additional Expert Requirements — DSC-MRI**

| Requirement | Priority | Rationale |
|-------------|----------|-----------|
| | | |

### ASL

| Feature | Description |
|---------|-------------|
| pCASL quantification | Pseudo-continuous ASL CBF estimation |
| PASL quantification | Pulsed ASL (FAIR, EPISTAR, PICORE) |
| CASL quantification | Continuous ASL |
| Multi-PLD fitting | ATT estimation from multiple post-labeling delays |
| M0 calibration | Absolute CBF quantification via M0 reference |
| Background suppression | Correction for suppression efficiency |

**Additional Expert Requirements — ASL**

| Requirement | Priority | Rationale |
|-------------|----------|-----------|
| | | |

### IVIM

| Feature | Description |
|---------|-------------|
| Segmented fitting | Two-step fitting (high-b mono-exp then full bi-exp) |
| Simultaneous fitting | Full bi-exponential least-squares |
| Bayesian fitting | MCMC with uncertainty estimates |
| Physiological bounds | Enforced parameter bounds (D, D*, f) |

**Additional Expert Requirements — IVIM**

| Requirement | Priority | Rationale |
|-------------|----------|-----------|
| | | |

### AIF (Arterial Input Function)

| Feature | Description |
|---------|-------------|
| Parker AIF | Population model (Parker et al., 2006) |
| Georgiou AIF | Population model |
| Fritz-Hansen AIF | Population model |
| Weinmann AIF | Population model (Weinmann et al., 1984) |
| McGrath AIF | Preclinical population model |
| Automatic detection | Peak-based AIF detection from image data |
| Arterial delay | `shift_aif()` utility for delay fitting |

### Convolution/Deconvolution Core

| Feature | Description |
|---------|-------------|
| Piecewise-linear convolution | Analytic integration for non-uniform time grids (dcmri approach) |
| Exponential convolution | Recursive expconv using Flouri et al. (2016) formulation |
| Bi-exponential convolution | Analytical for two-compartment models |
| N-exponential convolution | Gamma-variate for chain compartment models |
| FFT convolution | Optimized for large uniform datasets |
| Matrix deconvolution | Convolution matrix + regularized pseudo-inverse (TSVD/Tikhonov) |

### CLI

| Feature | Description |
|---------|-------------|
| YAML pipeline execution | Run any modality pipeline from a YAML config file via `osipy` |
| Config template generation | `--dump-defaults {dce\|dsc\|asl\|ivim}` prints commented YAML template |
| Config validation | `--validate` checks YAML against Pydantic models without running |
| Output override | `-o DIR` overrides the output directory |
| Verbose logging | `-v` enables DEBUG-level logging |

## 8. Technology Stack

### Core Dependencies (Required)

| Package | Version | Purpose |
|---------|---------|---------|
| Python | >= 3.12 | Runtime |
| numpy | >= 2.0.0 | Array operations via xp abstraction |
| nibabel | >= 5.2.0 | NIfTI I/O |
| pydicom | >= 2.4.0 | DICOM I/O |
| matplotlib | >= 3.8.0 | Visualization |
| pydantic | >= 2.5.0 | YAML config validation |
| pyyaml | >= 6.0 | YAML parsing |
| dcm2niix | >= 1.0.20250506 | DICOM→NIfTI conversion (bundled binary wheel) |

### Optional Dependencies

| Package | Version | Install Extra | Purpose |
|---------|---------|---------------|---------|
| cupy-cuda12x | >= 13.0.0 | `[gpu]` | GPU acceleration |

### Development Dependencies

| Package | Purpose |
|---------|---------|
| pytest >= 8.0 | Test runner |
| ruff | Linting (pycodestyle, pyflakes, isort, flake8-bugbear, pydocstyle NumPy convention) |
| mypy | Type checking (strict mode) |
| mkdocs + mkdocstrings | Documentation build |
| mkdocs-material | Documentation theme |

### Banned Dependencies

| Package | Reason | Replacement |
|---------|--------|-------------|
| scipy | Not GPU-compatible via `xp` abstraction | `LevenbergMarquardtFitter` (`common/fitting/`), `xp.linalg.*`, `xp.trapezoid()`, `xp.interp()` |

## 9. Security & Configuration

### Configuration Management

- **Backend selection**: `set_backend("cpu" | "gpu")` or automatic detection via `is_gpu_available()`
- **GPU config**: `GPUConfig` dataclass for batch sizes, memory limits, fallback behavior
- **Pipeline config**: Per-modality configuration (model selection, fitting method, parameter bounds, baseline frames)
- **Logging**: Python stdlib logging with configurable verbosity (DEBUG, INFO, WARNING, ERROR)

### Data Safety

- Input data is never modified in place; all operations produce new arrays
- Quality masks always accompany parameter maps — invalid voxels contain NaN or 0
- BIDS export preserves full provenance (model name, fitting method, software version, parameters used)

### Deployment

- Pure Python package installable via pip: `pip install osipy` or `pip install osipy[gpu]`
- Build system: hatchling
- No network access required at runtime
- No authentication or secrets management needed
- Cross-platform: Linux, macOS, Windows

## 10. API Specification

### Public API Summary

```python
# GPU/CPU Backend
get_array_module(*arrays)        # Returns numpy or cupy
to_gpu(array) / to_numpy(array)  # Transfer between CPU/GPU
is_gpu_available() / get_backend() / set_backend()

# I/O
load_nifti(path, modality=None, acquisition_params=None, sidecar_json=None) -> PerfusionDataset
discover_dicom(path) -> list[SeriesInfo]
load_dicom_series(series, modality=None) -> PerfusionDataset
export_bids(parameter_maps, output_dir, subject_id, ...) -> Path

# DCE-MRI
compute_t1_map(dataset, method="vfa") -> T1MappingResult
dce_signal_to_concentration(signal, t1_map, acquisition_params, ...) -> ndarray
fit_model(model_name, concentration, aif, time, ...) -> DCEFitResult
get_model(name) -> BasePerfusionModel
list_models() -> list[str]

# DSC-MRI
signal_to_delta_r2(signal, te, ...) -> ndarray
correct_leakage(delta_r2, aif, ...) -> ndarray
compute_perfusion_maps(delta_r2, aif, ...) -> dict[str, ParameterMap]
get_deconvolver(name).deconvolve(delta_r2, aif, ...) -> DeconvolutionResult

# ASL
quantify_cbf(difference, M0, ...) -> dict[str, ParameterMap]
apply_m0_calibration(data, M0, ...) -> ndarray

# IVIM
fit_ivim(signal, b_values, method="segmented") -> dict[str, ParameterMap]

# Pipelines
run_analysis(data, modality, **kwargs)
DCEPipeline / DSCPipeline / ASLPipeline / IVIMPipeline
```

### CLI Command Reference

```
osipy [options] config data_path

Positional arguments:
  config              Path to YAML config file
  data_path           Path to input data directory or file

Options:
  --version           Show version and exit
  --validate CONFIG   Validate a YAML config file and exit
  --dump-defaults MOD Print default YAML template for modality (dce|dsc|asl|ivim)
  -o DIR, --output DIR  Override output directory
  -v, --verbose       Enable DEBUG logging
```

### Core Data Entities

**PerfusionDataset**: Input container holding image data (3D/4D ndarray), affine matrix, modality enum, time points, modality-specific acquisition parameters, source provenance, and quality mask.

**ParameterMap**: Output container holding parameter values (3D ndarray), OSIPI-standard name/symbol/units, uncertainty estimates, quality mask with failure reasons, and model/fitting provenance.

**ArterialInputFunction**: AIF container with time/concentration arrays, type (measured/population/automatic), model parameters, and literature reference.

**FittingResult**: Per-voxel results with fitted parameters, uncertainties, residuals, R², convergence status, and iteration count.

### Error Handling

```python
class OsipyError(Exception): ...              # Base
class DataValidationError(OsipyError): ...    # Invalid input shapes/types/registry lookups
class FittingError(OsipyError): ...           # Model fitting failures
class MetadataError(OsipyError): ...          # Missing acquisition parameters
class AIFError(OsipyError): ...               # AIF-related errors
class IOError(OsipyError): ...                # File I/O errors
class ValidationError(OsipyError): ...        # DRO/reference validation failures
class MissingParametersError(OsipyError): ... # Required parameters not provided
```

Functions raise exceptions for structural errors. Individual voxel fitting failures are captured in quality masks, not raised as exceptions.

## 11. Success Criteria

### Accuracy

- ✅ **SC-001**: DCE Ktrans values match OSIPI DRO reference within 5% relative error
- ✅ **SC-002**: DSC rCBV values match reference implementations within 10% relative error
- ✅ **SC-003**: ASL CBF matches OSIPI DRO values within published tolerance ranges
- ✅ **SC-004**: IVIM parameters (D, D*, f) match synthetic ground truth within 15% relative error at clinical SNR

### Performance

- ✅ **SC-005**: End-to-end DCE analysis completes in <10 min (CPU) or <1 min (GPU) for 64x64x40x80 voxel dataset
- ✅ **SC-007**: Memory usage stays below 16GB for 128x128x30x60 voxel clinical datasets
- ✅ **SC-010**: GPU fitting achieves at least 10x speedup over single-threaded CPU for >10,000 voxels

### Reproducibility

- ✅ **SC-008**: Bit-for-bit reproducible results across Linux, macOS, and Windows
- ✅ **SC-011**: GPU and CPU produce identical parameter values within floating-point tolerance (relative error < 1e-6)
- ✅ **SC-012**: Library operates correctly on CPU when CuPy is not installed

### Usability

- ✅ **SC-006**: 95% of first-time users can run a sample analysis from documentation alone
- ✅ **SC-009**: All public functions include documented references to source literature

### Validation Tolerances (per OSIPI publications)

| Parameter | Absolute Tolerance | Relative Tolerance | Source |
|-----------|-------------------|-------------------|--------|
| Ktrans | 0.005 min-1 | 1% | QIBA DCE Profile v2.0 |
| ve | 0.05 | 5% | QIBA DCE Profile v2.0 |
| vp | 0.002 | 2% | QIBA DCE Profile v2.0 |
| CBF (DSC) | N/A | 10% | OSIPI TF4.1 |
| CBF (ASL) | 5 ml/100g/min | 10% | OSIPI ASL Consensus |
| D (IVIM) | 0.05e-3 mm2/s | 5% | IVIM literature consensus |

## 12. Implementation Phases

### Phase 1: Core Infrastructure (Complete)

**Goal**: Establish shared utilities, data containers, and I/O.

- ✅ Data containers (PerfusionDataset, ParameterMap, AcquisitionParams)
- ✅ NIfTI, DICOM, BIDS loading
- ✅ Custom exception hierarchy
- ✅ GPU/CPU backend abstraction (get_array_module)
- ✅ Population AIF models (Parker, Georgiou, Fritz-Hansen, Weinmann, McGrath)
- ✅ Automatic AIF detection
- ✅ Convolution core (piecewise-linear, exponential, FFT, matrix deconvolution)

### Phase 2: Modality Implementations (Complete)

**Goal**: Implement all four modality analysis pipelines.

- ✅ DCE: T1 mapping (VFA, Look-Locker), signal-to-concentration, 5 PK models (incl. 2CUM), arterial delay fitting
- ✅ DSC: Signal-to-deltaR2*, 3 SVD variants + Tikhonov, BSW + bidirectional leakage, bolus arrival detection, perfusion maps, normalization
- ✅ ASL: pCASL/PASL/CASL quantification, multi-PLD, M0 calibration
- ✅ IVIM: Segmented, simultaneous, and Bayesian fitting
- ✅ End-to-end pipeline wrappers per modality
- ✅ BIDS derivative export

### Phase 3: XP Compliance & Optimization (In Progress)

**Goal**: Achieve full GPU/CPU agnostic compliance and vectorize remaining sequential code.

- ✅ DCE models (Tofts, Extended Tofts, Patlak) xp-compliant
- ✅ DSC normalization, maps, leakage correction xp-compliant
- ✅ ASL CBF/multi-PLD quantification xp-compliant
- ✅ IVIM biexponential/Bayesian fitting xp-compliant
- ✅ FFT convolution and custom vectorized LM optimizer
- ❌ Remaining xp gaps: DCE `vfa.py` fallback, ASL `m0.py`/`schemes.py`, IVIM `estimators.py`, common `fitting/vectorized.py` setup, `fitting/bayesian.py`, `fitting/parallel.py`, `convolution/fft.py`
- ❌ Vectorize DSC SVD deconvolution (currently per-voxel loops, 10-50x speedup possible)

**Validation**: All refactored files must pass GPU equivalence tests (relative error < 1e-6).

### Phase 4: Quality & Release Readiness (Planned)

**Goal**: Testing completeness, CI/CD, and release preparation.

- ❌ Achieve 95% test coverage (100% for numerical algorithms)
- ❌ GitHub Actions CI pipeline (pytest, ruff, mypy on push/PR)
- ❌ Coverage reporting integration
- ❌ MkDocs build verification in CI
- ❌ Optional GPU test runner
- ❌ Package publication to PyPI
- ❌ API freeze and versioning policy

## 13. Future Considerations

### Post-MVP Enhancements (P2)

- **Model selection**: AIC/BIC framework for automated DCE model comparison
- **Hematocrit correction**: Adjust AIF for hematocrit (~42% underestimation without)
- **DSC clinical metrics**: PSR (percent signal recovery), rCBV normalization
- **DICOM 4D sorting**: Robust temporal sorting when metadata tags are missing
- **B1 map correction**: Improve VFA T1 mapping accuracy

### Advanced Features (P3)

- **Preprocessing integration**: Motion correction, skull stripping, registration (via FSL/ANTs wrappers)
- **Partial volume correction**: For ASL and DSC
- **Clustering AIF**: Data-driven AIF selection
- **Tri-exponential IVIM**: For liver applications
- **Kurtosis-IVIM**: Non-Gaussian diffusion component

### Research Extensions (P4)

- **Bayesian spatial priors**: Spatially-regularized fitting
- **Deep learning fitting**: IVIM-NET and similar approaches (optional PyTorch dependency)
- **Monte Carlo uncertainty**: Full uncertainty propagation
- **Spatial regularization**: TV/MRF-based regularization of parameter maps
- **SNR estimation**: Automated signal quality assessment

## 14. Risks & Mitigations

### Risk 1: XP Compliance Gaps Block GPU Users

**Impact**: High — GPU acceleration is a major feature.
**Likelihood**: Medium — Known gaps exist in several modules.
**Mitigation**: Remaining xp gaps are documented with specific file locations. Prioritize by impact (DSC SVD vectorization is highest value). CPU fallback ensures functionality while gaps are addressed.

### Risk 2: OSIPI Validation Data Availability

**Impact**: High — Validation against DROs is a core requirement.
**Likelihood**: Low — Synthetic DRO fixtures are already implemented in conftest.py.
**Mitigation**: Built-in synthetic DROs for all modalities. External OSIPI challenge data is gracefully skipped if unavailable. Consider bundling minimal DRO data with the package.

### Risk 3: Cross-Platform Reproducibility

**Impact**: Medium — Research reproducibility is a stated goal.
**Likelihood**: Medium — Floating-point behavior can vary across platforms.
**Mitigation**: Deterministic parallelism with fixed chunking (FR-048). Bit-for-bit reproducibility tests in CI across Linux/macOS/Windows. Fixed random seeds for stochastic operations.

### Risk 4: Scope Creep into Preprocessing

**Impact**: Medium — Preprocessing is the most common user request.
**Likelihood**: High — Users will expect motion correction, registration, etc.
**Mitigation**: Clearly documented scope: osipy handles analysis, not preprocessing. Documentation guides users to established tools (FSL, ANTs, SPM, dcm2niix). Consider thin wrapper functions in a future `osipy[preproc]` extra.

### Risk 5: API Stability Before 1.0

**Impact**: Medium — Breaking changes frustrate early adopters.
**Likelihood**: Medium — API is documented but not formally frozen.
**Mitigation**: Current API is well-documented in tutorials and reference docs. Phase 4 includes API freeze. Semantic versioning from 1.0 onward. Deprecation warnings for at least one minor version before removal.

## 15. Appendix

### Key Literature References

**DCE-MRI**:
- Tofts PS, Kermode AG (1991). Measurement of the blood-brain barrier permeability. *Magn Reson Med*.
- Tofts PS et al. (1999). Estimating kinetic parameters from DCE-MRI. *JMRI*.
- Parker GJM et al. (2006). Population-averaged AIF. *Magn Reson Med*.
- Weinmann HJ et al. (1984). Pharmacokinetics of GdDTPA. *Physiol Chem Phys Med NMR*.

**DSC-MRI**:
- Ostergaard L et al. (1996). High resolution CBF measurement. *Magn Reson Med*.
- Boxerman JL et al. (2006). Leakage-corrected rCBV maps. *AJNR*.
- Wu O et al. (2003). Timing-insensitive flow estimation (cSVD/oSVD). *Magn Reson Med*.

**ASL**:
- Buxton RB et al. (1998). General kinetic model for ASL. *Magn Reson Med*.
- Alsop DC et al. (2015). Recommended ASL implementation. *Magn Reson Med*.

**IVIM**:
- Le Bihan D et al. (1988). IVIM MR imaging. *Radiology*.
- Federau C (2017). IVIM as perfusion measure. *NMR Biomed*.

**Convolution**:
- Flouri D, Lesnic D, Mayrovitz HN (2016). Numerical solution of convolution integral equations in pharmacokinetics. *Comput Methods Biomech Biomed Eng*.
- Sourbron & Buckley (2013). Tracer kinetic modelling in MRI. *NMR Biomed*.

### Key Conventions

| Convention | Details |
|------------|---------|
| Time units | Seconds in public API, minutes internally for DCE models |
| Quality masks | Always check — invalid voxels contain NaN/0 |
| R² threshold | Fixed at 0.5 for valid fitting results |
| Error handling | Custom exceptions from `osipy.common.exceptions` |
| Naming | OSIPI CAPLEX (Ktrans, ve, vp, CBF, CBV, MTT, D, D*, f) |
| Docstrings | NumPy-style with OSIPI codes where applicable |
| Type hints | Python 3.12+ style with NDArray from numpy.typing |
| xp pattern | `xp = get_array_module(data, aif)` as first line of all numerical functions |