# How to Load Perfusion Data

Load perfusion MRI data from different vendors and formats.

## Quick Start

osipy exposes three direct loaders — pick the one that matches your input:

```python
from osipy.common.io import (
    discover_dicom,
    load_dicom_series,
    load_nifti,
    load_bids,
)
from osipy.common.types import Modality

# NIfTI (optionally with an adjacent BIDS sidecar JSON)
dataset = load_nifti("path/to/dce.nii.gz", modality=Modality.DCE)

# DICOM — two-step: observe, then load.
series = discover_dicom("path/to/dicom_dir")
dataset = load_dicom_series(series[0], modality=Modality.DCE)

# BIDS
dataset = load_bids("path/to/bids_dataset", subject="01", modality=Modality.ASL)
```

## Which Function to Use

| Format | Extension | Loader | Notes |
|--------|-----------|--------|-------|
| NIfTI | `.nii`, `.nii.gz` | `load_nifti()` | Auto-reads adjacent JSON sidecar |
| DICOM | `.dcm`, `.IMA`, enhanced | `discover_dicom()` + `load_dicom_series()` | Two-step for transparency |
| BIDS | Directory | `load_bids()`, `load_bids_with_m0()`, `load_asl_context()` | Full metadata from sidecars |

## DICOM: Discover, Then Load

DICOM loading is split into two stateless primitives so you can see
exactly what was found before any pixels are read:

```python
from osipy.common.io import discover_dicom, load_dicom_series

series = discover_dicom("path/to/dicom_dir")
for s in series:
    print(
        f"{s.description!r:30s} "
        f"role={s.role_hint:14s} "
        f"shape={s.rows}x{s.columns}  "
        f"tpi={s.n_temporal_positions}  "
        f"FA={s.flip_angle}"
    )
```

`discover_dicom()` walks the directory, groups files by
`SeriesInstanceUID`, and annotates each series with a best-effort
`role_hint` from headers alone. No pixel data is touched at this stage.

The role hints are:

| Hint | Meaning |
|------|---------|
| `dynamic` | A single series containing all timepoints (indexed by `TemporalPositionIdentifier`) — load it directly, `load_dicom_series` returns a 4D volume. |
| `dynamic_frame` | A single 3D volume that represents **one timepoint** of a dynamic acquisition exported one-series-per-frame (e.g. Siemens TWIST `TT=X.Xs`). Collect every frame that shares a `group_key` into a Python list and pass that list to `load_dicom_series()` — it stacks the per-timepoint volumes into one 4D dataset. |
| `vfa` | Single flip-angle structural, part of a VFA T1 mapping set. |
| `t1_look_locker` | Look-Locker T1 mapping series. |
| `unknown` | Anything else (structurals, localizers, non-image objects). |

Role hints are hints. You can select by any `SeriesInfo` attribute:

```python
# Load the first dynamic series
dyn = next(s for s in series if s.role_hint == "dynamic")
dataset = load_dicom_series(dyn, modality=Modality.DCE)

# Per-timepoint export: each dynamic frame is its own 3D series.
# Collect them into a list and load_dicom_series stacks them into 4D.
frames = [s for s in series if s.role_hint == "dynamic_frame"]
dataset = load_dicom_series(frames, modality=Modality.DCE)

# Or grab by SeriesInstanceUID
target = next(s for s in series if s.uid == "1.2.840.113619.…")
dataset = load_dicom_series(target, modality=Modality.DCE)
```

`load_dicom_series()` accepts either a single `SeriesInfo` (returns a
3D or 4D `PerfusionDataset` depending on the TPI count) or a list of
per-timepoint `SeriesInfo` (stacks into a 4D volume with a derived
time vector).

## Loading by Modality

### DCE-MRI

```python
from osipy.common.io import discover_dicom, load_dicom_series, load_nifti
from osipy.common.types import Modality

# From NIfTI (sidecar is picked up automatically when adjacent)
dataset = load_nifti("data/sub-01_dce.nii.gz", modality=Modality.DCE)

# From DICOM
series = discover_dicom("data/dicom/dce_series/")
dataset = load_dicom_series(
    next(s for s in series if s.role_hint == "dynamic"),
    modality=Modality.DCE,
)

print(f"Shape: {dataset.shape}")  # (x, y, z, t)
print(f"TR: {dataset.acquisition_params.tr} ms")
print(f"Flip angle: {dataset.acquisition_params.flip_angle}")
```

**Required Metadata:**
- TR (RepetitionTime)
- Flip angle (FlipAngle)

**Vendor Differences:**
- **GE**: TR in `RepetitionTime` tag (0018,0080)
- **Siemens**: TR in standard tag; may also expose CSA header
- **Philips**: Quantitative rescale lives in private tags (2005,100D/100E)

### DSC-MRI

```python
from osipy.common.io import load_nifti
from osipy.common.types import Modality

dataset = load_nifti("data/dsc_data.nii.gz", modality=Modality.DSC)
print(f"TE: {dataset.acquisition_params.te} ms")
```

**Required Metadata:**
- TE (EchoTime)
- TR (RepetitionTime)

### ASL

```python
from osipy.common.io import load_bids, load_bids_with_m0, load_asl_context
from osipy.common.types import Modality

# Load ASL with M0 calibration
asl_data, m0_data = load_bids_with_m0("data/bids_dataset/", subject="01")

# Load ASL context (control/label order)
context = load_asl_context("data/bids_dataset/", subject="01")
print(f"Volume types: {set(context)}")  # {'control', 'label'}

params = asl_data.acquisition_params
print(f"Labeling type: {params.labeling_type}")
print(f"PLD: {params.pld} ms")
print(f"Labeling duration: {params.labeling_duration} ms")
```

**Required BIDS metadata** (per [BIDS ASL appendix](https://bids-specification.readthedocs.io/en/stable/appendices/arterial-spin-labeling.html)):

For all ASL types:
- `ArterialSpinLabelingType` (PCASL, PASL, or CASL)
- `PostLabelingDelay`

For PCASL / CASL, additionally:
- `LabelingDuration`

For PASL, additionally (instead of `LabelingDuration`):
- `BolusCutOffFlag`
- when `BolusCutOffFlag` is `true`: `BolusCutOffTechnique` and `BolusCutOffDelayTime`

**ASL-BIDS Files:**
- `*_asl.nii.gz` — ASL timeseries
- `*_asl.json` — Sidecar with parameters
- `*_aslcontext.tsv` — Volume types (control, label, m0scan, deltam)
- `*_m0scan.nii.gz` — Separate M0 calibration (optional)

### IVIM

```python
from osipy.common.io import load_nifti
from osipy.common.types import Modality

dataset = load_nifti("data/dwi_ivim.nii.gz", modality=Modality.IVIM)
print(f"B-values: {dataset.acquisition_params.b_values}")
```

**Required Metadata:**
- B-values, provided in an FSL-format `*_dwi.bval` sibling file (not in
  the JSON sidecar). osipy also accepts a non-standard `DiffusionBValue`
  or `bValues` JSON key for flexibility, but BIDS-compliant writers
  must use `.bval`.

## Vendor-Specific Considerations

### GE Medical Systems

**DICOM Characteristics:**
- Manufacturer tag: "GE MEDICAL SYSTEMS"
- Private groups: 0027, 0043
- B-values: Often in private tag `(0043,1039)`. This is a multi-valued
  element; the b-value is in the first component and may be encoded
  with a vendor-specific offset (e.g. `b + 10⁶` or `b + 10⁹`). Decoding
  typically requires `value[0] % 100000` or equivalent; see
  [dcm2niix issue #149](https://github.com/rordenlab/dcm2niix/issues/149).
- ASL: 3D spiral pCASL product sequence common

Extract GE vendor metadata:

```python
import pydicom
from osipy.common.io.vendors import detect_vendor
from osipy.common.io.vendors.detection import extract_vendor_metadata

ds = pydicom.dcmread("ge_dicom.dcm")
vendor = detect_vendor(ds)  # "GE"
metadata = extract_vendor_metadata(ds)
print(f"Vendor: {metadata.vendor}")
print(f"TR: {metadata.tr} ms")
```

### Siemens

**DICOM Characteristics:**
- Manufacturer tag: "SIEMENS"
- Private groups: 0019, 0021, 0051
- CSA headers contain extended sequence information
- B-values: CSA header (Numaris VB/VD/VE) or private tag (0019,100C) on XA (where CSA is absent) — see the dcm2niix Siemens README for per-version details.
- ASL: WIP sequences common, PASL and pCASL available

**Per-timepoint exports:** Siemens TWIST/TWIST-VIBE typically export
one 3D series per timepoint with a `TT=X.Xs` suffix on the
`SeriesDescription`. `discover_dicom()` detects this pattern and tags
each frame as `dynamic_frame` with a shared `group_key`. Collect every
frame from the same `group_key` into a list and pass that list to
`load_dicom_series()` — it sorts by the embedded timing and stacks the
separate volumes into one 4D dataset.

### Philips

**DICOM Characteristics:**
- Manufacturer tag: "Philips Medical Systems" or "Philips"
- Private groups: 2001, 2005
- Enhanced DICOM format common
- B-values: Private tag (2001,1003)
- ASL: 2D EPI pCASL product sequence

**Pixel scaling:** Philips stores a private quantitative-rescale pair
`ScaleSlope (2005,100E)` and `ScaleIntercept (2005,100D)` alongside the
standard `RescaleSlope (0028,1053)` / `RescaleIntercept (0028,1052)`.
The canonical recovery of the floating-point value `FP` from the stored
value `PV` is (Chenevert et al., MAGMA 2014, PMC3998685; dcm2niix
Philips README):

```
FP = (PV + RI/RS) / SS_private
```

where `RI`, `RS` are the standard DICOM rescale tags and `SS_private` is
(2005,100E). `load_dicom_series()` currently applies a simplified
single-stage `(PV − SI_private) / SS_private` which is correct when
`RescaleSlope = 1` and `RescaleIntercept = 0` (the common case) but may
diverge for scans where the standard rescale is non-trivial. See
[GitHub issue tracker](https://github.com/OSIPI/osipy) if you observe a
scale offset with Philips data.

## Data Structures

### PerfusionDataset

All loading functions return a `PerfusionDataset`:

```python
@dataclass
class PerfusionDataset:
    data: NDArray[np.floating]      # Image data (3D or 4D)
    affine: NDArray[np.floating]    # 4x4 voxel-to-world transform
    modality: Modality              # DCE, DSC, ASL, or IVIM
    time_points: NDArray | None     # Time points for 4D data
    acquisition_params: AcquisitionParams  # Modality-specific params
    source_path: Path               # Original file path
    source_format: str              # "nifti", "dicom", or "bids"
```

### Acquisition Parameters

```python
# DCE
@dataclass
class DCEAcquisitionParams:
    tr: float
    te: float | None
    flip_angles: list[float]
    temporal_resolution: float | None

# ASL
@dataclass
class ASLAcquisitionParams:
    labeling_type: LabelingType
    pld: float | list[float]
    labeling_duration: float
    background_suppression: bool
    m0_scale: float | None

# IVIM
@dataclass
class IVIMAcquisitionParams:
    b_values: NDArray[np.floating]
    tr: float | None
    te: float | None
```

## Metadata Priority

When loading data, metadata is resolved in this priority order:

1. **Explicit `acquisition_params=` argument** (highest priority)
2. **BIDS sidecar JSON**
3. **Vendor-specific DICOM tags**
4. **Standard DICOM tags**
5. **Default values** (lowest priority)

## Loading DICOM via the CLI

The `osipy` CLI takes the same discovery + load path internally. Set
`data.format` to `dicom` (or `auto` for automatic detection):

```yaml
modality: dce
data:
  format: auto
pipeline:
  model: extended_tofts
  aif_source: population
  population_aif: parker
  acquisition:
    tr: 5.0
    flip_angles: [2, 5, 10, 15]
```

```bash
# Single-series directory
osipy config.yaml /path/to/dicom_series/

# Study directory with per-series subdirs
osipy config.yaml /path/to/study_dir/

# Per-timepoint exports — one 3D series per frame, auto-grouped by SeriesDescription
osipy config.yaml /path/to/twist_study/
```

For DCE specifically, the CLI additionally extracts VFA single-flip
series (`role_hint == "vfa"`) when present and uses them for T1 mapping
— no manual partitioning required.

## BIDS Batch Processing

osipy's BIDS reader is a hand-rolled parser over stdlib + nibabel; it
does **not** depend on `pybids`. The snippets below use `pybids` purely
as a convenience for iterating subjects — install it separately
(`pip install pybids`) if you need it.

```python
from bids import BIDSLayout
import osipy

layout = BIDSLayout("path/to/bids")

for subject in layout.get_subjects():
    dce_files = layout.get(subject=subject, suffix="dce", extension="nii.gz")
    if not dce_files:
        continue
    data = osipy.load_nifti(dce_files[0].path)
    metadata = dce_files[0].get_metadata()
```

### Multi-PLD ASL from BIDS

```python
from bids import BIDSLayout
import numpy as np

layout = BIDSLayout("path/to/bids")
asl_file = layout.get(subject="01", suffix="asl", extension="nii.gz")[0]
metadata = asl_file.get_metadata()

plds = metadata.get("PostLabelingDelay")
if isinstance(plds, list):
    plds = np.array(plds)
```

### Common BIDS Fields for Perfusion

| Field | Description | Modality |
|-------|-------------|----------|
| `RepetitionTime` | TR in seconds | All |
| `EchoTime` | TE in seconds | All |
| `FlipAngle` | Flip angle in degrees | DCE |
| `ArterialSpinLabelingType` | PASL, CASL, PCASL | ASL |
| `LabelingDuration` | Label duration in seconds | ASL |
| `PostLabelingDelay` | PLD in seconds | ASL |
| `M0Type` | `Included`, `Separate`, `Estimate`, `Absent` (BIDS 1.8+) | ASL |
| `_dwi.bval` (sibling file) | b-values, space-delimited s/mm² (not a JSON field) | IVIM/DWI |

## Troubleshooting

**"Data directory not found"**
- Check that subject/session directories exist
- osipy checks both `sub-XX/perf/` and `sub-XX/` for ASL data

**"4D data requires time_points array"**
- Ensure TR is specified in sidecar JSON
- Provide `RepetitionTime` or `RepetitionTimePreparation` in metadata

**`discover_dicom` returned no series**
- The directory contained no DICOM files, or all had unreadable headers
- Enable `INFO` logging to see the per-file rejection reasons

**Per-timepoint frames weren't grouped into one 4D volume**
- `discover_dicom()` groups by `(StudyInstanceUID, rows, columns, stripped_description)`
- If your exporter doesn't strip the `TT=X.Xs` suffix, frames may not cluster — inspect `SeriesInfo.trigger_time_hint` to verify

## See Also

- [Vendor DICOM Tag Reference](../reference/vendor-dicom-tags.md)
- [BIDS Specification](https://bids-specification.readthedocs.io/)
