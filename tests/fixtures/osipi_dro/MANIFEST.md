# OSIPI DRO Data Manifest

This directory contains Digital Reference Objects (DROs) for testing osipy pipelines.

## DCE-MRI Data

**Source**: [OSIPI TF6.2 DCE-DSC-MRI Challenges](https://github.com/OSIPI/TF6.2_DCE-DSC-MRI_Challenges)

| File | Description | Shape | Units | Range |
|------|-------------|-------|-------|-------|
| `Ktrans.nii` | Volume transfer constant | 256x256x16 | min^-1 | 0.0 - 11.3 |
| `kep.nii` | Rate constant (Ktrans/ve) | 256x256x16 | min^-1 | 0.0 - 3.6 |
| `vp.nii` | Plasma volume fraction | 256x256x16 | - | 0.0 - 0.83 |
| `R10.nii` | Pre-contrast R1 | 256x256x16 | s^-1 | 0.0 - 0.89 |
| `AIF.npz` | Arterial Input Function | - | - | - |
| `Ea.npz` | Enhancement data | - | - | - |

**Note**: ve (extravascular extracellular volume fraction) can be derived as `Ktrans / kep`.

## DSC-MRI Data

**Source**: Synthetic DRO with physiologically realistic values

| File | Description | Shape | Units | Range |
|------|-------------|-------|-------|-------|
| `cbv.npy` | Cerebral Blood Volume | 64x64x10 | mL/100g | 0.0 - 4.3 |
| `cbf.npy` | Cerebral Blood Flow | 64x64x10 | mL/100g/min | 0.0 - 63.0 |
| `mtt.npy` | Mean Transit Time | 64x64x10 | s | 0.0 - 20.0 |
| `aif.npy` | Arterial Input Function (delta R2*) | 60 | s^-1 | - |
| `time.npy` | Time vector | 60 | s | 0.0 - 60.0 |
| `mask.npy` | Brain mask | 64x64x10 | bool | - |

**Reference Regions**:
- Gray matter: CBV=4.0, CBF=60, MTT~4s
- White matter: CBV=2.0, CBF=25, MTT~5s

## ASL Data

**Source**: Synthetic DRO with ISMRM consensus values

| File | Description | Shape | Units | Range |
|------|-------------|-------|-------|-------|
| `cbf.npy` | Cerebral Blood Flow | 64x64x10 | mL/100g/min | 0.0 - 63.0 |
| `att.npy` | Arterial Transit Time | 64x64x10 | s | 0.3 - 1.9 |
| `m0.npy` | Equilibrium magnetization | 64x64x10 | a.u. | 0 - 1030 |
| `plds.npy` | Post-labeling delays | 6 | s | 0.25 - 2.0 |
| `mask.npy` | Brain mask | 64x64x10 | bool | - |

**Reference Values** (ISMRM Consensus):
- Gray matter: CBF=60, ATT=1.2s
- White matter: CBF=25, ATT=1.8s

## IVIM Data

**Source**: Synthetic DRO with literature values

| File | Description | Shape | Units | Range |
|------|-------------|-------|-------|-------|
| `d.npy` | Diffusion coefficient | 64x64x10 | mm^2/s | 0 - 1.05e-3 |
| `d_star.npy` | Pseudo-diffusion coefficient | 64x64x10 | mm^2/s | 0 - 16e-3 |
| `f.npy` | Perfusion fraction | 64x64x10 | - | 0 - 0.17 |
| `b_values.npy` | B-values | 10 | s/mm^2 | 0 - 1000 |
| `mask.npy` | Tissue mask | 64x64x10 | bool | - |

**Reference Values**:
- Region 1: D=1.0e-3, D*=10e-3, f=0.10
- Region 2: D=0.8e-3, D*=15e-3, f=0.15

## Usage

```python
import numpy as np
import nibabel as nib
from pathlib import Path

# Load DCE data
dce_dir = Path("tests/fixtures/osipi_dro/dce")
ktrans = nib.load(dce_dir / "Ktrans.nii").get_fdata()

# Load DSC data
dsc_dir = Path("tests/fixtures/osipi_dro/dsc")
cbv = np.load(dsc_dir / "cbv.npy")

# Load ASL data
asl_dir = Path("tests/fixtures/osipi_dro/asl")
cbf = np.load(asl_dir / "cbf.npy")

# Load IVIM data
ivim_dir = Path("tests/fixtures/osipi_dro/ivim")
d = np.load(ivim_dir / "d.npy")
```

## License

- DCE data: Apache 2.0 (OSIPI)
- DSC/ASL/IVIM synthetic data: Apache 2.0 (osipy)
