import hashlib
import urllib.request
import zipfile
from pathlib import Path
from typing import NamedTuple

from original.fitting.TF_reference.segmented_IVIMfit import segmented_IVIM_fit

import numpy as np
import pytest

import osipy
from osipy import IVIMFitParams
from osipy.common.validation import create_synthetic_dro
from osipy.ivim import FittingMethod
from osipy.ivim.fitting.segmented_wls import fit_segmented_IVIM_wrapper
from tests.conftest import generate_synthetic_ivim_data

ZENODO_URL = "https://zenodo.org/api/records/14605039/files/OSIPI_TF24_data_phantoms.zip/content"
ZENODO_MD5 = "e7b3fe1d811a7a45c5aaf6c604c82793"

TOLERANCE = 1e-4

class IVIMResults(NamedTuple):
    D: np.ndarray
    Dp: np.ndarray
    f: np.ndarray


CACHE_DIR = Path.home() / ".cache" / "osipy" / "osipi_tf24"


def _load_test_data():
    data_dir = CACHE_DIR / "data"

    if (data_dir / "Data" / "abdomen.nii.gz").exists():
        print(f"Using cached data in: {data_dir}")
    else:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = CACHE_DIR / "OSIPI_TF24_data_phantoms.zip"

        # Download
        urllib.request.urlretrieve(ZENODO_URL, zip_path)

        # Verify checksum
        md5 = hashlib.md5()
        with open(zip_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                md5.update(chunk)
        if md5.hexdigest() != ZENODO_MD5:
            zip_path.unlink()
            raise AssertionError("Checksum mismatch")

        # Unpack
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(data_dir)

        print(f"Data extracted to: {data_dir}")

    data_path = data_dir / "Data/abdomen.nii.gz"
    bval_path = data_dir / "Data/abdomen.bval"
    mask_path = data_dir / "Data/mask_abdomen_homogeneous.nii.gz"
    dataset = osipy.load_nifti(data_path)
    mask = osipy.load_nifti(mask_path)
    mask = mask.data.astype(bool)

    with open(bval_path, "r") as f:
        bvals = [int(x) for x in f.read().split()]

    bvals = np.array(bvals)

    # Average the signals taken at the same b-value
    unique_bvals = np.unique(bvals)
    averaged_signals = []
    for b in unique_bvals:
        indices = np.where(bvals == b)[0]
        averaged_signal = np.mean(dataset.data[..., indices], axis=-1)
        averaged_signals.append(averaged_signal)

    sorted_indices = np.argsort(unique_bvals)
    bvals = unique_bvals[sorted_indices]
    dataset.data = np.stack([averaged_signals[i] for i in sorted_indices], axis=-1)

    print(f"Number of b values: {len(bvals)}")
    print(f"data shape: {dataset.shape}")
    print(f"Mask shape: {mask.shape}")

    return dataset.data, mask, bvals

def _load_synthetic_test_data():
    synthetic_data = generate_synthetic_ivim_data()

    # float64: the generator returns float32, the reference fit works in float64
    signal = synthetic_data["signal"].astype(np.float64)
    bvals = synthetic_data["b_values"].astype(np.float64)

    return signal, synthetic_data["mask"], bvals



@pytest.mark.parametrize(
    "load_data",
    [_load_test_data, _load_synthetic_test_data],
    ids=["abdomen_phantom", "synthetic"],
)
def test_segmented_ivim_same_as_reference(load_data):
    """
    Test that the segmented IVIM fit implementation in this package produces the same results as the reference implementation.
    """
    data, mask, bvals = load_data()
    signals = _normalize_to_b0(data[mask], bvals)

    new_result = _fit_ivim_new(bvals, signals)
    reference_result = _fit_ivim_reference(bvals, signals)

    # Compare results
    np.testing.assert_allclose(new_result.D, reference_result.D, rtol=TOLERANCE)
    np.testing.assert_allclose(new_result.Dp, reference_result.Dp, rtol=TOLERANCE)
    np.testing.assert_allclose(new_result.f, reference_result.f, rtol=TOLERANCE)


def _normalize_to_b0(signals, bvalues):
    """Divide each voxel's signal by its b=0 signal (the segmented fit expects S(b=0)=1)."""
    b0 = signals[:, bvalues == 0].mean(axis=-1, keepdims=True)
    return signals / b0


def _fit_ivim_reference(bvalues, signals) -> IVIMResults:
    """
    Fit IVIM model using the reference implementation.
    This function uses the implementation form  the TF2.4 IVIM codecollection.
    """
    out = [segmented_IVIM_fit(bvalues, s) for s in
           signals]  # each -> (D, f, Dp)
    D, f, Dp = (np.array(x) for x in zip(*out))
    return IVIMResults(D=D, Dp=Dp, f=f)


def _fit_ivim_new(bvalues, signals) -> IVIMResults:
    maps = fit_segmented_IVIM_wrapper(signals, np.asarray(bvalues))

    # Maps are (n_voxels, 1, 1); flatten back to one value per voxel.
    return IVIMResults(
        D=maps["D"].values.reshape(-1),
        Dp=maps["D*"].values.reshape(-1),
        f=maps["f"].values.reshape(-1),
    )
