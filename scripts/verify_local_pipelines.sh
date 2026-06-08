#!/usr/bin/env bash
#
# verify_local_pipelines.sh — run the osipy CLI against the local datasets and
# verify every pipeline still produces valid output maps.
#
# This reproduces the 8 passing "localdata" checks used as the dead-code-removal
# regression baseline, but drives them through the *real* `osipy` command-line
# entry point (config YAML + data path) so you can verify them yourself:
#
#   5 end-to-end CLI pipelines        3 DICOM vendor-load checks
#   ---------------------------       --------------------------
#   1. DCE   (Clinical_P1, DICOM)     6. DICOM load: GE
#   2. IVIM  (brain, NIfTI)           7. DICOM load: Siemens
#   3. IVIM  (abdomen, NIfTI)         8. DICOM load: Philips
#   4. ASL   (ExploreASL, BIDS)
#   5. ASL   (OSIPI Dataset1, BIDS)
#
# (DSC has no local dataset, so it is covered only by the unit/integration suite.)
#
# Usage:
#   scripts/verify_local_pipelines.sh
#
# Environment overrides:
#   OSIPY_DATA   Path to the data directory   (default: /home/ltorres/projects/osipy/data)
#   OUTPUT_DIR   Where to write outputs       (default: a fresh temp dir)
#   PYTHON       Python interpreter           (default: .venv/bin/python)
#   OSIPY        osipy CLI executable         (default: .venv/bin/osipy)
#
set -u  # (intentionally NOT -e: we want to run every pipeline and tally failures)

# --- locations -------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OSIPY_DATA="${OSIPY_DATA:-/home/ltorres/projects/osipy/data}"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
OSIPY="${OSIPY:-$REPO_ROOT/.venv/bin/osipy}"
OUTPUT_DIR="${OUTPUT_DIR:-$(mktemp -d -t osipy_verify.XXXXXX)}"
CONFIG_DIR="$OUTPUT_DIR/configs"
mkdir -p "$CONFIG_DIR"

# Fall back to PATH executables if the venv ones are absent.
[ -x "$PYTHON" ] || PYTHON="python3"
[ -x "$OSIPY" ]  || OSIPY="osipy"

PASS=0
FAIL=0
declare -a RESULTS

echo "=================================================================="
echo " osipy local pipeline verification"
echo "   repo:    $REPO_ROOT"
echo "   data:    $OSIPY_DATA"
echo "   python:  $PYTHON"
echo "   osipy:   $OSIPY"
echo "   output:  $OUTPUT_DIR"
echo "=================================================================="

if [ ! -d "$OSIPY_DATA" ]; then
    echo "ERROR: data directory not found: $OSIPY_DATA" >&2
    echo "Set OSIPY_DATA to point at your data checkout." >&2
    exit 2
fi

# --- helpers ---------------------------------------------------------------

# assert_valid_nifti <file>  -> 0 if a loadable, non-empty, not-all-NaN NIfTI
assert_valid_nifti() {
    "$PYTHON" - "$1" <<'PY'
import sys
import numpy as np
import nibabel as nib
path = sys.argv[1]
img = nib.load(path)
data = np.asarray(img.dataobj)
assert data.size > 0, f"{path}: empty array"
assert np.isfinite(data).any(), f"{path}: entirely NaN/Inf"
PY
}

# record <name> <status 0|1> <detail>
record() {
    local name="$1" status="$2" detail="$3"
    if [ "$status" -eq 0 ]; then
        PASS=$((PASS + 1))
        RESULTS+=("PASS  $name")
        echo "  -> PASS: $name"
    else
        FAIL=$((FAIL + 1))
        RESULTS+=("FAIL  $name  ($detail)")
        echo "  -> FAIL: $name  ($detail)"
    fi
}

# run_cli_pipeline <name> <config.yaml> <data_path> <expected_file...>
run_cli_pipeline() {
    local name="$1" config="$2" data_path="$3"; shift 3
    local expected=("$@")
    local out="$OUTPUT_DIR/${name}"
    rm -rf "$out"

    echo
    echo "------------------------------------------------------------------"
    echo "[$name] osipy $(basename "$config") $data_path -o $out"
    echo "------------------------------------------------------------------"

    if [ ! -e "$data_path" ]; then
        record "$name" 1 "data not found: $data_path"
        return
    fi

    "$OSIPY" "$config" "$data_path" -o "$out"
    local rc=$?
    if [ $rc -ne 0 ]; then
        record "$name" 1 "CLI exited $rc"
        return
    fi

    local missing=""
    for f in "${expected[@]}"; do
        if [ ! -f "$out/$f" ]; then
            missing="$missing $f"
            continue
        fi
        if [[ "$f" == *.nii.gz ]] && ! assert_valid_nifti "$out/$f"; then
            missing="$missing $f(invalid)"
        fi
    done

    if [ -n "$missing" ]; then
        record "$name" 1 "missing/invalid outputs:$missing"
    else
        echo "  outputs: ${expected[*]}"
        record "$name" 0 ""
    fi
}

# --- 1. DCE (Clinical_P1 DICOM) -------------------------------------------
cat > "$CONFIG_DIR/dce.yaml" <<'YAML'
modality: dce
pipeline:
  model:
    method: extended_tofts
  t1_mapping_method:
    method: vfa
    fit_method: linear
  concentration:
    method: spgr
  aif_source: population
  population_aif:
    name: parker
  acquisition:
    tr: 5.0
    flip_angles: [5, 10, 15, 20, 25, 30]
    baseline_frames: 5
    relaxivity: 4.5
data:
  format: auto
YAML
run_cli_pipeline "1_dce_clinical_p1" "$CONFIG_DIR/dce.yaml" \
    "$OSIPY_DATA/dce/Clinical_P1/Visit1/09-15-1904-BRAINRESEARCH-89964" \
    osipy_run.json ktrans.nii.gz ve.nii.gz quality_mask.nii.gz

# --- 2 & 3. IVIM (brain + abdomen NIfTI) ----------------------------------
for region in brain abdomen; do
    cat > "$CONFIG_DIR/ivim_${region}.yaml" <<YAML
modality: ivim
pipeline:
  fitting:
    method: segmented
    b_threshold: 200.0
  model:
    model: biexponential
  normalize_signal: true
data:
  format: nifti
  b_values_file: $OSIPY_DATA/ivim/${region}/${region}.bval
YAML
done
run_cli_pipeline "2_ivim_brain" "$CONFIG_DIR/ivim_brain.yaml" \
    "$OSIPY_DATA/ivim/brain/brain.nii.gz" \
    osipy_run.json d.nii.gz f.nii.gz d_star.nii.gz
run_cli_pipeline "3_ivim_abdomen" "$CONFIG_DIR/ivim_abdomen.yaml" \
    "$OSIPY_DATA/ivim/abdomen/abdomen.nii.gz" \
    osipy_run.json d.nii.gz

# --- 4 & 5. ASL (BIDS) ----------------------------------------------------
# ExploreASL: PCASL, PLD 2.025 s, LD 1.65 s   (read from sidecar -> ms)
cat > "$CONFIG_DIR/asl_explore.yaml" <<'YAML'
modality: asl
pipeline:
  labeling_scheme: pcasl
  pld: 2025.0
  label_duration: 1650.0
  m0:
    method: single
  difference:
    method: pairwise
  quantification:
    mode: single_pld
  label_control_order: label_first
data:
  format: bids
  subject: Sub1
YAML
run_cli_pipeline "4_asl_exploreasl" "$CONFIG_DIR/asl_explore.yaml" \
    "$OSIPY_DATA/asl/ExploreASL_TestDataSet/rawdata" \
    osipy_run.json cbf.nii.gz

# OSIPI Dataset1: PCASL, PLD 2.025 s, LD 1.8 s
cat > "$CONFIG_DIR/asl_osipi1.yaml" <<'YAML'
modality: asl
pipeline:
  labeling_scheme: pcasl
  pld: 2025.0
  label_duration: 1800.0
  m0:
    method: single
  difference:
    method: pairwise
  quantification:
    mode: single_pld
  label_control_order: label_first
data:
  format: bids
  subject: "001"
YAML
run_cli_pipeline "5_asl_osipi_dataset1" "$CONFIG_DIR/asl_osipi1.yaml" \
    "$OSIPY_DATA/asl/OSIPI_TESTING/OSIPI_Dataset1/rawdata" \
    osipy_run.json cbf.nii.gz

# --- 6, 7, 8. DICOM vendor-load checks (GE / Siemens / Philips) -----------
# These exercise the DICOM loader (discover + load_dicom_series) directly,
# matching tests/unit/common/test_dicom.py::test_load_vendor_dce.
verify_vendor_dicom() {
    local vendor="$1"
    local vendor_dir="$OSIPY_DATA/test_dicom/${vendor}/dce"
    echo
    echo "------------------------------------------------------------------"
    echo "[${vendor}] DICOM load check: $vendor_dir"
    echo "------------------------------------------------------------------"
    if [ ! -d "$vendor_dir" ]; then
        record "${vendor}_dicom_load" 1 "data not found: $vendor_dir"
        return
    fi
    if OSIPY_VENDOR_DIR="$vendor_dir" "$PYTHON" - <<'PY'
import os
import sys
import numpy as np
from pathlib import Path
from osipy.common.dataset import PerfusionDataset
from osipy.common.io.discovery import discover_dicom, load_dicom_series

vendor_dir = Path(os.environ["OSIPY_VENDOR_DIR"])
# leaf = first directory under vendor_dir that contains .dcm files
leaf = None
for p in sorted(vendor_dir.rglob("*.dcm")):
    leaf = p.parent
    break
if leaf is None:
    print(f"  no .dcm files under {vendor_dir}", file=sys.stderr)
    sys.exit(1)

series_list = discover_dicom(leaf)
assert series_list, f"no series discovered under {leaf}"
ds = load_dicom_series(series_list[0])
assert isinstance(ds, PerfusionDataset), type(ds)
data = np.asarray(ds.data)
assert data.size > 0, "empty volume"
assert data.ndim >= 3, f"ndim={data.ndim}"
assert np.isfinite(data).any(), "all NaN/Inf"
assert ds.acquisition_params is not None, "no acquisition params"
print(f"  loaded {leaf.name}: shape={data.shape}")
PY
    then
        record "${vendor}_dicom_load" 0 ""
    else
        record "${vendor}_dicom_load" 1 "loader raised / empty volume"
    fi
}
verify_vendor_dicom ge
verify_vendor_dicom siemens
verify_vendor_dicom philips

# --- summary ---------------------------------------------------------------
echo
echo "=================================================================="
echo " SUMMARY"
echo "=================================================================="
for r in "${RESULTS[@]}"; do echo "  $r"; done
echo "------------------------------------------------------------------"
echo "  $PASS passed, $FAIL failed   (outputs under $OUTPUT_DIR)"
echo "=================================================================="

[ "$FAIL" -eq 0 ]
