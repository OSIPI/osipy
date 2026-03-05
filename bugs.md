# OSIPY — Bug Report (Deep Analysis v2)

> **Analysis Date:** 2026-03-05  
> **Analyzed Version:** 0.1.1 (from `_version.py`)  
> **Method:** Static code analysis of all ~140 Python source files  
> **Scope:** Full re-analysis post PR #85 overhaul + new bug discovery

---

## Summary Table

| ID | Severity | File | One-Line Description | Status |
|---|---|---|---|---|
| **BUG-01** | 🔴 HIGH | `array_module.py:40` | NumPy compat shim is a self-assignment no-op | Fix PR submitted ✅ |
| **BUG-05** | 🟢 LOW | `conftest.py:368` | `pld + label_duration` result discarded | Re-verified ✅ |
| **BUG-06** | 🟡 MEDIUM | `conv.py:266-276` | `uconv()` uses O(n²) Python loops | Re-verified ✅ |
| **BUG-07** | 🟡 MEDIUM | `batch.py:246-249` | Broad string matching catches non-memory GPU errors | Re-verified ✅ |
| **BUG-08** | 🟢 LOW | `config.py:142-149` | GPU cache skipped when env var forces CPU | Re-verified ✅ |
| **BUG-09** | 🟡 MEDIUM | `conv.py:146-174` | Non-uniform convolution is O(n² log n) | Re-verified ✅ |
| **BUG-10** | 🟢 LOW | `_version.py` | Version 0.1.1 may be behind PyPI 0.1.2 | Re-verified ✅ |
| **BUG-12** | 🟡 MEDIUM | `fitting.py:418-419` | Silent exception swallow hides R² failures | Re-verified ✅ |
| **BUG-15** |  MEDIUM | `normalization.py:215` | `np.errstate` has no effect on CuPy arrays | Re-verified ✅ |
| **BUG-17** | 🟡 MEDIUM | `fitting.py:281` | Quality mask excludes valid zero-parameter voxels | Re-verified ✅ |
| **BUG-18** | 🟢 LOW | `cbf.py:259` | PASL `ti1_s` fallback has ambiguous units | Re-verified ✅ |
| **BUG-19** | 🟢 LOW | `svd.py:183` | Threshold comparison readable but semantically inverted | Re-verified ✅ |
| **BUG-20** | 🟢 LOW | `correction.py:382` | `**kwargs` to dataclass gives unhelpful error on typo | Re-verified ✅ |
| **BUG-21** | 🔴 HIGH | `least_squares.py:384`| Silently swallows `LinAlgError` on matrix inversion failure | 🆕 NEW |
| **BUG-22** | 🟡 MEDIUM| `least_squares.py:165`| LM loop breaking doesn't update `r2` or `converged` properly | 🆕 NEW |

---

## BUG-01 — NumPy Compatibility Shim is a Self-Assignment No-Op

**File:** [`array_module.py`](file:///c:/Users/devgu/OneDrive/Desktop/osipi_gsoc_repos/osipi_gsoc_repo_7_osipy_updated/osipy/osipy/common/backend/array_module.py) — **Line 38-40**  
**Severity:** 🔴 HIGH

### Background

In NumPy 2.0, the function `np.trapz()` was renamed to `np.trapezoid()`. The codebase uses `xp.trapezoid()` everywhere (e.g., in `dsc/parameters/maps.py` lines 317 and 324: `xp.trapezoid(aif, time)`). The shim at the top of `array_module.py` is supposed to ensure that if someone runs this code on NumPy < 2.0 (where only `np.trapz` exists), it creates a `np.trapezoid` alias pointing to the old `np.trapz`.

### Current Code

```python
# Line 38-40 of array_module.py
# NumPy 2.0 renamed trapz -> trapezoid. Ensure compatibility with NumPy <2.0.
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapezoid  # type: ignore[attr-defined]
```

### What's Wrong

This is a **self-assignment**. The code says: "if `np.trapezoid` doesn't exist, set `np.trapezoid` to `np.trapezoid`". But we just checked that `np.trapezoid` **doesn't exist** — so the right-hand side `np.trapezoid` will raise an `AttributeError`.

The fix **should** be `np.trapezoid = np.trapz` — aliasing the OLD name to the NEW name.

**Contrast this with the CuPy shim** on lines 67-69 which is done **correctly**:

```python
# CuPy < 14 uses trapz; ensure trapezoid alias exists
if not hasattr(cp, "trapezoid") and hasattr(cp, "trapz"):
    cp.trapezoid = cp.trapz  # ← Correct: aliases old name to new name
```

### Why It's Masked

The project requires `numpy>=2.0.0` in `pyproject.toml`, so in practice `np.trapezoid` **always** exists, meaning the `if not hasattr` block never executes. The shim is dead-but-broken code.

### Impact

- If someone installs this package in an environment with NumPy 1.x (bypassing dependency checks), any call to `xp.trapezoid()` would crash with `AttributeError: module 'numpy' has no attribute 'trapezoid'`
- Harmless in practice because the dependency pin prevents NumPy <2.0

### Suggested Fix

**Option A — Fix the shim:**
```python
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore[attr-defined]
```

**Option B — Delete the shim entirely (recommended):**
Since `pyproject.toml` requires `numpy>=2.0.0`, this backward compatibility code is unnecessary. `np.trapezoid` always exists. Delete lines 38-40.

```diff
-# NumPy 2.0 renamed trapz -> trapezoid. Ensure compatibility with NumPy <2.0.
-if not hasattr(np, "trapezoid"):
-    np.trapezoid = np.trapezoid  # type: ignore[attr-defined]
```

---


## BUG-05 — Dead Expression in ASL Test Fixture

**File:** `tests/conftest.py` — **Line 368**  
**Severity:** 🟢 LOW (test code only)

### Current Code

```python
pld + label_duration    # Computed, never stored
```

### What's Wrong

The sum is computed but never assigned to a variable. It was likely intended to compute a total delay time for the DRO (Digital Reference Object) generation.

### Suggested Fix

Either assign it: `total_delay = pld + label_duration` and use it, or delete the line.

---

## BUG-06 — `uconv()` Uses O(n²) Pure Python Loops

**File:** [`conv.py`](file:///c:/Users/devgu/OneDrive/Desktop/osipi_gsoc_repos/osipi_gsoc_repo_7_osipy_updated/osipy/osipy/common/convolution/conv.py) — **Lines 266-276**  
**Severity:** 🟡 MEDIUM

### Background

`uconv()` performs convolution on **uniform** time grids. The docstring says "Optimized convolution for uniform time sampling" — but the implementation uses a pure Python double-loop, which is extremely slow.

### Current Code

```python
def uconv(f, h, dt):
    """Optimized convolution for uniform time sampling..."""   # ← Docstring says "optimized"
    xp = get_array_module(f)
    # ...
    result = xp.zeros(n, dtype=f.dtype)

    for i in range(1, n):                              # ← O(n) outer loop
        total = 0.5 * f[0] * h[i]                     # Trapezoidal weight at endpoint
        for j in range(1, i):                          # ← O(n) inner loop
            total += f[j] * h[i - j]                   # Pure Python scalar ops
        total += 0.5 * f[i] * h[0]
        result[i] = total * dt

    return result
```

### What's Wrong

**O(n²) time complexity:** For `n = 1000` time points (typical for a clinical DCE-MRI temporal resolution), this executes ~500,000 Python-level multiply+add operations. For `n = 10000`, it's ~50 million operations. Python loops are ~100× slower than vectorized NumPy/CuPy operations.

On a uniform grid, convolution can be done using `np.convolve` (O(n²) but in C) or `np.fft.fft`-based convolution (O(n log n)).

### Impact

- Clinical DCE-MRI datasets with many time points will be very slow
- Completely negates the GPU acceleration design — even if `xp` is CuPy, the Python loop serializes everything

### Suggested Fix

Replace with vectorized trapezoidal convolution:

```python
def uconv(f, h, dt):
    xp = get_array_module(f)
    f = xp.asarray(f)
    h = xp.asarray(h)
    n = len(f)
    if n <= 1:
        return xp.zeros(n, dtype=f.dtype)

    # Trapezoidal weights: half-weight at endpoints
    w = xp.ones(n, dtype=f.dtype)
    w[0] = 0.5
    w[-1] = 0.5   # Technically not needed for causal conv but mathematically correct

    # Full convolution using vectorized ops, then truncate to causal part
    full_conv = xp.convolve(f * w, h, mode='full')[:n] * dt

    return full_conv
```

Or even simpler — use FFT:
```python
from numpy.fft import fft, ifft
F = fft(f * w, n=2*n)
H = fft(h, n=2*n)
result = ifft(F * H).real[:n] * dt
```

---

## BUG-07 — Overly Broad GPU Error Detection Masks Real Errors

**File:** [`batch.py`](file:///c:/Users/devgu/OneDrive/Desktop/osipi_gsoc_repos/osipi_gsoc_repo_7_osipy_updated/osipy/osipy/common/backend/batch.py) — **Lines 243-268**  
**Severity:** 🟡 MEDIUM

### Background

The `BatchProcessor.map()` method catches exceptions during GPU processing and tries to detect memory errors so it can fall back to CPU. The detection works by converting the exception message to a string and checking for keywords.

### Current Code

```python
except Exception as e:
    error_str = str(e).lower()
    is_memory_error = any(
        phrase in error_str
        for phrase in ["out of memory", "memory", "cuda", "allocation"]
    )
    if is_memory_error and self.auto_fallback:
        # ... silently falls back to CPU ...
```

### What's Wrong

The phrases `"memory"` and `"cuda"` are extremely broad:

- `"memory"` matches: `"shared memory not supported"`, `"memory order not valid"`, `"virtual memory exhausted"`, etc. — these aren't OOM errors
- `"cuda"` matches: `"CUDA driver version is too old"`, `"CUDA error: invalid device ordinal"`, `"CUDA initialization failure"` — these aren't memory issues and shouldn't trigger silent fallback

**Real example:** If a user has a CUDA version mismatch, the error `"CUDA driver version is insufficient for CUDA runtime version"` contains `"cuda"` → the code silently falls back to CPU, and the user never knows their GPU isn't actually working.

### Suggested Fix

Use more specific patterns and the actual exception types:

```python
except Exception as e:
    is_memory_error = isinstance(e, MemoryError)

    if not is_memory_error:
        # CuPy-specific memory error detection
        error_str = str(e).lower()
        is_memory_error = any(
            phrase in error_str
            for phrase in ["out of memory", "cudaerrormemoryal location"]
        )

    if is_memory_error and self.auto_fallback:
        # ... fall back to CPU ...
    else:
        raise  # Don't swallow non-memory GPU errors
```

---

## BUG-08 — GPU Availability Cache Skipped When Env Var Forces CPU

**File:** [`config.py`](file:///c:/Users/devgu/OneDrive/Desktop/osipi_gsoc_repos/osipi_gsoc_repo_7_osipy_updated/osipy/osipy/common/backend/config.py) — **Lines 142-149**  
**Severity:** 🟢 LOW

### Current Code

```python
def is_gpu_available() -> bool:
    global _gpu_available_cache

    # Check environment variable for forced CPU mode
    if os.environ.get("OSIPY_FORCE_CPU", "0") == "1":
        return False                       # ← Returns early, never sets cache

    if _gpu_available_cache is not None:
        return _gpu_available_cache
    # ... expensive GPU detection logic ...
```

### What's Wrong

When `OSIPY_FORCE_CPU=1`, the function returns `False` directly **without** caching the result. This means every call to `is_gpu_available()` re-reads the environment variable. While `os.environ.get()` is cheap, this breaks the caching contract described in the docstring ("The result is cached after the first call for performance").

More importantly, if someone later **removes** `OSIPY_FORCE_CPU`, the function will proceed to the expensive GPU detection code below (CuPy import, device count, test allocation) on every subsequent call, because `_gpu_available_cache` was never set.

### Suggested Fix

Set the cache before returning:

```diff
     if os.environ.get("OSIPY_FORCE_CPU", "0") == "1":
+        _gpu_available_cache = False
         return False
```

---

## BUG-09 — Non-Uniform Convolution is O(n² log n)

**File:** [`conv.py`](file:///c:/Users/devgu/OneDrive/Desktop/osipi_gsoc_repos/osipi_gsoc_repo_7_osipy_updated/osipy/osipy/common/convolution/conv.py) — **Lines 146-174**  
**Severity:** 🟡 MEDIUM

### Background

`_conv_nonuniform()` handles convolution on non-uniform time grids where FFT isn't applicable (time steps aren't equal).

### Current Code

```python
for i in range(1, n):                      # O(n)
    total = 0.0
    for j in range(i):                     # O(n)
        # ...
        h0 = _interp_value(h, t, tau0)     # _interp_value uses xp.searchsorted → O(log n)
        h1 = _interp_value(h, t, tau1)     # O(log n)
        total += (f0 * h1 + f1 * h0) * dt_j / 2.0
    result[i] = total
```

### What's Wrong

Triple-nested complexity: O(n) × O(n) × O(log n) = **O(n² log n)**. For clinical data with 60 time points, this means ~3,600 iterations × 2 binary searches each = ~7,200 `searchsorted` calls. For 1000 time points, it's ~1 million × 2 = 2 million binary searches.

### Impact

Practically unusable for large non-uniform time series. The function is called as a fallback from `conv()` when the time grid isn't uniform.

### Suggested Fix

Pre-compute a lookup table or use `xp.interp()` (vectorized interpolation) instead of per-point `_interp_value()`:

```python
def _conv_nonuniform(f, h, t):
    xp = get_array_module(f)
    n = len(t)
    result = xp.zeros(n, dtype=f.dtype)

    for i in range(1, n):
        # Vectorize: evaluate h at all (t[i] - t[0..i]) points at once
        tau = t[i] - t[:i+1]
        h_vals = xp.interp(tau, t, h)       # Vectorized interpolation, O(n log n) total
        # Trapezoidal integration of f[0..i] * h_vals
        integrand = f[:i+1] * h_vals
        result[i] = float(xp.trapezoid(integrand, t[:i+1]))

    return result
```

This reduces complexity to **O(n² log n) → O(n²)** by eliminating per-point binary search.

---

## BUG-10 — Version Mismatch

**File:** `osipy/_version.py`  
**Severity:** 🟢 LOW

### What's Wrong

`__version__ = "0.1.1"` in `_version.py` may lag behind PyPI (0.1.2). This can happen if the release workflow bumps the version on the release branch but the change doesn't get merged back to main.

### Suggested Fix

Check the release workflow (`release.yaml`) to ensure the version bump step creates a PR back to main after release. If it already does, this may just need a manual sync.

---


## BUG-12 — Silent Exception Swallow Hides R² Computation Failures

**File:** [`fitting.py`](file:///c:/Users/devgu/OneDrive/Desktop/osipi_gsoc_repos/osipi_gsoc_repo_7_osipy_updated/osipy/osipy/dce/fitting.py) — **Lines 403-419**  
**Severity:** 🟡 MEDIUM

### Background

After fitting DCE pharmacokinetic models (Tofts, Extended Tofts, 2CXM, Patlak), the code computes an R² (coefficient of determination) goodness-of-fit map. This map tells users how well the model fits each voxel — critical for quality control in clinical research.

### Current Code

```python
def _compute_r_squared_vectorized(ct_4d, bound_model, param_maps, quality_mask, xp):
    # ... setup ...
    try:
        ct_pred = bound_model.predict_array_batch(params_batch, xp)
        ct_pred = ct_pred.T
        residuals = ct_masked - ct_pred
        ss_res = xp.sum(residuals**2, axis=1)
        ct_mean = xp.mean(ct_masked, axis=1, keepdims=True)
        ss_tot = xp.sum((ct_masked - ct_mean) ** 2, axis=1)
        r2_values = xp.where(ss_tot > 1e-10, 1.0 - ss_res / ss_tot, 0.0)
        r_squared[quality_mask] = r2_values

    except Exception:
        pass                    # ← Catches EVERYTHING, does NOTHING

    return r_squared            # Returns zeros if exception occurred
```

### What's Wrong

`except Exception: pass` is one of the worst Python anti-patterns. It catches:
- `TypeError`, `ValueError` — bugs in the code (shape mismatches, wrong dtypes)
- `MemoryError` — GPU/CPU out of memory
- `RuntimeError` — CuPy kernel errors
- Any future bug introduced anywhere in the prediction pipeline

When any of these happens, `r_squared` remains all zeros, and the caller (`_fit_model_impl`) has no idea. The user sees an R² map of all zeros and might think their model fits are terrible, when actually the R² code itself crashed.

### Suggested Fix

Log the error and let users know something went wrong:

```python
    except Exception:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            "R² computation failed; returning zero map. "
            "Parameter maps are still valid.",
            exc_info=True,       # Logs the full traceback at WARNING level
        )

    return r_squared
```

Or, better, catch only the specific exceptions you expect (e.g., `xp.linalg.LinAlgError`, `ValueError`) and let real bugs propagate.

---


## BUG-15 — `np.errstate` Has No Effect on CuPy Arrays

**File:** [`normalization.py`](file:///c:/Users/devgu/OneDrive/Desktop/osipi_gsoc_repos/osipi_gsoc_repo_7_osipy_updated/osipy/osipy/dsc/normalization.py) — **Line 215**  
**Severity:** 🟡 MEDIUM

### Background

`normalize_to_white_matter()` divides parameter values by a reference value. When the reference value is zero (which shouldn't happen due to earlier validation, but defensive coding is good), this would produce `inf` or `nan`. The code uses `np.errstate` to suppress the division warning.

### Current Code

```python
# Normalize the map
with np.errstate(divide="ignore", invalid="ignore"):
    normalized_values = values / ref_value
```

### What's Wrong

`np.errstate` is a **NumPy-only** context manager that controls NumPy's internal floating-point error flags. When `values` is a **CuPy array** (GPU execution path), this context manager has absolutely no effect because:

1. CuPy operations run on the GPU via CUDA kernels
2. CuPy doesn't read or respect NumPy's error state flags
3. GPU floating-point behavior follows IEEE 754 rules independently

So on the GPU path, if `ref_value` is zero, CuPy will produce `inf`/`nan` **without** the division being suppressed. The `nan_to_num` call on line 218 catches these, so no crash occurs — but the intent of the errstate (suppressing warnings) is not achieved on GPU.

### Suggested Fix

Use an xp-agnostic approach that avoids division by zero entirely:

```diff
-    with np.errstate(divide="ignore", invalid="ignore"):
-        normalized_values = values / ref_value
+    # xp-agnostic: avoid division by zero instead of suppressing warnings
+    normalized_values = xp.where(
+        ref_value != 0, values / ref_value, 0.0
+    )
```

Or, since `ref_value` is a scalar that's already validated to be > 0 (the function raises `DataValidationError` if all WM values are zero), you could just remove the `errstate` entirely — it's protecting against something that can't happen:

```diff
-    with np.errstate(divide="ignore", invalid="ignore"):
-        normalized_values = values / ref_value
+    normalized_values = values / ref_value
```

---

## BUG-16 — Double Memory Allocation in `signal_to_concentration()`

**File:** [`signal_to_conc.py`](file:///c:/Users/devgu/OneDrive/Desktop/osipi_gsoc_repos/osipi_gsoc_repo_7_osipy_updated/osipy/osipy/dce/concentration/signal_to_conc.py) — **Lines 250, 253**  
**Severity:** 🟡 MEDIUM (memory waste)

### Background

This function is the entry point for converting DCE-MRI signal to contrast agent concentration. It pre-allocates an output array, passes it to the converter, and gets back a result.

### Current Code

```python
# Line 250: Allocate a zeros array the same shape as signal (4D)
concentration = xp.zeros_like(signal)        # ← Allocation #1: e.g., 64×64×20×60 float64 = 37 MB

# Line 252-264: Call the converter, which creates its OWN array internally
converter = get_concentration_model(method)
concentration = converter(                  # ← Allocation #2: converter returns new array
    signal, s0, t1_pre, tr, cos_a, sin_a,
    relaxivity, baseline_frames,
    concentration,                          # ← Passes allocation #1, but converter ignores it (BUG-04)
    xp,
)
```

### What's Wrong

This is the **caller-side** of BUG-04. The flow is:

1. Allocate 37 MB of zeros → `concentration` points to this array
2. Pass it to `_convert_spgr()` as the `concentration` parameter
3. `_convert_spgr()` internally does `concentration = delta_r1 / relaxivity` — creates a **new** array, doesn't touch the zeros
4. Returns the new array
5. Line 253 assigns the return value to `concentration`, replacing the pointer to the zeros array
6. The zeros array has zero references → garbage collected

**Net effect:** 37 MB allocated, never written to, immediately freed.

### Suggested Fix

See the combined fix in BUG-04. Remove line 250 and the `concentration` parameter from the converter call.

---

## BUG-17 — Quality Mask Logic Excludes Valid Zero-Parameter Voxels

**File:** [`fitting.py`](file:///c:/Users/devgu/OneDrive/Desktop/osipi_gsoc_repos/osipi_gsoc_repo_7_osipy_updated/osipy/osipy/dce/fitting.py) — **Line 278-281**  
**Severity:** 🟡 MEDIUM

### Background

After DCE model fitting, the code builds a "quality mask" that identifies which voxels were successfully fitted. This mask is used downstream for statistics and R² computation.

### Current Code

```python
# Build quality mask (voxels with valid fits)
# Standardized to > 0 for all models
first_param_name = next(iter(param_maps.keys()))      # Gets whatever parameter comes first
first_map = param_maps[first_param_name]
first_values = xp.asarray(first_map.values)
quality_mask = xp.isfinite(first_values) & (first_values > 0)
```

### What's Wrong

The quality mask requires the **first parameter in the dictionary** to be strictly `> 0`. This relies on dictionary insertion order (guaranteed in Python 3.7+) and assumes the first parameter is always positive.

**Problem scenarios:**

1. **With `_DelayAwareModel`**: When `fit_delay=True` is used, the model wraps the base model and appends `"delay"` to the parameters list. But `_DelayAwareModel.parameters` is `[*self._base.parameters, "delay"]`, so the first parameter is still from the base model (e.g., `"Ktrans"`). However, if someone subclasses and puts delay first, voxels with zero delay would be excluded.

2. **Edge case with Patlak model**: Patlak's first parameter is `Ktrans`, which is > 0 for valid fits. But for 2CXM, if the parameter order changes in a future refactor, this breaks silently.

The comment says "Standardized to > 0 for all models" — but this is only true for Ktrans. Using the **first** parameter is fragile.

### Suggested Fix

Explicitly check the parameter that's known to be positive:

```python
# Build quality mask using the primary perfusion parameter
# Ktrans / PS / Fp are always > 0 for valid fits
primary_params = ["Ktrans", "PS", "Fp"]
primary = next((p for p in primary_params if p in param_maps), next(iter(param_maps)))
primary_values = xp.asarray(param_maps[primary].values)
quality_mask = xp.isfinite(primary_values) & (primary_values > 0)
```

---

## BUG-18 — PASL `ti1_s` Fallback Has Ambiguous Units

**File:** [`cbf.py`](file:///c:/Users/devgu/OneDrive/Desktop/osipi_gsoc_repos/osipi_gsoc_repo_7_osipy_updated/osipy/osipy/asl/quantification/cbf.py) — **Line 259**  
**Severity:** 🟢 LOW

### Current Code

```python
def _quantify_pasl(delta_m, m0, params):
    ti_s = params.pld / 1000.0           # ms → s ✓ (all params are in ms)
    t1b_s = params.t1_blood / 1000.0     # ms → s ✓

    # Bolus duration (TI1 for QUIPSS)
    ti1_s = params.bolus_duration / 1000.0 if params.bolus_duration is not None else 0.7
    #       ↑ Converts ms → s ✓                                                    ↑
    #                                                                    Is this 0.7 s or 0.7 ms?
```

### What's Wrong

The variable name `ti1_s` clearly means "TI1 in seconds". The value `0.7` is **correct** — the QUIPSS II default TI1 is 700 ms = 0.7 seconds. But the inconsistency with the conversion pattern on the left side (`/ 1000.0`) makes it confusing:

- Left side of `if`: `params.bolus_duration` comes in **milliseconds** and is divided by 1000 to get seconds
- Right side of `else`: `0.7` is directly in **seconds** — no conversion needed

A future maintainer might think "all the other params come in ms, this fallback should also be in ms" and change it to `700`, which would make TI1 = 700 seconds and produce wildly incorrect CBF values.

### Suggested Fix

Add a clarifying comment:

```python
ti1_s = (
    params.bolus_duration / 1000.0
    if params.bolus_duration is not None
    else 0.7  # Default 700 ms = 0.7 s (QUIPSS II, Alsop et al. 2015)
)
```

---

## BUG-19 — SVD Threshold Comparison Direction (Clarity Issue)

**File:** [`svd.py`](file:///c:/Users/devgu/OneDrive/Desktop/osipi_gsoc_repos/osipi_gsoc_repo_7_osipy_updated/osipy/osipy/dsc/deconvolution/svd.py) — **Lines 183, 203, 447**  
**Severity:** 🟢 LOW (code clarity, mathematically correct)

### Current Code

```python
S_inv = xp.where(s_thresh < S, 1.0 / S, 0.0)     # Keep S values that are > s_thresh
```

### What's Wrong

This is **mathematically correct** but reads backwards. SVD truncation convention is: "zero out singular values **below** the threshold." The natural reading is:

```python
# Natural reading: keep singular values ABOVE the threshold
S_inv = xp.where(S > s_thresh, 1.0 / S, 0.0)
```

`s_thresh < S` is equivalent to `S > s_thresh`, but the former reads as "is the threshold less than S?" which is indirect. This appears in three places (lines 183, 203, 447).

### Suggested Fix

Swap the comparison for readability:

```diff
-    S_inv = xp.where(s_thresh < S, 1.0 / S, 0.0)
+    S_inv = xp.where(S > s_thresh, 1.0 / S, 0.0)
```

---

## BUG-20 — BSW `**kwargs` Passed Directly to Dataclass Constructor

**File:** [`correction.py`](file:///c:/Users/devgu/OneDrive/Desktop/osipi_gsoc_repos/osipi_gsoc_repo_7_osipy_updated/osipy/osipy/dsc/leakage/correction.py) — **Lines 380-403**  
**Severity:** 🟢 LOW

### Current Code

```python
@register_leakage_corrector("bsw")
class BSWCorrector(BaseLeakageCorrector):
    def correct(self, delta_r2, aif, time, mask=None, **kwargs):
        params = LeakageCorrectionParams(method="bsw", **kwargs)    # ← Any typo crashes
        return correct_leakage(delta_r2, aif, time, mask, params)
```

### What's Wrong

If a user passes a typo like `use_t1_corection=True` (missing 'r'), Python raises:
```
TypeError: LeakageCorrectionParams.__init__() got an unexpected keyword argument 'use_t1_corection'
```

This is a confusing error — it comes from the `@dataclass` constructor and doesn't tell the user what the valid fields are.

### Suggested Fix

Validate kwargs explicitly:

```python
def correct(self, delta_r2, aif, time, mask=None, **kwargs):
    valid_keys = {f.name for f in LeakageCorrectionParams.__dataclass_fields__.values()} - {"method"}
    invalid = set(kwargs.keys()) - valid_keys
    if invalid:
        raise DataValidationError(
            f"Unknown parameters: {invalid}. Valid: {sorted(valid_keys)}"
        )
    params = LeakageCorrectionParams(method="bsw", **kwargs)
    return correct_leakage(delta_r2, aif, time, mask, params)
```

---

## Dead Code Summary

| Location | Code | Lines | What Replaced It |
|----------|------|-------|-----------------|
| `array_module.py` | `np.trapezoid = np.trapezoid` | 38-40 | Not needed — `numpy>=2.0` guarantees `np.trapezoid` exists |
| `conv.py` | `df / dt_safe; dh / dt_safe` | 141-142 | Never used — function uses `_interp_value()` instead |
| `expconv.py` | `xp.zeros(n, dtype=f.dtype)` | 278 | Never used — function uses numerical differentiation instead of manual loop |
| `svd.py` | `_apply_svd_truncation_xp()` | 586-625 | Inline vectorized solve in `_vectorized_svd_solve()` line 452 |
| `svd.py` | `_compute_oscillation_index_xp()` | 628-661 | Batch version `_compute_oscillation_index_batch()` at line 487 |
| `correction.py` | `delta_r2.shape[:-1]` | 291 | Was probably going to be `spatial_shape =` but spatial_shape not needed |
| `conftest.py` | `pld + label_duration` | 368 | Likely meant `total_delay = pld + label_duration` but never used |
| `signal_to_conc.py` | `concentration = xp.zeros_like(signal)` | 250 | Converter creates its own array — pre-allocation is unused |
 
 