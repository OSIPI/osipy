# Contributing to osipy

## Adding a New Model

One file, one decorator, done. Here's a complete DCE pharmacokinetic model:

```python
"""My custom pharmacokinetic model."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from osipy.common.convolution import convolve_aif
from osipy.dce.models.base import BasePerfusionModel, ModelParameters
from osipy.dce.models.registry import register_model

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class MyModelParams(ModelParameters):
    ktrans: float = 0.1
    ve: float = 0.2


@register_model("my_model")
class MyModel(BasePerfusionModel[MyModelParams]):

    @property
    def name(self) -> str:
        return "My Custom Model"

    @property
    def parameters(self) -> list[str]:
        return ["Ktrans", "ve"]

    @property
    def parameter_units(self) -> dict[str, str]:
        return {"Ktrans": "1/min", "ve": "fraction"}

    @property
    def reference(self) -> str:
        return "Author et al. (2025). Journal Name."

    @property
    def time_unit(self) -> str:
        return "minutes"

    def _predict(self, t, aif, params, xp):
        ktrans = params[0]  # scalar OR (n_voxels,) — broadcasting handles both
        ve = params[1]

        t_min = self._convert_time(t, xp)
        ve_safe = xp.where(ve > 0, ve, xp.asarray(1e-10))
        kep = ktrans / ve_safe
        dt = float(t_min.ravel()[1] - t_min.ravel()[0]) if t_min.size > 1 else 1.0
        irf = ktrans * xp.exp(-kep * t_min)
        return convolve_aif(aif, irf, dt=dt)

    def get_bounds(self):
        return {"Ktrans": (0.0, 5.0), "ve": (0.001, 1.0)}

    def get_initial_guess(self, ct, aif, t):
        return MyModelParams(ktrans=0.1, ve=0.2)
```

That's it. Your model works for single voxels, batch fitting, and GPU — automatically.

```python
from osipy.dce.fitting import fit_model
import osipy.dce.models.my_model  # triggers registration

result = fit_model("my_model", concentration, aif, time)
```

### How it works

You only implement `_predict()`. The base class provides everything else:

| Method | What it does | You write it? |
|--------|-------------|---------------|
| `_predict(t, aif, params, xp)` | Core math with array params | **Yes** |
| `predict(t, aif, params)` | Accepts dicts/dataclasses/arrays, calls `_predict()` | No (inherited) |
| `predict_batch(t, aif, params_batch, xp)` | Reshapes for broadcasting, calls `_predict()` | No (inherited) |

### Rules for `_predict()`

1. **Index params by row**: `params[0]`, `params[1]`, etc. — not dict keys or attributes
2. **Use `xp` for everything**: `xp.where()` not `if`, `xp.abs()` not `abs()`, `xp.maximum()` not `max()`
3. **Use `t.ravel()`** when extracting `dt` — `t` may be `(n_time, 1)` in batch mode
4. **Call convolution helpers normally**: `convolve_aif()`, `expconv()` handle both 1D and 2D

### More how-to guides

- [Add an ASL Model](how-to/add-asl-quantification-model.md)
- [Add a Deconvolution Method](how-to/add-deconvolution-method.md)
- [Add a BBB ASL Model](how-to/add-bbb-asl-model.md)

## Key Conventions

### Array module (`xp`)

All numerical code uses `xp = get_array_module()` so it runs on both CPU and GPU:

```python
from osipy.common.backend.array_module import get_array_module

def my_function(data, other):
    xp = get_array_module(data, other)
    return xp.exp(-data)
```

Direct `np.*` is fine for type hints, constants (`np.pi`), and I/O.

| Instead of | Use |
|------------|-----|
| `scipy.optimize` | `LevenbergMarquardtFitter` from `osipy.common.fitting` |
| `scipy.linalg` | `xp.linalg` |
| `scipy.integrate` | `xp.trapezoid()` |

### Time units

Public API uses **seconds**. DCE models convert to **minutes** internally via `_convert_time()`. Ktrans is in 1/min.

### Quality masks

Always filter with the quality mask — invalid voxels contain NaN/0:

```python
quality = result.quality_mask
mean_ktrans = result.parameter_maps["Ktrans"].values[quality].mean()
```

### OSIPI standards

Use CAPLEX parameter names (Ktrans, ve, vp, CBF). Include OSIPI codes in docstrings. See [OSIPI Standards](explanation/osipi-standards.md).

## Testing

```bash
pytest                             # All tests
pytest tests/unit/                 # Unit tests only
pytest tests/unit/dce/test_models.py  # Specific file
```

### Writing model tests

```python
import numpy as np
import pytest
from osipy.dce.models import ToftsModel

class TestMyModel:
    @pytest.fixture
    def model(self):
        return ToftsModel()

    def test_predict_shape(self, model):
        time = np.linspace(0, 300, 60)
        aif = np.exp(-time / 30)
        result = model.predict(time, aif, {"Ktrans": 0.1, "ve": 0.2})
        assert result.shape == time.shape

    @pytest.mark.parametrize("ktrans", [0.01, 0.1, 0.5])
    def test_ktrans_range(self, model, ktrans):
        time = np.linspace(0, 300, 60)
        aif = np.exp(-time / 30)
        result = model.predict(time, aif, {"Ktrans": ktrans, "ve": 0.2})
        assert np.isfinite(result).all()
```

## Development Setup

```bash
git clone https://github.com/OSIPI/osipy.git
cd osipy
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -e ".[dev]"
```

### Code quality

```bash
ruff check .          # Lint
mypy osipy            # Type check
```

### Docstrings

NumPy-style with OSIPI references:

```python
def fit_model(model_name: str, concentration: NDArray, ...) -> DCEFitResult:
    """Fit a pharmacokinetic model (OSIPI: M.IC1.004).

    Parameters
    ----------
    model_name : str
        Registered model name (e.g., "tofts", "extended_tofts").

    References
    ----------
    .. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
    """
```

### Validating against DROs

OSIPI provides Digital Reference Objects for validation:

```python
dro = load_dro("path/to/dro")
result = fit_model("extended_tofts", dro.concentration, dro.aif, dro.time)
ktrans_error = np.abs(result.parameter_maps["Ktrans"].values - dro.ktrans_true)
```

See the [OSIPI DCE Challenge (OSF)](https://osf.io/u7a6f/) for DCE DROs and the [OSIPI IVIM Code Collection](https://github.com/OSIPI/TF2.4_IVIM-MRI_CodeCollection) for IVIM test data.
