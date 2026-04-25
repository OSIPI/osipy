# How to Add a Custom DCE Pharmacokinetic Model

Add a new pharmacokinetic model to osipy in a single file using the `@register_model` decorator.

## Steps

### 1. Create your model file

Create a new file in `osipy/dce/models/`, e.g., `osipy/dce/models/my_model.py`:

!!! example "Define a custom pharmacokinetic model"

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
        """Parameters for my custom model."""

        ktrans: float = 0.1
        ve: float = 0.2


    @register_model("my_model")
    class MyModel(BasePerfusionModel[MyModelParams]):
        """My custom pharmacokinetic model."""

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

The key method is `_predict(self, t, aif, params, xp)`:

- **`params`** is an array: `params[0]` is Ktrans, `params[1]` is ve (matching `self.parameters` order)
- **`xp`** is the array module (numpy or cupy) — use it for all math
- Works for **single voxels and batches automatically** — when `params[i]` is a scalar you get a single curve; when it's `(n_voxels,)` you get `(n_time, n_voxels)` output via broadcasting
- Use `xp.where()` instead of `if`, `xp.abs()` instead of `abs()`
- Use `t.ravel()` when extracting `dt` since `t` may be `(n_time, 1)` in batch mode

You do **not** need to implement `predict()` or `predict_batch()` — the base class handles parameter conversion and shape setup.

### 2. Use it

That's it. Your model is now available everywhere:

!!! example "Use the registered model via fit_model()"

    ```python
    from osipy.dce.fitting import fit_model
    from osipy.dce.models import get_model, list_models

    # Import your model file to trigger registration
    import osipy.dce.models.my_model

    # Verify it's registered
    print(list_models())  # [..., 'my_model', ...]

    # Use via fit_model()
    result = fit_model("my_model", concentration, aif, time)

    # Or get the model instance directly
    model = get_model("my_model")
    ct = model.predict(time, aif, {"Ktrans": 0.1, "ve": 0.2})
    ```

The DCE pipeline also works automatically with any registered model:

!!! example "Use custom model in DCE pipeline"

    ```python
    from osipy.pipeline import DCEPipeline, DCEPipelineConfig

    config = DCEPipelineConfig(model="my_model")
    pipeline = DCEPipeline(config)
    result = pipeline.run(dce_data, time)
    ```

## What you get for free

- Registry-driven lookup via `get_model("my_model")`
- Unified fitting via `fit_model("my_model", ...)`
- Pipeline integration (no code changes needed)
- GPU acceleration if you use `xp` operations in `_predict()`
- Batch processing — base class `predict_batch()` calls your `_predict()` with broadcasting
- Quality masks, R² maps, and fitting statistics
