# How to Add a BBB ASL Water Exchange Model

Implement Blood-Brain Barrier (BBB) water exchange measurement using multi-echo ASL
and a two-compartment model. This guide extends osipy across four registry extension
points: quantification model, ATT model, difference method, and result dataclass.

## Prerequisites

- Familiar with [Add ASL Model](add-asl-quantification-model.md) (simpler template)
- Understand [ASL Physics](../explanation/asl-physics.md), especially the single-compartment assumption and its limitations
- Understand the [registry architecture](../explanation/architecture.md)

## Background: BBB ASL Physics

Standard ASL assumes instantaneous water exchange between intravascular and
extravascular compartments. In reality, water crosses the BBB at a finite rate
characterized by the exchange rate $K_w$ (units: 1/s). Multi-echo ASL exploits
the T2 difference between blood and tissue water to separate the two compartments.

### Two-compartment signal model

The multi-echo ASL difference signal at echo time TE and post-labeling delay PLD is:

$$
\Delta S(\text{TE}, \text{PLD}) = \Delta M_{iv}(\text{PLD}) \cdot e^{-\text{TE}/T_{2,blood}} + \Delta M_{ev}(\text{PLD}) \cdot e^{-\text{TE}/T_{2,tissue}}
$$

where $\Delta M_{iv}$ and $\Delta M_{ev}$ are the intravascular and extravascular
labeled magnetization, governed by the coupled ODEs:

$$
\frac{d \Delta M_{iv}}{dt} = -\left(\frac{1}{T_{1,blood}} + K_w\right) \Delta M_{iv} + \frac{CBF}{\lambda} \cdot \alpha \cdot M_0
$$

$$
\frac{d \Delta M_{ev}}{dt} = K_w \cdot \Delta M_{iv} - \frac{1}{T_{1,tissue}} \Delta M_{ev}
$$

The permeability-surface area product for water relates to $K_w$ via:

$$
PS_w = K_w \cdot v_{iv}
$$

### Key parameters

| Parameter | Symbol | Units | Typical value (3T) |
|-----------|--------|-------|-------------------|
| CBF | CBF | mL/100g/min | 40-80 (gray matter) |
| Water exchange rate | Kw | 1/s | 2-4 (healthy brain) |
| Water PS product | PS_w | mL/100g/min | ~120 |
| Intravascular volume fraction | v_iv | mL/100mL | 2-5 |
| T2 of blood | T2_blood | ms | ~150 |
| T2 of tissue | T2_tissue | ms | ~80 |

!!! warning "Simplified for documentation"

    The implementation below is pedagogical. A production BBB ASL model would
    need Bayesian priors to regularize the ill-conditioned multi-parameter fit,
    proper noise propagation, and validation against Monte Carlo simulations.

## Step 1: Define BBBASLParams

Extend `ASLQuantificationParams` with multi-echo and exchange parameters.

!!! example "BBB ASL parameter dataclass"

    ```python
    """BBB ASL water exchange model parameters and implementation."""

    from dataclasses import dataclass, field
    from typing import TYPE_CHECKING, Any

    import numpy as np

    from osipy.asl.quantification.cbf import ASLQuantificationParams
    from osipy.common.backend.array_module import get_array_module

    if TYPE_CHECKING:
        from numpy.typing import NDArray


    @dataclass
    class BBBASLParams(ASLQuantificationParams):
        """Parameters for BBB water exchange ASL quantification.

        Extends standard ASL parameters with multi-echo and exchange
        rate parameters.

        Attributes
        ----------
        echo_times : list[float]
            Echo times in milliseconds for multi-echo acquisition.
        t2_blood : float
            T2 of arterial blood in milliseconds. Default 150 ms at 3T.
        t2_tissue : float
            T2 of gray matter tissue in milliseconds. Default 80 ms at 3T.
        kw_init : float
            Initial guess for water exchange rate Kw in 1/s.

        References
        ----------
        .. [1] OSIPI ASL Lexicon, https://osipi.github.io/ASL-Lexicon/
        .. [2] St. Lawrence KS, Owen D, Wang DJJ. MRM 2012;67(5):1275-1284. doi:10.1002/mrm.23104
        """

        echo_times: list[float] = field(default_factory=lambda: [10.0, 30.0, 50.0])
        t2_blood: float = 150.0   # ms at 3T
        t2_tissue: float = 80.0   # ms at 3T
        kw_init: float = 3.0      # 1/s initial guess
    ```

## Step 2: Implement BBB Quantification Model

Register a quantification model that separates intravascular and extravascular
compartments via T2-weighted least squares, then computes CBF.

All ASL models inherit from `BaseASLModel`, which provides the full
`BaseSignalModel` interface (`parameters`, `parameter_units`, `get_bounds()`,
`name`, `reference`) plus the ASL-specific `labeling_type` property and
optional `quantify()` method.

!!! example "BBB multi-TE quantification model"

    ```python
    from osipy.asl.quantification.base import BaseASLModel
    from osipy.asl.quantification.registry import register_quantification_model


    @register_quantification_model("bbb_multi_te")
    class BBBMultiTEModel(BaseASLModel):
        """BBB water exchange quantification from multi-echo ASL.

        Separates intravascular and extravascular signal using T2
        decay differences, then quantifies CBF and Kw from the
        two-compartment model.

        References
        ----------
        .. [1] OSIPI ASL Lexicon, https://osipi.github.io/ASL-Lexicon/
        .. [2] St. Lawrence KS, Owen D, Wang DJJ. MRM 2012;67(5):1275-1284. doi:10.1002/mrm.23104
        .. [3] Wengler K et al. NeuroImage 2019;189:401-414.
        """

        @property
        def name(self) -> str:
            return "bbb_multi_te"

        @property
        def parameters(self) -> list[str]:
            return ["CBF"]

        @property
        def parameter_units(self) -> dict[str, str]:
            return {"CBF": "mL/100g/min"}

        @property
        def reference(self) -> str:
            return "St. Lawrence KS, Owen D, Wang DJJ. MRM 2012;67(5):1275-1284. doi:10.1002/mrm.23104"

        @property
        def labeling_type(self) -> str:
            return "pcasl"

        def get_bounds(self) -> dict[str, tuple[float, float]]:
            return {"CBF": (0.0, 200.0)}

        def quantify(self, delta_m, m0, params):
            """Compute CBF from multi-echo ASL difference signal.

            Parameters
            ----------
            delta_m : NDArray
                Multi-echo difference signal, shape (..., n_echoes).
            m0 : NDArray
                Equilibrium magnetization, shape (...).
            params : BBBASLParams
                Parameters including echo_times, t2_blood, t2_tissue.

            Returns
            -------
            NDArray
                CBF values in mL/100g/min, shape (...).
            """
            xp = get_array_module(delta_m, m0)

            te_ms = xp.asarray(params.echo_times, dtype=xp.float64)
            n_echoes = len(te_ms)

            # Build T2 weighting matrix: [exp(-TE/T2b), exp(-TE/T2t)]
            # Shape: (n_echoes, 2)
            t2_matrix = xp.stack([
                xp.exp(-te_ms / params.t2_blood),
                xp.exp(-te_ms / params.t2_tissue),
            ], axis=1)

            # Reshape delta_m for least-squares: (..., n_echoes) -> (n_voxels, n_echoes)
            spatial_shape = delta_m.shape[:-1]
            dm_2d = delta_m.reshape(-1, n_echoes)  # (n_voxels, n_echoes)

            # Solve for [delta_M_iv, delta_M_ev] per voxel via least squares
            # A @ x = b  =>  x = (A^T A)^{-1} A^T b
            ata = t2_matrix.T @ t2_matrix        # (2, 2)
            ata_inv = xp.linalg.inv(ata)          # (2, 2)
            proj = ata_inv @ t2_matrix.T          # (2, n_echoes)

            # Separate compartments: (2, n_echoes) @ (n_echoes, n_voxels) -> (2, n_voxels)
            components = proj @ dm_2d.T
            delta_m_iv = components[0, :]  # intravascular
            delta_m_ev = components[1, :]  # extravascular

            # Total perfusion signal (sum of compartments)
            delta_m_total = delta_m_iv + delta_m_ev

            # Reshape M0 for division
            m0_flat = m0.reshape(-1)

            # Standard pCASL CBF equation applied to total signal
            pld_s = params.pld / 1000.0
            tau_s = params.label_duration / 1000.0
            t1b_s = params.t1_blood / 1000.0

            numerator = 6000.0 * params.partition_coefficient * delta_m_total * xp.exp(pld_s / t1b_s)
            denominator = (
                2.0 * params.labeling_efficiency * t1b_s * m0_flat
                * (1.0 - xp.exp(-tau_s / t1b_s))
            )
            cbf = numerator / (denominator + 1e-10)
            cbf = xp.where(~xp.isfinite(cbf), 0.0, cbf)

            return cbf.reshape(spatial_shape)
    ```

!!! tip "Why `xp.linalg.inv()` instead of scipy?"

    osipy bans scipy for numerical operations. The `xp.linalg` module is
    available in both numpy and cupy, keeping this GPU-compatible. For the
    small 2x2 system here, direct inversion is efficient and numerically stable.

## Step 3: Implement BBB Multi-PLD Model

Register a multi-PLD model that extends the Buxton kinetic model with water exchange.
This model inherits from `BaseASLModel` and provides a forward prediction via its
`parameters`, `get_bounds()`, and optionally works with `BoundASLModel` + fitter.

!!! example "BBB multi-PLD model with water exchange"

    ```python
    from osipy.asl.quantification.base import BaseASLModel
    from osipy.asl.quantification.registry import register_quantification_model


    @register_quantification_model("bbb_multi_pld")
    class BBBMultiPLDModel(BaseASLModel):
        """Multi-PLD ASL model with BBB water exchange.

        Extends the Buxton general kinetic model with a water exchange
        rate Kw as an additional fitted parameter. Supports iterative
        fitting via BoundASLModel + LevenbergMarquardtFitter.

        References
        ----------
        .. [1] OSIPI ASL Lexicon, https://osipi.github.io/ASL-Lexicon/
        .. [2] Buxton RB et al. MRM 1998;40(3):383-396.
        .. [3] St. Lawrence KS, Owen D, Wang DJJ. MRM 2012;67(5):1275-1284. doi:10.1002/mrm.23104
        """

        @property
        def name(self) -> str:
            return "bbb_multi_pld"

        @property
        def parameters(self) -> list[str]:
            return ["CBF", "ATT", "Kw"]

        @property
        def parameter_units(self) -> dict[str, str]:
            return {"CBF": "mL/100g/min", "ATT": "s", "Kw": "1/s"}

        @property
        def reference(self) -> str:
            return "St. Lawrence KS, Owen D, Wang DJJ. MRM 2012;67(5):1275-1284. doi:10.1002/mrm.23104"

        @property
        def labeling_type(self) -> str:
            return "pcasl"

        def get_bounds(self) -> dict[str, tuple[float, float]]:
            return {
                "CBF": (0.0, 200.0),
                "ATT": (0.1, 5.0),
                "Kw": (0.1, 10.0),
            }

        # No quantify() override — requires iterative fitting via BoundASLModel
    ```

!!! note "Forward model for fitting"

    This model does not override `quantify()` because it has 3 free parameters
    and requires iterative fitting. To use it with the shared fitter infrastructure,
    create a `BoundASLModel` that wraps it and implements `predict_array_batch()`.
    See the [multi-PLD binding adapter](../reference/asl/quantification/binding.md) for details.

## Step 4: Multi-TE Difference Method

Register a difference method that preserves the echo-time dimension for
multi-echo processing.

!!! example "Multi-TE pairwise difference method"

    ```python
    from osipy.asl.quantification.cbf import register_difference_method
    from osipy.common.exceptions import DataValidationError


    @register_difference_method("multi_te_pairwise")
    def _difference_multi_te_pairwise(controls, labels, control_indices,
                                       label_indices, asl_data, xp):
        """Pair-wise subtraction preserving echo-time dimension.

        For multi-echo ASL data with shape (x, y, z, n_echoes, n_volumes),
        computes control-label differences while keeping the echo dimension.

        Parameters
        ----------
        controls : NDArray
            Control volumes, shape (..., n_echoes, n_controls).
        labels : NDArray
            Label volumes, shape (..., n_echoes, n_labels).
        control_indices : list[int]
            Indices of control volumes.
        label_indices : list[int]
            Indices of label volumes.
        asl_data : NDArray
            Raw ASL data (unused, required by signature).
        xp : module
            Array module (numpy or cupy).

        Returns
        -------
        NDArray
            Mean difference, shape (..., n_echoes).

        Raises
        ------
        DataValidationError
            If no control-label pairs found.
        """
        n_pairs = min(len(control_indices), len(label_indices))

        if n_pairs == 0:
            raise DataValidationError("No control-label pairs found for multi-TE data")

        differences = []
        for i in range(n_pairs):
            diff = controls[..., i] - labels[..., i]
            differences.append(diff)

        # Stack and average: preserves echo dimension
        return xp.mean(xp.stack(differences, axis=-1), axis=-1)
    ```

## Step 5: BBB Result Dataclass

Define a result container with `ParameterMap` instances for all BBB ASL outputs.

!!! example "BBB ASL result dataclass"

    ```python
    from dataclasses import dataclass

    from osipy.common.parameter_map import ParameterMap


    @dataclass
    class BBBASLResult:
        """Result of BBB ASL water exchange quantification.

        All ParameterMap fields use ASCII-only names, symbols, and units
        per osipy convention.

        Attributes
        ----------
        cbf_map : ParameterMap
            CBF in mL/100g/min (OSIPI ASL Lexicon).
        kw_map : ParameterMap
            Water exchange rate Kw in 1/s.
        ps_w_map : ParameterMap
            Water permeability-surface area product in mL/100g/min.
        v_iv_map : ParameterMap
            Intravascular volume fraction in mL/100mL.
        quality_mask : NDArray
            Mask of reliable voxels.

        References
        ----------
        .. [1] OSIPI ASL Lexicon, https://osipi.github.io/ASL-Lexicon/
        """

        cbf_map: ParameterMap
        kw_map: ParameterMap
        ps_w_map: ParameterMap
        v_iv_map: ParameterMap
        quality_mask: "NDArray"
    ```

    Build the result from fitted maps:

    ```python
    import numpy as np
    from osipy.common.parameter_map import ParameterMap

    # Ensure 3D shapes for ParameterMap validation
    cbf_3d = cbf_values[..., np.newaxis] if cbf_values.ndim == 2 else cbf_values

    result = BBBASLResult(
        cbf_map=ParameterMap(
            name="CBF", symbol="CBF", units="mL/100g/min",
            values=cbf_3d, affine=np.eye(4),
            quality_mask=quality_mask,
        ),
        kw_map=ParameterMap(
            name="Kw", symbol="Kw", units="1/s",
            values=kw_3d, affine=np.eye(4),
            quality_mask=quality_mask,
        ),
        ps_w_map=ParameterMap(
            name="PS_w", symbol="PS_w", units="mL/100g/min",
            values=ps_w_3d, affine=np.eye(4),
            quality_mask=quality_mask,
        ),
        v_iv_map=ParameterMap(
            name="v_iv", symbol="v_iv", units="mL/100mL",
            values=v_iv_3d, affine=np.eye(4),
            quality_mask=quality_mask,
        ),
        quality_mask=quality_mask,
    )
    ```

## Step 6: Verification

Import the module to trigger registration, then verify all components are available.

!!! example "Verify registration and run smoke test"

    ```python
    # Import to trigger registration
    import osipy.asl.quantification.bbb_asl  # your module

    from osipy.asl.quantification.registry import list_quantification_models
    from osipy.asl.quantification.cbf import list_difference_methods

    # Check registries
    assert "bbb_multi_te" in list_quantification_models()
    assert "bbb_multi_pld" in list_quantification_models()
    assert "multi_te_pairwise" in list_difference_methods()

    # Smoke test with synthetic data
    import numpy as np

    n_echoes = 3
    delta_m = np.random.rand(8, 8, 4, n_echoes) * 50.0
    m0 = np.full((8, 8, 4), 1000.0)

    params = BBBASLParams(
        echo_times=[10.0, 30.0, 50.0],
        t2_blood=150.0,
        t2_tissue=80.0,
        pld=1800.0,
        label_duration=1800.0,
    )

    from osipy.asl.quantification.registry import get_quantification_model

    model = get_quantification_model("bbb_multi_te")
    cbf = model.quantify(delta_m, m0, params)
    assert cbf.shape == (8, 8, 4)
    print(f"CBF range: {cbf.min():.1f} - {cbf.max():.1f} mL/100g/min")

    # Verify isinstance checks
    from osipy.common.models.base import BaseComponent, BaseSignalModel

    assert isinstance(model, BaseComponent)
    assert isinstance(model, BaseSignalModel)
    assert model.reference
    assert model.get_bounds() == {"CBF": (0.0, 200.0)}
    ```

## Pipeline Integration

Use the BBB multi-PLD model with the existing multi-PLD quantification pipeline:

!!! example "Use BBB model in multi-PLD pipeline"

    ```python
    from osipy.asl.quantification import quantify_multi_pld

    result = quantify_multi_pld(
        data, m0, params, mask=mask,
    )
    ```

## What You Get for Free

By using the `BaseASLModel` hierarchy, your BBB ASL model automatically gets:

- **Registry lookup** — `get_quantification_model("bbb_multi_te")` works everywhere
- **Unified interface** — `isinstance(model, BaseSignalModel)` and `isinstance(model, BaseComponent)` both work
- **Parameter introspection** — `model.parameters`, `model.parameter_units`, `model.get_bounds()` available for all models
- **Multi-PLD integration** — works with `quantify_multi_pld()` via `BoundASLModel` + fitter
- **GPU acceleration** — all code uses `xp = get_array_module()`, runs on CUDA if available
- **Quality masks** — standard CBF range filtering (0-200 mL/100g/min)
- **BIDS export** — `ParameterMap` instances export via the standard pipeline

## References

1. St. Lawrence KS, Owen D, Wang DJJ. A two-stage approach for measuring vascular water exchange and arterial transit time by diffusion-weighted perfusion MRI. *Magn Reson Med*. 2012;67(5):1275-1284.
2. Wengler K, Bangiyev L, Canli T, et al. 3D MRI of whole-brain water permeability with intrinsic diffusivity encoding of arterial labeled spin (IDEALS). *NeuroImage*. 2019;189:401-414.
3. Shao X, Ma SJ, Casey M, et al. Mapping water exchange across the blood-brain barrier using 3D diffusion-prepared arterial spin labeled perfusion MRI. *Magn Reson Med*. 2019;81(5):3065-3079.
4. OSIPI ASL Lexicon, https://osipi.github.io/ASL-Lexicon/
5. Suzuki Y et al. *MRM* 2024;91(5):1743-1760. doi:10.1002/mrm.29815
