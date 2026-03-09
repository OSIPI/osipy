"""Base class for pharmacokinetic perfusion models.

This module provides the abstract base class for all DCE/DSC
pharmacokinetic models following the OSIPI CAPLEX lexicon.

References
----------
.. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
.. [2] Tofts PS et al. J Magn Reson Imaging 1999;10(3):223-232.
.. [3] Sourbron SP, Buckley DL. MRM 2011;66(3):735-745.
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from osipy.common.backend.array_module import get_array_module, to_numpy
from osipy.common.models.base import BaseSignalModel

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@dataclass
class ModelParameters:
    """Base class for model parameter dataclasses."""

    pass


P = TypeVar("P", bound=ModelParameters)


class BasePerfusionModel(BaseSignalModel, Generic[P]):
    """Abstract base class for perfusion pharmacokinetic models.

    All DCE/DSC models must inherit from this class and implement
    the required methods.

    Notes
    -----
    Type parameter ``P`` is a ``ModelParameters`` dataclass
    type for model-specific parameters.

    Examples
    --------
    >>> class ToftsParams(ModelParameters):
    ...     ktrans: float
    ...     ve: float
    >>> class ToftsModel(BasePerfusionModel[ToftsParams]):
    ...     # Implementation
    ...     pass

    References
    ----------
    All implementations MUST include primary literature reference.
    """

    @property
    def time_unit(self) -> str:
        """Return the time unit used internally by this model.

        Returns
        -------
        str
            ``"seconds"`` or ``"minutes"``.
        """
        return "seconds"

    def _convert_time(
        self, t: "NDArray[np.floating[Any]]", xp: Any
    ) -> "NDArray[np.floating[Any]]":
        """Convert time array based on model's time_unit.

        Parameters
        ----------
        t : NDArray[np.floating]
            Time points in seconds.
        xp : module
            Array module (numpy or cupy).

        Returns
        -------
        NDArray[np.floating]
            Time in model's native unit.
        """
        if self.time_unit == "minutes":
            return t / 60.0
        return t

    @property
    @abstractmethod
    def name(self) -> str:
        """Return human-readable model name.

        Returns
        -------
        str
            Model name for display and logging.
        """
        ...

    @property
    @abstractmethod
    def parameters(self) -> list[str]:
        """Return list of parameter names following OSIPI terminology.

        Returns
        -------
        list[str]
            Parameter names (e.g., ['Ktrans', 've', 'vp']).
        """
        ...

    # parameter_units and reference are inherited from BaseSignalModel

    def predict(
        self,
        t: "NDArray[np.floating[Any]]",
        aif: "NDArray[np.floating[Any]]",
        params: "Any",
        xp: Any = None,
    ) -> "NDArray[np.floating[Any]]":
        """Predict tissue concentration given parameters.

        Accepts both dict/dataclass and array parameters transparently.

        Parameters
        ----------
        t : NDArray[np.floating]
            Time points in seconds.
        aif : NDArray[np.floating]
            Arterial input function concentration.
        params : NDArray or dict or ModelParameters
            Parameter array ``(n_params,)`` / ``(n_params, n_voxels)``,
            or a dict / dataclass that will be converted automatically.
        xp : module, optional
            Array module (numpy or cupy). Inferred from *t* when omitted.

        Returns
        -------
        NDArray[np.floating]
            Predicted tissue concentration.
        """
        if xp is None:
            xp = get_array_module(t, aif)
        if not hasattr(params, "ndim"):
            params = xp.asarray(self.params_to_array(params), dtype=xp.float64)
        return self._predict(t, aif, params, xp)

    @abstractmethod
    def _predict(
        self,
        t: "NDArray[np.floating[Any]]",
        aif: "NDArray[np.floating[Any]]",
        params: "NDArray[np.floating[Any]]",
        xp: Any,
    ) -> "NDArray[np.floating[Any]]":
        """Core prediction — model implementers override this.

        Uses ``xp`` operations for both single-voxel and batch
        prediction via broadcasting.

        Parameters
        ----------
        t : NDArray[np.floating]
            Time points in seconds.
            Shape ``(n_timepoints,)`` for single voxel,
            ``(n_timepoints, 1)`` for batch (set up by ``predict_batch``).
        aif : NDArray[np.floating]
            Arterial input function concentration.
            Shape ``(n_timepoints,)`` for single voxel,
            ``(n_timepoints, 1)`` for batch.
        params : NDArray[np.floating]
            Parameter array.
            Shape ``(n_params,)`` for single voxel,
            ``(n_params, n_voxels)`` for batch.
        xp : module
            Array module (numpy or cupy).

        Returns
        -------
        NDArray[np.floating]
            Predicted tissue concentration.
        """
        ...

    @abstractmethod
    def get_bounds(self) -> dict[str, tuple[float, float]]:
        """Return physiological parameter bounds.

        Returns
        -------
        dict[str, tuple[float, float]]
            Mapping of parameter names to (lower, upper) bounds.
        """
        ...

    @abstractmethod
    def get_initial_guess(
        self,
        ct: "NDArray[np.floating[Any]]",
        aif: "NDArray[np.floating[Any]]",
        t: "NDArray[np.floating[Any]]",
    ) -> P:
        """Compute initial parameter guess from data.

        Parameters
        ----------
        ct : NDArray[np.floating]
            Tissue concentration curve.
        aif : NDArray[np.floating]
            Arterial input function.
        t : NDArray[np.floating]
            Time points.

        Returns
        -------
        P
            Initial parameter estimates.
        """
        ...

    def get_initial_guess_batch(
        self,
        ct_batch: "NDArray[np.floating[Any]]",
        aif: "NDArray[np.floating[Any]]",
        t: "NDArray[np.floating[Any]]",
        xp: Any,
    ) -> "NDArray[np.floating[Any]]":
        """Compute initial parameter guesses for a batch of voxels.

        Default implementation loops over voxels using get_initial_guess().
        Subclasses should override with vectorized implementations.

        Parameters
        ----------
        ct_batch : NDArray[np.floating]
            Tissue concentration curves, shape (n_timepoints, n_voxels).
        aif : NDArray[np.floating]
            Arterial input function, shape (n_timepoints,).
        t : NDArray[np.floating]
            Time points in seconds, shape (n_timepoints,).
        xp : module
            Array module (numpy or cupy).

        Returns
        -------
        NDArray[np.floating]
            Initial parameter guesses, shape (n_params, n_voxels).
        """
        n_voxels = ct_batch.shape[1]
        n_params = len(self.parameters)
        params_batch = xp.zeros((n_params, n_voxels), dtype=ct_batch.dtype)
        aif_np = to_numpy(aif)
        t_np = to_numpy(t)
        for v in range(n_voxels):
            guess = self.get_initial_guess(to_numpy(ct_batch[:, v]), aif_np, t_np)
            params_batch[:, v] = xp.asarray(self.params_to_array(guess))
        return params_batch

    # params_to_array, array_to_params, bounds_to_arrays inherited from BaseSignalModel

    def predict_batch(
        self,
        t: "NDArray[np.floating[Any]]",
        aif: "NDArray[np.floating[Any]]",
        params_batch: "NDArray[np.floating[Any]]",
        xp: Any,
    ) -> "NDArray[np.floating[Any]]":
        """Predict tissue concentration for multiple voxels simultaneously.

        Reshapes *t* and *aif* to column vectors and delegates to
        :meth:`predict`, which uses broadcasting to handle the batch
        dimension automatically.  No per-voxel loop is needed.

        Parameters
        ----------
        t : NDArray[np.floating]
            Time points in seconds, shape (n_timepoints,).
        aif : NDArray[np.floating]
            Arterial input function, shape ``(n_timepoints,)`` or
            ``(n_timepoints, n_voxels)`` for per-voxel shifted AIFs.
        params_batch : NDArray[np.floating]
            Parameter values for all voxels, shape (n_params, n_voxels).
            Row order must match self.parameters.
        xp : module
            Array module to use (numpy or cupy).

        Returns
        -------
        NDArray[np.floating]
            Predicted tissue concentrations, shape (n_timepoints, n_voxels).
        """
        t_col = t[:, xp.newaxis]  # (n_time, 1)
        aif_col = (
            aif[:, xp.newaxis] if aif.ndim == 1 else aif
        )  # (n_time, 1) or (n_time, n_voxels)
        return self._predict(t_col, aif_col, params_batch, xp)
