"""Time-course visualization functions.

This module provides functions for visualizing time-course data,
AIFs, and residue functions.
"""

from typing import TYPE_CHECKING, Any

import numpy as np

from osipy.common.exceptions import DataValidationError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from osipy.common.aif import ArterialInputFunction

# Check for matplotlib availability
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_time_course(
    time: "NDArray[np.floating[Any]]",
    signal: "NDArray[np.floating[Any]]",
    fitted: "NDArray[np.floating[Any]] | None" = None,
    title: str = "Time Course",
    xlabel: str = "Time (s)",
    ylabel: str = "Signal",
    label_measured: str = "Measured",
    label_fitted: str = "Fitted",
    ax: Any | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> Any:
    """Plot time-course data with optional fitted curve.

    Parameters
    ----------
    time : NDArray
        Time points.
    signal : NDArray
        Measured signal values.
    fitted : NDArray, optional
        Fitted/predicted signal values.
    title : str
        Plot title.
    xlabel, ylabel : str
        Axis labels.
    label_measured, label_fitted : str
        Legend labels.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    figsize : tuple
        Figure size if creating new figure.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    if not HAS_MATPLOTLIB:
        msg = "matplotlib is required for visualization"
        raise ImportError(msg)

    if ax is None:
        _fig, ax = plt.subplots(figsize=figsize)

    ax.plot(time, signal, "o-", label=label_measured, markersize=4)

    if fitted is not None:
        ax.plot(time, fitted, "-", label=label_fitted, linewidth=2)
        ax.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return ax


def plot_aif(
    aif: "ArterialInputFunction | NDArray[np.floating[Any]]",
    time: "NDArray[np.floating[Any]] | None" = None,
    population_aif: "NDArray[np.floating[Any]] | None" = None,
    title: str = "Arterial Input Function",
    xlabel: str = "Time (s)",
    ylabel: str = "Concentration (mM)",
    ax: Any | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> Any:
    """Plot arterial input function.

    Parameters
    ----------
    aif : ArterialInputFunction or NDArray
        AIF to plot (or measured AIF array if using time parameter).
    time : NDArray, optional
        Time points (required if aif is an array).
    population_aif : NDArray, optional
        Population-based AIF to compare.
    title : str
        Plot title.
    xlabel, ylabel : str
        Axis labels.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure or matplotlib.axes.Axes
        The figure or axes with the plot.
    """
    if not HAS_MATPLOTLIB:
        msg = "matplotlib is required for visualization"
        raise ImportError(msg)

    from osipy.common.aif import ArterialInputFunction

    if isinstance(aif, ArterialInputFunction):
        t = aif.time
        conc = aif.concentration
        if aif.model_name:
            title = f"AIF ({aif.model_name})"
    else:
        if time is None:
            msg = "time is required when aif is an array"
            raise DataValidationError(msg)
        t = time
        conc = aif

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(
        t,
        conc,
        "r-",
        linewidth=2,
        label="Measured AIF" if population_aif is not None else None,
    )
    ax.fill_between(t, 0, conc, alpha=0.3, color="red")

    if population_aif is not None:
        ax.plot(t, population_aif, "b--", linewidth=2, label="Population AIF")
        ax.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(0, None)

    return fig


def plot_residue_function(
    time: "NDArray[np.floating[Any]]",
    residue: "NDArray[np.floating[Any]]",
    title: str = "Residue Function",
    xlabel: str = "Time (s)",
    ylabel: str = "R(t)",
    show_mtt: bool = True,
    show_oscillation_warning: bool = False,
    ax: Any | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> Any:
    """Plot residue function from deconvolution.

    Parameters
    ----------
    time : NDArray
        Time points.
    residue : NDArray
        Residue function values.
    title : str
        Plot title.
    xlabel, ylabel : str
        Axis labels.
    show_mtt : bool
        Whether to mark MTT on the plot.
    show_oscillation_warning : bool
        Whether to check and warn about oscillations.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure or matplotlib.axes.Axes
        The figure or axes with the plot.
    """
    if not HAS_MATPLOTLIB:
        msg = "matplotlib is required for visualization"
        raise ImportError(msg)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(time, residue, "b-", linewidth=2)
    ax.fill_between(time, 0, residue, alpha=0.3, color="blue")

    # Mark CBF (max of R)
    cbf_idx = np.argmax(residue)
    cbf = residue[cbf_idx]
    ax.axhline(cbf, color="green", linestyle="--", alpha=0.7, label=f"CBF = {cbf:.2f}")

    if show_mtt:
        # MTT = area / max
        dt = time[1] - time[0]
        area = np.sum(residue) * dt
        mtt = area / cbf if cbf > 0 else 0
        ax.axvline(
            mtt, color="orange", linestyle="--", alpha=0.7, label=f"MTT = {mtt:.2f} s"
        )

    if show_oscillation_warning:
        # Check for oscillations (sign changes after peak)
        post_peak = residue[cbf_idx:]
        sign_changes = np.sum(np.diff(np.sign(post_peak)) != 0)
        if sign_changes > 2:
            ax.text(
                0.02,
                0.98,
                "⚠ Oscillations detected",
                transform=ax.transAxes,
                fontsize=10,
                color="red",
                verticalalignment="top",
                fontweight="bold",
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, time[-1])
    ax.set_ylim(0, None)

    return fig


def plot_multi_curves(
    time: "NDArray[np.floating[Any]]",
    curves: list["NDArray[np.floating[Any]]"],
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    title: str = "Time Courses",
    xlabel: str = "Time (s)",
    ylabel: str = "Signal",
    ax: Any | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> Any:
    """Plot multiple time courses on the same axes.

    Parameters
    ----------
    time : NDArray
        Time points.
    curves : list[NDArray]
        List of time courses to plot.
    labels : list[str], optional
        Labels for each curve.
    colors : list[str], optional
        Colors for each curve.
    title : str
        Plot title.
    xlabel, ylabel : str
        Axis labels.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    if not HAS_MATPLOTLIB:
        msg = "matplotlib is required for visualization"
        raise ImportError(msg)

    if ax is None:
        _fig, ax = plt.subplots(figsize=figsize)

    if labels is None:
        labels = [f"Curve {i + 1}" for i in range(len(curves))]

    if colors is None:
        colors = plt.cm.tab10.colors[: len(curves)]

    for curve, label, color in zip(curves, labels, colors, strict=False):
        ax.plot(time, curve, label=label, color=color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax
