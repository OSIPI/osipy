"""Parameter map visualization functions.

This module provides functions for visualizing perfusion parameter maps.

NO scipy dependency - uses numpy for linear regression.
"""

from typing import TYPE_CHECKING, Any

import numpy as np

from osipy.common.exceptions import DataValidationError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from osipy.common.parameter_map import ParameterMap


# Check for matplotlib availability
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_parameter_map(
    parameter_map: "ParameterMap | NDArray[np.floating[Any]]",
    slice_idx: int | None = None,
    axis: int = 2,
    orientation: str | None = None,
    title: str | None = None,
    colormap: str = "viridis",
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    mask: "NDArray[np.bool_] | None" = None,
    show_mask: bool = False,
    show_colorbar: bool = True,
    underlay: "NDArray[np.floating[Any]] | None" = None,
    alpha: float = 1.0,
    ax: Any | None = None,
    figsize: tuple[float, float] = (8, 6),
) -> Any:
    """Plot a 2D slice of a parameter map.

    Parameters
    ----------
    parameter_map : ParameterMap or NDArray
        Parameter map to visualize.
    slice_idx : int, optional
        Slice index to display. If None, uses middle slice.
    axis : int
        Axis perpendicular to the slice (0, 1, or 2).
    orientation : str, optional
        Orientation name: "axial", "coronal", or "sagittal".
        If provided, overrides axis parameter.
    title : str, optional
        Plot title. Uses parameter name if not provided.
    colormap : str
        Matplotlib colormap name.
    cmap : str, optional
        Alias for colormap (for compatibility).
    vmin, vmax : float, optional
        Color scale limits.
    mask : NDArray, optional
        Mask to apply (show only masked region).
    show_mask : bool
        Whether to show quality mask from ParameterMap.
    show_colorbar : bool
        Whether to show colorbar.
    underlay : NDArray, optional
        Anatomical underlay image.
    alpha : float
        Transparency for overlay (when underlay is provided).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Creates new figure if None.
    figsize : tuple
        Figure size if creating new figure.

    Returns
    -------
    matplotlib.figure.Figure or matplotlib.axes.Axes
        The figure or axes with the plot.

    Raises
    ------
    ImportError
        If matplotlib is not available.
    """
    if not HAS_MATPLOTLIB:
        msg = "matplotlib is required for visualization"
        raise ImportError(msg)

    # Handle cmap alias
    if cmap is not None:
        colormap = cmap

    # Handle orientation -> axis mapping
    if orientation is not None:
        orientation_map = {"axial": 2, "coronal": 1, "sagittal": 0}
        axis = orientation_map.get(orientation.lower(), 2)

    # Extract data and metadata
    from osipy.common.parameter_map import ParameterMap

    if isinstance(parameter_map, ParameterMap):
        data = parameter_map.values
        default_title = f"{parameter_map.name}"
        if parameter_map.units:
            default_title += f" ({parameter_map.units})"
        quality_mask = parameter_map.quality_mask if show_mask else None
    else:
        data = parameter_map
        default_title = "Parameter Map"
        quality_mask = None

    title = title or default_title

    # Handle 2D data
    if data.ndim == 2:
        slice_data = data
        mask_slice = mask if mask is not None else None
        underlay_slice = underlay if underlay is not None else None
        quality_mask_slice = quality_mask
    elif data.ndim == 3:
        # Get slice
        if slice_idx is None:
            slice_idx = data.shape[axis] // 2

        if axis == 0:
            slice_data = data[slice_idx, :, :]
            mask_slice = mask[slice_idx, :, :] if mask is not None else None
            underlay_slice = underlay[slice_idx, :, :] if underlay is not None else None
            quality_mask_slice = (
                quality_mask[slice_idx, :, :] if quality_mask is not None else None
            )
        elif axis == 1:
            slice_data = data[:, slice_idx, :]
            mask_slice = mask[:, slice_idx, :] if mask is not None else None
            underlay_slice = underlay[:, slice_idx, :] if underlay is not None else None
            quality_mask_slice = (
                quality_mask[:, slice_idx, :] if quality_mask is not None else None
            )
        else:
            slice_data = data[:, :, slice_idx]
            mask_slice = mask[:, :, slice_idx] if mask is not None else None
            underlay_slice = underlay[:, :, slice_idx] if underlay is not None else None
            quality_mask_slice = (
                quality_mask[:, :, slice_idx] if quality_mask is not None else None
            )
    else:
        msg = f"Cannot plot {data.ndim}D data"
        raise DataValidationError(msg)

    # Apply mask
    if mask_slice is not None:
        slice_data = np.where(mask_slice, slice_data, np.nan)

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Plot underlay first if provided
    if underlay_slice is not None:
        ax.imshow(
            underlay_slice.T,
            origin="lower",
            cmap="gray",
            interpolation="nearest",
        )

    # Plot parameter map
    im = ax.imshow(
        slice_data.T,
        origin="lower",
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        alpha=alpha if underlay_slice is not None else 1.0,
    )

    # Overlay quality mask contour if requested
    if quality_mask_slice is not None and show_mask:
        ax.contour(
            quality_mask_slice.T,
            levels=[0.5],
            colors=["red"],
            linewidths=[1.5],
            origin="lower",
        )

    ax.set_title(title)
    ax.axis("off")

    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    return fig


def plot_parameter_comparison(
    *maps_args: "ParameterMap",
    maps: list["ParameterMap"] | None = None,
    slice_idx: int | None = None,
    axis: int = 2,
    colormap: str = "viridis",
    figsize: tuple[float, float] | None = None,
) -> Any:
    """Plot multiple parameter maps side by side.

    Parameters
    ----------
    *maps_args : ParameterMap
        Parameter maps to compare (positional arguments).
    maps : list[ParameterMap], optional
        List of parameter maps (alternative to positional args).
    slice_idx : int, optional
        Slice index to display.
    axis : int
        Slice axis.
    colormap : str
        Colormap name.
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        The figure with the plots.
    """
    if not HAS_MATPLOTLIB:
        msg = "matplotlib is required for visualization"
        raise ImportError(msg)

    # Handle both positional args and maps parameter
    if maps is None:
        maps = list(maps_args)
        # If the first argument is already a list, use it directly
        if len(maps) == 1 and isinstance(maps[0], list):
            maps = maps[0]

    n_maps = len(maps)
    if figsize is None:
        figsize = (4 * n_maps, 4)

    fig, axes = plt.subplots(1, n_maps, figsize=figsize)
    if n_maps == 1:
        axes = [axes]

    for ax, pmap in zip(axes, maps, strict=False):
        plot_parameter_map(
            pmap, slice_idx=slice_idx, axis=axis, colormap=colormap, ax=ax
        )

    plt.tight_layout()
    return fig


def create_montage(
    data: "NDArray[np.floating[Any]]",
    n_cols: int = 5,
    colormap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[float, float] | None = None,
) -> Any:
    """Create a montage of all slices in a 3D volume.

    Parameters
    ----------
    data : NDArray
        3D data volume.
    n_cols : int
        Number of columns in montage.
    colormap : str
        Colormap name.
    vmin, vmax : float, optional
        Color limits.
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        The figure with the montage.
    """
    if not HAS_MATPLOTLIB:
        msg = "matplotlib is required for visualization"
        raise ImportError(msg)

    n_slices = data.shape[2]
    n_rows = int(np.ceil(n_slices / n_cols))

    if figsize is None:
        figsize = (3 * n_cols, 3 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes)

    for i in range(n_slices):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].imshow(
            data[:, :, i].T, cmap=colormap, vmin=vmin, vmax=vmax, origin="lower"
        )
        axes[row, col].axis("off")
        axes[row, col].set_title(f"Slice {i}")

    # Hide empty axes
    for i in range(n_slices, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis("off")

    plt.tight_layout()
    return fig
