"""Automatic arterial input function detection.

This module implements automatic AIF detection from DCE-MRI data using
signal characteristics to identify vascular voxels.

GPU/CPU agnostic using the xp array module pattern.
NO scipy dependency - uses custom implementations for filtering and labeling.

References
----------
Mouridsen K et al. (2006). Automatic selection of arterial input function
using cluster analysis. Magn Reson Med 55(3):524-531.

Peruzzo D et al. (2011). Automatic selection of arterial input function
on dynamic contrast-enhanced MR images. Comput Methods Programs Biomed
104(3):e148-e157.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from osipy.common.aif.base import ArterialInputFunction
from osipy.common.backend.array_module import get_array_module, to_numpy
from osipy.common.dataset import PerfusionDataset
from osipy.common.exceptions import AIFError

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class AIFDetectionParams:
    """Parameters for automatic AIF detection.

    Attributes
    ----------
    min_peak_enhancement : float
        Minimum peak enhancement ratio relative to baseline.
    max_fwhm : float
        Maximum full-width at half-maximum (seconds).
    early_arrival_percentile : float
        Percentile for early arrival time selection.
    n_candidates : int
        Number of candidate voxels to consider.
    cluster_size : int
        Minimum cluster size for valid AIF region.
    """

    min_peak_enhancement: float = 3.0
    max_fwhm: float = 15.0  # seconds
    early_arrival_percentile: float = 10.0
    n_candidates: int = 100
    cluster_size: int = 3
    use_mask: bool = True
    smoothing_sigma: float = 0.5


@dataclass
class AIFDetectionResult:
    """Result of automatic AIF detection.

    Attributes
    ----------
    aif : ArterialInputFunction
        Detected arterial input function.
    voxel_mask : NDArray
        Binary mask of voxels used for AIF.
    voxel_indices : list
        List of (x, y, z) indices of selected voxels.
    quality_score : float
        Quality metric for the detected AIF (0-1).
    detection_params : AIFDetectionParams
        Parameters used for detection.
    """

    aif: ArterialInputFunction
    voxel_mask: "NDArray[np.bool_]"
    voxel_indices: list[tuple[int, int, int]] = field(default_factory=list)
    quality_score: float = 0.0
    detection_params: AIFDetectionParams = field(default_factory=AIFDetectionParams)


def detect_aif(
    dataset: PerfusionDataset,
    params: AIFDetectionParams | None = None,
    roi_mask: "NDArray[np.bool_] | None" = None,
    method: str | None = None,
) -> AIFDetectionResult:
    """Automatically detect AIF from DCE-MRI data.

    Uses a multi-criteria approach:
    1. High peak enhancement
    2. Early arrival time
    3. Narrow peak width (short FWHM)
    4. Spatial clustering

    Parameters
    ----------
    dataset : PerfusionDataset
        DCE-MRI dataset with shape (x, y, z, t) or (x, y, t).
    params : AIFDetectionParams, optional
        Detection parameters.
    roi_mask : NDArray[np.bool_], optional
        Region of interest mask to restrict search.

    Returns
    -------
    AIFDetectionResult
        Detection result with AIF and quality metrics.

    Raises
    ------
    AIFError
        If AIF detection fails.
    """
    if method is not None:
        from osipy.common.aif.detection_registry import get_aif_detector

        detector = get_aif_detector(method)
        return detector.detect(dataset, params=params, roi_mask=roi_mask)

    params = params or AIFDetectionParams()
    data = dataset.data

    # Handle 3D vs 4D data
    if data.ndim == 3:
        # (x, y, t) -> add z dimension
        data = data[:, :, np.newaxis, :]
        spatial_shape = data.shape[:3]
        is_3d = False
    elif data.ndim == 4:
        spatial_shape = data.shape[:3]
        is_3d = True
    else:
        msg = f"Invalid data shape: {data.shape}. Expected 3D or 4D."
        raise AIFError(msg)

    n_timepoints = data.shape[-1]

    # Get time vector
    if dataset.time is not None:
        time = dataset.time
    else:
        # Assume uniform sampling
        tr = dataset.acquisition_params.tr if dataset.acquisition_params else 1.0
        time = np.arange(n_timepoints) * tr

    # Create or validate ROI mask
    if roi_mask is not None:
        if roi_mask.ndim == 2:
            roi_mask = roi_mask[:, :, np.newaxis]
        search_mask = roi_mask.astype(bool)
    else:
        search_mask = np.ones(spatial_shape, dtype=bool)

    # Flatten spatial dimensions for analysis
    flat_data = data.reshape(-1, n_timepoints)
    flat_mask = search_mask.ravel()

    # Compute signal characteristics for each voxel
    baseline_indices = slice(0, min(5, n_timepoints // 4))

    # Baseline signal
    baseline = np.mean(flat_data[:, baseline_indices], axis=1)
    baseline = np.maximum(baseline, 1e-10)  # Avoid division by zero

    # Peak enhancement
    peak_signal = np.max(flat_data, axis=1)
    enhancement_ratio = peak_signal / baseline

    # Time to peak (arrival time proxy)
    time_to_peak = np.argmax(flat_data, axis=1).astype(float)

    # Full-width at half-maximum
    fwhm = _compute_fwhm(flat_data, time)

    # Score voxels based on arterial characteristics
    # High enhancement score
    enhancement_score = np.clip(enhancement_ratio / params.min_peak_enhancement, 0, 2)

    # Early arrival score (lower is better)
    arrival_score = 1.0 - (time_to_peak / n_timepoints)

    # Narrow width score (lower FWHM is better)
    width_score = np.clip(1.0 - (fwhm / params.max_fwhm), 0, 1)

    # Combined score
    scores = enhancement_score * arrival_score * width_score

    # Apply mask
    scores[~flat_mask] = 0

    # Filter by minimum criteria
    valid_mask = (
        (enhancement_ratio >= params.min_peak_enhancement)
        & (fwhm <= params.max_fwhm)
        & flat_mask
    )
    scores[~valid_mask] = 0

    if np.sum(valid_mask) < params.cluster_size:
        msg = (
            f"Insufficient valid voxels for AIF detection. "
            f"Found {np.sum(valid_mask)}, need {params.cluster_size}."
        )
        raise AIFError(msg)

    # Select top candidates
    candidate_indices = np.argsort(scores)[::-1][: params.n_candidates]
    candidate_indices = candidate_indices[scores[candidate_indices] > 0]

    if len(candidate_indices) < params.cluster_size:
        msg = "Not enough high-scoring candidates for AIF."
        raise AIFError(msg)

    # Convert flat indices to 3D coordinates
    candidate_coords = np.unravel_index(candidate_indices, spatial_shape)

    # Spatial clustering: find connected components
    candidate_mask = np.zeros(spatial_shape, dtype=bool)
    for i in range(len(candidate_indices)):
        x, y, z = (
            candidate_coords[0][i],
            candidate_coords[1][i],
            candidate_coords[2][i],
        )
        candidate_mask[x, y, z] = True

    # Label connected components using xp-compatible implementation
    labeled_array, n_clusters = _label_connected_components(candidate_mask)

    # Find best cluster
    best_cluster = None
    best_cluster_score = 0

    for cluster_id in range(1, n_clusters + 1):
        cluster_mask = labeled_array == cluster_id
        cluster_size = np.sum(cluster_mask)

        if cluster_size >= params.cluster_size:
            cluster_indices = np.where(cluster_mask.ravel())[0]
            cluster_score = np.mean(scores[cluster_indices])

            if cluster_score > best_cluster_score:
                best_cluster_score = cluster_score
                best_cluster = cluster_id

    if best_cluster is None:
        # Fall back to top individual voxels
        selected_indices = candidate_indices[: params.cluster_size]
        final_mask = np.zeros(spatial_shape, dtype=bool)
        coords = np.unravel_index(selected_indices, spatial_shape)
        for i in range(len(selected_indices)):
            final_mask[coords[0][i], coords[1][i], coords[2][i]] = True
    else:
        final_mask = labeled_array == best_cluster

    # Extract and average AIF from selected voxels
    final_indices = np.where(final_mask.ravel())[0]
    aif_signals = flat_data[final_indices, :]
    mean_aif = np.mean(aif_signals, axis=0)

    # Apply light smoothing if requested
    if params.smoothing_sigma > 0:
        mean_aif = _gaussian_filter1d_xp(mean_aif, params.smoothing_sigma)

    # Compute quality score
    quality_score = _compute_aif_quality(mean_aif, time, params)

    # Build voxel index list
    voxel_list: list[tuple[int, int, int]] = []
    coords = np.unravel_index(final_indices, spatial_shape)
    for i in range(len(final_indices)):
        voxel_list.append((int(coords[0][i]), int(coords[1][i]), int(coords[2][i])))

    # Create AIF object
    aif = ArterialInputFunction(
        time=time,
        concentration=mean_aif,
        source="detected",
        model_name="automatic",
    )

    # Reshape mask if input was 3D
    output_mask = final_mask if is_3d else final_mask[:, :, 0]

    return AIFDetectionResult(
        aif=aif,
        voxel_mask=output_mask,
        voxel_indices=voxel_list,
        quality_score=quality_score,
        detection_params=params,
    )


def _compute_fwhm(
    data: "NDArray[np.floating[Any]]", time: "NDArray[np.floating[Any]]"
) -> "NDArray[np.floating[Any]]":
    """Compute full-width at half-maximum for each voxel.

    Parameters
    ----------
    data : NDArray
        Signal data with shape (n_voxels, n_timepoints).
    time : NDArray
        Time vector.

    Returns
    -------
    NDArray
        FWHM values for each voxel.
    """
    n_voxels, n_timepoints = data.shape
    fwhm = np.full(n_voxels, np.inf)

    # Baseline
    baseline = np.mean(data[:, :3], axis=1, keepdims=True)
    data_centered = data - baseline

    # Peak value and index
    peak_values = np.max(data_centered, axis=1)
    peak_indices = np.argmax(data_centered, axis=1)

    half_max = peak_values / 2

    for i in range(n_voxels):
        if peak_values[i] <= 0:
            continue

        signal = data_centered[i, :]
        hm = half_max[i]
        peak_idx = peak_indices[i]

        # Find left crossing
        left_idx = peak_idx
        for j in range(peak_idx, -1, -1):
            if signal[j] <= hm:
                left_idx = j
                break

        # Find right crossing
        right_idx = peak_idx
        for j in range(peak_idx, n_timepoints):
            if signal[j] <= hm:
                right_idx = j
                break

        if right_idx > left_idx:
            fwhm[i] = time[right_idx] - time[left_idx]

    return fwhm


def _compute_aif_quality(
    aif: "NDArray[np.floating[Any]]",
    time: "NDArray[np.floating[Any]]",
    params: AIFDetectionParams,
) -> float:
    """Compute quality score for detected AIF.

    Parameters
    ----------
    aif : NDArray
        AIF signal.
    time : NDArray
        Time vector.
    params : AIFDetectionParams
        Detection parameters.

    Returns
    -------
    float
        Quality score between 0 and 1.
    """
    # Baseline
    baseline = np.mean(aif[:3])
    baseline = max(baseline, 1e-10)

    # Enhancement ratio
    peak = np.max(aif)
    enhancement = peak / baseline

    # FWHM
    half_max = (peak + baseline) / 2
    above_half = aif >= half_max
    if np.any(above_half):
        indices = np.where(above_half)[0]
        fwhm = time[indices[-1]] - time[indices[0]]
    else:
        fwhm = np.inf

    # Score components
    enhancement_quality = min(enhancement / params.min_peak_enhancement, 1.0)
    width_quality = max(1.0 - fwhm / params.max_fwhm, 0.0) if fwhm < np.inf else 0.0

    # SNR estimate
    noise_std = np.std(aif[:5])
    snr = (peak - baseline) / max(noise_std, 1e-10)
    snr_quality = min(snr / 10, 1.0)

    # Combined quality
    quality = (enhancement_quality + width_quality + snr_quality) / 3

    return float(np.clip(quality, 0, 1))


def _label_connected_components(
    mask: "NDArray[np.bool_]",
) -> tuple["NDArray[np.int32]", int]:
    """Label connected components in a binary mask.

    XP-compatible implementation without scipy.

    Parameters
    ----------
    mask : NDArray
        Binary mask.

    Returns
    -------
    labeled : NDArray
        Labeled array where each component has a unique integer.
    n_components : int
        Number of connected components.
    """
    xp = get_array_module(mask)

    # Use numpy for connected component labeling (CPU operation)
    # This is acceptable since AIF detection is a small operation
    mask_np = to_numpy(mask)
    labeled = np.zeros_like(mask_np, dtype=np.int32)
    current_label = 0

    # 3D 6-connectivity or 2D 4-connectivity flood fill
    if mask_np.ndim == 3:
        neighbors = [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ]
    else:
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    # Simple flood fill algorithm
    for idx in np.ndindex(mask_np.shape):
        if mask_np[idx] and labeled[idx] == 0:
            current_label += 1
            _flood_fill(mask_np, labeled, idx, current_label, neighbors)

    return xp.asarray(labeled), current_label


def _flood_fill(
    mask: "NDArray[np.bool_]",
    labeled: "NDArray[np.int32]",
    start: tuple,
    label: int,
    neighbors: list,
) -> None:
    """Flood fill to label connected component.

    Parameters
    ----------
    mask : NDArray
        Binary mask.
    labeled : NDArray
        Output labeled array (modified in place).
    start : tuple
        Starting index.
    label : int
        Label to assign.
    neighbors : list
        Neighbor offsets for connectivity.
    """
    stack = [start]
    shape = mask.shape
    ndim = len(shape)

    while stack:
        current = stack.pop()
        if labeled[current] != 0:
            continue
        labeled[current] = label

        for offset in neighbors:
            neighbor = tuple(current[i] + offset[i] for i in range(ndim))
            if (
                all(0 <= neighbor[i] < shape[i] for i in range(ndim))
                and mask[neighbor]
                and labeled[neighbor] == 0
            ):
                stack.append(neighbor)


def _gaussian_filter1d_xp(
    signal: "NDArray[np.floating[Any]]",
    sigma: float,
) -> "NDArray[np.floating[Any]]":
    """Apply 1D Gaussian smoothing.

    XP-compatible implementation without scipy.

    Parameters
    ----------
    signal : NDArray
        Input signal.
    sigma : float
        Standard deviation of Gaussian kernel.

    Returns
    -------
    NDArray
        Smoothed signal.
    """
    xp = get_array_module(signal)

    # Create Gaussian kernel
    # Kernel size should be at least 6*sigma to capture most of the Gaussian
    kernel_size = max(3, int(6 * sigma) | 1)  # Ensure odd
    half_size = kernel_size // 2

    x = xp.arange(-half_size, half_size + 1, dtype=signal.dtype)
    kernel = xp.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / xp.sum(kernel)

    # Pad signal for convolution
    padded = xp.pad(signal, half_size, mode="reflect")

    # Convolve
    n = len(signal)
    result = xp.zeros(n, dtype=signal.dtype)
    for i in range(n):
        result[i] = xp.sum(padded[i : i + kernel_size] * kernel)

    return result


# --- Registry-based detector wrapping the existing logic ---

from osipy.common.aif.base_detector import BaseAIFDetector
from osipy.common.aif.detection_registry import register_aif_detector


@register_aif_detector("multi_criteria")
class MultiCriteriaAIFDetector(BaseAIFDetector):
    """Multi-criteria AIF detection algorithm."""

    @property
    def name(self) -> str:
        """Detector display name."""
        return "Multi-criteria AIF"

    @property
    def reference(self) -> str:
        """Literature reference for the detection algorithm."""
        return "Peruzzo D et al. MRM 2011;66(2):461-472."

    def detect(self, dataset, params=None, roi_mask=None):
        """Detect AIF using multi-criteria approach.

        Parameters
        ----------
        dataset : PerfusionDataset
            Input dataset.
        params : AIFDetectionParams | None
            Detection parameters.
        roi_mask : NDArray | None
            ROI mask to restrict search.

        Returns
        -------
        AIFDetectionResult
            Detection result.
        """
        return detect_aif(dataset, params=params, roi_mask=roi_mask)
