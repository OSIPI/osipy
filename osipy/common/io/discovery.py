"""DICOM discovery and loading.

Two primitives:

* :func:`discover_dicom` walks a directory, groups files by
  ``SeriesInstanceUID``, and annotates each series with a best-effort
  role hint (dynamic / dynamic_frame / vfa / t1_look_locker / unknown).
  It never touches pixel data and logs a transparent summary of what
  it found and why.
* :func:`load_dicom_series` loads one :class:`SeriesInfo` (3D or 4D,
  depending on TPI structure) or stacks a list of per-timepoint series
  into a 4D volume with a derived time vector.

Design: discovery is observation only. Role hints are hints — callers
may ignore them and select series by any attribute on ``SeriesInfo``.
No modality-specific behaviour is baked into loading; ``modality`` is
just carried through onto the returned :class:`PerfusionDataset`.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from osipy.common.dataset import PerfusionDataset
from osipy.common.exceptions import DataValidationError
from osipy.common.exceptions import IOError as OsipyIOError
from osipy.common.io.dicom import (
    _NON_IMAGE_MODALITIES,
    _apply_pixel_scaling,
    _extract_time_from_series_description,
    build_affine_from_dicom,
)
from osipy.common.types import AcquisitionParams, DCEAcquisitionParams, Modality

logger = logging.getLogger(__name__)

RoleHint = Literal["dynamic", "dynamic_frame", "vfa", "t1_look_locker", "unknown"]

# Minimum cluster size to treat a set of sibling series as one multi-series
# dynamic acquisition. Below this we refuse to guess — two same-shape
# structurals don't make a DCE series.
_MIN_DYNAMIC_FRAME_CLUSTER = 3

# Regex stripped from SeriesDescription to discover Mode-B clusters. Matches
# the trigger-time suffix vendors (Siemens TWIST in particular) stamp onto
# per-timepoint series, e.g. "…_TT=49.6s".
_TT_SUFFIX_RE = re.compile(r"[_\s-]*TT[=_]?\d+\.?\d*\s*s?\s*$", re.IGNORECASE)


@dataclass
class SeriesInfo:
    """Metadata for one DICOM series, produced by :func:`discover_dicom`.

    Attributes
    ----------
    uid : str
        SeriesInstanceUID (``0020,000E``).
    study_instance_uid : str | None
        StudyInstanceUID (``0020,000D``). Used to keep per-timepoint
        frame clustering from crossing study boundaries (e.g. multiple
        visits exported under one root).
    files : list[Path]
        All files belonging to this series, sorted.
    description : str
        SeriesDescription (``0008,103E``).
    series_number : int | None
        SeriesNumber (``0020,0011``).
    dicom_modality : str | None
        DICOM Modality (``0008,0060``) — e.g. ``"MR"``, ``"KO"``.
    manufacturer : str | None
        Manufacturer (``0008,0070``).
    flip_angle : float | None
        FlipAngle (``0018,1314``), only set when uniform across the series.
    tr, te, field_strength : float | None
        RepetitionTime, EchoTime, MagneticFieldStrength.
    rows, columns : int | None
        Image matrix size (from first file).
    n_temporal_positions : int
        Count of unique ``TemporalPositionIdentifier`` values. ``0`` when
        the tag is absent on all files. ``>1`` marks a single-series
        dynamic (all timepoints packaged inside one ``SeriesInstanceUID``).
    n_acquisition_numbers : int
        Count of unique ``AcquisitionNumber`` values. Used as a fallback
        single-series dynamic signal when ``TemporalPositionIdentifier``
        is absent (common on older TCIA DCE exports).
    n_slice_locations : int
        Count of unique ``SliceLocation`` values.
    image_types : set[tuple[str, ...]]
        Unique ``ImageType`` (``0008,0008``) combinations. Used to detect
        mixed magnitude/phase series (Philips dual-output exports).
    acquisition_time : str | None
        First file's AcquisitionTime (``HHMMSS.FFFFFF``).
    acquisition_time_sec : float | None
        Parsed ``acquisition_time`` in seconds since midnight.
    trigger_times : set[float]
        Unique ``TriggerTime`` values across the series (ms).
    trigger_time_hint : float | None
        Seconds parsed from the SeriesDescription ``TT=X.Xs`` suffix,
        when present — only emitted by per-timepoint exports (one 3D
        series per dynamic frame).
    role_hint : RoleHint
        Best-effort classification. See module docstring for meanings.
    group_key : str | None
        For ``dynamic_frame`` series: a key shared by every frame of the
        same multi-series dynamic. ``None`` otherwise.
    reason : str
        Human-readable justification for the ``role_hint``.
    """

    uid: str
    study_instance_uid: str | None
    files: list[Path]
    description: str
    series_number: int | None
    dicom_modality: str | None
    manufacturer: str | None
    flip_angle: float | None
    tr: float | None
    te: float | None
    field_strength: float | None
    rows: int | None
    columns: int | None
    n_temporal_positions: int
    n_acquisition_numbers: int
    n_slice_locations: int
    image_types: set[tuple[str, ...]] = field(default_factory=set)
    acquisition_time: str | None = None
    acquisition_time_sec: float | None = None
    trigger_times: set[float] = field(default_factory=set)
    trigger_time_hint: float | None = None
    role_hint: RoleHint = "unknown"
    group_key: str | None = None
    reason: str = ""


# ---------------------------------------------------------------------------
# Header scanning
# ---------------------------------------------------------------------------


def _parse_dicom_time(time_str: str) -> float | None:
    """Parse a DICOM TM string (HHMMSS[.FFFFFF]) to seconds since midnight."""
    if not time_str:
        return None
    try:
        h = int(time_str[:2]) if len(time_str) >= 2 else 0
        m = int(time_str[2:4]) if len(time_str) >= 4 else 0
        s = float(time_str[4:]) if len(time_str) > 4 else 0.0
        return h * 3600 + m * 60 + s
    except (ValueError, IndexError):
        return None


def _iter_candidate_files(path: Path, recursive: bool) -> list[Path]:
    """Return files under ``path`` that are candidates for DICOM reads.

    Skips dotfiles and entries like ``__MACOSX``/``DICOMDIR`` detritus.
    """
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []

    walker = path.rglob("*") if recursive else path.iterdir()
    out: list[Path] = []
    for f in walker:
        if not f.is_file():
            continue
        if f.name.startswith("."):
            continue
        if f.name == "DICOMDIR":
            continue
        # Skip macOS resource-fork clutter.
        if "__MACOSX" in f.parts:
            continue
        out.append(f)
    return out


def _scan_headers(files: list[Path]) -> dict[str, SeriesInfo]:
    """Read each file's header once and group into :class:`SeriesInfo`."""
    try:
        import pydicom
        from pydicom.errors import InvalidDicomError
    except ImportError as e:
        msg = "pydicom is required for DICOM discovery"
        raise ImportError(msg) from e

    groups: dict[str, dict[str, Any]] = {}
    skipped = 0

    for f in files:
        try:
            dcm = pydicom.dcmread(f, stop_before_pixels=True)
        except (InvalidDicomError, OSError, ValueError):
            skipped += 1
            continue

        uid = str(dcm.get("SeriesInstanceUID", ""))
        if not uid:
            skipped += 1
            continue

        g = groups.setdefault(
            uid,
            {
                "files": [],
                "study_uids": set(),
                "descriptions": set(),
                "series_numbers": set(),
                "dicom_modalities": set(),
                "manufacturers": set(),
                "flip_angles": set(),
                "trs": set(),
                "tes": set(),
                "field_strengths": set(),
                "rows": set(),
                "columns": set(),
                "tpis": set(),
                "acquisition_numbers": set(),
                "slice_locs": set(),
                "image_types": set(),
                "acq_times": [],
                "trigger_times": set(),
            },
        )
        g["files"].append(f)
        if (study_uid := dcm.get("StudyInstanceUID")) is not None:
            g["study_uids"].add(str(study_uid))

        # Best-effort grabs — malformed tags fall through silently.
        try:
            if (v := dcm.get("SeriesDescription")) is not None:
                g["descriptions"].add(str(v))
            if (v := dcm.get("SeriesNumber")) is not None:
                g["series_numbers"].add(int(v))
            if (v := dcm.get("Modality")) is not None:
                g["dicom_modalities"].add(str(v))
            if (v := dcm.get("Manufacturer")) is not None:
                g["manufacturers"].add(str(v))
            if (v := dcm.get("FlipAngle")) is not None:
                g["flip_angles"].add(float(v))
            if (v := dcm.get("RepetitionTime")) is not None:
                g["trs"].add(float(v))
            if (v := dcm.get("EchoTime")) is not None:
                g["tes"].add(float(v))
            if (v := dcm.get("MagneticFieldStrength")) is not None:
                g["field_strengths"].add(float(v))
            if (v := dcm.get("Rows")) is not None:
                g["rows"].add(int(v))
            if (v := dcm.get("Columns")) is not None:
                g["columns"].add(int(v))
            if (v := dcm.get("TemporalPositionIdentifier")) is not None:
                g["tpis"].add(int(v))
            if (v := dcm.get("AcquisitionNumber")) is not None:
                g["acquisition_numbers"].add(int(v))
            if (v := dcm.get("SliceLocation")) is not None:
                g["slice_locs"].add(round(float(v), 3))
            if (v := dcm.get("ImageType")) is not None:
                g["image_types"].add(tuple(str(x).upper() for x in v))
            if (v := dcm.get("AcquisitionTime")) is not None:
                g["acq_times"].append(str(v))
            if (v := dcm.get("TriggerTime")) is not None:
                g["trigger_times"].add(float(v))
        except (TypeError, ValueError):
            continue

    if skipped:
        logger.debug("discover_dicom: skipped %d non-DICOM / unreadable files", skipped)

    infos: dict[str, SeriesInfo] = {}
    for uid, g in groups.items():
        if not g["files"]:
            continue
        acq_time = min(g["acq_times"]) if g["acq_times"] else None
        description = next(iter(g["descriptions"]), "")
        infos[uid] = SeriesInfo(
            uid=uid,
            study_instance_uid=next(iter(g["study_uids"]), None),
            files=sorted(g["files"]),
            description=description,
            series_number=min(g["series_numbers"]) if g["series_numbers"] else None,
            dicom_modality=next(iter(g["dicom_modalities"]), None),
            manufacturer=next(iter(g["manufacturers"]), None),
            flip_angle=next(iter(g["flip_angles"]))
            if len(g["flip_angles"]) == 1
            else None,
            tr=next(iter(g["trs"])) if g["trs"] else None,
            te=next(iter(g["tes"])) if g["tes"] else None,
            field_strength=next(iter(g["field_strengths"]))
            if g["field_strengths"]
            else None,
            rows=next(iter(g["rows"])) if len(g["rows"]) == 1 else None,
            columns=next(iter(g["columns"])) if len(g["columns"]) == 1 else None,
            n_temporal_positions=len(g["tpis"]),
            n_acquisition_numbers=len(g["acquisition_numbers"]),
            n_slice_locations=len(g["slice_locs"]),
            image_types=g["image_types"],
            acquisition_time=acq_time,
            acquisition_time_sec=_parse_dicom_time(acq_time) if acq_time else None,
            trigger_times=g["trigger_times"],
            trigger_time_hint=_extract_time_from_series_description(description),
        )
    return infos


# ---------------------------------------------------------------------------
# Role classification
# ---------------------------------------------------------------------------


def _strip_tt_suffix(description: str) -> str:
    """Normalize a per-timepoint SeriesDescription by dropping a TT=X.Xs tail."""
    return _TT_SUFFIX_RE.sub("", description).strip()


def _looks_like_t1_mapping(description: str) -> bool:
    d = description.lower()
    return (
        "look" in d or "locker" in d or "t1-map" in d or "t1_map" in d or "t1 map" in d
    )


def _classify(series_list: list[SeriesInfo]) -> None:
    """Assign ``role_hint``, ``group_key``, ``reason`` on each series in place."""
    # Filter out non-image series up front — they stay "unknown" with a reason.
    for s in series_list:
        if s.dicom_modality in _NON_IMAGE_MODALITIES:
            s.role_hint = "unknown"
            s.reason = f"non-image modality ({s.dicom_modality})"

    imaged = [s for s in series_list if s.role_hint != "unknown" or s.reason == ""]

    # Pass 1 — intra-series dynamic (Mode A). Anything with multiple TPIs is
    # a dynamic candidate. We don't require "the most TPIs"; callers can pick.
    mode_a_dynamic_uids: set[str] = set()
    for s in imaged:
        if s.n_temporal_positions > 1:
            s.role_hint = "dynamic"
            s.reason = (
                f"{s.n_temporal_positions} unique TemporalPositionIdentifier values"
            )
            mode_a_dynamic_uids.add(s.uid)

    # Pass 1b — fallback Mode-A detection via AcquisitionNumber. Older TCIA
    # exports (e.g. Clinical_P1) use AcquisitionNumber to index temporal
    # frames and omit TemporalPositionIdentifier entirely. If TPI is absent
    # but we see at least _MIN_DYNAMIC_FRAME_CLUSTER unique AcquisitionNumber
    # values, treat the series as dynamic.
    for s in imaged:
        if s.uid in mode_a_dynamic_uids:
            continue
        if (
            s.n_temporal_positions <= 1
            and s.n_acquisition_numbers >= _MIN_DYNAMIC_FRAME_CLUSTER
        ):
            s.role_hint = "dynamic"
            s.reason = (
                f"{s.n_acquisition_numbers} unique AcquisitionNumber values "
                f"(TemporalPositionIdentifier absent)"
            )
            mode_a_dynamic_uids.add(s.uid)

    # Pass 2 — inter-series dynamic clusters (Mode B). Group by
    # (study_uid, rows, cols, stripped_description). Slice count is
    # deliberately *not* part of the key — vendors occasionally drop the
    # last slice on one frame (e.g. 127 vs 128) and we don't want that
    # to shatter a dynamic into separate clusters. StudyInstanceUID is
    # part of the key so multi-visit exports don't cross-pollinate.
    # A cluster of >= _MIN_DYNAMIC_FRAME_CLUSTER same-shape series with
    # differing AcquisitionTime (or distinct TT hints) is a multi-series
    # dynamic.
    clusters: dict[tuple[Any, ...], list[SeriesInfo]] = defaultdict(list)
    for s in imaged:
        if s.uid in mode_a_dynamic_uids:
            continue
        if s.role_hint != "unknown" and s.reason and "non-image" in s.reason:
            continue
        key = (
            s.study_instance_uid,
            s.rows,
            s.columns,
            _strip_tt_suffix(s.description),
        )
        clusters[key].append(s)

    # Count how many clusters share a stripped description — if multiple
    # studies produced the same description, we suffix the study UID into
    # the group key so callers can distinguish them.
    desc_counts: dict[str, int] = defaultdict(int)
    for (_study, _rows, _cols, desc), members in clusters.items():
        if len(members) >= _MIN_DYNAMIC_FRAME_CLUSTER:
            desc_counts[desc] += 1

    for key, members in clusters.items():
        if len(members) < _MIN_DYNAMIC_FRAME_CLUSTER:
            continue
        # Require distinct temporal signals across members.
        tt_hints = {
            m.trigger_time_hint for m in members if m.trigger_time_hint is not None
        }
        acq_times = {
            m.acquisition_time_sec
            for m in members
            if m.acquisition_time_sec is not None
        }
        if len(tt_hints) < len(members) and len(acq_times) < len(members):
            # Can't establish a unique temporal ordering for each frame.
            continue

        study_uid, _rows, _cols, desc = key
        base = desc or f"group_{members[0].rows}x{members[0].columns}"
        if desc_counts.get(desc, 0) > 1 and study_uid:
            group_id = f"{base}@{study_uid[-8:]}"
        else:
            group_id = base
        for m in members:
            m.role_hint = "dynamic_frame"
            m.group_key = group_id
            m.reason = (
                f"cluster of {len(members)} same-shape series "
                f"({'TT-suffix' if tt_hints else 'AcquisitionTime'} varies)"
            )

    # Pass 3 — VFA. Requires a coherent cluster: ≥2 same-study, same-shape
    # series with distinct FlipAngles. Mis-shaped structurals that happen
    # to carry a distinct FA (post-contrast GRE, FLAIR, etc.) stay
    # "unknown", so callers can do ``[s for s in series if s.role_hint ==
    # "vfa"]`` and get only series usable for T1 mapping.
    candidates = [
        s
        for s in imaged
        if s.role_hint == "unknown" and s.flip_angle is not None and not s.reason
    ]
    vfa_groups: dict[tuple[Any, ...], list[SeriesInfo]] = defaultdict(list)
    for s in candidates:
        vfa_groups[
            (s.study_instance_uid, s.rows, s.columns, s.n_slice_locations)
        ].append(s)
    for members in vfa_groups.values():
        distinct_fas = {m.flip_angle for m in members}
        if len(members) < 2 or len(distinct_fas) < 2:
            continue
        for m in members:
            m.role_hint = "vfa"
            m.reason = (
                f"FlipAngle={m.flip_angle:.0f}° within same-shape VFA cluster "
                f"of {len(members)} series (FAs={sorted(distinct_fas)})"
            )

    # Pass 4 — Look-Locker T1 mapping. A single-series acquisition whose
    # description mentions look-locker / T1 mapping and has multiple
    # TriggerTime values at a single FlipAngle.
    for s in imaged:
        if s.role_hint != "unknown" or s.reason:
            continue
        if _looks_like_t1_mapping(s.description) and len(s.trigger_times) > 1:
            s.role_hint = "t1_look_locker"
            s.reason = (
                f"description matches T1-mapping and {len(s.trigger_times)} "
                f"unique TriggerTime values"
            )

    # Pass 5 — promote mislabeled Mode-A series whose description advertises
    # Look-Locker / T1 mapping (e.g. QIN-Breast-01 "multi-flip_T1-map_smartTX"
    # has TPI>1 but is a T1 map, not a dynamic).
    for s in imaged:
        if s.role_hint != "dynamic":
            continue
        if _looks_like_t1_mapping(s.description):
            s.role_hint = "t1_look_locker"
            s.reason = (
                f"description matches T1-mapping; {s.n_temporal_positions} "
                f"TPIs interpreted as inversion times, not dynamic frames"
            )

    # Anything still unknown with no reason: mark it.
    for s in series_list:
        if s.role_hint == "unknown" and not s.reason:
            s.reason = "no matching heuristic"


# ---------------------------------------------------------------------------
# Transparent summary logging
# ---------------------------------------------------------------------------


def _log_discovery_summary(series_list: list[SeriesInfo], path: Path) -> None:
    """Emit a human-readable digest of what ``discover_dicom`` found."""
    if not series_list:
        logger.info("discover_dicom: no DICOM series found under %s", path)
        return

    logger.info(
        "discover_dicom: %d series under %s",
        len(series_list),
        path,
    )
    for s in series_list:
        desc = (s.description or "<no description>")[:50]
        sn = s.series_number if s.series_number is not None else "?"
        fa = f"{s.flip_angle:.0f}°" if s.flip_angle is not None else "?"
        tpi = s.n_temporal_positions
        acq = s.n_acquisition_numbers
        nfiles = len(s.files)
        group = f" group={s.group_key!r}" if s.group_key else ""
        logger.info(
            "  [%s] SN=%s n=%d FA=%s TPIs=%d AcqNums=%d  %-50s  →  %s (%s)%s",
            s.uid[-8:],
            sn,
            nfiles,
            fa,
            tpi,
            acq,
            desc,
            s.role_hint,
            s.reason,
            group,
        )


# ---------------------------------------------------------------------------
# Public: discover
# ---------------------------------------------------------------------------


def discover_dicom(
    path: str | Path,
    modality: Modality | str | None = None,
    recursive: bool = True,
) -> list[SeriesInfo]:
    """Walk ``path``, group DICOM files by series, and classify each series.

    Parameters
    ----------
    path : str | Path
        Directory (or single file) to scan.
    modality : Modality | str | None
        Reserved for future modality-specific classification tweaks.
        Currently ignored — the heuristics do not change by modality
        because the signals they rely on (TPI, FlipAngle, ImageType,
        description patterns) are modality-agnostic.
    recursive : bool, default True
        Recurse into subdirectories. Handles both flat (NKI) and nested
        (TCIA) layouts.

    Returns
    -------
    list[SeriesInfo]
        One entry per unique ``SeriesInstanceUID`` under ``path``. Sorted
        by ``series_number`` (when available), then UID.

    Notes
    -----
    Role hints (``SeriesInfo.role_hint``) are advisory. Callers can — and
    should — inspect ``SeriesInfo`` directly when the heuristic is wrong.
    A summary of the classification is logged at INFO level.
    """
    path = Path(path)
    if not path.exists():
        msg = f"Path not found: {path}"
        raise FileNotFoundError(msg)

    files = _iter_candidate_files(path, recursive=recursive)
    infos = _scan_headers(files)
    series_list = sorted(
        infos.values(),
        key=lambda s: (
            s.series_number if s.series_number is not None else 10**9,
            s.uid,
        ),
    )
    _classify(series_list)
    _log_discovery_summary(series_list, path)
    return series_list


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _is_magnitude(image_type: tuple[str, ...]) -> bool:
    """Return False when an ImageType marks a phase/derived output."""
    if not image_type:
        return True
    return all(tag not in image_type for tag in ("PHASE MAP", "PHASE", "P"))


def _temporal_index(dcm: Any) -> int:
    """TPI > AcquisitionNumber > InstanceNumber, matching CLI convention."""
    if hasattr(dcm, "TemporalPositionIdentifier"):
        try:
            return int(dcm.TemporalPositionIdentifier)
        except (TypeError, ValueError):
            pass
    if hasattr(dcm, "AcquisitionNumber"):
        try:
            return int(dcm.AcquisitionNumber)
        except (TypeError, ValueError):
            pass
    return int(getattr(dcm, "InstanceNumber", 0))


def _acquisition_params(
    dcm: Any, existing: SeriesInfo | None = None
) -> AcquisitionParams:
    _info_attr = {
        "RepetitionTime": "tr",
        "EchoTime": "te",
        "FlipAngle": "flip_angle",
        "MagneticFieldStrength": "field_strength",
    }

    def _f(dicom_attr: str) -> float | None:
        if existing is not None:
            v = getattr(existing, _info_attr[dicom_attr], None)
            if v is not None:
                return float(v)
        if hasattr(dcm, dicom_attr):
            try:
                return float(getattr(dcm, dicom_attr))
            except (TypeError, ValueError):
                return None
        return None

    return AcquisitionParams(
        tr=_f("RepetitionTime"),
        te=_f("EchoTime"),
        flip_angle=_f("FlipAngle"),
        field_strength=_f("MagneticFieldStrength"),
    )


def _read_dicom_slices(
    files: list[Path],
    drop_phase: bool = True,
) -> tuple[list[tuple[int, float, Any, np.ndarray]], int]:
    """Read pixel arrays and key header fields from a list of DICOM files.

    Returns ``(records, n_dropped)`` where each record is
    ``(temporal_index, slice_location, dcm, pixel_array)``.
    """
    import pydicom

    records: list[tuple[int, float, Any, np.ndarray]] = []
    n_dropped = 0
    for f in files:
        try:
            dcm = pydicom.dcmread(f)
        except Exception:
            n_dropped += 1
            continue
        image_type = tuple(str(x).upper() for x in getattr(dcm, "ImageType", ()) or ())
        if drop_phase and not _is_magnitude(image_type):
            n_dropped += 1
            continue
        t_idx = _temporal_index(dcm)
        loc = float(getattr(dcm, "SliceLocation", 0))
        records.append((t_idx, loc, dcm, _apply_pixel_scaling(dcm, dcm.pixel_array)))
    return records, n_dropped


def _load_single_series(
    series: SeriesInfo,
    modality: Modality | None,
) -> PerfusionDataset:
    """Load one series as 3D or 4D depending on TPI structure."""
    records, n_dropped = _read_dicom_slices(series.files, drop_phase=True)
    if n_dropped:
        logger.info(
            "load_dicom_series[%s]: dropped %d phase/unreadable frames",
            series.uid[-8:],
            n_dropped,
        )
    if not records:
        msg = f"No readable magnitude DICOM frames in series {series.uid}"
        raise OsipyIOError(msg)

    # Group by temporal index.
    by_tpi: dict[int, list[tuple[float, Any, np.ndarray]]] = defaultdict(list)
    for t_idx, loc, dcm, arr in records:
        by_tpi[t_idx].append((loc, dcm, arr))

    sorted_tpis = sorted(by_tpi)
    n_timepoints = len(sorted_tpis)
    first_record = by_tpi[sorted_tpis[0]][0]
    first_dcm = first_record[1]
    rows, cols = int(first_dcm.Rows), int(first_dcm.Columns)
    slice_thickness = float(getattr(first_dcm, "SliceThickness", 1.0))
    affine = build_affine_from_dicom(first_dcm, slice_thickness, transpose_slices=True)

    if n_timepoints == 1:
        slices = sorted(by_tpi[sorted_tpis[0]], key=lambda x: x[0])
        data = np.zeros((cols, rows, len(slices)), dtype=np.float32)
        for i, (_loc, _dcm, arr) in enumerate(slices):
            data[:, :, i] = arr.astype(np.float32).T
        time_points = None
        logger.info(
            "load_dicom_series[%s]: 3D volume %s",
            series.uid[-8:],
            data.shape,
        )
    else:
        # 4D — Mode A dynamic.
        n_slices = max(len(by_tpi[t]) for t in sorted_tpis)
        # All timepoints should have the same slice count; warn if not.
        slice_counts = {len(by_tpi[t]) for t in sorted_tpis}
        if len(slice_counts) > 1:
            logger.warning(
                "load_dicom_series[%s]: inconsistent slice counts across "
                "temporal positions (%s); using %d and padding with zeros",
                series.uid[-8:],
                sorted(slice_counts),
                n_slices,
            )
        data = np.zeros((cols, rows, n_slices, n_timepoints), dtype=np.float32)
        time_points = _build_time_vector(by_tpi, sorted_tpis, first_dcm, n_slices)
        for t_out, t_idx in enumerate(sorted_tpis):
            slices = sorted(by_tpi[t_idx], key=lambda x: x[0])
            for z_idx, (_loc, _dcm, arr) in enumerate(slices):
                data[:, :, z_idx, t_out] = arr.astype(np.float32).T
        logger.info(
            "load_dicom_series[%s]: 4D volume %s, t=[%.1f..%.1f]s",
            series.uid[-8:],
            data.shape,
            float(time_points[0]),
            float(time_points[-1]),
        )

    acq = _acquisition_params(first_dcm, existing=series)
    return PerfusionDataset(
        data=data,
        affine=affine,
        modality=modality or Modality.DCE,
        time_points=time_points,
        acquisition_params=acq,
        source_path=series.files[0].parent,
        source_format="dicom",
    )


def _build_time_vector(
    by_tpi: dict[int, list[tuple[float, Any, np.ndarray]]],
    sorted_tpis: list[int],
    first_dcm: Any,
    n_slices: int,
) -> np.ndarray:
    """Derive per-volume timing for a Mode-A dynamic series.

    Preference order:
        TriggerTime (ms, per-frame dynamic timing — what Philips uses)
        → AcquisitionTime (wall clock)
        → synthetic ``TR × n_slices`` fallback.
    """
    # Try TriggerTime — one value per volume, from first slice.
    trigger_per_volume: list[float | None] = []
    for t_idx in sorted_tpis:
        first = by_tpi[t_idx][0][1]
        tt = getattr(first, "TriggerTime", None)
        trigger_per_volume.append(float(tt) if tt is not None else None)
    if all(t is not None for t in trigger_per_volume) and len(
        set(trigger_per_volume)
    ) == len(sorted_tpis):
        arr = np.array(trigger_per_volume) / 1000.0
        arr = arr - arr[0]
        return arr

    # Try AcquisitionTime.
    acq_per_volume: list[float | None] = []
    for t_idx in sorted_tpis:
        first = by_tpi[t_idx][0][1]
        at = getattr(first, "AcquisitionTime", None)
        acq_per_volume.append(_parse_dicom_time(str(at)) if at is not None else None)
    if all(t is not None for t in acq_per_volume) and len(set(acq_per_volume)) == len(
        sorted_tpis
    ):
        arr = np.array(acq_per_volume, dtype=float)
        arr = arr - arr[0]
        return arr

    # Fall back to TR × slices.
    tr_ms = float(getattr(first_dcm, "RepetitionTime", 0.0)) or 0.0
    dt = tr_ms * n_slices / 1000.0 if tr_ms > 0 else 1.0
    logger.warning(
        "load_dicom_series: no usable TriggerTime/AcquisitionTime; "
        "synthesizing time vector from TR × n_slices (dt=%.2fs)",
        dt,
    )
    return np.arange(len(sorted_tpis), dtype=float) * dt


def _load_multi_series(
    frames: list[SeriesInfo],
    modality: Modality | None,
) -> PerfusionDataset:
    """Stack per-timepoint series into a 4D dynamic (Mode B)."""
    if len(frames) < 2:
        msg = f"load_dicom_series: need >=2 series for a multi-series dynamic, got {len(frames)}"
        raise DataValidationError(msg)

    # Order frames: TT-suffix hint → AcquisitionTime → SeriesNumber.
    def _sort_key(s: SeriesInfo) -> tuple[int, float]:
        if s.trigger_time_hint is not None:
            return (0, float(s.trigger_time_hint))
        if s.acquisition_time_sec is not None:
            return (1, float(s.acquisition_time_sec))
        if s.series_number is not None:
            return (2, float(s.series_number))
        return (3, 0.0)

    ordered = sorted(frames, key=_sort_key)
    timing_source = (
        "TT=X.Xs suffix"
        if ordered[0].trigger_time_hint is not None
        else "AcquisitionTime"
        if ordered[0].acquisition_time_sec is not None
        else "SeriesNumber"
    )
    logger.info(
        "load_dicom_series: assembling %d per-timepoint series "
        "(group=%r, ordering by %s)",
        len(ordered),
        ordered[0].group_key,
        timing_source,
    )

    volumes: list[np.ndarray] = []
    times: list[float] = []
    first_dcm: Any = None
    reference_xy: tuple[int, int] | None = None
    reference_affine: np.ndarray | None = None

    for i, frame in enumerate(ordered):
        records, n_dropped = _read_dicom_slices(frame.files, drop_phase=True)
        if n_dropped:
            logger.debug(
                "  frame %d: dropped %d phase/unreadable slices",
                i,
                n_dropped,
            )
        if not records:
            msg = f"No readable frames in series {frame.uid}"
            raise OsipyIOError(msg)

        records.sort(key=lambda r: r[1])  # by slice location
        _t_idx, _loc, dcm, _arr = records[0]
        if first_dcm is None:
            first_dcm = dcm
            slice_thickness = float(getattr(dcm, "SliceThickness", 1.0))
            reference_affine = build_affine_from_dicom(
                dcm,
                slice_thickness,
                transpose_slices=True,
            )
        rows, cols = int(dcm.Rows), int(dcm.Columns)
        vol = np.zeros((cols, rows, len(records)), dtype=np.float32)
        for z, (_t, _l, _d, arr) in enumerate(records):
            vol[:, :, z] = arr.astype(np.float32).T

        if reference_xy is None:
            reference_xy = (cols, rows)
        elif (cols, rows) != reference_xy:
            msg = (
                f"Frame in-plane shape mismatch: series {frame.uid} has "
                f"{(cols, rows)}, expected {reference_xy}"
            )
            raise DataValidationError(msg)

        volumes.append(vol)

        # Resolve time for this frame.
        if frame.trigger_time_hint is not None:
            times.append(float(frame.trigger_time_hint))
        elif frame.acquisition_time_sec is not None:
            times.append(float(frame.acquisition_time_sec))
        elif frame.series_number is not None:
            times.append(float(frame.series_number))
        else:
            times.append(float(i))

    # Equalize Z across frames — vendors occasionally drop an end slice
    # (e.g. 127 vs 128 from Siemens TWIST). Trim to the common minimum.
    slice_counts = [v.shape[2] for v in volumes]
    if len(set(slice_counts)) > 1:
        n_z = min(slice_counts)
        logger.warning(
            "load_dicom_series: frames have inconsistent slice counts %s; "
            "trimming all to %d to enable stacking",
            sorted(set(slice_counts)),
            n_z,
        )
        volumes = [v[:, :, :n_z] for v in volumes]

    data_4d = np.stack(volumes, axis=-1)
    time_arr = np.array(times, dtype=float)
    if time_arr[0] != 0.0:
        logger.info(
            "load_dicom_series: zero-referencing time vector (offset %.1fs)",
            float(time_arr[0]),
        )
        time_arr = time_arr - time_arr[0]

    acq = _acquisition_params(first_dcm, existing=ordered[0])
    logger.info(
        "load_dicom_series: 4D multi-series volume %s, t=[%.1f..%.1f]s",
        data_4d.shape,
        float(time_arr[0]),
        float(time_arr[-1]),
    )

    return PerfusionDataset(
        data=data_4d,
        affine=reference_affine,
        modality=modality or Modality.DCE,
        time_points=time_arr,
        acquisition_params=acq,
        source_path=ordered[0].files[0].parent.parent,
        source_format="dicom",
    )


def _load_vfa_stack(
    series_list: list[SeriesInfo],
    modality: Modality | None,
) -> PerfusionDataset:
    """Stack a list of single-flip-angle series into a VFA 4D volume.

    The last axis is flip angle (not time). ``time_points`` is set to an
    integer placeholder because :class:`PerfusionDataset` requires a time
    vector for 4D data; downstream VFA consumers read ``flip_angles`` from
    ``acquisition_params`` instead.
    """
    if len(series_list) < 2:
        msg = f"VFA stacking requires >= 2 series, got {len(series_list)}"
        raise DataValidationError(msg)
    for s in series_list:
        if s.flip_angle is None:
            msg = (
                f"Series {s.uid} has no (or non-uniform) FlipAngle; cannot stack as VFA"
            )
            raise DataValidationError(msg)

    ordered = sorted(series_list, key=lambda s: float(s.flip_angle))
    volumes = [_load_single_series(s, modality) for s in ordered]

    reference_shape = volumes[0].data.shape
    for v, s in zip(volumes, ordered, strict=True):
        if v.data.shape != reference_shape:
            msg = (
                f"VFA shape mismatch: series {s.uid} has {v.data.shape}, "
                f"expected {reference_shape}"
            )
            raise DataValidationError(msg)

    data = np.stack([v.data for v in volumes], axis=-1)
    flip_angles = [float(s.flip_angle) for s in ordered]
    first_acq = volumes[0].acquisition_params
    acq = DCEAcquisitionParams(
        tr=getattr(first_acq, "tr", None),
        te=getattr(first_acq, "te", None),
        flip_angles=flip_angles,
        field_strength=getattr(first_acq, "field_strength", None),
    )
    logger.info(
        "load_dicom_series: stacked %d VFA series along flip-angle axis "
        "(FAs=%s°), shape=%s",
        len(ordered),
        [f"{a:.0f}" for a in flip_angles],
        data.shape,
    )

    return PerfusionDataset(
        data=data,
        affine=volumes[0].affine,
        modality=modality or Modality.DCE,
        time_points=np.arange(len(ordered), dtype=float),
        acquisition_params=acq,
        source_path=ordered[0].files[0].parent,
        source_format="dicom",
    )


def load_dicom_series(
    series: SeriesInfo | list[SeriesInfo],
    modality: Modality | str | None = None,
) -> PerfusionDataset:
    """Load one or more DICOM series as a :class:`PerfusionDataset`.

    Parameters
    ----------
    series : SeriesInfo | list[SeriesInfo]
        Either a single :class:`SeriesInfo` (loaded as 3D or 4D depending
        on its TPI structure), or a list of series that should be stacked
        into one 4D volume. When every member of the list is VFA
        (``role_hint == "vfa"``), they are stacked along the flip-angle
        axis (sorted by flip angle) with ``acquisition_params.flip_angles``
        populated from the source series. Otherwise the list is treated
        as per-timepoint frames — each a separate 3D volume representing
        one dynamic timepoint — which are sorted by the embedded timing
        and stacked into a 4D dataset with a derived time vector.
    modality : Modality | str | None
        Attached to the returned ``PerfusionDataset``. Does not alter
        loading behaviour.

    Returns
    -------
    PerfusionDataset
        For single 3D series: ``time_points=None``. For 4D (single-series
        dynamic, per-timepoint stack of separate 3D volumes, or VFA
        stack): populated ``time_points``. VFA stacks use an integer
        placeholder because the last axis is flip angle, not time.

    Raises
    ------
    IOError
        If no magnitude frames can be read.
    DataValidationError
        If a multi-series load sees inconsistent shapes across frames,
        or if fewer than 2 series are passed as a list.
    """
    mod = _resolve_modality(modality)
    if isinstance(series, list):
        if not series:
            msg = "load_dicom_series: empty series list"
            raise DataValidationError(msg)
        if all(s.role_hint == "vfa" for s in series):
            return _load_vfa_stack(series, mod)
        return _load_multi_series(series, mod)
    return _load_single_series(series, mod)


def _resolve_modality(modality: Modality | str | None) -> Modality | None:
    if modality is None or isinstance(modality, Modality):
        return modality
    m = str(modality).upper().replace("-MRI", "")
    for candidate in Modality:
        if candidate.name == m or candidate.value.upper() == m:
            return candidate
    msg = f"Unknown modality: {modality}"
    raise OsipyIOError(msg)
