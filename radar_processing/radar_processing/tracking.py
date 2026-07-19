"""TOBAC-native persistent track linking and mask relabeling."""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import xarray as xr


def link_tracks(
    all_features: list[pd.DataFrame],
    z_composites: list[np.ndarray],
    scan_times: list[pd.Timestamp],
    grid_spacing_m: float,
    max_distance_px: int = 20,
    memory: int = 2,
) -> pd.DataFrame | None:
    """Link TOBAC features across all scans. Returns DataFrame with persistent `cell` IDs, or None."""
    import tobac  # lazy — keeps tobac/iris out of the parent process

    logging.getLogger("trackpy").setLevel(logging.WARNING)

    if not all_features:
        return None

    features_all = pd.concat(all_features, ignore_index=True)

    # Per-scan detection calls feature_detection_multithreshold with a single-frame
    # DataArray each time, so every feature gets frame=0.  Remap to the actual index
    # in the full scan_times sequence so TOBAC sees features in different frames.
    t64_to_idx = {t.to_datetime64(): i for i, t in enumerate(scan_times)}
    features_all["frame"] = [
        t64_to_idx.get(np.datetime64(t, "ns"), -1) for t in features_all["time"]
    ]

    # Median scan interval in seconds (NEXRAD varies 4–6 min between scans)
    if len(scan_times) > 1:
        try:
            dt_s = float(np.median(np.diff([t.timestamp() for t in scan_times])))
        except Exception:
            dt_s = 300.0
    else:
        dt_s = 300.0

    # Convert pixel displacement → m/s so the search radius is always
    # max_distance_px grid cells regardless of how long the scan interval is.
    v_max_ms = (max_distance_px * grid_spacing_m) / dt_s

    # Build (n_scans, ny, nx) DataArray with real timestamps so TOBAC assigns
    # correct frame indices when matching features to scans.
    times_np = np.array([t.to_datetime64() for t in scan_times])
    field_da = xr.DataArray(
        np.stack(z_composites).astype(np.float32),
        dims=["time", "y", "x"],
        coords={"time": times_np},
    )

    tracks = tobac.linking_trackpy(
        features_all,
        field_da,
        dt=dt_s,
        dxy=grid_spacing_m,
        v_max=v_max_ms,
        memory=memory,
        time_cell_min=dt_s,
        method_linking="random",  # O(N log N) KD-tree; 'predict' uses recursive subnet solver → slow on dense days
        adaptive_stop=0.1,  # floor: 10% of original search range before giving up on a subnet
        adaptive_step=0.96,  # shrink search radius by 4% per iteration when subnet exceeds size limit
    )
    return tracks


def merge_split_flags(
    tracks: pd.DataFrame | None,
    grid_spacing_m: float,
    max_distance_px: int = 25,
    frame_len: int = 5,
) -> dict[int, dict[str, int | bool]]:
    """Return TOBAC MEST merge/split metadata keyed by original cell ID."""
    if tracks is None or tracks.empty:
        return {}

    try:
        from tobac.merge_split import merge_split_MEST

        ms = merge_split_MEST(
            tracks,
            dxy=grid_spacing_m,
            distance=max_distance_px * grid_spacing_m,
            frame_len=frame_len,
        )
    except Exception as exc:
        logging.warning("TOBAC merge/split detection failed: %s", exc)
        return {}

    result: dict[int, dict[str, int | bool]] = {}
    parent_ids = ms["cell_parent_track_id"].to_series()
    child_counts = ms["track_child_cell_count"].to_series()
    starts_split = ms["cell_starts_with_split"].to_series()
    ends_merge = ms["cell_ends_with_merge"].to_series()

    for cell_id_raw, parent_id in parent_ids.items():
        try:
            cell_id = np.asarray(cell_id_raw).item()
            parent = int(parent_id)
            n_children = int(child_counts.loc[parent])
            has_split = bool(starts_split.loc[cell_id]) or n_children > 1
            has_merge = bool(ends_merge.loc[cell_id]) or n_children > 1
            result[int(cell_id)] = {
                "track_parent_id": parent,
                "track_child_cell_count": n_children,
                "track_has_split": has_split,
                "track_has_merge": has_merge,
                "track_is_isolated": n_children == 1
                and not has_split
                and not has_merge,
            }
        except Exception as exc:
            logging.warning("Skipping invalid merge/split row %r: %s", cell_id_raw, exc)
    return result


def build_track_masks(
    tracks: pd.DataFrame | None,
    local_masks: list[np.ndarray],
    scan_times: list[pd.Timestamp],
) -> list[np.ndarray]:
    """Relabel per-scan TOBAC masks (local feature IDs) with persistent cell IDs from linking."""
    if tracks is None:
        return [np.zeros_like(m) for m in local_masks]

    # Drop unassigned features (cell == -1 means TOBAC could not link them)
    assigned = tracks[tracks["cell"] >= 0]

    result: list[np.ndarray] = []
    for local_mask, scan_time in zip(local_masks, scan_times):
        t_np = scan_time.to_datetime64()
        frame_rows = assigned[assigned["time"] == t_np]
        feature_to_cell = dict(
            zip(frame_rows["feature"].astype(int), frame_rows["cell"].astype(int))
        )
        result.append(relabel_mask(local_mask, feature_to_cell))
    return result


def relabel_mask(mask: np.ndarray, feature_to_cell: dict[int, int]) -> np.ndarray:
    """Vectorized LUT relabeling of feature IDs → persistent cell IDs. Absent features → 0."""
    if not feature_to_cell:
        return np.zeros_like(mask)
    try:
        max_mask = int(mask.max())
    except Exception:
        return np.zeros_like(mask)
    lut = np.zeros(max_mask + 1, dtype=np.int32)
    for fid, cid in feature_to_cell.items():
        if 0 < fid <= max_mask:
            lut[fid] = cid
    return lut[mask]


def _geodetic_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Bearing in degrees clockwise from north, from (lat1,lon1) to (lat2,lon2)."""
    dlon = np.radians(lon2 - lon1)
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    try:
        return float((np.degrees(np.arctan2(x, y)) + 360) % 360)
    except Exception:
        return float("nan")


def compute_track_bearings(all_obs: list[dict]) -> dict[int, float]:
    # Vector mean of step bearings avoids circular-statistics wrap-around (e.g. 359°+1° → 0°, not 180°).
    by_track: dict[int, list[tuple]] = {}
    for o in all_obs:
        try:
            track_id = int(o["track_id"])
            frame = (
                int(o["scan_idx"]),
                float(o["centroid_lat"]),
                float(o["centroid_lon"]),
            )
        except Exception:
            continue
        by_track.setdefault(track_id, []).append(frame)

    result: dict[int, float] = {}
    for track_id, frames in by_track.items():
        frames.sort()
        if len(frames) < 2:
            result[track_id] = np.nan
            continue

        step_bearings = []
        for (_, lat0, lon0), (_, lat1, lon1) in zip(frames, frames[1:]):
            if lat0 != lat1 or lon0 != lon1:
                step_bearings.append(_geodetic_bearing(lat0, lon0, lat1, lon1))

        if not step_bearings:
            result[track_id] = np.nan
        else:
            rads = np.radians(step_bearings)
            try:
                result[track_id] = float(
                    (
                        np.degrees(
                            np.arctan2(np.mean(np.sin(rads)), np.mean(np.cos(rads)))
                        )
                        + 360
                    )
                    % 360
                )
            except Exception:
                result[track_id] = np.nan

    return result
