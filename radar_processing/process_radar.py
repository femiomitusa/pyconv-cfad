#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportPossiblyUnboundVariable=false, reportAttributeAccessIssue=false, reportArgumentType=false
"""Stage 2: NEXRAD gridding, TOBAC cell detection + tracking → tracking.zarr."""

import calendar
import glob
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from skimage.measure import regionprops
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    BASE_DATA_DIR,
    RADAR_VISUALIZATION,
    SKIP_EXISTING_PROCESSING,
    YEAR_START,
    YEAR_END,
    VALID_MONTHS,
    TARGET_MODE,
    TARGET_YEAR,
    TARGET_MONTH,
    TARGET_DAY,
    GRID_BOUNDS,
    GRID_POINTS,
    GRID_SPACING,
    GRID_SPACING_M,
    TOBAC_THRESHOLDS,
    TOBAC_SEGMENTATION_THRESHOLD,
    TOBAC_MIN_PIXELS,
    TOBAC_MAX_DISTANCE_PIXELS,
    TOBAC_MAX_GAP,
    GRID_N_WORKERS,
    RUN_RADAR_PROCESSING,
    VERBOSE_BATCH_LOGGING,
)
from utils import get_data_directory, get_array_directory, get_figures_directory

try:
    from radar_processing import (
        setup_radar_grid,
        process_radar_file,
        has_usable_reflectivity,
        DetectionConfig,
        detect_cells,
        compute_eth_maps,
        link_tracks,
        merge_split_flags,
        build_track_masks,
        compute_track_bearings,
        get_datetime_from_filename,
        create_radar_plot,
    )

    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Radar processing modules not available: {e}")
    MODULES_AVAILABLE = False

_SAVE_FIELDS = [
    "reflectivity",
    "differential_reflectivity",
    "cross_correlation_ratio",
    "kdp",
]


def _n_workers() -> int:
    return GRID_N_WORKERS or min(cpu_count(), 4)


# Module-level worker functions (must be at module level for ProcessPoolExecutor pickling)


def _remove_tree(path: Path) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception as exc:
        print(f"  Cleanup failed for {path}: {exc}")


def _fields_to_dataset(radar_fields: dict, z_levels: np.ndarray) -> xr.Dataset:
    """Convert one scan's gridded fields to a compact xarray Dataset for Zarr storage."""
    data_vars = {
        fname: xr.Variable(
            ["level", "y", "x"],
            np.ma.filled(radar_fields[fname].astype(np.float32), np.nan),
        )
        for fname in _SAVE_FIELDS
    }
    data_vars["lat"] = xr.Variable(
        ["y", "x"], radar_fields["lat_2d"].astype(np.float32)
    )
    data_vars["lon"] = xr.Variable(
        ["y", "x"], radar_fields["lon_2d"].astype(np.float32)
    )
    data_vars["z_levels"] = xr.Variable("level", z_levels.astype(np.float32))
    return xr.Dataset(data_vars)


def _save_scan_grid_zarr(path: Path, radar_fields: dict, z_levels: np.ndarray) -> None:
    """Write one scan's full gridded fields to its own Zarr store."""
    if path.exists():
        _remove_tree(path)
    ds = _fields_to_dataset(radar_fields, z_levels)
    encoding = {fname: {"chunks": (len(z_levels), 128, 128)} for fname in _SAVE_FIELDS}
    encoding["lat"] = {"chunks": (128, 128)}
    encoding["lon"] = {"chunks": (128, 128)}
    ds.to_zarr(path, mode="w", encoding=encoding, consolidated=False)


def _load_scan_grid_zarr(path: str | Path) -> dict:
    """Load one scan's gridded fields from a Zarr store into the existing dict shape."""
    with xr.open_zarr(path, consolidated=False) as ds:
        radar_fields = {
            fname: np.ma.masked_invalid(ds[fname].values) for fname in _SAVE_FIELDS
        }
        radar_fields["lat_2d"] = ds["lat"].values.astype(np.float32)
        radar_fields["lon_2d"] = ds["lon"].values.astype(np.float32)
    return radar_fields


def _pass1_worker(args: tuple) -> tuple:
    """Grid one radar file once, persist full fields to Zarr, and detect cells."""
    filename, scan_time, detection_cfg, temp_grid_path, z_levels = args
    try:
        # Pre-check: skip files with no usable reflectivity (avoid wasting resources on gridding)
        if not has_usable_reflectivity(
            filename, threshold_dbz=detection_cfg.segmentation_threshold
        ):
            raise ValueError("No reflectivity above threshold")

        # kdp_parallel=False: prevents joblib from spawning threads inside each
        # worker process, which would oversubscribe CPUs when N workers are active.
        radar_fields = process_radar_file(
            filename, kdp_parallel=False, include_kdp=True
        )
        _save_scan_grid_zarr(Path(temp_grid_path), radar_fields, z_levels)
        z_composite, local_mask, features = detect_cells(
            radar_fields, scan_time, detection_cfg
        )
        return (
            filename,
            scan_time,
            str(temp_grid_path),
            z_composite,
            local_mask,
            features,
            radar_fields["lat_2d"],
            radar_fields["lon_2d"],
            None,
        )
    except Exception as exc:
        return (
            filename,
            scan_time,
            str(temp_grid_path),
            None,
            None,
            None,
            None,
            None,
            str(exc),
        )


def _pass2_worker(args: tuple) -> list[dict]:
    """Extract cell statistics for one scan from the one-pass gridded Zarr store."""
    (
        filename,
        scan_time,
        temp_grid_path,
        track_mask,
        z_composite,
        z_levels,
        pixel_area_km2,
        scan_idx,
    ) = args
    try:
        radar_fields = _load_scan_grid_zarr(temp_grid_path)
    except Exception as exc:
        print(f"  Failed (pass 2): {Path(filename).name} — {exc}")
        return []

    Z_filled = np.ma.filled(radar_fields["reflectivity"].astype(float), np.nan)
    eth_maps = compute_eth_maps(Z_filled, z_levels)

    return _collect_frame_obs(
        track_mask,
        radar_fields,
        Z_filled,
        z_composite,
        radar_fields["lat_2d"],
        radar_fields["lon_2d"],
        eth_maps,
        z_levels,
        pixel_area_km2,
        scan_idx,
    )


# Per-cell-per-scan statistics


def _collect_frame_obs(
    track_mask: np.ndarray,
    radar_fields: dict,
    Z_filled: np.ndarray,
    z_composite: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    eth_maps: dict,
    z_levels: np.ndarray,
    pixel_area_km2: float,
    scan_idx: int,
) -> list[dict]:
    """Extract per-cell statistics and vertical profiles for one scan."""
    regions = regionprops(track_mask)
    if not regions:
        return []

    filled_fields = {
        fname: np.ma.filled(radar_fields[fname].astype(float), np.nan)
        for fname in _SAVE_FIELDS
    }
    try:
        dz = (
            float(z_levels[1] - z_levels[0])
            if len(z_levels) > 1
            else float(z_levels[0])
        )
    except Exception:
        dz = float(GRID_SPACING_M)

    obs_list: list[dict] = []
    for p in regions:
        try:
            ys, xs = p.coords[:, 0], p.coords[:, 1]
            cy, cx = int(round(p.centroid[0])), int(round(p.centroid[1]))

            valid_z = z_composite[ys, xs]
            valid_z = valid_z[np.isfinite(valid_z)]
            core_pixels = int(np.sum(valid_z >= 45.0))
        except Exception as exc:
            print(
                f"  Failed cell statistics setup for label {getattr(p, 'label', '?')}: {exc}"
            )
            continue

        with np.errstate(all="ignore"):
            eth20m, eth30m, eth40m = np.nanmax(
                [eth_maps[t][ys, xs] for t in (20, 30, 40)], axis=1
            )

        profiles = {
            fname: filled_fields[fname][:, ys, xs] for fname in _SAVE_FIELDS
        }  # (n_levels, n_pts)
        ref_flat = profiles["reflectivity"]
        zdr_flat = profiles["differential_reflectivity"]
        rho_flat = profiles["cross_correlation_ratio"]
        kdp_flat = profiles["kdp"]

        # Height of max reflectivity — mean altitude of the column-max across cell pixels
        finite_cols = np.isfinite(ref_flat)
        valid_cols = finite_cols.any(axis=0)
        try:
            if valid_cols.any():
                safe = np.where(finite_cols, ref_flat, -np.inf)
                height_max_ref_m = float(
                    np.mean(z_levels[np.argmax(safe, axis=0)[valid_cols]])
                )
            else:
                height_max_ref_m = np.nan

            # VIL (kg m⁻²) — column-integrated liquid, averaged across cell pixels
            Z_lin = np.where(finite_cols, 10.0 ** (ref_flat / 10.0), 0.0)
            vil_kg_m2 = float(
                np.mean(3.44e-6 * np.sum(Z_lin ** (4.0 / 7.0) * dz, axis=0))
            )
        except Exception:
            height_max_ref_m = np.nan
            vil_kg_m2 = np.nan

        try:
            track_id = int(p.label)
            centroid_lat = float(lat_grid[cy, cx])
            centroid_lon = float(lon_grid[cy, cx])
            ref_max = float(np.nanmax(ref_flat)) if ref_flat.size else np.nan
            ref_mean = float(np.nanmean(ref_flat)) if ref_flat.size else np.nan
            ref_p75 = float(np.nanpercentile(ref_flat, 75)) if ref_flat.size else np.nan
            eth20 = float(eth20m)
            eth30 = float(eth30m)
            eth40 = float(eth40m)
            eccentricity = float(p.eccentricity)
            orientation = float(p.orientation)
            zdr_mean = float(np.nanmean(zdr_flat)) if zdr_flat.size else np.nan
            rhohv_mean = float(np.nanmean(rho_flat)) if rho_flat.size else np.nan
            kdp_mean = float(np.nanmean(kdp_flat)) if kdp_flat.size else np.nan
        except Exception as exc:
            print(
                f"  Failed cell scalar statistics for label {getattr(p, 'label', '?')}: {exc}"
            )
            continue

        obs_list.append(
            {
                "scan_idx": scan_idx,
                "track_id": track_id,
                "centroid_y": p.centroid[0],
                "centroid_x": p.centroid[1],
                "centroid_lat": centroid_lat,
                "centroid_lon": centroid_lon,
                "area_km2": p.area * pixel_area_km2,
                "n_points": p.area,
                "ref_max_dbz": ref_max,
                "ref_mean_dbz": ref_mean,
                "ref_p75_dbz": ref_p75,
                "eth_20dbz_m": eth20,
                "eth_30dbz_m": eth30,
                "eth_40dbz_m": eth40,
                "core_area_45dbz_km2": core_pixels * pixel_area_km2,
                "height_max_ref_m": height_max_ref_m,
                "vil_kg_m2": vil_kg_m2,
                "eccentricity": eccentricity,
                "orientation_rad": orientation,
                "zdr_mean": zdr_mean,
                "rhohv_mean": rhohv_mean,
                "kdp_mean": kdp_mean,
                "profile_y": ys.astype(np.int16),
                "profile_x": xs.astype(np.int16),
                **{f"{fname}_profile": profiles[fname] for fname in _SAVE_FIELDS},
            }
        )
    return obs_list


# Zarr output


def _add_track_motion_variables(
    data_vars: dict,
    scan_times_np: np.ndarray,
    all_obs: list[dict],
    n_obs: int,
) -> None:
    """Add cross-scan lifetime/age/motion variables to the output dataset."""
    sec = scan_times_np.astype("datetime64[s]").astype(
        np.float64
    )  # epoch seconds per scan index
    by_track: dict = {}
    for i, o in enumerate(all_obs):
        by_track.setdefault(o["track_id"], []).append(
            (o["scan_idx"], o["centroid_y"], o["centroid_x"], i)
        )

    lifetime_arr = np.full(n_obs, np.nan, np.float32)
    age_arr = np.full(n_obs, np.nan, np.float32)
    motion_u_arr = np.full(n_obs, np.nan, np.float32)
    motion_v_arr = np.full(n_obs, np.nan, np.float32)
    for scans in by_track.values():
        scans.sort()
        try:
            t0 = sec[scans[0][0]]
            lifetime_s = float(sec[scans[-1][0]] - t0)
        except Exception:
            continue
        for j, (sidx, cy, cx, i_obs) in enumerate(scans):
            lifetime_arr[i_obs] = lifetime_s
            try:
                age_arr[i_obs] = float(sec[sidx] - t0)
            except Exception:
                age_arr[i_obs] = np.nan
            if len(scans) < 2:
                continue
            j0, j1 = max(0, j - 1), min(len(scans) - 1, j + 1)
            p0, p1 = scans[j0], scans[j1]
            try:
                dt = float(sec[p1[0]] - sec[p0[0]])
                if dt > 0:
                    motion_u_arr[i_obs] = float((p1[2] - p0[2]) * GRID_SPACING_M / dt)
                    motion_v_arr[i_obs] = float((p1[1] - p0[1]) * GRID_SPACING_M / dt)
            except Exception:
                continue

    data_vars["lifetime_s"] = xr.Variable("obs", lifetime_arr)
    data_vars["age_s"] = xr.Variable("obs", age_arr)
    data_vars["motion_u_ms"] = xr.Variable("obs", motion_u_arr)
    data_vars["motion_v_ms"] = xr.Variable("obs", motion_v_arr)


def _ragged_profile_vars(all_obs: list[dict], n_levels: int) -> dict:
    """Build ragged profile variables with one sample per actual cell pixel."""
    try:
        n_samples = int(sum(int(o["n_points"]) for o in all_obs))
    except Exception:
        n_samples = 0
    data_vars: dict = {
        "sample_obs": xr.Variable("sample", np.empty(n_samples, dtype=np.int32)),
        "sample_y": xr.Variable("sample", np.empty(n_samples, dtype=np.int16)),
        "sample_x": xr.Variable("sample", np.empty(n_samples, dtype=np.int16)),
    }
    for fname in _SAVE_FIELDS:
        data_vars[fname] = xr.Variable(
            ["sample", "level"],
            np.empty((n_samples, n_levels), dtype=np.float32),
        )

    sample_obs = data_vars["sample_obs"].data
    sample_y = data_vars["sample_y"].data
    sample_x = data_vars["sample_x"].data
    profile_data = {fname: data_vars[fname].data for fname in _SAVE_FIELDS}

    cursor = 0
    for obs_idx, obs in enumerate(all_obs):
        try:
            n_pts = int(obs["n_points"])
        except Exception:
            continue
        end = cursor + n_pts
        sample_obs[cursor:end] = obs_idx

        # Keep the actual grid-point locations for downstream regional/CFAD extraction.
        ys = np.asarray(obs["profile_y"], dtype=np.int16)
        xs = np.asarray(obs["profile_x"], dtype=np.int16)
        sample_y[cursor:end] = ys
        sample_x[cursor:end] = xs

        for fname in _SAVE_FIELDS:
            # Stored in observations as (level, point); write Zarr-friendly (sample, level).
            profile_data[fname][cursor:end, :] = np.asarray(
                obs[f"{fname}_profile"], dtype=np.float32
            ).T
        cursor = end

    return data_vars


def _write_tracking_zarr(
    zarr_path: Path,
    valid_scans: list[tuple[str, pd.Timestamp]],
    track_masks: list[np.ndarray],
    z_composites: list[np.ndarray],
    all_obs: list[dict],
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    z_levels: np.ndarray,
) -> None:
    """Write one day's tracking results to a Zarr store using ragged profile arrays."""
    if zarr_path.exists():
        _remove_tree(zarr_path)

    scan_times_np = np.array([t.to_datetime64() for _, t in valid_scans])
    masks_arr = np.stack(track_masks).astype(np.int32)
    zcomp_arr = np.stack(z_composites).astype(np.float32)

    n_obs = len(all_obs)
    n_levels = len(z_levels)

    scalar_vars = [
        "scan_idx",
        "track_id",
        "centroid_y",
        "centroid_x",
        "centroid_lat",
        "centroid_lon",
        "area_km2",
        "n_points",
        "ref_max_dbz",
        "ref_mean_dbz",
        "ref_p75_dbz",
        "eth_20dbz_m",
        "eth_30dbz_m",
        "eth_40dbz_m",
        "core_area_45dbz_km2",
        "height_max_ref_m",
        "vil_kg_m2",
        "eccentricity",
        "orientation_rad",
        "zdr_mean",
        "rhohv_mean",
        "kdp_mean",
        "motion_bearing_deg",
        "track_parent_id",
        "track_child_cell_count",
        "track_n_obs",
        "track_has_merge",
        "track_has_split",
        "track_is_isolated",
    ]

    data_vars: dict = {
        "scan_time": xr.Variable("scan", scan_times_np),
        "mask": xr.Variable(
            ["scan", "y", "x"],
            masks_arr,
            attrs={"long_name": "persistent track IDs (0 = background)"},
        ),
        "z_composite": xr.Variable(
            ["scan", "y", "x"],
            zcomp_arr,
            attrs={"units": "dBZ", "long_name": "column-max reflectivity"},
        ),
        "lat": xr.Variable(["y", "x"], lat_grid.astype(np.float32)),
        "lon": xr.Variable(["y", "x"], lon_grid.astype(np.float32)),
        "z_levels": xr.Variable(
            "level",
            z_levels.astype(np.float32),
            attrs={"units": "m", "long_name": "altitude AGL"},
        ),
    }

    if n_obs > 0:
        for vname in scalar_vars:
            dtype = (
                np.int32
                if vname
                in (
                    "scan_idx",
                    "track_id",
                    "n_points",
                    "track_parent_id",
                    "track_child_cell_count",
                    "track_n_obs",
                    "track_has_merge",
                    "track_has_split",
                    "track_is_isolated",
                )
                else np.float32
            )
            data_vars[vname] = xr.Variable(
                "obs", np.array([o[vname] for o in all_obs], dtype=dtype)
            )

        data_vars.update(_ragged_profile_vars(all_obs, n_levels))
        _add_track_motion_variables(data_vars, scan_times_np, all_obs, n_obs)

    ds = xr.Dataset(data_vars)
    ds.attrs.update(
        {
            "format": "PyMOOSAIC tracking Zarr",
            "profile_storage": "ragged sample arrays: sample_obs/sample_y/sample_x map samples to obs/grid points",
        }
    )

    encoding = {
        "mask": {"chunks": (1, GRID_POINTS, GRID_POINTS)},
        "z_composite": {"chunks": (1, GRID_POINTS, GRID_POINTS)},
        "lat": {"chunks": (GRID_POINTS, GRID_POINTS)},
        "lon": {"chunks": (GRID_POINTS, GRID_POINTS)},
    }
    if n_obs > 0:
        try:
            sample_chunk = min(max(int(ds.sizes.get("sample", 1)), 1), 50_000)
        except Exception:
            sample_chunk = 1
        for fname in _SAVE_FIELDS:
            encoding[fname] = {"chunks": (sample_chunk, n_levels)}
        for vname in ("sample_obs", "sample_y", "sample_x"):
            encoding[vname] = {"chunks": (sample_chunk,)}

    ds.to_zarr(zarr_path, mode="w", encoding=encoding, consolidated=False)


def _regen_figures(
    zarr_path: Path, fig_dir: str, date_str: str, xx: np.ndarray, yy: np.ndarray
) -> None:
    """Regenerate any missing scan figures from an existing tracking Zarr store."""
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    with xr.open_zarr(zarr_path, consolidated=False) as ds:
        if "z_composite" not in ds:
            print(f"  [viz] z_composite missing from {zarr_path.name}.")
            print("  [viz] Delete the .zarr store and rerun to regenerate figures.")
            return
        z_comps = ds["z_composite"].values  # (scan, y, x)
        masks = ds["mask"].values  # (scan, y, x) int32
        scan_times = pd.DatetimeIndex(ds["scan_time"].values)

    missing = [
        (i, f"KHGX{date_str}_{pd.Timestamp(t).strftime('%H%M%S')}_plot.png")
        for i, t in enumerate(scan_times)
        if not (
            Path(fig_dir)
            / f"KHGX{date_str}_{pd.Timestamp(t).strftime('%H%M%S')}_plot.png"
        ).exists()
    ]
    if not missing:
        return

    print(f"  [viz] Regenerating {len(missing)} missing figure(s) from tracking Zarr …")
    for i, fig_name in missing:
        fig_path = Path(fig_dir) / fig_name
        try:
            create_radar_plot(xx, yy, z_comps[i], masks[i], fig_name, fig_path)
        except Exception as exc:
            print(f"  [viz] Failed: {fig_name} — {exc}")


# Day processing


def _was_day_skipped(year: int, month: int, day: int) -> bool:
    """Check if day output already exists."""
    if not SKIP_EXISTING_PROCESSING:
        return False
    date_str = f"{year}{month:02d}{day:02d}"
    return (
        Path(get_array_directory(year, month, day, BASE_DATA_DIR))
        / f"KHGX{date_str}_tracking.zarr"
    ).exists()


def process_day(year: int, month: int, day: int) -> bool:
    """One-pass gridding pipeline for one day. Returns True if output was written."""
    date_str = f"{year}{month:02d}{day:02d}"
    array_dir = get_array_directory(year, month, day, BASE_DATA_DIR)
    fig_dir = get_figures_directory(year, month, day, BASE_DATA_DIR)

    # --- scan enumeration: NEXRAD Level-2 ---
    data_dir = get_data_directory(year, month, day, BASE_DATA_DIR)
    nexrad_files = sorted(glob.glob(f"{data_dir}/KHGX{date_str}_*_V06"))
    scan_items = [
        (fn, t)
        for fn in nexrad_files
        if (t := get_datetime_from_filename(Path(fn).name)) is not None
    ]

    if not scan_items:
        return False

    tracking_zarr = Path(array_dir) / f"KHGX{date_str}_tracking.zarr"
    if SKIP_EXISTING_PROCESSING and tracking_zarr.exists():
        if RADAR_VISUALIZATION:
            xx, yy, _ = setup_radar_grid(GRID_BOUNDS, GRID_POINTS, GRID_SPACING)
            _regen_figures(tracking_zarr, fig_dir, date_str, xx, yy)
        return True

    Path(array_dir).mkdir(parents=True, exist_ok=True)
    temp_grid_dir = Path(array_dir) / f".KHGX{date_str}_tmp_grids"
    if temp_grid_dir.exists():
        _remove_tree(temp_grid_dir)
    temp_grid_dir.mkdir(parents=True, exist_ok=True)

    xx, yy, z_levels = setup_radar_grid(GRID_BOUNDS, GRID_POINTS, GRID_SPACING)
    pixel_area_km2 = (GRID_SPACING_M / 1000.0) ** 2

    detection_cfg = DetectionConfig(
        thresholds=TOBAC_THRESHOLDS,
        segmentation_threshold=TOBAC_SEGMENTATION_THRESHOLD,
        min_pixels=TOBAC_MIN_PIXELS,
        grid_spacing_m=GRID_SPACING_M,
    )

    n_workers = _n_workers()

    # Build pass-1 args for NEXRAD Level-2 files.
    pass1_args = [
        (fn, t, detection_cfg, temp_grid_dir / f"{Path(fn).name}.zarr", z_levels)
        for fn, t in scan_items
    ]
    pass1_fn = _pass1_worker

    if not pass1_args:
        _remove_tree(temp_grid_dir)
        return False

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        pass1_results = list(pool.map(pass1_fn, pass1_args))

    valid_scans: list[tuple[str, pd.Timestamp]] = []
    temp_grid_paths: list[str] = []
    local_masks: list[np.ndarray] = []
    z_composites: list[np.ndarray] = []
    all_features: list[pd.DataFrame] = []
    lat_grid = lon_grid = None

    for (
        filename,
        scan_time,
        temp_grid_path,
        z_composite,
        local_mask,
        features,
        lat,
        lon,
        error,
    ) in pass1_results:
        if error:
            print(f"  Failed (pass 1): {Path(filename).name} — {error}")
            continue
        if lat_grid is None:
            lat_grid, lon_grid = lat, lon
        valid_scans.append((filename, scan_time))
        temp_grid_paths.append(temp_grid_path)
        z_composites.append(z_composite)
        local_masks.append(local_mask)
        if features is not None:
            all_features.append(features)

    if not valid_scans or lat_grid is None:
        _remove_tree(temp_grid_dir)
        return False

    # Link tracks — sequential, needs all frames at once
    scan_times = [t for _, t in valid_scans]
    tracks = link_tracks(
        all_features,
        z_composites,
        scan_times,
        grid_spacing_m=GRID_SPACING_M,
        max_distance_px=TOBAC_MAX_DISTANCE_PIXELS,
        memory=TOBAC_MAX_GAP,
    )
    merge_split_by_cell = merge_split_flags(
        tracks,
        grid_spacing_m=GRID_SPACING_M,
        max_distance_px=TOBAC_MAX_DISTANCE_PIXELS,
        frame_len=TOBAC_MAX_GAP + 1,
    )
    track_masks = build_track_masks(tracks, local_masks, scan_times)

    # Extract stats + profiles from the one-pass gridded Zarr stores.
    pass2_args = [
        (
            filename,
            scan_time,
            temp_grid_paths[i],
            track_masks[i],
            z_composites[i],
            z_levels,
            pixel_area_km2,
            i,
        )
        for i, (filename, scan_time) in enumerate(valid_scans)
        if np.any(track_masks[i] > 0)
    ]

    all_obs: list[dict] = []
    if pass2_args:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for obs_list in pool.map(_pass2_worker, pass2_args):
                all_obs.extend(obs_list)

    # Inject per-track mean motion bearing derived from consecutive centroid positions.
    # Singleton tracks (only one scan) receive NaN and are later marked unclassified.
    if all_obs:
        cell_bearings = compute_track_bearings(all_obs)
        track_counts = (
            pd.Series([obs["track_id"] for obs in all_obs]).value_counts().to_dict()
        )
        for obs in all_obs:
            try:
                track_id = int(obs["track_id"])
            except Exception:
                track_id = -1
            flags = merge_split_by_cell.get(track_id, {})
            obs["motion_bearing_deg"] = cell_bearings.get(track_id, np.nan)
            obs["track_parent_id"] = flags.get("track_parent_id", track_id)
            obs["track_child_cell_count"] = flags.get("track_child_cell_count", 1)
            obs["track_n_obs"] = track_counts.get(track_id, 1)
            obs["track_has_merge"] = 1 if flags.get("track_has_merge", False) else 0
            obs["track_has_split"] = 1 if flags.get("track_has_split", False) else 0
            obs["track_is_isolated"] = 1 if flags.get("track_is_isolated", True) else 0

    # Visualization — main process only (matplotlib is not fork-safe)
    if RADAR_VISUALIZATION:
        Path(fig_dir).mkdir(parents=True, exist_ok=True)
        print(f"  [viz] Writing {len(valid_scans)} figures to {fig_dir}")
        for i, (filename, _) in enumerate(valid_scans):
            basename = Path(filename).name
            fig_path = Path(fig_dir) / basename.replace("_V06", "_plot.png")
            try:
                create_radar_plot(
                    xx, yy, z_composites[i], track_masks[i], basename, fig_path
                )
            except Exception as exc:
                print(f"  [viz] Failed: {basename} — {exc}")

    _write_tracking_zarr(
        tracking_zarr,
        valid_scans,
        track_masks,
        z_composites,
        all_obs,
        lat_grid,
        lon_grid,
        z_levels,
    )

    _remove_tree(temp_grid_dir)
    return True


# Entry point


def main() -> None:
    if not RUN_RADAR_PROCESSING:
        print("Radar processing disabled in config.py")
        return

    if not MODULES_AVAILABLE:
        print(
            "Radar processing modules unavailable. Run: pip install -e radar_processing/"
        )
        sys.exit(1)

    if TARGET_MODE:
        ok = process_day(TARGET_YEAR, TARGET_MONTH, TARGET_DAY)
        print("Done." if ok else "No output written — check data directory.")
        return

    month_map = {calendar.month_abbr[i]: i for i in range(1, 13)}
    valid = set(VALID_MONTHS)
    skipped = processed = failed = 0

    for year in range(YEAR_START, YEAR_END + 1):
        year_dir = Path(BASE_DATA_DIR) / str(year)
        if not year_dir.is_dir():
            continue

        day_dirs = sorted(
            d
            for d in year_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".") and d.name[:3] in valid
        )

        with tqdm(day_dirs, desc=str(year), unit="day", ncols=100) as pbar:
            for day_dir in pbar:
                month_num = month_map[day_dir.name[:3]]
                try:
                    day_num = int(day_dir.name[3:])
                except Exception:
                    continue
                if _was_day_skipped(year, month_num, day_num):
                    skipped += 1
                elif process_day(year, month_num, day_num):
                    processed += 1
                else:
                    failed += 1
                if VERBOSE_BATCH_LOGGING:
                    pbar.set_postfix_str(f"✓ {processed} | ⊘ {skipped} | ✗ {failed}")

    print(f"\n{'=' * 55}")
    print(f"Batch {YEAR_START}–{YEAR_END} | ✓ {processed} | ⊘ {skipped} | ✗ {failed}")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
