#!/usr/bin/env python3
"""Stage 2: NEXRAD gridding, TOBAC cell detection + tracking → tracking.nc."""

import calendar
import glob
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
    BASE_DATA_DIR, RADAR_VISUALIZATION,
    SKIP_EXISTING_PROCESSING, YEAR_START, YEAR_END, VALID_MONTHS,
    TARGET_MODE, TARGET_YEAR, TARGET_MONTH, TARGET_DAY,
    GRID_BOUNDS, GRID_POINTS, GRID_SPACING, GRID_SPACING_M, VERTICAL_LIMIT,
    TOBAC_THRESHOLDS, TOBAC_SEGMENTATION_THRESHOLD, TOBAC_MIN_PIXELS,
    TOBAC_MAX_DISTANCE_PIXELS, TOBAC_MAX_GAP, GRID_N_WORKERS, RUN_RADAR_PROCESSING,
)
from utils import get_data_directory, get_array_directory, get_figures_directory

try:
    from radar_processing import (
        setup_radar_grid, process_radar_file,
        DetectionConfig, detect_cells, compute_eth_maps,
        link_tracks, build_track_masks,
        get_datetime_from_filename, create_radar_plot,
    )
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Radar processing modules not available: {e}")
    MODULES_AVAILABLE = False

_SAVE_FIELDS = ["reflectivity", "differential_reflectivity", "cross_correlation_ratio", "kdp"]


def _n_workers() -> int:
    return GRID_N_WORKERS or min(cpu_count(), 4)


# Module-level worker functions (must be at module level for ProcessPoolExecutor pickling)


def _pass1_worker(args: tuple) -> tuple:
    """Grid one radar file and detect cells. Returns 8-tuple ending with error_str (None on success)."""
    filename, scan_time, detection_cfg = args
    try:
        # kdp_parallel=False: prevents joblib from spawning threads inside each
        # worker process, which would oversubscribe CPUs when N workers are active.
        radar_fields = process_radar_file(filename, kdp_parallel=False, include_kdp=False)
        z_composite, local_mask, features = detect_cells(
            radar_fields, scan_time, detection_cfg
        )
        return filename, scan_time, z_composite, local_mask, features, radar_fields['lat_2d'], radar_fields['lon_2d'], None
    except Exception as exc:
        return filename, scan_time, None, None, None, None, None, str(exc)


def _pass2_worker(args: tuple) -> list[dict]:
    """Extract cell statistics for one scan. Reruns process_radar_file — full 3-D fields are too large to cache across Pass 1."""
    filename, scan_time, track_mask, z_composite, z_levels, pixel_area_km2, scan_idx = args
    try:
        radar_fields = process_radar_file(filename, kdp_parallel=False)
    except Exception as exc:
        print(f"  Failed (pass 2): {Path(filename).name} — {exc}")
        return []

    Z_filled = np.ma.filled(radar_fields["reflectivity"].astype(float), np.nan)
    eth_maps = compute_eth_maps(Z_filled, z_levels)

    return _collect_frame_obs(
        track_mask, radar_fields, Z_filled, z_composite,
        radar_fields['lat_2d'], radar_fields['lon_2d'], eth_maps, z_levels, pixel_area_km2, scan_idx,
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

    filled_fields = {fname: np.ma.filled(radar_fields[fname].astype(float), np.nan)
                     for fname in _SAVE_FIELDS}
    dz = float(z_levels[1] - z_levels[0]) if len(z_levels) > 1 else float(z_levels[0])

    obs_list: list[dict] = []
    for p in regions:
        ys, xs = p.coords[:, 0], p.coords[:, 1]
        cy, cx = int(round(p.centroid[0])), int(round(p.centroid[1]))

        valid_z = z_composite[ys, xs]
        valid_z = valid_z[np.isfinite(valid_z)]
        core_pixels = int(np.sum(valid_z >= 45.0))

        with np.errstate(all="ignore"):
            eth20m, eth30m, eth40m = np.nanmax(
                [eth_maps[t][ys, xs] for t in (20, 30, 40)], axis=1
            )

        profiles = {fname: filled_fields[fname][:, ys, xs] for fname in _SAVE_FIELDS}  # (n_levels, n_pts)
        ref_flat = profiles["reflectivity"]
        zdr_flat = profiles["differential_reflectivity"]
        rho_flat = profiles["cross_correlation_ratio"]
        kdp_flat = profiles["kdp"]

        # Height of max reflectivity — mean altitude of the column-max across cell pixels
        finite_cols = np.isfinite(ref_flat)
        valid_cols = finite_cols.any(axis=0)
        if valid_cols.any():
            safe = np.where(finite_cols, ref_flat, -np.inf)
            height_max_ref_m = float(np.mean(z_levels[np.argmax(safe, axis=0)[valid_cols]]))
        else:
            height_max_ref_m = np.nan

        # VIL (kg m⁻²) — column-integrated liquid, averaged across cell pixels
        Z_lin = np.where(finite_cols, 10.0 ** (ref_flat / 10.0), 0.0)
        vil_kg_m2 = float(np.mean(3.44e-6 * np.sum(Z_lin ** (4.0 / 7.0) * dz, axis=0)))

        obs_list.append({
            "scan_idx":            scan_idx,
            "track_id":            int(p.label),
            "centroid_y":          p.centroid[0],
            "centroid_x":          p.centroid[1],
            "centroid_lat":        float(lat_grid[cy, cx]),
            "centroid_lon":        float(lon_grid[cy, cx]),
            "area_km2":            p.area * pixel_area_km2,
            "n_points":            p.area,
            "ref_max_dbz":         float(np.nanmax(ref_flat))            if ref_flat.size else np.nan,
            "ref_mean_dbz":        float(np.nanmean(ref_flat))           if ref_flat.size else np.nan,
            "ref_p75_dbz":         float(np.nanpercentile(ref_flat, 75)) if ref_flat.size else np.nan,
            "eth_20dbz_m":         float(eth20m),
            "eth_30dbz_m":         float(eth30m),
            "eth_40dbz_m":         float(eth40m),
            "core_area_45dbz_km2": core_pixels * pixel_area_km2,
            "height_max_ref_m":    height_max_ref_m,
            "vil_kg_m2":           vil_kg_m2,
            "eccentricity":        float(p.eccentricity),
            "orientation_rad":     float(p.orientation),
            "zdr_mean":            float(np.nanmean(zdr_flat))           if zdr_flat.size else np.nan,
            "rhohv_mean":          float(np.nanmean(rho_flat))           if rho_flat.size else np.nan,
            "kdp_mean":            float(np.nanmean(kdp_flat))           if kdp_flat.size else np.nan,
            **{f"{fname}_profile": profiles[fname] for fname in _SAVE_FIELDS},
        })
    return obs_list


# NetCDF output


def _write_tracking_netcdf(
    nc_path: Path,
    valid_scans: list[tuple[str, pd.Timestamp]],
    track_masks: list[np.ndarray],
    z_composites: list[np.ndarray],
    all_obs: list[dict],
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    z_levels: np.ndarray,
) -> None:
    """Write one day's tracking results to a compressed NetCDF file."""
    scan_times_np = np.array([t.to_datetime64() for _, t in valid_scans])
    masks_arr = np.stack(track_masks).astype(np.int32)
    zcomp_arr = np.stack(z_composites).astype(np.float32)

    n_obs = len(all_obs)
    n_levels = len(z_levels)
    max_pts = int(max(o["n_points"] for o in all_obs)) if all_obs else 1

    scalar_vars = [
        "scan_idx", "track_id",
        "centroid_y", "centroid_x", "centroid_lat", "centroid_lon",
        "area_km2", "n_points",
        "ref_max_dbz", "ref_mean_dbz", "ref_p75_dbz",
        "eth_20dbz_m", "eth_30dbz_m", "eth_40dbz_m",
        "core_area_45dbz_km2",
        "height_max_ref_m", "vil_kg_m2",
        "eccentricity", "orientation_rad",
        "zdr_mean", "rhohv_mean", "kdp_mean",
    ]

    data_vars: dict = {
        "scan_time":   xr.Variable("scan", scan_times_np),
        "mask":        xr.Variable(["scan", "y", "x"], masks_arr,
                                    attrs={"long_name": "persistent track IDs (0 = background)"}),
        "z_composite": xr.Variable(["scan", "y", "x"], zcomp_arr,
                                    attrs={"units": "dBZ", "long_name": "column-max reflectivity"}),
        "lat":         xr.Variable(["y", "x"], lat_grid),
        "lon":         xr.Variable(["y", "x"], lon_grid),
        "z_levels":    xr.Variable("level", z_levels.astype(np.float32),
                                    attrs={"units": "m", "long_name": "altitude AGL"}),
    }

    if n_obs > 0:
        for vname in scalar_vars:
            dtype = np.int32 if vname in ("scan_idx", "track_id", "n_points") else np.float32
            data_vars[vname] = xr.Variable("obs", np.array([o[vname] for o in all_obs], dtype=dtype))

        for fname in _SAVE_FIELDS:
            arr = np.full((n_obs, n_levels, max_pts), np.nan, dtype=np.float32)
            for i, obs in enumerate(all_obs):
                prof = obs[f"{fname}_profile"]
                n_pts = prof.shape[1]
                arr[i, :, :n_pts] = prof
            data_vars[fname] = xr.Variable(["obs", "level", "point"], arr)

        # Cross-scan stats: lifetime, age since initiation, storm motion u/v
        sec = scan_times_np.astype("datetime64[s]").astype(np.float64)  # epoch seconds per scan index
        by_track: dict = {}
        for i, o in enumerate(all_obs):
            by_track.setdefault(o["track_id"], []).append(
                (o["scan_idx"], o["centroid_y"], o["centroid_x"], i)
            )
        lifetime_arr = np.full(n_obs, np.nan, np.float32)
        age_arr      = np.full(n_obs, np.nan, np.float32)
        motion_u_arr = np.full(n_obs, np.nan, np.float32)
        motion_v_arr = np.full(n_obs, np.nan, np.float32)
        for scans in by_track.values():
            scans.sort()
            t0 = sec[scans[0][0]]
            lifetime_s = float(sec[scans[-1][0]] - t0)
            for j, (sidx, cy, cx, i_obs) in enumerate(scans):
                lifetime_arr[i_obs] = lifetime_s
                age_arr[i_obs]      = float(sec[sidx] - t0)
                if len(scans) < 2:
                    continue
                j0, j1 = max(0, j - 1), min(len(scans) - 1, j + 1)
                p0, p1 = scans[j0], scans[j1]
                dt = float(sec[p1[0]] - sec[p0[0]])
                if dt > 0:
                    motion_u_arr[i_obs] = float((p1[2] - p0[2]) * GRID_SPACING_M / dt)
                    motion_v_arr[i_obs] = float((p1[1] - p0[1]) * GRID_SPACING_M / dt)
        data_vars["lifetime_s"]   = xr.Variable("obs", lifetime_arr)
        data_vars["age_s"]        = xr.Variable("obs", age_arr)
        data_vars["motion_u_ms"]  = xr.Variable("obs", motion_u_arr)
        data_vars["motion_v_ms"]  = xr.Variable("obs", motion_v_arr)

    ds = xr.Dataset(data_vars)
    enc = {fname: {"zlib": True, "complevel": 9} for fname in _SAVE_FIELDS if fname in ds}
    enc["mask"]        = {"zlib": True, "complevel": 9, "dtype": "int32"}
    enc["z_composite"] = {"zlib": True, "complevel": 4}
    ds.to_netcdf(nc_path, encoding=enc)


def _regen_figures(nc_path: Path, fig_dir: str, date_str: str, xx: np.ndarray, yy: np.ndarray) -> None:
    """Regenerate any missing scan figures from an existing tracking.nc."""
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(nc_path) as ds:
        if "z_composite" not in ds:
            print(f"  [viz] z_composite missing from {nc_path.name} (old format).")
            print("  [viz] Delete the .nc file and rerun to regenerate figures.")
            return
        z_comps    = ds["z_composite"].values          # (scan, y, x)
        masks      = ds["mask"].values                 # (scan, y, x) int32
        scan_times = pd.DatetimeIndex(ds["scan_time"].values)

    missing = [
        (i, f"KHGX{date_str}_{pd.Timestamp(t).strftime('%H%M%S')}_plot.png")
        for i, t in enumerate(scan_times)
        if not (Path(fig_dir) / f"KHGX{date_str}_{pd.Timestamp(t).strftime('%H%M%S')}_plot.png").exists()
    ]
    if not missing:
        return

    print(f"  [viz] Regenerating {len(missing)} missing figure(s) from tracking.nc …")
    for i, fig_name in missing:
        fig_path = Path(fig_dir) / fig_name
        try:
            create_radar_plot(xx, yy, z_comps[i], masks[i], fig_name, fig_path)
        except Exception as exc:
            print(f"  [viz] Failed: {fig_name} — {exc}")


# Day processing


def process_day(year: int, month: int, day: int) -> bool:
    """Two-pass parallel pipeline for one day. Returns True if output was written."""
    date_str = f"{year}{month:02d}{day:02d}"
    data_dir = get_data_directory(year, month, day, BASE_DATA_DIR)
    array_dir = get_array_directory(year, month, day, BASE_DATA_DIR)
    fig_dir = get_figures_directory(year, month, day, BASE_DATA_DIR)

    radar_files = sorted(glob.glob(f"{data_dir}/KHGX{date_str}_*_V06"))
    if not radar_files:
        return False

    tracking_nc = Path(array_dir) / f"KHGX{date_str}_tracking.nc"
    if SKIP_EXISTING_PROCESSING and tracking_nc.exists():
        if RADAR_VISUALIZATION:
            xx, yy, _ = setup_radar_grid(GRID_BOUNDS, GRID_POINTS, GRID_SPACING)
            _regen_figures(tracking_nc, fig_dir, date_str, xx, yy)
        return True

    Path(array_dir).mkdir(parents=True, exist_ok=True)

    xx, yy, z_levels = setup_radar_grid(GRID_BOUNDS, GRID_POINTS, GRID_SPACING)
    pixel_area_km2 = (GRID_SPACING_M / 1000.0) ** 2

    detection_cfg = DetectionConfig(
        thresholds=TOBAC_THRESHOLDS,
        segmentation_threshold=TOBAC_SEGMENTATION_THRESHOLD,
        min_pixels=TOBAC_MIN_PIXELS,
        grid_spacing_m=GRID_SPACING_M,
    )

    n_workers = _n_workers()

    # Pass 1 — grid + detect
    pass1_args = [
        (fn, t, detection_cfg)
        for fn in radar_files
        if (t := get_datetime_from_filename(Path(fn).name)) is not None
    ]

    if not pass1_args:
        return False

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        pass1_results = list(pool.map(_pass1_worker, pass1_args))

    valid_scans: list[tuple[str, pd.Timestamp]] = []
    local_masks:  list[np.ndarray] = []
    z_composites: list[np.ndarray] = []
    all_features: list[pd.DataFrame] = []
    lat_grid = lon_grid = None

    for filename, scan_time, z_composite, local_mask, features, lat, lon, error in pass1_results:
        if error:
            print(f"  Failed (pass 1): {Path(filename).name} — {error}")
            continue
        if lat_grid is None:
            lat_grid, lon_grid = lat, lon
        valid_scans.append((filename, scan_time))
        z_composites.append(z_composite)
        local_masks.append(local_mask)
        if features is not None:
            all_features.append(features)

    if not valid_scans or lat_grid is None:
        return False

    # Link tracks — sequential, needs all frames at once
    scan_times = [t for _, t in valid_scans]
    tracks = link_tracks(
        all_features, z_composites, scan_times,
        grid_spacing_m=GRID_SPACING_M,
        max_distance_px=TOBAC_MAX_DISTANCE_PIXELS,
        memory=TOBAC_MAX_GAP,
    )
    track_masks = build_track_masks(tracks, local_masks, scan_times)

    # Pass 2 — extract stats + profiles
    pass2_args = [
        (filename, scan_time, track_masks[i], z_composites[i], z_levels, pixel_area_km2, i)
        for i, (filename, scan_time) in enumerate(valid_scans)
        if np.any(track_masks[i] > 0)
    ]

    all_obs: list[dict] = []
    if pass2_args:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for obs_list in pool.map(_pass2_worker, pass2_args):
                all_obs.extend(obs_list)

    # Visualization — main process only (matplotlib is not fork-safe)
    if RADAR_VISUALIZATION:
        Path(fig_dir).mkdir(parents=True, exist_ok=True)
        print(f"  [viz] Writing {len(valid_scans)} figures to {fig_dir}")
        for i, (filename, _) in enumerate(valid_scans):
            basename = Path(filename).name
            fig_path = Path(fig_dir) / basename.replace("_V06", "_plot.png")
            try:
                create_radar_plot(xx, yy, z_composites[i], track_masks[i], basename, fig_path)
            except Exception as exc:
                print(f"  [viz] Failed: {basename} — {exc}")

    _write_tracking_netcdf(
        tracking_nc, valid_scans, track_masks, z_composites, all_obs, lat_grid, lon_grid, z_levels
    )
    return True


# Entry point


def main() -> None:
    if not RUN_RADAR_PROCESSING:
        print("Radar processing disabled in config.py")
        return

    if not MODULES_AVAILABLE:
        print("Radar processing modules unavailable. Run: pip install -e radar_processing/")
        sys.exit(1)

    if TARGET_MODE:
        ok = process_day(TARGET_YEAR, TARGET_MONTH, TARGET_DAY)
        print("Done." if ok else "No output written — check data directory.")
        return

    month_map = {calendar.month_abbr[i]: i for i in range(1, 13)}
    valid = set(VALID_MONTHS)
    success = failed = 0

    for year in range(YEAR_START, YEAR_END + 1):
        year_dir = Path(BASE_DATA_DIR) / str(year)
        if not year_dir.is_dir():
            continue

        day_dirs = sorted(
            d for d in year_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".") and d.name[:3] in valid
        )

        for day_dir in tqdm(day_dirs, desc=str(year), unit="day", ncols=80):
            month_num = month_map[day_dir.name[:3]]
            day_num = int(day_dir.name[3:])
            if process_day(year, month_num, day_num):
                success += 1
            else:
                failed += 1

    print(f"\nDone: {success} days processed, {failed} failed.")


if __name__ == "__main__":
    main()
