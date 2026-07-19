#!/usr/bin/env python3
"""
CFAD Analysis Runner with Configuration Integration and Comprehensive Validation
Integrates cfad_analysis_latest.py with the main project configuration with safeguards
"""

import sys
import os
import glob
import calendar
import shutil
import numpy as np
import pandas as pd
import xarray as xr
from typing import Tuple, List, Dict, Any

# Add parent directory to path to import config
sys.path.append("..")
sys.path.append(".")

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, script_dir)

try:
    import config
    from config import (
        RUN_CFAD_ANALYSIS,
        SKIP_EXISTING_ANALYSIS,
        REGIONAL_CFAD_ENABLED,
        REGIONS,
    )
    from utils import get_array_directory, get_data_directory
except ImportError:
    print(
        "Error: Could not import config.py. Make sure you're running from the project root or cfad_analysis directory."
    )
    sys.exit(1)

# Import the CFAD analysis functions
try:
    from cfad_analysis_latest import (
        compute_all_stats,
        plot_cfads,
        plot_profiles,
        plot_percentiles,
        bin_centers_dict,
    )
except ImportError:
    # If direct import fails, try importing from current directory
    import cfad_analysis_latest

    compute_all_stats = cfad_analysis_latest.compute_all_stats
    plot_cfads = cfad_analysis_latest.plot_cfads
    plot_profiles = cfad_analysis_latest.plot_profiles
    plot_percentiles = cfad_analysis_latest.plot_percentiles
    bin_centers_dict = cfad_analysis_latest.bin_centers_dict


def configure_cfad_bins(cfad_module) -> None:
    """Synchronize CFAD bin edges with the central config.py settings."""
    bin_specs = {
        "Z": (config.CFAD_Z_LIMITS, config.CFAD_DZ, "z"),
        "ZDR": (config.CFAD_ZDR_LIMITS, config.CFAD_DZDR, "zdr"),
        "rho": (config.CFAD_RHO_LIMITS, config.CFAD_DRHO, "rho"),
        "kdp": (config.CFAD_KDP_LIMITS, config.CFAD_DKDP, "kdp"),
    }

    cfad_module.bin_centers_dict.clear()
    for var, (limits, step, attr_prefix) in bin_specs.items():
        start, stop = limits
        bins = np.arange(start, stop + step * 0.5, step)
        centers = (bins[:-1] + bins[1:]) / 2
        setattr(cfad_module, f"{attr_prefix}_bins", bins)
        setattr(cfad_module, f"bins_{attr_prefix}", centers)
        cfad_module.bin_centers_dict[var] = centers


def _tracking_zarr_path(year: int, month: int, day: int, region: str = None) -> str:
    """Return the Stage 2/3 Zarr store used as CFAD input.

    For regional lifecycle CFADs, use the annotated ``all`` regional store, not
    the per-region subset. This preserves the full track history so target index
    0 means true track initiation, not first entry into a region.
    """
    month_name = calendar.month_abbr[month]
    date_str = f"{year:04d}{month:02d}{day:02d}"
    if region:
        return os.path.join(
            config.BASE_DATA_DIR,
            "Arrays_Regional",
            "all",
            str(year),
            f"{month_name}{day:02d}",
            f"KHGX{date_str}_regional.zarr",
        )
    return os.path.join(
        config.ARRAY_OUTPUT_DIR,
        str(year),
        f"{month_name}{day:02d}",
        f"KHGX{date_str}_tracking.zarr",
    )


def validate_cfad_input_arrays(
    year: int, month: int, day: int, region: str = None
) -> Tuple[bool, List[str], str]:
    """Validate that the processed Zarr tracking/regional store exists for CFAD analysis."""
    zarr_path = _tracking_zarr_path(year, month, day, region)
    array_dir = os.path.dirname(zarr_path)

    if not os.path.exists(zarr_path):
        print(
            f"❌ CFAD input validation failed: Zarr store does not exist: {zarr_path}"
        )
        return False, [], array_dir

    try:
        with xr.open_zarr(zarr_path, consolidated=False) as ds:
            n_obs = int(ds.sizes.get("obs", 0))
            n_samples = int(ds.sizes.get("sample", 0))
            if n_obs == 0 or n_samples == 0:
                print(f"⚠️  CFAD input has no observations for this target: {zarr_path}")
                return True, [], array_dir
            required = (
                "sample_obs",
                "reflectivity",
                "differential_reflectivity",
                "cross_correlation_ratio",
                "kdp",
            )
            missing = [name for name in required if name not in ds]
            if missing:
                print(
                    f"❌ CFAD input validation failed: {os.path.basename(zarr_path)} missing {missing}"
                )
                return False, [], array_dir
    except Exception as e:
        print(f"❌ CFAD input validation failed: Could not open {zarr_path}: {e}")
        return False, [], array_dir

    print(
        f"✅ CFAD input validation passed: {zarr_path} ({n_obs} obs, {n_samples} samples)"
    )
    return True, [zarr_path], array_dir


def _date_output_dir(year: int, month: int, day: int, region: str = None) -> str:
    """Date-scoped CFAD output directory."""
    day_dir = f"{calendar.month_abbr[month]}{day:02d}"
    base = os.path.join(config.CFAD_OUTPUT_DIR, str(year), day_dir)
    return os.path.join(base, region.capitalize()) if region else base


def _target_label(target_value: float | int) -> str:
    mode = getattr(config, "CFAD_TARGET_MODE", "index")
    text = str(target_value).replace(".", "p")
    if mode == "elapsed_minutes":
        return f"elapsed_{text}min"
    if mode == "lifetime_fraction":
        return f"lifetime_{text}"
    return f"index_{text}"


def check_cfad_output_exists(
    year: int, month: int, day: int, region: str = None
) -> Tuple[bool, Dict[str, bool], str]:
    """Check if CFAD analysis results already exist for the specified date.

    Args:
        year: Year (e.g., 2019)
        month: Month as integer (e.g., 6 for June)
        day: Day as integer (e.g., 29)

    Returns:
        Tuple[bool, Dict[str, bool], str]: (results_exist, output_status, output_directory)
    """
    output_dir = _date_output_dir(year, month, day, region)
    plot_output = os.path.join(output_dir, "cfad_plots")
    profile_output = os.path.join(output_dir, "profile_plots")

    # Check for various output files
    hist_files = glob.glob(os.path.join(output_dir, "hist_mean_*.npz"))
    hist_zarr = os.path.exists(os.path.join(output_dir, "cfad_histograms.zarr"))
    output_status = {
        "histograms": len(hist_files) > 0 or hist_zarr,
        "cfad_plots": False,
        "profile_plots": False,
    }

    # Check for plot files in appropriate directories
    if os.path.exists(plot_output):
        plot_files = glob.glob(f"{plot_output}/*.png")
        output_status["cfad_plots"] = len(plot_files) > 0

    if os.path.exists(profile_output):
        profile_files = glob.glob(f"{profile_output}/*.png")
        output_status["profile_plots"] = len(profile_files) > 0

    # Also check main output directory for plot files (fallback)
    if not output_status["cfad_plots"]:
        cfad_files = glob.glob(f"{output_dir}/cfad_plots*.png")
        output_status["cfad_plots"] = len(cfad_files) > 0

    if not output_status["profile_plots"]:
        profile_files = glob.glob(f"{output_dir}/profile_plots*.png")
        output_status["profile_plots"] = len(profile_files) > 0

    # Consider results to exist if we have histogram data and at least some plots
    results_exist = output_status["histograms"] and (
        output_status["cfad_plots"] or output_status["profile_plots"]
    )

    return results_exist, output_status, output_dir


def validate_array_data_quality(array_files: List[str]) -> Tuple[bool, Dict[str, Any]]:
    """Validate radar-profile quality in Zarr CFAD inputs."""
    quality_metrics = {
        "total_files": len(array_files),
        "valid_files": 0,
        "total_data_points": 0,
        "non_zero_points": 0,
        "suspicious_files": [],
    }

    try:
        for zarr_path in array_files:
            try:
                with xr.open_zarr(zarr_path, consolidated=False) as ds:
                    refl = ds["reflectivity"]
                    quality_metrics["valid_files"] += 1
                    quality_metrics["total_data_points"] += int(np.prod(refl.shape))
                    sample = refl.isel(
                        sample=slice(0, min(1000, refl.sizes.get("sample", 0)))
                    ).values
                    quality_metrics["non_zero_points"] += int(
                        np.count_nonzero(np.isfinite(sample))
                    )
            except Exception as e:
                quality_metrics["suspicious_files"].append(
                    f"{os.path.basename(zarr_path)}: {e}"
                )

        data_valid = (
            quality_metrics["valid_files"] > 0
            and quality_metrics["non_zero_points"] > 0
        )
        if data_valid:
            print("✅ Data quality validation passed:")
            print(
                f"   Valid stores: {quality_metrics['valid_files']}/{quality_metrics['total_files']}"
            )
            print(
                f"   Finite sampled data points: {quality_metrics['non_zero_points']}"
            )
        else:
            print("❌ Data quality validation failed:")
            print(
                f"   Valid stores: {quality_metrics['valid_files']}/{quality_metrics['total_files']}"
            )
            if quality_metrics["suspicious_files"]:
                print(f"   Issues: {quality_metrics['suspicious_files'][:3]}...")

        return data_valid, quality_metrics

    except Exception as e:
        print(f"❌ Error during data quality validation: {e}")
        return False, quality_metrics


def run_cfad_analysis_with_safeguards(
    year: int, month: int, day: int, force: bool = False, region: str = None
) -> bool:
    """Run CFAD analysis for a specific date with comprehensive safeguards.

    Args:
        year: Year (e.g., 2019)
        month: Month as integer (e.g., 6 for June)
        day: Day as integer (e.g., 29)
        force: If True, run analysis even if results already exist

    Returns:
        bool: True if analysis successful or results already exist, False on failure
    """
    print(f"\n{'=' * 60}")
    if region:
        print(f"CFAD ANALYSIS ({region.upper()}) - {year}-{month:02d}-{day:02d}")
    else:
        print(f"CFAD ANALYSIS - {year}-{month:02d}-{day:02d}")
    print(f"{'=' * 60}")

    # Check if CFAD analysis is enabled
    if not RUN_CFAD_ANALYSIS:
        print("CFAD analysis is disabled in configuration")
        return True

    # SAFEGUARD 1: Validate input arrays exist
    arrays_valid, array_files, array_dir = validate_cfad_input_arrays(
        year, month, day, region
    )
    if not arrays_valid:
        return False

    if not array_files:
        print(
            "No CFAD input observations for this target/region; skipping as successful."
        )
        return True

    # SAFEGUARD 2: Check if output already exists (unless forced)
    if not force and SKIP_EXISTING_ANALYSIS:
        results_exist, output_status, _ = check_cfad_output_exists(
            year, month, day, region
        )
        if results_exist:
            print(f"CFAD results already exist for {year}-{month:02d}-{day:02d}")
            print(f"Output status: {output_status}")
            print("Skipping analysis due to SKIP_EXISTING_ANALYSIS=True")
            return True

    # SAFEGUARD 3: Validate data quality
    data_valid, quality_metrics = validate_array_data_quality(array_files)
    if not data_valid:
        print("❌ Data quality validation failed - aborting CFAD analysis")
        return False

    print()
    print(f"Output directory: {_date_output_dir(year, month, day, region)}")
    print(f"Processing {len(array_files)} array files...")
    print(
        f"Data quality: {quality_metrics['valid_files']} valid files with {quality_metrics['non_zero_points']} data points"
    )

    try:
        # Update configuration for this specific date
        original_values = {}
        original_values["target_year"] = config.TARGET_YEAR
        original_values["target_month"] = config.TARGET_MONTH
        original_values["target_day"] = config.TARGET_DAY

        # Temporarily update config for this run
        config.TARGET_YEAR = year
        config.TARGET_MONTH = month
        config.TARGET_DAY = day

        # Run the analysis
        success = run_single_day_analysis(region)

        # Restore original configuration
        config.TARGET_YEAR = original_values["target_year"]
        config.TARGET_MONTH = original_values["target_month"]
        config.TARGET_DAY = original_values["target_day"]

        if success is None:
            print(
                "✅ CFAD analysis skipped: no observations matched this lifecycle/region selection"
            )
            return True
        if success:
            # Verify output was created
            results_exist, output_status, _ = check_cfad_output_exists(
                year, month, day, region
            )
            if results_exist:
                print("✅ CFAD analysis successful!")
                print(f"   Results: {output_status}")
                return True
            else:
                print("❌ CFAD analysis completed but no output found")
                return False
        else:
            print("❌ CFAD analysis failed")
            return False

    except Exception as e:
        print(f"❌ Error during CFAD analysis: {e}")
        import traceback

        traceback.print_exc()
        return False


def setup_cfad_environment():
    """Setup environment variables based on config.py"""

    # Create output directories
    os.makedirs(config.CFAD_OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CFAD_PLOT_OUTPUT, exist_ok=True)
    os.makedirs(config.CFAD_PROFILE_OUTPUT, exist_ok=True)

    # Set environment variables for CFAD analysis
    os.environ["CFAD_FOLDER_PATH"] = config.ARRAY_OUTPUT_DIR
    os.environ["CFAD_STATS_PATH"] = config.STATS_FILE
    os.environ["CFAD_OUTPUT_PATH"] = config.CFAD_OUTPUT_DIR
    os.environ["CFAD_PARENT_DIR"] = config.ARRAY_OUTPUT_DIR
    os.environ["CFAD_SAVE_DIR"] = config.CFAD_OUTPUT_DIR
    os.environ["CFAD_PLOT_OUTPUT"] = config.CFAD_PLOT_OUTPUT
    os.environ["CFAD_PROFILE_OUTPUT"] = config.CFAD_PROFILE_OUTPUT
    os.environ["CFAD_TARGET_INDICES"] = " ".join(map(str, config.CFAD_TARGET_INDICES))

    print("CFAD environment configured successfully")
    print(f"Output directory: {config.CFAD_OUTPUT_DIR}")
    print(f"Data source: {config.ARRAY_OUTPUT_DIR}")
    print(f"Stats file: {config.STATS_FILE}")


REGION_TO_CODE = {
    "unclassified": -1,
    "urban": 0,
    "downwind": 1,
    "right": 2,
    "upwind": 3,
    "left": 4,
}


def _selected_obs_indices(ds: xr.Dataset, target_value: float | int) -> np.ndarray:
    """Select one observation per track using the configured lifecycle target mode.

    Supported modes via ``CFAD_TARGET_MODE``:
    - ``index`` (default): Nth scan in each track; 0 = initiation.
    - ``elapsed_minutes``: observation nearest N minutes after track initiation.
    - ``lifetime_fraction``: observation nearest fraction of observed track lifetime
      (0.0 = initiation, 1.0 = final observed scan).

    For regional CFADs, regional filtering is applied after this global track
    lifecycle selection.
    """
    n_obs = int(ds.sizes.get("obs", 0))
    if n_obs == 0:
        return np.array([], dtype=np.int64)
    if "track_id" not in ds:
        return np.arange(n_obs, dtype=np.int64)

    mode = getattr(config, "CFAD_TARGET_MODE", "index")
    isolated_only = bool(getattr(config, "CFAD_ISOLATED_ONLY", False))
    track_ids = ds["track_id"].values
    scan_idx = ds["scan_idx"].values if "scan_idx" in ds else np.arange(n_obs)
    isolated = (
        ds["track_is_isolated"].values
        if isolated_only and "track_is_isolated" in ds
        else None
    )
    selected = []
    for track_id in pd.unique(track_ids):
        obs_for_track = np.where(track_ids == track_id)[0]
        obs_for_track = obs_for_track[np.argsort(scan_idx[obs_for_track])]
        if len(obs_for_track) == 0:
            continue
        if isolated is not None and not bool(
            np.all(isolated[obs_for_track].astype(bool))
        ):
            continue

        if mode == "index":
            idx = int(target_value)
            if len(obs_for_track) > idx:
                selected.append(obs_for_track[idx])
        elif mode == "elapsed_minutes":
            if "scan_time" in ds and "scan_idx" in ds:
                times = pd.to_datetime(ds["scan_time"].values[scan_idx[obs_for_track]])
                elapsed_min = (times - times[0]).total_seconds() / 60.0
            else:
                elapsed_min = np.arange(len(obs_for_track), dtype=float) * 5.0
            selected.append(
                obs_for_track[
                    int(np.nanargmin(np.abs(elapsed_min - float(target_value))))
                ]
            )
        elif mode == "lifetime_fraction":
            frac = min(max(float(target_value), 0.0), 1.0)
            idx = int(round(frac * (len(obs_for_track) - 1)))
            selected.append(obs_for_track[idx])
        else:
            raise ValueError(f"Unsupported CFAD_TARGET_MODE: {mode!r}")
    return np.array(selected, dtype=np.int64)


def _filter_lifecycle_obs_by_region(
    ds: xr.Dataset, obs_indices: np.ndarray, region: str | None
) -> np.ndarray:
    """Keep selected lifecycle observations whose own centroid region matches region."""
    if not region:
        return obs_indices
    if "region_code" not in ds:
        raise ValueError(
            "Regional CFAD requires region_code in the annotated all-regions Zarr store"
        )
    code = REGION_TO_CODE[region]
    region_codes = ds["region_code"].values
    return obs_indices[region_codes[obs_indices] == code]


def _field_matrix_for_obs(
    ds: xr.Dataset, obs_indices: np.ndarray, sample_obs: np.ndarray, field: str
) -> np.ndarray:
    """Return concatenated ragged profiles for selected obs as (sample, level)."""
    keep_sample = np.isin(sample_obs, obs_indices)
    sample_idx = np.where(keep_sample)[0]
    if len(sample_idx) == 0:
        return np.empty((0, int(ds.sizes.get("level", 0))), dtype=float)
    return ds[field].isel(sample=sample_idx).values.astype(float)


def _write_cfad_histograms_zarr(
    save_data: str,
    label: str,
    save_dict: dict,
    heights: np.ndarray,
    n_observations: int,
    region: str | None = None,
) -> None:
    """Write CFAD histogram products as a Zarr dataset alongside legacy NPZ files."""
    data_vars = {}
    for var in ("Z", "ZDR", "rho", "kdp"):
        hist_key = f"hist_{var}"
        raw_key = f"raw_hist_{var}"
        ctr_key = f"{var}ctr"
        if hist_key not in save_dict:
            continue
        bin_dim = f"{var}_bin"
        data_vars[hist_key] = (
            (bin_dim, "level"),
            save_dict[hist_key].astype(np.float32),
        )
        data_vars[raw_key] = ((bin_dim, "level"), save_dict[raw_key].astype(np.float32))
        data_vars[ctr_key] = (bin_dim, save_dict[ctr_key].astype(np.float32))

    if not data_vars:
        return
    ds_out = xr.Dataset(
        data_vars,
        coords={
            "level": np.arange(len(heights), dtype=np.int16),
            "height_m": ("level", np.asarray(heights, dtype=np.float32)),
        },
        attrs={
            "target_label": label,
            "target_mode": getattr(config, "CFAD_TARGET_MODE", "index"),
            "n_observations": int(n_observations),
            "region": region or "all",
            "description": "CFAD normalized and raw histograms computed directly from ragged profile Zarr inputs",
        },
    )
    out_path = os.path.join(save_data, "cfad_histograms.zarr")
    shutil.rmtree(os.path.join(out_path, label), ignore_errors=True)
    ds_out.to_zarr(out_path, mode="a", group=label, consolidated=False)


def process_cfad_zarr_data(
    cfad_module,
    zarr_path: str,
    save_data: str,
    target_time_index: float | int,
    heights: np.ndarray,
    norm_opt: int,
    kdp_calc: bool,
    region: str | None = None,
    write_products: bool = True,
) -> bool | None:
    """Compute CFAD histograms directly from ragged Zarr profiles; no .npy intermediate.

    Returns True when histograms were produced, False on processing failure, and
    None when the lifecycle/region selection is scientifically valid but empty.
    """
    os.makedirs(save_data, exist_ok=True)
    vars_to_process = ["Z", "ZDR", "rho"] + (["kdp"] if kdp_calc else [])
    field_for_var = {
        "Z": "reflectivity",
        "ZDR": "differential_reflectivity",
        "rho": "cross_correlation_ratio",
        "kdp": "kdp",
    }
    bins_for_var = {
        "Z": cfad_module.z_bins,
        "ZDR": cfad_module.zdr_bins,
        "rho": cfad_module.rho_bins,
        "kdp": cfad_module.kdp_bins,
    }

    label = _target_label(target_time_index)
    with xr.open_zarr(zarr_path, consolidated=False) as ds:
        sample_obs = ds["sample_obs"].values.astype(np.int64)
        obs_indices = _selected_obs_indices(ds, target_time_index)
        obs_indices = _filter_lifecycle_obs_by_region(ds, obs_indices, region)
        if len(obs_indices) == 0:
            suffix = f" in region {region}" if region else ""
            print(
                f"No observations available for CFAD target index {target_time_index}{suffix}"
            )
            return None

        suffix = f" in region {region}" if region else ""
        print(
            f"Selected {len(obs_indices)} lifecycle observations for CFAD target index {target_time_index}{suffix}"
        )
        save_dict = {}
        for var in vars_to_process:
            matrix = _field_matrix_for_obs(
                ds, obs_indices, sample_obs, field_for_var[var]
            )
            if matrix.size == 0:
                print(f"Warning: No samples for variable {var}")
                continue

            # cfad_calc expects (height, point); Zarr stores (sample, level).
            data_var = matrix.T[: len(heights), :]
            hist, ctr, raw_hist = cfad_module.cfad_calc(
                data_var, bins_for_var[var], norm_opt
            )
            save_dict[f"hist_{var}"] = hist
            save_dict[f"raw_hist_{var}"] = raw_hist
            save_dict[f"{var}ctr"] = ctr
            cfad_module.aggregated_data[target_time_index]["total_raw_hist"][var] = (
                raw_hist
            )
            cfad_module.aggregated_data[target_time_index]["final_means"][
                f"hist_{var}"
            ] = hist

    if not save_dict:
        return False
    if write_products:
        _write_cfad_histograms_zarr(
            save_data, label, save_dict, heights, len(obs_indices), region
        )
    return True


def run_single_day_analysis(region: str = None):
    """Run single-day CFAD analysis using Zarr tracking/regional inputs."""

    # Override global variables in cfad_analysis_latest with config values
    import cfad_analysis_latest as cfad

    # Update configuration
    configure_cfad_bins(cfad)
    cfad.target_indices = config.CFAD_TARGET_INDICES
    cfad.target_year = config.TARGET_YEAR
    cfad.target_month = config.TARGET_MONTH
    cfad.target_day = config.TARGET_DAY
    cfad.ymax = config.CFAD_YMAX
    cfad.include_iqr_on_mean = config.CFAD_INCLUDE_IQR_ON_MEAN
    cfad.percentiles = config.CFAD_PERCENTILES
    cfad.zlims = config.CFAD_Z_LIMITS
    cfad.zdrlims = config.CFAD_ZDR_LIMITS
    cfad.rholims = config.CFAD_RHO_LIMITS
    cfad.kdplims = config.CFAD_KDP_LIMITS
    cfad.norm_opt = config.CFAD_NORM_OPT
    cfad.kdp_calc = config.CFAD_KDP_CALC
    cfad.profile_colors = config.CFAD_PROFILE_COLORS
    cfad.percentile_display = config.CFAD_PERCENTILE_DISPLAY

    # Set heights from the global variable in cfad module
    cfad.heights = cfad.heights

    # Date-scoped output paths prevent cross-day overwrite/skip collisions.
    cfad.folder_path = config.ARRAY_OUTPUT_DIR
    cfad.stats_path = config.STATS_FILE
    cfad.parent_directory = config.ARRAY_OUTPUT_DIR
    output_dir = _date_output_dir(
        config.TARGET_YEAR, config.TARGET_MONTH, config.TARGET_DAY, region
    )
    os.makedirs(output_dir, exist_ok=True)
    cfad.output_path = output_dir
    cfad.save_data = output_dir
    plot_dir = os.path.join(output_dir, "cfad_plots")
    profile_dir = os.path.join(output_dir, "profile_plots")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(profile_dir, exist_ok=True)
    cfad.plot_output = os.path.join(plot_dir, "cfad_plots")
    cfad.profile_output = os.path.join(profile_dir, "profile_plots")

    target_zarr = _tracking_zarr_path(
        config.TARGET_YEAR, config.TARGET_MONTH, config.TARGET_DAY, region
    )
    if not os.path.exists(target_zarr):
        print(f"Warning: Target Zarr store not found: {target_zarr}")
        return False

    with xr.open_zarr(target_zarr, consolidated=False) as ds_meta:
        if "z_levels" in ds_meta:
            cfad.heights = ds_meta["z_levels"].values.astype(float)

    print()
    print(f"Processing data from: {target_zarr}")
    if region:
        print(f"Regional selection: {region} (after global lifecycle-index selection)")
    print(
        f"Target date: {config.TARGET_YEAR}-{config.TARGET_MONTH:02d}-{config.TARGET_DAY:02d}"
    )
    print(f"Target indices: {config.CFAD_TARGET_INDICES}")

    # Initialize aggregated_data
    cfad.aggregated_data = {
        idx: {"total_raw_hist": {}, "final_means": {}, "stats": {}}
        for idx in config.CFAD_TARGET_INDICES
    }

    try:
        # Run the analysis for each target index
        for target_time_index in config.CFAD_TARGET_INDICES:
            print(f"Processing time index: {target_time_index}")

            try:
                success = process_cfad_zarr_data(
                    cfad,
                    target_zarr,
                    cfad.save_data,
                    target_time_index,
                    cfad.heights,
                    config.CFAD_NORM_OPT,
                    config.CFAD_KDP_CALC,
                    region=region,
                )
            except Exception as e:
                print(f"Zarr CFAD processing failed: {e}")
                print("Skipping CFAD analysis for this region.")
                return True  # Return success to continue with other regions

            if success is None:
                print(
                    f"No CFAD data for time index {target_time_index}; skipping this target/region."
                )
                return None
            if not success:
                print(f"❌ CFAD processing failed for time index {target_time_index}")
                return False

        # Compute statistics and generate plots
        print("Computing statistics...")
        compute_all_stats(
            config.CFAD_TARGET_INDICES,
            cfad.aggregated_data,
            bin_centers_dict,
            config.CFAD_PERCENTILES,
        )

        print("Generating CFAD plots...")
        plot_cfads(
            config.CFAD_TARGET_INDICES,
            cfad.aggregated_data,
            cfad.plot_output,
            cfad.heights,
            config.CFAD_YMAX,
            config.CFAD_Z_LIMITS,
            config.CFAD_ZDR_LIMITS,
            config.CFAD_RHO_LIMITS,
            config.CFAD_KDP_LIMITS,
            region,
        )

        print("Generating profile plots...")
        plot_profiles(
            config.CFAD_TARGET_INDICES,
            cfad.aggregated_data,
            cfad.profile_output,
            cfad.heights,
            config.CFAD_YMAX,
            config.CFAD_INCLUDE_IQR_ON_MEAN,
            region,
        )

        print("Generating percentile plots...")
        plot_percentiles(
            config.CFAD_TARGET_INDICES,
            cfad.aggregated_data,
            cfad.profile_output,
            cfad.heights,
            config.CFAD_YMAX,
            region,
        )

        print("Single-day CFAD analysis completed successfully!")
        return True

    except Exception as e:
        print(f"Error during CFAD analysis: {e}")
        import traceback

        traceback.print_exc()
        return False


def _discover_zarr_inputs(multi_cfg: dict, region: str | None = None) -> list[str]:
    """Discover tracking/regional Zarr stores for multi-temporal CFAD."""
    base = config.BASE_DATA_DIR if region else config.ARRAY_OUTPUT_DIR
    years = multi_cfg.get("years", "all")
    months = multi_cfg.get("months", "all")
    days = multi_cfg.get("days", "all")

    if years == "all":
        year_values = sorted(
            d for d in os.listdir(config.ARRAY_OUTPUT_DIR) if d.isdigit()
        )
    else:
        year_values = [str(y) for y in years]
    month_values = None if months == "all" else set(str(m) for m in months)
    day_values = None if days == "all" or days == ["all"] else set(str(d) for d in days)

    paths = []
    for year in year_values:
        year_dir = (
            os.path.join(base, "Arrays_Regional", "all", year)
            if region
            else os.path.join(base, year)
        )
        if not os.path.isdir(year_dir):
            continue
        for day_dir_name in sorted(os.listdir(year_dir)):
            day_dir = os.path.join(year_dir, day_dir_name)
            if not os.path.isdir(day_dir) or len(day_dir_name) < 5:
                continue
            if month_values is not None and day_dir_name[:3] not in month_values:
                continue
            if (
                day_values is not None
                and day_dir_name[3:] not in day_values
                and day_dir_name not in day_values
            ):
                continue
            pattern = "*_regional.zarr" if region else "*_tracking.zarr"
            paths.extend(sorted(glob.glob(os.path.join(day_dir, pattern))))
    return paths


def _normalized_from_raw(raw_hist: np.ndarray, norm_opt: int) -> np.ndarray:
    hist = raw_hist.astype(float).copy()
    if norm_opt == 1:
        level_sums = np.sum(hist, axis=0)
        valid = level_sums > 0
        hist[:, valid] /= level_sums[valid]
    elif norm_opt == 2:
        max_val = np.nanmax(hist) if hist.size else 0
        if max_val > 0:
            hist /= max_val
    return hist


def _run_multi_temporal_for_region(cfad, region: str | None) -> bool:
    multi_cfg = config.CFAD_MULTI_TEMPORAL.copy()
    paths = _discover_zarr_inputs(multi_cfg, region)
    label_region = region or "all"
    print(f"Found {len(paths)} Zarr stores for multi-temporal CFAD ({label_region})")
    if not paths:
        return True if region else False

    target_values = config.CFAD_TARGET_INDICES
    vars_to_process = ["Z", "ZDR", "rho"] + (["kdp"] if config.CFAD_KDP_CALC else [])
    bin_centers = {
        "Z": cfad.bins_z,
        "ZDR": cfad.bins_zdr,
        "rho": cfad.bins_rho,
        "kdp": cfad.bins_kdp,
    }
    cfad.aggregated_data = {
        idx: {"total_raw_hist": {}, "final_means": {}, "stats": {}}
        for idx in target_values
    }
    counts = {idx: 0 for idx in target_values}

    for zarr_path in paths:
        try:
            with xr.open_zarr(zarr_path, consolidated=False) as ds:
                if "z_levels" in ds:
                    cfad.heights = ds["z_levels"].values.astype(float)
        except Exception as exc:
            print(f"  Skipping unreadable store {zarr_path}: {exc}")
            continue

        tmp_dir = os.path.join(config.CFAD_OUTPUT_DIR, ".tmp_multi")
        os.makedirs(tmp_dir, exist_ok=True)
        for idx in target_values:
            before = {
                k: v.copy()
                for k, v in cfad.aggregated_data[idx]["total_raw_hist"].items()
            }
            ok = process_cfad_zarr_data(
                cfad,
                zarr_path,
                tmp_dir,
                idx,
                cfad.heights,
                config.CFAD_NORM_OPT,
                config.CFAD_KDP_CALC,
                region=region,
                write_products=False,
            )
            if ok:
                counts[idx] += 1
                # process_cfad_zarr_data wrote this period's raw hist into aggregated_data;
                # add to prior sums instead of replacing them.
                for var in vars_to_process:
                    current = cfad.aggregated_data[idx]["total_raw_hist"].get(var)
                    if current is None:
                        continue
                    cfad.aggregated_data[idx]["total_raw_hist"][var] = (
                        before.get(var, 0) + current
                    )

    any_data = False
    for idx in target_values:
        for var, raw_hist in list(cfad.aggregated_data[idx]["total_raw_hist"].items()):
            cfad.aggregated_data[idx]["final_means"][f"hist_{var}"] = (
                _normalized_from_raw(raw_hist, config.CFAD_NORM_OPT)
            )
            any_data = True
    if not any_data:
        shutil.rmtree(
            os.path.join(config.CFAD_OUTPUT_DIR, ".tmp_multi"), ignore_errors=True
        )
        print(f"No observations matched multi-temporal CFAD selection ({label_region})")
        return True if region else False

    suffix = multi_cfg.get("output_suffix", "multi_temporal")
    out_dir = os.path.join(
        config.CFAD_OUTPUT_DIR, "multi_temporal", suffix, label_region.capitalize()
    )
    os.makedirs(out_dir, exist_ok=True)
    plot_dir = os.path.join(out_dir, "cfad_plots")
    profile_dir = os.path.join(out_dir, "profile_plots")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(profile_dir, exist_ok=True)

    for idx in target_values:
        save_dict = {}
        for var in vars_to_process:
            raw = cfad.aggregated_data[idx]["total_raw_hist"].get(var)
            hist = cfad.aggregated_data[idx]["final_means"].get(f"hist_{var}")
            if raw is None or hist is None:
                continue
            save_dict[f"raw_hist_{var}"] = raw
            save_dict[f"hist_{var}"] = hist
            save_dict[f"{var}ctr"] = bin_centers[var]
        if save_dict:
            _write_cfad_histograms_zarr(
                out_dir,
                _target_label(idx),
                save_dict,
                cfad.heights,
                counts[idx],
                region,
            )

    print("Computing statistics from aggregated Zarr data...")
    compute_all_stats(
        target_values, cfad.aggregated_data, bin_centers_dict, config.CFAD_PERCENTILES
    )
    plot_cfads(
        target_values,
        cfad.aggregated_data,
        os.path.join(plot_dir, "cfad_plots"),
        cfad.heights,
        config.CFAD_YMAX,
        config.CFAD_Z_LIMITS,
        config.CFAD_ZDR_LIMITS,
        config.CFAD_RHO_LIMITS,
        config.CFAD_KDP_LIMITS,
        region,
    )
    plot_profiles(
        target_values,
        cfad.aggregated_data,
        os.path.join(profile_dir, "profile_plots"),
        cfad.heights,
        config.CFAD_YMAX,
        config.CFAD_INCLUDE_IQR_ON_MEAN,
        region,
    )
    plot_percentiles(
        target_values,
        cfad.aggregated_data,
        os.path.join(profile_dir, "profile_plots"),
        cfad.heights,
        config.CFAD_YMAX,
        region,
    )
    shutil.rmtree(
        os.path.join(config.CFAD_OUTPUT_DIR, ".tmp_multi"), ignore_errors=True
    )
    print(f"Multi-temporal CFAD completed for {label_region}: {counts}")
    return True


def run_multi_temporal_analysis():
    """Run multi-temporal CFAD analysis directly from Zarr stores."""
    print("=" * 60)
    print("RUNNING CFAD ANALYSIS - MULTI-TEMPORAL ZARR MODE")
    print("=" * 60)
    import cfad_analysis_latest as cfad

    configure_cfad_bins(cfad)
    cfad.norm_opt = config.CFAD_NORM_OPT
    cfad.kdp_calc = config.CFAD_KDP_CALC
    cfad.profile_colors = config.CFAD_PROFILE_COLORS
    cfad.percentile_display = config.CFAD_PERCENTILE_DISPLAY

    try:
        if REGIONAL_CFAD_ENABLED:
            ok = True
            for region in REGIONS:
                ok = _run_multi_temporal_for_region(cfad, region) and ok
            return ok
        return _run_multi_temporal_for_region(cfad, None)
    except Exception as e:
        print(f"Error during multi-temporal CFAD analysis: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_cfad_for_config_targets() -> bool:
    """Run CFAD analysis for targets specified in configuration.

    Uses CFAD_TARGET_YEAR, CFAD_TARGET_MONTH, CFAD_TARGET_DAY from config.

    Returns:
        bool: True if analysis successful, False otherwise
    """
    year = config.TARGET_YEAR
    month = config.TARGET_MONTH
    day = config.TARGET_DAY

    print(f"Running CFAD analysis for configured target: {year}-{month:02d}-{day:02d}")

    # Check if regional CFAD is enabled
    if REGIONAL_CFAD_ENABLED:
        print()
        print(f"Regional CFAD analysis enabled - processing {len(REGIONS)} regions")
        all_success = True

        # Process each region
        for region in REGIONS:
            success = run_cfad_analysis_with_safeguards(year, month, day, region=region)
            if not success:
                print(f"❌ Failed to process region: {region}")
                all_success = False
            else:
                print(f"✅ Successfully processed region: {region}")

        return all_success
    else:
        # Run standard analysis only
        return run_cfad_analysis_with_safeguards(year, month, day)


def main():
    """Main function to run CFAD analysis based on configuration with comprehensive validation"""

    # Check if CFAD analysis is enabled
    if not getattr(config, "RUN_CFAD_ANALYSIS", False):
        print("CFAD analysis is disabled in config.py")
        return False

    print()
    print(f"Base data directory: {config.BASE_DATA_DIR}")
    print(f"Array directory: {config.ARRAY_OUTPUT_DIR}")
    print(f"Output directory: {config.CFAD_OUTPUT_DIR}")
    print(f"Safeguards enabled: {getattr(config, 'SKIP_EXISTING_ANALYSIS', True)}")
    print()

    try:
        # Setup environment
        setup_cfad_environment()
        print()

        # Check if multi-temporal analysis is enabled (can be overridden by environment variable)
        multi_temporal_enabled = config.CFAD_MULTI_TEMPORAL.get("enabled", False)
        if os.environ.get("CFAD_MULTI_TEMPORAL_ENABLED", "").lower() == "false":
            multi_temporal_enabled = False

        if multi_temporal_enabled:
            print("Running multi-temporal CFAD analysis...")
            success = run_multi_temporal_analysis()
        else:
            print("Running single-day CFAD analysis with safeguards...")
            success = run_cfad_for_config_targets()

        print(f"\n{'=' * 80}")
        if success:
            print("CFAD ANALYSIS COMPLETED SUCCESSFULLY!")
            print(f"{'=' * 80}")
            print(f"Results saved to: {config.CFAD_OUTPUT_DIR}")
            if os.path.exists(config.CFAD_PLOT_OUTPUT):
                plot_count = len(glob.glob(f"{config.CFAD_PLOT_OUTPUT}/*.png"))
                print(f"CFAD plots: {plot_count} files in {config.CFAD_PLOT_OUTPUT}")
            if os.path.exists(config.CFAD_PROFILE_OUTPUT):
                profile_count = len(glob.glob(f"{config.CFAD_PROFILE_OUTPUT}/*.png"))
                print(
                    f"Profile plots: {profile_count} files in {config.CFAD_PROFILE_OUTPUT}"
                )
        else:
            print("CFAD ANALYSIS FAILED!")
        print(f"{'=' * 80}")

        return success

    except Exception as e:
        print(f"\\n{'=' * 80}")
        print("CFAD ANALYSIS PIPELINE ERROR!")
        print(f"{'=' * 80}")
        print(f"Unexpected error in main pipeline: {e}")
        import traceback

        traceback.print_exc()
        print(f"{'=' * 80}")
        return False


if __name__ == "__main__":
    main()
