#!/usr/bin/env python3
"""
Stage 0b — ERA5 Wind Download and Climatological Mean

Downloads 700 hPa u/v wind components from the Copernicus Climate Data Store
for June–August 2017–2025 over the Houston area, then computes the JJA
vector-mean steering-wind direction.

Prerequisites
-------------
1. Register at https://cds.climate.copernicus.eu and accept the ERA5 terms.
2. Install the CDS API key in ~/.cdsapirc:

       url: https://cds.climate.copernicus.eu/api
       key: <your-personal-access-token>

   The token is shown on your CDS profile page after login.

Usage
-----
    python3.12 stage0_download_era5.py            # download + compute mean
    python3.12 stage0_download_era5.py --compute-only  # recompute from cached file
"""

import argparse
import calendar
import json
import os
import sys
from pathlib import Path

# PROJ database fix — must precede any pyproj/geopandas import
_PROJ_DB = Path("/home/oomitusa/miniforge3/envs/metstat/share/proj")
if _PROJ_DB.exists():
    os.environ.setdefault("PROJ_DATA", str(_PROJ_DB))
    os.environ.setdefault("PROJ_LIB", str(_PROJ_DB))

import numpy as np

sys.path.append(str(Path(__file__).parent))
from config import REGIONAL_CONFIG, OUTPUT_DIR

# ── settings ──────────────────────────────────────────────────────────────────

ERA5_OUTPUT_DIR = Path(OUTPUT_DIR) / "era5"
ERA5_PRESSURE_LEVEL = "700"
ERA5_MONTHLY_DIR = ERA5_OUTPUT_DIR / "monthly_700hpa"
WIND_SUMMARY_FILE = ERA5_OUTPUT_DIR / "wind_climatology_700hpa.json"

CITY_LAT = REGIONAL_CONFIG["city_center_lat"]  # 29.4719
CITY_LON = REGIONAL_CONFIG["city_center_lon"]  # -95.0787

# Houston analysis box from config.py. REGIONAL_CONFIG stores (W, E, S, N),
# while the CDS API expects area as [N, W, S, E].
WEST, EAST, SOUTH, NORTH = REGIONAL_CONFIG["bounding_box_limits"]
AREA = [NORTH, WEST, SOUTH, EAST]
YEARS = [str(y) for y in range(2017, 2026)]
MONTHS = ["06", "07", "08"]  # JJA warm season


# ── download ──────────────────────────────────────────────────────────────────


def _monthly_file(year: str, month: str) -> Path:
    return ERA5_MONTHLY_DIR / f"era5_{ERA5_PRESSURE_LEVEL}hpa_{year}{month}.nc"


def _requested_files() -> list[Path]:
    return [_monthly_file(year, month) for year in YEARS for month in MONTHS]


def download_era5() -> None:
    """Download ERA5 in monthly chunks to stay under CDS request-size limits."""
    try:
        import cdsapi
    except ImportError:
        sys.exit("cdsapi not installed. Run: pip install cdsapi")

    ERA5_MONTHLY_DIR.mkdir(parents=True, exist_ok=True)

    print("  Connecting to Copernicus CDS …")
    print("  (requires ~/.cdsapirc with your personal access token)")
    c = cdsapi.Client()

    print(
        f"  Requesting ERA5 {ERA5_PRESSURE_LEVEL} hPa u/v for JJA {YEARS[0]}–{YEARS[-1]} in monthly chunks …"
    )
    print(f"  Area: N={AREA[0]} W={AREA[1]} S={AREA[2]} E={AREA[3]}")

    for year in YEARS:
        for month in MONTHS:
            output_file = _monthly_file(year, month)
            if output_file.exists():
                print(f"  Cached: {output_file}")
                continue

            last_day = calendar.monthrange(int(year), int(month))[1]
            print(f"  Downloading {year}-{month} → {output_file.name}")
            c.retrieve(
                "reanalysis-era5-pressure-levels",
                {
                    "product_type": "reanalysis",
                    "variable": ["u_component_of_wind", "v_component_of_wind"],
                    "pressure_level": ERA5_PRESSURE_LEVEL,
                    "year": year,
                    "month": month,
                    "day": [f"{d:02d}" for d in range(1, last_day + 1)],
                    "time": [f"{h:02d}:00" for h in range(24)],
                    "area": AREA,
                    "format": "netcdf",
                },
                str(output_file),
            )
            print(f"  Saved → {output_file}")


# ── compute climatological mean ───────────────────────────────────────────────


def compute_wind_climatology(nc_files: list[Path]) -> dict:
    """
    Compute the JJA vector-mean 700 hPa steering wind over the Houston analysis
    box and return a summary dict with from/to directions and component means.

    Vector mean is used instead of averaging angles directly to avoid the
    circular statistics wrap-around problem (e.g. averaging 359° and 1°
    should give 0°, not 180°).
    """
    try:
        import xarray as xr
    except ImportError:
        sys.exit("xarray not installed. Run: pip install xarray")

    if not nc_files:
        sys.exit(f"No ERA5 monthly files found in {ERA5_MONTHLY_DIR}")

    u_sum = 0.0
    v_sum = 0.0
    n_sum = 0

    for nc_file in nc_files:
        print(f"  Loading {nc_file} …")
        with xr.open_dataset(nc_file) as ds:
            # Variable names can differ between ERA5 versions
            u_name = "u" if "u" in ds else "u_component_of_wind"
            v_name = "v" if "v" in ds else "v_component_of_wind"

            u = ds[u_name]
            v = ds[v_name]
            valid = np.isfinite(u) & np.isfinite(v)
            count = int(valid.sum().values)
            if count == 0:
                continue
            u_sum += float(u.where(valid).sum(skipna=True).values)
            v_sum += float(v.where(valid).sum(skipna=True).values)
            n_sum += count

    if n_sum == 0:
        sys.exit("ERA5 files contain no valid u/v wind values.")

    # Spatial + temporal vector mean over all monthly files.
    u_mean = u_sum / n_sum
    v_mean = v_sum / n_sum

    # Meteorological convention:
    #   direction FROM which wind blows  → atan2(-v, -u) rotated to compass
    #   direction TO which wind blows (downwind centre) = from_dir + 180°
    wind_from = (270 - np.degrees(np.arctan2(v_mean, u_mean))) % 360
    wind_to = (wind_from + 180) % 360
    speed_ms = np.sqrt(u_mean**2 + v_mean**2)

    result = {
        "u_mean_ms": round(u_mean, 3),
        "v_mean_ms": round(v_mean, 3),
        "speed_ms": round(speed_ms, 2),
        "wind_from_deg": round(wind_from, 2),
        "wind_to_deg": round(wind_to, 2),
        "downwind_bearing": round(wind_to, 2),
        "season": "JJA",
        "years": f"{YEARS[0]}–{YEARS[-1]}",
        "level_hPa": int(ERA5_PRESSURE_LEVEL),
        "area_N_W_S_E": AREA,
    }

    return result


def print_summary(r: dict) -> None:
    print()
    print(f"  {'ERA5 JJA 700 hPa Steering Wind':^45}")
    print(f"  {'─' * 45}")
    print(f"  Period        : {r['season']} {r['years']}")
    print(f"  Mean u        : {r['u_mean_ms']:+.2f} m/s  (positive = westerly)")
    print(f"  Mean v        : {r['v_mean_ms']:+.2f} m/s  (positive = southerly)")
    print(f"  Mean speed    : {r['speed_ms']:.2f} m/s")
    print(f"  Wind FROM     : {r['wind_from_deg']:.1f}°")
    print(f"  Wind TO       : {r['wind_to_deg']:.1f}°  ← downwind bearing")
    print()
    print("  Update config.py:")
    print(f"    'temporal_wind': {r['downwind_bearing']},")
    print()


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Stage 0b — download ERA5 winds and compute downwind bearing"
    )
    parser.add_argument(
        "--compute-only",
        action="store_true",
        help="Skip download, recompute from cached NetCDF file.",
    )
    args = parser.parse_args()

    print(f"\n{'=' * 55}")
    print("  Stage 0b — ERA5 Wind Climatology")
    print(f"{'=' * 55}")

    expected_files = _requested_files()
    if not args.compute_only:
        download_era5()
    else:
        missing = [p for p in expected_files if not p.exists()]
        if missing:
            sys.exit(
                f"Missing {len(missing)} cached monthly ERA5 files in {ERA5_MONTHLY_DIR}. "
                "Run without --compute-only first."
            )

    print("\nComputing climatological mean …")
    available_files = [p for p in expected_files if p.exists()]
    result = compute_wind_climatology(available_files)
    print_summary(result)

    WIND_SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(WIND_SUMMARY_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Summary saved → {WIND_SUMMARY_FILE}")

    print(f"\n{'=' * 55}")
    print("  Stage 0b complete.")
    print(f"  700 hPa downwind bearing: {result['downwind_bearing']}°")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
