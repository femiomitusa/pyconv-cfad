#!/usr/bin/env python3
"""Stage 0c — HRRR target-day environmental fields.

Downloads selected HRRR analysis (F00) fields for the configured target day,
subsets them to the Houston analysis box, and writes an hourly NetCDF file for
cell-level environmental attachment.

Notes
-----
HRRR analysis fields are hourly. HRRR sub-hourly products are forecast fields,
not analyses, so this script intentionally uses F00 analysis files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import OUTPUT_DIR, REGIONAL_CONFIG, TARGET_DAY, TARGET_MONTH, TARGET_YEAR

HRRR_OUTPUT_DIR = Path(OUTPUT_DIR) / "hrrr"
HRRR_CACHE_DIR = HRRR_OUTPUT_DIR / "cache"

# Herbie/cfgrib search pattern for HRRR sfc F00 messages.
HRRR_SEARCH = (
    r":(CAPE|CIN):surface:"
    r"|:PWAT:"
    r"|:HPBL:"
    r"|:(UGRD|VGRD):(10 m above ground|500 mb|700 mb|850 mb):"
    r"|:(TMP|DPT):2 m above ground:"
)

OUTPUT_VARIABLES = [
    "cape",
    "cin",
    "pwat",
    "blh",
    "t2m",
    "d2m",
    "u10",
    "v10",
    "u500",
    "v500",
    "u700",
    "v700",
    "u850",
    "v850",
    "shear_500_850_ms",
    "wind700_from_deg",
    "wind700_to_deg",
]


def _as_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts is pd.NaT:
        raise ValueError(f"Invalid timestamp: {value!r}")
    return cast(pd.Timestamp, ts)


def _target_date_from_config() -> pd.Timestamp:
    return _as_timestamp(
        pd.Timestamp(year=TARGET_YEAR, month=TARGET_MONTH, day=TARGET_DAY)
    )


def _output_file(date: pd.Timestamp) -> Path:
    return HRRR_OUTPUT_DIR / f"hrrr_environment_{date:%Y%m%d}.nc"


def _analysis_hours(date: pd.Timestamp) -> list[pd.Timestamp]:
    start = _as_timestamp(date.date())
    return [_as_timestamp(start + pd.Timedelta(hours=h)) for h in range(24)]


def _hrrr_area_bounds() -> tuple[float, float, float, float]:
    """Return bounding box as west, east, south, north with a small margin."""
    west, east, south, north = REGIONAL_CONFIG["bounding_box_limits"]
    margin = 0.25
    return west - margin, east + margin, south - margin, north + margin


def _combine_herbie_result(result: Any) -> xr.Dataset:
    if isinstance(result, list):
        return xr.merge(result, compat="override")
    return cast(xr.Dataset, result)


def _crop_to_houston(ds: xr.Dataset) -> xr.Dataset:
    west, east, south, north = _hrrr_area_bounds()
    lon = ds["longitude"]
    lon180 = xr.where(lon > 180, lon - 360, lon)
    lat = ds["latitude"]
    mask = (lat >= south) & (lat <= north) & (lon180 >= west) & (lon180 <= east)
    y_idx, x_idx = np.where(mask.values)
    if len(y_idx) == 0:
        raise ValueError(
            "HRRR subset mask is empty for configured Houston bounding box"
        )
    y_min = y_idx.min().item()
    y_max = y_idx.max().item()
    x_min = x_idx.min().item()
    x_max = x_idx.max().item()
    cropped = ds.isel(
        y=slice(y_min, y_max + 1),
        x=slice(x_min, x_max + 1),
    ).copy()
    cropped["longitude"] = xr.where(
        cropped["longitude"] > 180, cropped["longitude"] - 360, cropped["longitude"]
    )
    return cropped


def _get_2d(ds: xr.Dataset, name: str) -> xr.DataArray:
    if name not in ds:
        raise KeyError(f"HRRR variable {name!r} not found in downloaded dataset")
    da = ds[name]
    drop_dims = [d for d in da.dims if d not in ("y", "x")]
    for dim in drop_dims:
        da = da.isel({dim: 0})
    return da.astype("float32")


def _get_level(ds: xr.Dataset, base_name: str, level_hpa: int) -> xr.DataArray:
    if base_name not in ds:
        raise KeyError(f"HRRR pressure-level variable {base_name!r} not found")
    da = ds[base_name]
    if "isobaricInhPa" not in da.dims:
        raise ValueError(f"{base_name!r} does not have isobaricInhPa dimension")
    return da.sel(isobaricInhPa=level_hpa).astype("float32")


def _wind_from_to(
    u: xr.DataArray, v: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    # Meteorological FROM direction and downwind/TO bearing.
    angle = xr.apply_ufunc(np.arctan2, v, u)
    wind_from = (270.0 - xr.apply_ufunc(np.degrees, angle)) % 360.0
    wind_to = (wind_from + 180.0) % 360.0
    return wind_from.astype("float32"), wind_to.astype("float32")


def _hour_dataset(valid_time: pd.Timestamp) -> xr.Dataset:
    try:
        from herbie.core import Herbie
    except ImportError:
        sys.exit(
            "herbie-data is not installed. Run: python -m pip install herbie-data cfgrib eccodes"
        )

    h = Herbie(
        valid_time, model="hrrr", product="sfc", fxx=0, save_dir=str(HRRR_CACHE_DIR)
    )
    raw = h.xarray(search=HRRR_SEARCH, remove_grib=False)
    ds = _crop_to_houston(_combine_herbie_result(raw))

    out = xr.Dataset(
        coords={
            "y": ds["latitude"].coords["y"]
            if "y" in ds["latitude"].coords
            else np.arange(ds.sizes["y"]),
            "x": ds["latitude"].coords["x"]
            if "x" in ds["latitude"].coords
            else np.arange(ds.sizes["x"]),
            "latitude": (("y", "x"), ds["latitude"].values.astype("float32")),
            "longitude": (("y", "x"), ds["longitude"].values.astype("float32")),
        }
    )

    for name in ("cape", "cin", "pwat", "blh", "t2m", "d2m", "u10", "v10"):
        out[name] = _get_2d(ds, name)

    for level in (500, 700, 850):
        out[f"u{level}"] = _get_level(ds, "u", level)
        out[f"v{level}"] = _get_level(ds, "v", level)

    out["shear_500_850_ms"] = np.hypot(
        out["u500"] - out["u850"], out["v500"] - out["v850"]
    ).astype("float32")
    out["wind700_from_deg"], out["wind700_to_deg"] = _wind_from_to(
        out["u700"], out["v700"]
    )
    out = out.expand_dims(time=[np.datetime64(valid_time.to_datetime64())])
    return out[OUTPUT_VARIABLES]


def download_hrrr_target_day(date: pd.Timestamp, force: bool = False) -> Path:
    HRRR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = _output_file(date)
    if out_file.exists() and not force:
        print(f"Cached HRRR environment exists: {out_file}")
        return out_file

    hourly = []
    for hour in _analysis_hours(date):
        print(f"Downloading/reading HRRR analysis: {hour:%Y-%m-%d %H:%M UTC}")
        hourly.append(_hour_dataset(hour))

    ds_day = xr.concat(hourly, dim="time")
    ds_day.attrs.update(
        {
            "description": "HRRR F00 analysis environmental fields subset to Houston box",
            "model": "HRRR",
            "product": "sfc analysis f00",
            "date": f"{date:%Y-%m-%d}",
            "bounding_box_limits_west_east_south_north": str(
                REGIONAL_CONFIG["bounding_box_limits"]
            ),
            "note": "Hourly HRRR analysis fields; interpolate in time/space when attaching to radar cells.",
        }
    )
    encoding = {
        name: {"zlib": True, "complevel": 1, "dtype": "float32"}
        for name in ds_day.data_vars
    }
    ds_day.to_netcdf(out_file, encoding=encoding)
    print(f"Saved HRRR environment -> {out_file}")
    return out_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download target-day HRRR analysis environmental fields"
    )
    parser.add_argument(
        "--date", help="Target date YYYY-MM-DD. Defaults to config TARGET date."
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing output NetCDF."
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    date = _as_timestamp(args.date) if args.date else _target_date_from_config()
    print("=" * 60)
    print("Stage 0c — HRRR target-day environment")
    print("=" * 60)
    print(f"Date: {date:%Y-%m-%d}")
    print(f"Output: {_output_file(date)}")
    download_hrrr_target_day(date, force=args.force)


if __name__ == "__main__":
    main()
