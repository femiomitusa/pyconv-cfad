#!/usr/bin/env python3
"""Attach HRRR environment to tracked cells using each cell's segmentation footprint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import OUTPUT_DIR

HRRR_VARS = [
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
STATS = ("mean", "min", "max", "std")


def _default_paths(date: str) -> tuple[Path, Path]:
    t = pd.Timestamp(date)
    day_dir = (
        Path(OUTPUT_DIR) / "Arrays_Regional" / "all" / f"{t.year}" / f"{t:%b}{t.day}"
    )
    cells = day_dir / f"KHGX{t:%Y%m%d}_regional.zarr"
    hrrr = Path(OUTPUT_DIR) / "hrrr" / f"hrrr_environment_{t:%Y%m%d}.nc"
    return cells, hrrr


def _xy(lon: np.ndarray, lat: np.ndarray, ref_lat: float) -> np.ndarray:
    return np.column_stack([lon * np.cos(np.deg2rad(ref_lat)), lat])


def _env_at_time(env: xr.Dataset, when: np.datetime64) -> xr.Dataset:
    first, last = env.time.values[0], env.time.values[-1]
    if first <= when <= last:
        return env[HRRR_VARS].interp(time=when)
    return env[HRRR_VARS].sel(time=when, method="nearest")


def attach_environment(cells_path: Path, hrrr_path: Path) -> None:
    cells = xr.open_zarr(cells_path, consolidated=False).load()
    env = xr.open_dataset(hrrr_path)

    ref_lat = env.latitude.mean().item()
    tree = cKDTree(
        _xy(env.longitude.values.ravel(), env.latitude.values.ravel(), ref_lat)
    )
    nearest = tree.query
    shape = env.latitude.shape

    out = {
        f"hrrr_{v}_{s}": np.full(cells.sizes["obs"], np.nan, "float32")
        for v in HRRR_VARS
        for s in STATS
    }
    out["hrrr_n_gridpoints"] = np.zeros(cells.sizes["obs"], "int16")

    radar_lat = cells.lat.values
    radar_lon = cells.lon.values
    sample_obs = cells.sample_obs.values
    sample_y = cells.sample_y.values
    sample_x = cells.sample_x.values
    scan_time = cells.scan_time.values

    for obs in range(cells.sizes["obs"]):
        use = sample_obs == obs
        if use.any():
            lat = radar_lat[sample_y[use], sample_x[use]]
            lon = radar_lon[sample_y[use], sample_x[use]]
        else:
            lat = np.array([cells.centroid_lat.values[obs]])
            lon = np.array([cells.centroid_lon.values[obs]])

        _, idx = nearest(_xy(lon, lat, ref_lat))
        yy, xx = np.unravel_index(np.unique(idx), shape)
        now = _env_at_time(env, scan_time[cells.scan_idx.values[obs]])
        out["hrrr_n_gridpoints"][obs] = len(yy)

        for var in HRRR_VARS:
            vals = now[var].values[yy, xx].astype("float32")
            out[f"hrrr_{var}_mean"][obs] = np.nanmean(vals)
            out[f"hrrr_{var}_min"][obs] = np.nanmin(vals)
            out[f"hrrr_{var}_max"][obs] = np.nanmax(vals)
            out[f"hrrr_{var}_std"][obs] = np.nanstd(vals)

    xr.Dataset({name: ("obs", values) for name, values in out.items()}).to_zarr(
        cells_path, mode="a"
    )
    print(f"Attached HRRR footprint environment to {cells_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Sample HRRR fields with cell segmentation footprints"
    )
    p.add_argument("--date", default="2022-07-10")
    p.add_argument("--cells")
    p.add_argument("--hrrr")
    args = p.parse_args()
    cells, hrrr = _default_paths(args.date)
    attach_environment(
        Path(args.cells) if args.cells else cells,
        Path(args.hrrr) if args.hrrr else hrrr,
    )


if __name__ == "__main__":
    main()
