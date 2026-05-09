"""TOBAC-based convective cell detection on NEXRAD column-max reflectivity."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr


@dataclass(frozen=True)
class DetectionConfig:
    thresholds: tuple[float, ...] = (25.0, 30.0, 35.0, 40.0, 45.0)
    segmentation_threshold: float = 25.0
    min_pixels: int = 10
    grid_spacing_m: float = 500.0


def detect_cells(
    radar_fields: dict,
    scan_time: pd.Timestamp,
    config: DetectionConfig,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame | None]:
    """Detect convective cells via TOBAC. Returns (z_composite, local_mask, features)."""
    import tobac  # lazy — keeps tobac/iris out of the parent process

    Z_filled = np.ma.filled(radar_fields["reflectivity"].astype(float), np.nan)
    z_composite = np.nanmax(Z_filled, axis=0).astype(np.float32)

    finite = z_composite[np.isfinite(z_composite)]
    if finite.size == 0:
        return z_composite, np.zeros(z_composite.shape, dtype=np.int32), None

    field_max = float(finite.max())
    usable = [t for t in config.thresholds if field_max >= t]
    if not usable:
        return z_composite, np.zeros(z_composite.shape, dtype=np.int32), None

    da = _as_tobac_da(z_composite, scan_time)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*segmentation with time length 1.*")
        features = tobac.feature_detection_multithreshold(
            da,
            config.grid_spacing_m,
            threshold=usable,
            target="maximum",
            position_threshold="extreme",
            n_min_threshold=config.min_pixels,
            n_erosion_threshold=0,
            sigma_threshold=0.0,
        )

    if features is None or features.empty:
        return z_composite, np.zeros(z_composite.shape, dtype=np.int32), None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*segmentation with time length 1.*")
        mask_da, _ = tobac.segmentation_2D(
            features,
            da,
            config.grid_spacing_m,
            threshold=config.segmentation_threshold,
            target="maximum",
            method="watershed",
        )

    local_mask = np.asarray(mask_da.values[0], dtype=np.int32)
    return z_composite, local_mask, features


def compute_eth_maps(Z_3d_filled: np.ndarray, z_levels: np.ndarray) -> dict[int, np.ndarray]:
    """Pixel-wise echo-top height (m) for thresholds 20/30/40 dBZ. Returns {thr: (ny, nx) array}."""
    result = {}
    z_broadcast = z_levels[:, None, None]  # broadcast over y, x
    for thr in (20, 30, 40):
        above = np.where(np.isfinite(Z_3d_filled) & (Z_3d_filled >= thr), z_broadcast, np.nan)
        result[thr] = np.nanmax(above, axis=0).astype(np.float32)
    return result


def _as_tobac_da(field: np.ndarray, t: pd.Timestamp) -> xr.DataArray:
    return xr.DataArray(
        field[np.newaxis],
        dims=["time", "y", "x"],
        coords={"time": [t.to_datetime64()]},
    )
