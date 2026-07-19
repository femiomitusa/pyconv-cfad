import os
os.environ['PYART_QUIET'] = "1"

import sys
import numpy as np
import pyart
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    GRID_SHAPE, GRID_LIMITS, RHOHV_THRESHOLD, KDP_PARAMS,
    VERTICAL_LIMIT, WEIGHTING_FUNCTION,
)

FIELD_NAMES = ['reflectivity', 'differential_reflectivity', 'cross_correlation_ratio', 'kdp']


def has_usable_reflectivity(filename: str, threshold_dbz: float = 25.0) -> bool:
    """Quick pre-check: max reflectivity >= threshold? Avoids gridding useless files."""
    try:
        radar = pyart.io.read(filename)
        Z = radar.fields.get('reflectivity')
        if Z is None:
            return False
        Z_data = np.ma.filled(Z['data'].astype(float), np.nan)
        max_val = np.nanmax(Z_data)
        return bool(np.isfinite(max_val) and max_val >= threshold_dbz)
    except Exception:
        return True


def setup_radar_grid(grid_bounds, grid_points, grid_spacing):
    xx, yy = np.meshgrid(
        np.linspace(grid_bounds[0], grid_bounds[1], grid_points),
        np.linspace(grid_bounds[0], grid_bounds[1], grid_points),
    )
    z = np.linspace(0, VERTICAL_LIMIT, grid_spacing)
    return xx, yy, z


def process_radar_file(
    filename,
    *,
    kdp_parallel: bool | None = None,
    include_kdp: bool = True,
    field_names: tuple[str, ...] | None = None,
):
    # kdp_parallel=False prevents joblib spawning threads inside worker processes (CPU oversubscription).
    radar = pyart.io.read(filename)

    if field_names is None:
        field_names = tuple(FIELD_NAMES if include_kdp else ("reflectivity", "cross_correlation_ratio"))

    if include_kdp and "kdp" in field_names:
        kdp_params = dict(KDP_PARAMS)
        if kdp_parallel is not None:
            kdp_params["parallel"] = kdp_parallel
        kdp, _ = pyart.retrieve.kdp_vulpiani(radar, gatefilter=None, **kdp_params)
        radar.add_field('kdp', kdp)

    grid = pyart.map.grid_from_radars(
        radar,
        GRID_SHAPE,
        GRID_LIMITS,
        fields=list(field_names),
        roi=None,
        weighting_function=WEIGHTING_FUNCTION,
    )
    del radar

    # Validate that grid has valid data before accessing
    lat_data = grid.point_latitude['data']
    lon_data = grid.point_longitude['data']
    
    if lat_data.size == 0 or lon_data.size == 0 or lat_data.shape[0] == 0:
        # Grid is empty/invalid — skip this file (likely incomplete/corrupted)
        raise ValueError("Empty grid")

    fields = {name: grid.fields[name]['data'] for name in field_names}
    lat_2d = lat_data[0].astype(np.float32)
    lon_2d = lon_data[0].astype(np.float32)
    del grid

    # keep raw copy for visualization before RhoHV masking
    fields['reflectivity_raw'] = fields['reflectivity'].copy()

    mask = fields['cross_correlation_ratio'] < RHOHV_THRESHOLD
    for name in field_names:
        if hasattr(fields[name], 'mask'):
            fields[name].mask = fields[name].mask | mask
        else:
            fields[name] = np.ma.masked_array(fields[name], mask=mask)

    fields['lat_2d'] = lat_2d
    fields['lon_2d'] = lon_2d
    return fields
