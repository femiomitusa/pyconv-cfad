import os
os.environ['PYART_QUIET'] = "1"  # Suppress welcome message after first time

import numpy as np
import pyart
import sys
from typing import Tuple, Dict, List

# Add myutils directory to Python path
sys.path.insert(1, os.path.join(os.path.dirname(__file__), 'myutils'))
import multiprocessing as mp
from sort import latlon2cart, extract_and_save
# Add project root to Python path for config import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import (
    GRID_SHAPE, GRID_LIMITS, RHOHV_THRESHOLD, KDP_PARAMS,
    VERTICAL_LIMIT, WEIGHTING_FUNCTION
)

def setup_radar_grid(grid_bounds: Tuple[float, float], grid_points: int, grid_spacing: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a matching grid for radar data."""
    xx, yy = np.meshgrid(
        np.linspace(grid_bounds[0], grid_bounds[1], grid_points),
        np.linspace(grid_bounds[0], grid_bounds[1], grid_points)
    )
    z = np.linspace(0, VERTICAL_LIMIT, grid_spacing)
    return xx, yy, z

def process_radar_file(filename: str) -> Dict:
    """Process radar file and return gridded data."""
    radar = pyart.io.read(filename)
    
    # Calculate KDP
    kdp_vulpiani = pyart.retrieve.kdp_vulpiani(
        radar, 
        gatefilter=None,
        **KDP_PARAMS
    )
    radar.add_field('kdp', kdp_vulpiani[0])
    
    # Grid the radar
    grid_radar = pyart.map.grid_from_radars(
        radar,
        GRID_SHAPE,
        GRID_LIMITS,
        roi=None,
        weighting_function=WEIGHTING_FUNCTION
    )
    
    # Extract fields
    fields = {}
    field_names = ['reflectivity', 'differential_reflectivity', 'cross_correlation_ratio', 'kdp']
    for field_name in field_names:
        fields[field_name] = grid_radar.fields[field_name]['data']
    
    # Create mask based on rhohv threshold
    mask = fields['cross_correlation_ratio'] < RHOHV_THRESHOLD
    
    # Apply masking to all fields
    for field_name in field_names:
        if hasattr(fields[field_name], 'mask'):
            fields[field_name].mask = fields[field_name].mask | mask
        else:
            fields[field_name] = np.ma.masked_array(fields[field_name], mask=mask)
    
    return fields

def process_cell(args: Tuple) -> Dict:
    """Process a single cell's data (for parallel processing)."""
    cell_params, radar_fields, xx, yy = args
    
    xctr, yctr = cell_params['gridlon'], cell_params['gridlat']
    rad_lim = cell_params['radius']
    
    rad = np.sqrt((xx - xctr) ** 2 + (yy - yctr) ** 2)
    storm_loc = rad <= rad_lim
    num_grid_pts = np.sum(storm_loc)
    
    # Initialize storage arrays for each field
    field_data = {}
    for field_name, field_values in radar_fields.items():
        save_array = np.full((num_grid_pts, field_values.shape[0]), np.nan)
        
        # Extract data for each vertical level
        for j in range(field_values.shape[0]):
            save_array = extract_and_save(
                np.squeeze(field_values[j, :, :]),
                save_array,
                storm_loc,
                j
            )
        field_data[f'{field_name}_save'] = save_array.tolist()
    
    return {
        'basetime': cell_params['base_time'],
        'radius': rad_lim,
        'tracks': cell_params['tracks'],
        'area': cell_params['area'],
        'gridlon': xctr,
        'gridlat': yctr,
        'longitude': cell_params['longitudes'],
        'latitude': cell_params['latitudes'],
        **field_data
    }

def parallel_process_cells(cell_params_list: List[Dict], radar_fields: Dict, xx: np.ndarray, yy: np.ndarray) -> List[Dict]:
    """Process multiple cells in parallel."""
    # Prepare arguments for parallel processing
    args_list = [
        ({k: v[i] for k, v in cell_params_list.items()}, radar_fields, xx, yy)
        for i in range(len(cell_params_list['radius']))
    ]
    
    # Use multiprocessing to process cells in parallel
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(process_cell, args_list)
    
    return results
