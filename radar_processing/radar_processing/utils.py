import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta

CELL_TIME_MATCH_WINDOW_SECONDS = 150
import re
import os
import glob
import sys
from typing import List, Tuple, Dict

# Add project root to Python path for config import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# Add myutils directory to Python path
sys.path.insert(1, os.path.join(os.path.dirname(__file__), 'myutils'))

from sort import latlon2cart
from config import FIGURE_OUTPUT_DIR, ARRAY_OUTPUT_DIR

def setup_output_directories(date: str) -> Tuple[str, str]:
    """Create and return figure and data output directories."""
    save_fig = os.path.join(FIGURE_OUTPUT_DIR, date)
    save_data = os.path.join(ARRAY_OUTPUT_DIR, date)
    os.makedirs(save_fig, exist_ok=True)
    os.makedirs(save_data, exist_ok=True)
    return save_fig, save_data

def get_datetime_from_filename(filename: str) -> Tuple[str, str]:
    """Extract a matching time window from a radar filename.

    Tracked-cell timestamps are not always identical to the exact seconds in
    the NEXRAD filename. Use a narrow centered window to avoid dropping valid
    cells because of timestamp rounding or scan-time differences.
    """
    match = re.search(r"(\d{8})_(\d{6})", filename)
    if match:
        dt = datetime.strptime(f"{match.group(1)}_{match.group(2)}", "%Y%m%d_%H%M%S")
        return (
            (dt - timedelta(seconds=CELL_TIME_MATCH_WINDOW_SECONDS)).strftime('%Y-%m-%dT%H:%M:%S'),
            (dt + timedelta(seconds=CELL_TIME_MATCH_WINDOW_SECONDS)).strftime('%Y-%m-%dT%H:%M:%S')
        )
    return None, None

def get_output_paths(save_data: str, save_fig: str, filename: str) -> Tuple[str, str, str]:
    """Generate output file paths for data and figures using original filename."""
    # Extract just the filename without path
    base_filename = os.path.basename(filename)
    
    data_output_path = os.path.join(save_data, f"{base_filename}.npy")
    fig_output_path = os.path.join(save_fig, f"{base_filename}.png")
    
    return base_filename, data_output_path, fig_output_path

def filter_tracked_cells(tracked_cells: xr.Dataset, start_date: str, end_date: str) -> pd.DataFrame:
    """Filter tracked cells data for the given time period."""
    filtered_data = tracked_cells[
        ['cell_area', 'max_dbz', 'maxETH_20dbz', 'base_time', 'tracks', 
         'cell_meanlon', 'cell_meanlat']
    ].to_dataframe().dropna(how='any').reset_index()
    
    filtered_data['base_time'] = pd.to_datetime(filtered_data['base_time'])
    cells = filtered_data[
        (filtered_data['base_time'] >= start_date) & 
        (filtered_data['base_time'] <= end_date)
    ]
    return cells

def extract_cell_parameters(cells: pd.DataFrame) -> Dict:
    """Extract relevant parameters from cell data."""
    params = {
        'radius': (np.sqrt(cells.cell_area / np.pi) * 1000).values,
        'area': cells.cell_area.values,
        'longitudes': cells.cell_meanlon.values,
        'latitudes': cells.cell_meanlat.values,
        'tracks': cells.tracks.values,
        'base_time': cells.base_time.dt.strftime('%Y-%m-%d %H:%M:%S').values
    }
    
    # Convert coordinates
    params['gridlon'], params['gridlat'] = latlon2cart(params['latitudes'], params['longitudes'])
    return params
