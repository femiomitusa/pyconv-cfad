"""
Radar Cell Processing Package

A Python package for processing weather radar data and tracking storm cells.
"""

from .radar_processing import (
    process_radar_file,
    setup_radar_grid,
    process_cell,
    parallel_process_cells
)

from .utils import (
    setup_output_directories,
    get_datetime_from_filename,
    get_output_paths,
    filter_tracked_cells,
    extract_cell_parameters
)

from .visualization import create_radar_plot

__version__ = "0.1.0"
__author__ = "Oluwafemi Omitusa"

__all__ = [
    'process_radar_file',
    'setup_radar_grid',
    'process_cell',
    'parallel_process_cells',
    'setup_output_directories',
    'get_datetime_from_filename',
    'get_output_paths',
    'filter_tracked_cells',
    'extract_cell_parameters',
    'create_radar_plot'
]
