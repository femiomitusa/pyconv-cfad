# Radar Processing Module

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyART](https://img.shields.io/badge/PyART-Latest-green.svg)](https://arm-doe.github.io/pyart/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python module for processing weather radar data with a focus on cell tracking and analysis. Part of the larger Weather Radar Analysis Suite.

## Features

- **Configurable Radar Processing**
  - Flexible grid configurations
  - Adjustable vertical limits
  - Multiple weighting function options (Barnes, Cressman, etc.)
  - Quality control via RhoHV thresholding

- **Parallel Processing**
  - Multi-core processing of radar cells
  - Efficient handling of large datasets
  - Automatic CPU core optimization

- **Comprehensive Data Analysis**
  - Multiple radar field processing (Z, ZDR, ρHV, KDP)
  - Storm cell tracking
  - Automated masking based on quality thresholds

## Installation

From the project root:
```bash
cd radar_processing
pip install -e .
cd ..
```

## Package Structure

```
radar_processing/
├── radar_processing/     # Core package
│   ├── __init__.py      # Package initialization
│   ├── config.py        # Configuration (imports from root)
│   ├── radar_processing.py # Core processing functions
│   ├── utils.py         # Utility functions
│   ├── visualization.py # Visualization functions
│   └── myutils/        
│       └── sort.py     # Coordinate conversion/extraction
├── process_radar.py     # Radar processing script
├── setup.py            # Package installation
└── requirements.txt    # Dependencies
```

## Configuration

This module uses the project-wide configuration from the root `config.py`. Key parameters include:

```python
# Project Control
RUN_RADAR_PROCESSING = True/False
RADAR_VISUALIZATION = True/False

# Data Paths
DATA_DIR = "./Data/2019"
STATS_FILE = "./Data/stats_all.nc"
ARRAY_OUTPUT_DIR = "./Data/Arrays"
FIGURE_OUTPUT_DIR = "./Data/Figures"

# Processing Parameters
VERTICAL_LIMIT = 20000
GRID_SHAPE = (41, 401, 401)
WEIGHTING_FUNCTION = 'Barnes'
RHOHV_THRESHOLD = 0.90
```

## Usage

This module is meant to be run through the main cell_analysis.sh script at the project root:

```bash
# From project root
bash cell_analysis.sh
```

The script will:
1. Check if radar processing is enabled in config.py
2. Process all radar files in the input directory
3. Generate output arrays and visualizations as configured

## Output

- **Data Arrays**: `Data/Arrays/<date>/KHGX*.npy`
  - Processed radar data for each timestep
  - All radar fields and cell parameters
  - Format: NumPy arrays with fields:
    - reflectivity
    - differential_reflectivity
    - correlation_coefficient
    - kdp

- **Visualizations**: `Data/Figures/<date>/KHGX*.png`
  - Reflectivity plots with cell locations
  - Quality-controlled radar fields
  - Automatic cell tracking visualization

## Dependencies

- numpy>=1.20.0
- arm-pyart>=1.13.0
- pandas>=1.3.0
- xarray>=0.19.0
- matplotlib>=3.4.0
- tqdm>=4.65.0

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## Author

Oluwafemi Omitusa - [GitHub](https://github.com/oluwafemiomitusa)
