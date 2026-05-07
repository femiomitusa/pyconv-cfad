# CFAD Analysis Module

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Py-ART](https://img.shields.io/badge/Py--ART-1.13%2B-green.svg)](https://arm-doe.github.io/pyart/)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)]()

The CFAD (Contoured Frequency by Altitude Diagrams) Analysis module processes radar data arrays to generate statistical analysis and publication-ready visualizations of dual-polarization radar variables as a function of altitude.

## Module Structure

```
cfad_analysis/
├── README.md                    # This file
├── cfad_analysis_latest.py      # Main CFAD analysis engine with advanced features
├── run_cfad_with_config.py      # Configuration integration wrapper
├── test_cfad_analysis_latest.py # Core functionality tests (14 tests)
├── test_cfad_extended.py        # Extended functionality tests (8 tests)
└── myutils/                     # Utility modules
    ├── cfad.py                  # CFAD computation and statistics functions
    └── sort.py                  # Data processing and coordinate conversion
```

## Configuration

The module uses a dual configuration system with environment variables and fallback defaults.

### Environment Variables
```bash
# Core Paths
export CFAD_FOLDER_PATH="/path/to/radar/arrays/Jun29"
export CFAD_STATS_PATH="/path/to/stats_all.nc"
export CFAD_OUTPUT_PATH="/path/to/filtered_cfad_data.npy"

# Directory Structure
export CFAD_PARENT_DIR="/path/to/radar/arrays"
export CFAD_DIR="Jun29"
export CFAD_BASE_DIR="/path/to/results/results_Jun29_1/"

# Output Paths
export CFAD_PLOT_OUTPUT="/path/to/results/cfad"
export CFAD_PROFILE_OUTPUT="/path/to/results/profiles"

# Analysis Parameters
export CFAD_TARGET_INDICES="0 1 2"  # Space-separated time indices
```

### Fallback Configuration
If environment variables are not set, the module uses hardcoded defaults:
```python
# Default paths (automatically used if env vars not set)
folder_path = "/Users/paulomits/Documents/pyconv/Data/Arrays/Jun29"
stats_path = "/Users/paulomits/Documents/pyconv/Data/stats_all.nc"
base_dir = "/Users/paulomits/Documents/pyconv/Data/Results/results_Jun29_1/"
```

### Analysis Mode Configuration
```python
# Single-Day Analysis
target_indices = [0, 1, 2]      # Time indices to analyze
target_year, target_month, target_day = 2019, 6, 29

# Multi-Temporal Analysis
multi_temporal_config = {
    'enabled': True,             # Enable multi-temporal mode
    'base_data_path': '/path/to/arrays',
    'years': ['2018', '2019'],   # Years to process or 'all'
    'months': ['Jun', 'Jul'],    # Months to process or 'all'
    'days': 'all',               # Days to process or 'all'
    'aggregation_method': 'average',  # 'sum' or 'average'
    'output_suffix': 'multi_temporal'
}
```

## Usage

### Method 1: Environment Variable Configuration (Recommended)
```bash
# Set environment variables
export CFAD_FOLDER_PATH="/path/to/your/data/Jun29"
export CFAD_STATS_PATH="/path/to/your/stats_all.nc"
export CFAD_BASE_DIR="/path/to/your/results/"
export CFAD_TARGET_INDICES="0 1 2"

# Activate environment
conda activate metstat

# Run analysis
cd cfad_analysis
python cfad_analysis_latest.py
```

### Method 2: Multi-Temporal Analysis
```bash
# Configure multi-temporal analysis in code
# Set multi_temporal_config['enabled'] = True
# Set years, months, aggregation method

# Run multi-temporal analysis
python cfad_analysis_latest.py
```

### Method 3: Integrated Pipeline
```bash
# Run as part of complete analysis pipeline
./cell_analysis.sh
```

## Processing Architecture

### Data Flow
```
Load Data → Filter Quality → Process Histograms → Compute Statistics → Aggregate → Visualize
    ↓            ↓              ↓                  ↓               ↓         ↓
.npy files → stats.nc → cfad_calc() → compute_stats_from_hist() → aggregate → plots
```

## Multi-Temporal Analysis

### Data Structure Requirements
The module expects a specific directory hierarchy:
```
base_data_path/
├── 2018/
│   ├── Jun/
│   │   ├── file1.npy
│   │   └── file2.npy
│   └── Jul/
│       ├── file1.npy
│       └── file2.npy
└── 2019/
    ├── Jun/
    │   ├── file1.npy
    │   └── file2.npy
    └── Aug/
        ├── file1.npy
        └── file2.npy
```

### Aggregation Recommendations
- **Sum method**: Use for total activity analysis across same time periods
- **Average method**: Use for typical pattern analysis across different periods
- **Multi-year analysis**: Always use 'average' to account for data availability differences

## Output Files

Results are saved to configured output directories:

### CFAD Contour Plots
- `cfad_plots_index_{X}.png` - 2×2 CFAD plots for each time index (300 DPI)
- `cfad_plots_multi_temporal.png` - Multi-temporal aggregated CFAD plots

### Vertical Profile Plots
- `profile_plots_mean_2x2.png` - Mean vertical profiles with IQR bands
- `profile_plots_median_2x2.png` - Median vertical profiles
- `profile_plots_mode_2x2.png` - Mode vertical profiles  
- `profile_plots_percentiles_2x2.png` - Multi-percentile family plots

### Data Files
- `filtered_cfad_data.npy` - Quality-filtered input data
- `hist_mean_index_{X}.npz` - Histogram data for each time index
- `total_and_lengths_index_{X}.npz` - Processing statistics and metadata


## Dependencies

### Core Requirements
```python
numpy >= 1.20.0          # Array operations and statistical computing
pandas >= 1.3.0          # Data manipulation and analysis
xarray >= 0.19.0         # N-dimensional labeled arrays
matplotlib >= 3.4.0      # Publication-quality plotting
arm-pyart >= 1.13.0      # Radar data processing
```

### Standard Library
```python
os                       # Environment variable handling
glob                     # File pattern matching
warnings                 # Warning suppression
tempfile                 # Temporary file operations (testing)
shutil                   # File operations (testing)
```

### Optional Dependencies
```python
scipy                    # Advanced statistical functions
cartopy                  # Cartographic projections (if needed)
netCDF4                  # NetCDF file support
```

## Environment Setup

### Create Conda Environment
```bash
conda create -n metstat python=3.9
conda activate metstat
pip install arm-pyart numpy pandas xarray matplotlib
```

### Set Environment Variables
```bash
# Add to ~/.bashrc or ~/.zshrc
export CFAD_FOLDER_PATH="/your/data/path"
export CFAD_STATS_PATH="/your/stats/path"
export CFAD_BASE_DIR="/your/results/path"
```

## References

- Yuter, S. E., & Houze Jr, R. A. (1995). Three-dimensional kinematic and microphysical evolution of Florida cumulonimbus. Part II: Frequency distributions of vertical velocity, reflectivity, and differential reflectivity. *Monthly Weather Review*, 123(7), 1941-1963.
- ARM PyART documentation: https://arm-doe.github.io/pyart/
- Project main repository: https://github.com/oluwafemiomitusa/pyconv
- HomeyerRainbow colormap: https://github.com/CSU-Radarmet/CSU_RadarTools

## License

This module is part of the Weather Radar Analysis Suite and is licensed under the MIT License. See the main project [LICENSE](../LICENSE) file for details.