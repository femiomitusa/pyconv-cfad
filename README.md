# Weather Radar Analysis Suite

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A comprehensive modular pipeline for processing NEXRAD Level 2 radar data, featuring regional storm cell classification and advanced CFAD statistical analysis.

## Quick Start

### **Run Complete Pipeline**
```bash
bash cell_analysis.sh
```

### **Configure Analysis**
Edit `config.py` for your target period:
```python
# Single-day mode
TARGET_MODE = True
TARGET_YEAR = 2022
TARGET_MONTH = 6  # June
TARGET_DAY = 8

# Multi-temporal mode
CFAD_MULTI_TEMPORAL = {
    'enabled': True,
    'years': ['2016', '2017', '2018'],
    'months': ['Jul', 'Aug'],
    'aggregation_method': 'average'
}
```

## Architecture Overview

This suite implements a **4-stage modular pipeline** with centralized configuration:

### **Pipeline Flow**
```
Raw NEXRAD → Gridded Arrays → Regional Arrays → CFAD Statistics/Plots
   (AWS S3)    (ARM PyART)     (Geographic)       (Dual-Pol Stats)
```

### **Stage Details**
1. **NEXRAD Download** - Parallel downloads from AWS S3 with robust error handling
2. **Radar Processing** - Gridding, KDP calculation, quality control via ARM PyART  
3. **Regional Filtering** - Geographic storm cell classification (5 regions)
4. **CFAD Analysis** - Statistical analysis with single-day and multi-temporal modes

## Project Structure

```
.
├── config.py                    # Centralized configuration
├── utils.py                     # Path utilities and date formatting
├── cell_analysis.sh            # Main pipeline execution with safeguards
├── nexrad_download/             # Data download module
│   ├── data_download.sh         #     Parallel AWS S3 downloads
│   └── organize_data.sh         #     Data organization utilities
├── radar_processing/            # Radar processing package
│   ├── process_radar.py         #     Main processing script
│   ├── setup.py                #     Installable package setup
│   └── radar_processing/        #     Core processing package
│       ├── radar_processing.py  #     ARM PyART integration
│       ├── utils.py             #     Processing utilities
│       └── visualization.py     #     Radar visualization
├── filter_by_region.py          # Regional classification (pycellstats)
└── cfad_analysis/               # CFAD analysis module
    ├── cfad_analysis_latest.py  #     Main CFAD engine
    ├── run_cfad_with_config.py  #     Configuration integration
    └── myutils/                 #     CFAD utilities
        ├── cfad.py              #     CFAD computation functions
        └── sort.py              #     Data processing utilities
```

## Configuration System

The **centralized configuration** in `config.py` controls all pipeline components:

### **Global Settings**
```python
# Data Storage
BASE_DATA_DIR = '/Volumes/My Book/NEXRAD_1'
STATS_FILE = f"{BASE_DATA_DIR}/stats_all.nc"

# Pipeline Control
RUN_NEXRAD_DOWNLOAD = True
RUN_RADAR_PROCESSING = True  
RUN_REGIONAL_FILTERING = True
RUN_CFAD_ANALYSIS = True

# Safeguards
SKIP_EXISTING_DOWNLOADS = True
SKIP_EXISTING_PROCESSING = True
SKIP_EXISTING_FILTERING = True
SKIP_EXISTING_ANALYSIS = True
```

### **Time Period Configuration**
```python
# Bulk Processing Range
YEAR_START = 2016
YEAR_END = 2018
VALID_MONTHS = ['Jul', 'Aug']

# Single-Day Target
TARGET_MODE = True
TARGET_YEAR = 2022
TARGET_MONTH = 6
TARGET_DAY = 8
```

### **Radar Processing Settings**
```python
# Grid Configuration
VERTICAL_LIMIT = 20000  # 20 km maximum height
GRID_SHAPE = (41, 401, 401)  # (z, y, x) - 500m vertical resolution
WEIGHTING_FUNCTION = 'Barnes'
RHOHV_THRESHOLD = 0.90

# KDP Calculation (Vulpiani method)
KDP_PARAMS = {
    'band': 'S',
    'windsize': 10,
    'n_iter': 10,
    'interp': False,
    'parallel': True
}
```

### **CFAD Analysis Settings**
```python
# Variable Limits and Binning
CFAD_Z_LIMITS = [-20, 65]      # Reflectivity (dBZ)  
CFAD_ZDR_LIMITS = [-1, 5]      # Differential Reflectivity (dB)
CFAD_RHO_LIMITS = [0.87, 1.03] # Correlation Coefficient
CFAD_KDP_LIMITS = [-1, 3]      # Specific Differential Phase (deg/km)

# Statistical Analysis
CFAD_PERCENTILES = [0, 10, 25, 50, 75, 90, 100]
CFAD_YMAX = 20  # Maximum altitude for plots (km)

# Multi-Temporal Configuration
CFAD_MULTI_TEMPORAL = {
    'enabled': True,
    'years': [str(y) for y in range(YEAR_START, YEAR_END + 1)],
    'months': VALID_MONTHS,
    'days': ['all'],
    'aggregation_method': 'average',
    'output_suffix': 'multi_temporal'
}
```

### **Regional Filtering Settings**
```python
# Geographic Configuration (Houston KHGX radar)
REGIONAL_CONFIG = {
    'city_center_lat': 29.4719,
    'city_center_lon': -95.0787,
    'downwind_start': 100,       # km from radar
    'downwind_end': 150,         # km from radar
    'sector_angle': 160,         # degrees total width
    'temporal_wind': 138.37,     # SE wind direction (degrees)
    'bounding_box_limits': (-96.22, -93.94, 28.48, 30.46),
    'shapefile_path': '/path/to/urban_shapefile.shp'
}
```

## Data Flow and Directory Structure

### **Expected Input Structure**
```
/Volumes/My Book/NEXRAD_1/
├── 2016/Jul01/KHGX20160701_HHMMSS_V06    # Raw radar files
├── 2017/Aug15/KHGX20170815_HHMMSS_V06
├── 2018/Jun29/KHGX20180629_HHMMSS_V06
└── stats_all.nc                          # Storm cell statistics
```

### **Generated Output Structure**
```
/Volumes/My Book/NEXRAD_1/
├── Arrays/2018/Jul01/KHGX*.npy           # Processed radar arrays
├── Arrays/urban/2018/Jul01/*.npy         # Urban region storm cells
├── Arrays/upwind/2018/Jul01/*.npy        # Upwind region storm cells  
├── Arrays/downwind/2018/Jul01/*.npy      # Downwind region storm cells
├── Arrays/left/2018/Jul01/*.npy          # Left crosswind region
├── Arrays/right/2018/Jul01/*.npy         # Right crosswind region
├── Figures/2018/Jul01/KHGX*_plot.png     # Radar visualizations
├── CFAD_Results/                         # Overall CFAD analysis
│   ├── filtered_cfad_data.npy
│   ├── cfad_plots_index_0.png
│   ├── profile_plots_mean_2x2.png
│   └── profile_plots_multi_temporal_*.png
└── CFAD_Regional_Results/                # Regional CFAD analysis
    ├── Urban/cfad_plots_index_0.png
    ├── Downwind/profile_plots_mean_2x2.png
    └── [other regions]/
```

## Usage

### **1. Complete Pipeline (Recommended)**
```bash
bash cell_analysis.sh
```
**Features:**
- Comprehensive error handling and progress tracking
- Validates dependencies and data availability before execution
- Handles conda environment activation automatically
- Colored output with stage-by-stage timing

### **2. Individual Module Execution**

#### **Install Processing Module**
```bash
cd radar_processing
pip install -e .
cd ..
```

#### **NEXRAD Download Only**
```bash
bash nexrad_download/data_download.sh
```

#### **Radar Processing Only**
```bash
python radar_processing/process_radar.py
```

#### **Regional Filtering Only**
```bash
python filter_by_region.py
```

#### **CFAD Analysis Only**
```bash
conda activate metstat
python cfad_analysis/run_cfad_with_config.py
```

### **3. Configuration Modes**

#### **Single-Day Analysis**
```python
# In config.py
TARGET_MODE = True
TARGET_YEAR = 2022
TARGET_MONTH = 6
TARGET_DAY = 8
CFAD_MULTI_TEMPORAL = {'enabled': False}
```

#### **Multi-Temporal Analysis**
```python
# In config.py  
TARGET_MODE = False
CFAD_MULTI_TEMPORAL = {
    'enabled': True,
    'years': ['2016', '2017', '2018'],
    'months': ['Jul', 'Aug'],
    'aggregation_method': 'average'
}
```

## Regional Analysis System

### **5-Region Classification**
Uses pycellstats methodology for geographic storm cell classification:

- **Urban**: Circular area based on shapefile analysis
- **Upwind**: 318.37° ± 80° (NW, opposite wind direction)  
- **Downwind**: 138.37° ± 80° (SE, wind direction)
- **Left**: 228.37° ± 80° (SW, 90° counterclockwise)
- **Right**: 48.37° ± 80° (NE, 90° clockwise)

### **Processing Features**
- Parallel processing with automatic CPU detection
- Individual storm cell coordinate-based classification
- UTM projection for accurate geometric calculations
- Comprehensive error handling and progress reporting

## CFAD Analysis Capabilities

### **Single-Day Mode**
- Processes specific target date
- Groups data by basetime for temporal organization
- Generates histograms for dual-polarization variables
- Computes comprehensive statistics with edge case handling

### **Multi-Temporal Mode** 
- Auto-discovers data across multiple years/months
- Supports 'sum' or 'average' aggregation methods
- Maintains statistical framework consistency
- Outputs with configurable suffix identification

### **Statistical Analysis**
- **Variables**: Z, ZDR, ρHV, KDP with configurable limits
- **Statistics**: Mean, median, mode, percentiles, IQR
- **Vertical Range**: 0-20 km with 500m resolution
- **Quality Control**: Sophisticated histogram processing

## Requirements

### **Core Dependencies**
```bash
python>=3.8
numpy>=1.20.0
arm-pyart>=1.13.0
pandas>=1.3.0
xarray>=0.19.0
matplotlib>=3.4.0
tqdm>=4.65.0
```

### **Regional Analysis Dependencies**
```bash
shapely>=1.8.0          # Geometric operations
geopandas>=0.10.0       # Shapefile processing
```

### **Environment Setup**
```bash
# Create conda environment
conda create -n metstat python=3.8
conda activate metstat
pip install numpy xarray matplotlib netCDF4 arm-pyart pandas tqdm shapely geopandas

# Install radar processing module
cd radar_processing && pip install -e . && cd ..
```

## Testing

The system uses **direct Python execution** for testing:

```bash
# CFAD analysis tests (requires metstat environment)
conda activate metstat
cd cfad_analysis
python test_cfad_analysis_latest.py      # Core tests (14 tests)
python test_cfad_extended.py             # Extended tests (8 tests)  
python test_regional_cfad_integration.py # Regional integration tests
```

**Note**: Test files are referenced in documentation but may not be present in current repository state.

## Key Features

### **Comprehensive Safeguards**
- Prevents duplicate processing via configurable `SKIP_EXISTING_*` flags
- Validates data quality and dependencies before execution
- Handles missing files and directories gracefully
- Provides detailed error reporting with colored output

### **Performance Optimization**
- Parallel downloads with configurable worker limits (max 8 concurrent)
- Parallel regional processing with automatic CPU detection
- Efficient date-based file filtering using filename patterns
- Smart skipping of existing outputs to minimize reprocessing

### **Advanced Statistical Analysis**
- Publication-quality CFAD plots with HomeyerRainbow colormap
- Comprehensive vertical profiles (mean, median, mode, percentiles)
- Edge case handling in histogram computation
- Both single-day and multi-temporal analysis modes

### **Regional Classification**
- Sophisticated geometric operations using UTM projection
- Wind-direction-based sector generation
- Shapefile-based urban area definition
- Individual storm cell coordinate classification

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-analysis`
3. Implement changes with proper testing
4. Follow Angular commit style: `git commit -m 'feat: add multi-temporal analysis'`
5. Push and create Pull Request

**Commit Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `build`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Oluwafemi Omitusa - [GitHub](https://github.com/oluwafemiomitusa)

## Acknowledgments

- ARM PyART developers for the radar processing toolkit
- AWS for providing NEXRAD data access via S3
- HomeyerRainbow colormap from CSU RadarTools
