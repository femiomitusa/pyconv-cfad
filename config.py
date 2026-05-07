# ============================================================================
# WEATHER RADAR ANALYSIS SUITE - CONFIGURATION
# ============================================================================

# ============================================================================
# GLOBAL PROJECT SETTINGS
# ============================================================================

# Data Storage
BASE_DATA_DIR = '/Volumes/My Book/NEXRAD'
DATA_DIR = BASE_DATA_DIR
STATS_FILE = f"{BASE_DATA_DIR}/stats_all.nc"
FIGURE_OUTPUT_DIR = f'{BASE_DATA_DIR}/Figures'
ARRAY_OUTPUT_DIR = f'{BASE_DATA_DIR}/Arrays'
RADAR_FILE_PATTERN = "KHGX*"

# Pipeline Control
RUN_NEXRAD_DOWNLOAD = True
RUN_RADAR_PROCESSING = False
RUN_REGIONAL_FILTERING = False
RUN_CFAD_ANALYSIS = False

# Safeguards
SKIP_EXISTING_DOWNLOADS = True
SKIP_EXISTING_PROCESSING = True
SKIP_EXISTING_FILTERING = True
SKIP_EXISTING_ANALYSIS = True

# Visualization
RADAR_VISUALIZATION = True
CFAD_VISUALIZATION = True

# Output Control
QUIET_MODE = True  # Set to True to reduce output verbosity

# ============================================================================
# TIME PERIOD CONFIGURATION
# ============================================================================

TARGET_MODE = False  # Set to True for single-day analysis, False for multi-period

# Date Range for Bulk Processing
YEAR_START = 2025
YEAR_END = 2026
VALID_MONTHS = ['Jun', 'Jul', 'Aug']  # Summer months to process

# Target Date for Single-Day Analysis
TARGET_YEAR = 2022
TARGET_MONTH = 6
TARGET_DAY = 8

# ============================================================================
# RADAR PROCESSING SETTINGS
# ============================================================================

# Grid Configuration
VERTICAL_LIMIT = 20000  # Maximum height in meters
GRID_SHAPE = (41, 401, 401)  # (z, y, x)
GRID_LIMITS = ((0.0, VERTICAL_LIMIT), (-100000, 100000), (-100000, 100000))
GRID_SPACING = 41  # Number of vertical levels
GRID_BOUNDS = (-100000, 100000)  # Grid bounds in meters
GRID_POINTS = 401  # Number of points in x/y dimensions

# Processing Parameters
WEIGHTING_FUNCTION = 'Barnes'  # Options: 'Barnes', 'Cressman', 'Nearest'
RHOHV_THRESHOLD = 0.90

# KDP Calculation
KDP_PARAMS = {
    'band': 'S',
    'windsize': 10,
    'n_iter': 10,
    'interp': False,
    'parallel': True
}

# Radar Visualization
PLOT_FIGSIZE = (8, 8)
REFLECTIVITY_LIMITS = (-5, 50)
COLORMAP = 'HomeyerRainbow'

# ============================================================================
# CFAD ANALYSIS SETTINGS
# ============================================================================

# Analysis Control
CFAD_ENABLED = True
CFAD_TARGET_INDICES = [0]  # Time indices to analyze
CFAD_KDP_CALC = True
CFAD_NORM_OPT = 2

# Variable Limits and Binning
CFAD_Z_LIMITS = [-20, 65]      # Reflectivity (dBZ)
CFAD_ZDR_LIMITS = [-1, 5]      # Differential Reflectivity (dB)
CFAD_RHO_LIMITS = [0.87, 1.03] # Correlation Coefficient
CFAD_KDP_LIMITS = [-1, 3]      # Specific Differential Phase (deg/km)

CFAD_DZ = 5        # Reflectivity bin size
CFAD_DZDR = 0.5    # ZDR bin size
CFAD_DRHO = 0.0025 # RhoHV bin size
CFAD_DKDP = 0.1    # KDP bin size

# Output Configuration
CFAD_OUTPUT_DIR = f'{BASE_DATA_DIR}/CFAD_Results'
CFAD_PLOT_OUTPUT = f'{CFAD_OUTPUT_DIR}/cfad_plots'
CFAD_PROFILE_OUTPUT = f'{CFAD_OUTPUT_DIR}/profile_plots'
CFAD_YMAX = 20  # Maximum altitude for plots (km)
CFAD_INCLUDE_IQR_ON_MEAN = True

# Statistical Analysis
CFAD_PERCENTILES = [0, 10, 25, 50, 75, 90, 100]
CFAD_PERCENTILE_DISPLAY = [10, 25, 50, 75, 90]

# Single-Day Analysis
CFAD_TARGET_YEAR = TARGET_YEAR
CFAD_TARGET_MONTH = TARGET_MONTH
CFAD_TARGET_DAY = TARGET_DAY

# Multi-temporal Analysis
CFAD_MULTI_TEMPORAL = {
    'enabled': True,
    'base_data_path': ARRAY_OUTPUT_DIR,
    'years': [str(y) for y in range(YEAR_START, YEAR_END + 1)],  # ['2016', '2017', '2018']
    'months': VALID_MONTHS,  # ['Jul', 'Aug']
    'days': ['all'],
    'aggregation_method': 'average',  # 'sum' or 'average'
    'output_suffix': 'multi_temporal'
}

# CFAD Plot Colors
CFAD_PROFILE_COLORS = {
    'Z': 'blue',
    'ZDR': 'green',
    'rho': 'red',
    'kdp': 'purple'
}

# ============================================================================
# REGIONAL FILTERING SETTINGS
# ============================================================================

# Regional Analysis Control
REGIONAL_CFAD_ENABLED = True
REGIONAL_USE_PARALLEL = True
REGIONAL_MAX_WORKERS = None  # None = use CPU count

# Regional Output Directories
REGIONAL_OUTPUT_DIR = f'{BASE_DATA_DIR}/Arrays_Regional'
REGIONAL_CFAD_OUTPUT_DIR = f'{BASE_DATA_DIR}/CFAD_Regional_Results'

# Standard Regions
REGIONS = ['urban', 'upwind', 'downwind', 'left', 'right']

# Regional Configuration (pycellstats approach)
REGIONAL_CONFIG = {
    # Reference coordinates (Houston KHGX radar)
    'city_center_lat': 29.4719,
    'city_center_lon': -95.0787,
    
    # Wind Configuration
    'wind_data_path': '/Volumes/My Book/NEXRAD/wind_data/wind_data_1.nc',
    'default_wind_direction': 138.37,  # SE wind direction (degrees)
    'temporal_wind': 138.37,
    'pressure_level': 850,  # hPa for wind direction calculation
    'temporal_avg': 'overall',  # 'overall', 'daily', or 'hourly'
    
    # Distance Parameters
    'downwind_start': 100,  # km
    'downwind_end': 150,    # km
    'sector_angle': 160,    # degrees (total sector width)
    
    # Geographic Bounds
    'bounding_box_limits': (-96.22093200683594, -93.9366226196289, 28.483173370361328, 30.45787239074707),
    'shapefile_path': '/Volumes/My Book/NEXRAD/2022_Shapefile/2022_Developed_Shapefile_1.shp',
    
    # Processing
    'num_workers': None,  # Auto-detect CPU count
    'regions': ['urban', 'upwind', 'downwind', 'left', 'right'],
    
    # CFAD Regional Output
    'cfad_enabled': True,
    'cfad_output_dir': f'{BASE_DATA_DIR}/CFAD_Regional_Results',
    'cfad_plot_output_dir': f'{BASE_DATA_DIR}/CFAD_Regional_Results/plots',
    'cfad_profile_output_dir': f'{BASE_DATA_DIR}/CFAD_Regional_Results/profiles'
}

# Backward compatibility
REGIONAL_CFAD_CONFIG = REGIONAL_CONFIG