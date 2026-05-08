from pathlib import Path

# ============================================================================
# WEATHER RADAR ANALYSIS SUITE - CONFIGURATION
# ============================================================================

# ============================================================================
# GLOBAL PROJECT SETTINGS
# ============================================================================

# All pipeline outputs go under OUTPUT_DIR.
# Change this to any path you prefer (e.g. an external drive).
OUTPUT_DIR = str(Path(__file__).parent / "output")

BASE_DATA_DIR = OUTPUT_DIR
STATS_FILE = f"{OUTPUT_DIR}/stats_all.nc"
FIGURE_OUTPUT_DIR = f"{OUTPUT_DIR}/Figures"
ARRAY_OUTPUT_DIR = f"{OUTPUT_DIR}/Arrays"
RADAR_FILE_PATTERN = "KHGX*"

# Pipeline Control
RUN_NEXRAD_DOWNLOAD = True
RUN_RADAR_PROCESSING = True
RUN_REGIONAL_FILTERING = False
RUN_CFAD_ANALYSIS = False

# Safeguards — set to False to force reprocessing of existing files
SKIP_EXISTING_DOWNLOADS = True
SKIP_EXISTING_PROCESSING = True
SKIP_EXISTING_FILTERING = True
SKIP_EXISTING_ANALYSIS = True

# Visualization
RADAR_VISUALIZATION = True
CFAD_VISUALIZATION = True

QUIET_MODE = True

# ============================================================================
# TIME PERIOD CONFIGURATION
# ============================================================================

TARGET_MODE = True  # True = single-day, False = full date range

# Date range for bulk processing
YEAR_START = 2017
YEAR_END = 2026
VALID_MONTHS = ['Jun', 'Jul', 'Aug']

# Target date for single-day analysis
TARGET_YEAR = 2022
TARGET_MONTH = 6
TARGET_DAY = 27

# ============================================================================
# RADAR PROCESSING SETTINGS
# ============================================================================

# Grid configuration
VERTICAL_LIMIT = 20000  # metres
GRID_SHAPE = (41, 401, 401)  # (z, y, x)
GRID_LIMITS = ((0.0, VERTICAL_LIMIT), (-100000, 100000), (-100000, 100000))
GRID_SPACING = 41
GRID_BOUNDS = (-100000, 100000)
GRID_POINTS = 401

# Processing parameters
WEIGHTING_FUNCTION = 'Barnes'  # Options: 'Barnes', 'Cressman', 'Nearest'
RHOHV_THRESHOLD = 0.85

# KDP calculation
KDP_PARAMS = {
    'band': 'S',
    'windsize': 10,
    'n_iter': 10,
    'interp': False,
    'parallel': True,
}

# Derived grid spacing (metres per pixel)
GRID_SPACING_M = (GRID_BOUNDS[1] - GRID_BOUNDS[0]) / (GRID_POINTS - 1)  # 500.0

# Parallel gridding — number of radar files processed simultaneously.
# Each worker runs PyART gridding independently. Set to None to use
# min(cpu_count, 4). Set to 1 to disable (sequential, useful for debugging).
GRID_N_WORKERS = None

# TOBAC cell detection thresholds (dBZ)
TOBAC_THRESHOLDS = (25.0, 30.0, 35.0, 40.0, 45.0)
TOBAC_SEGMENTATION_THRESHOLD = 25.0   # outer cell boundary
TOBAC_MIN_PIXELS = 5                 

# TOBAC cell tracker
TOBAC_MAX_DISTANCE_PIXELS = 20        # max cell displacement per scan (pixels); 20 px = 10 km
TOBAC_MAX_GAP = 2                     # missed scans before track terminates

# Radar visualization
PLOT_FIGSIZE = (8, 8)
REFLECTIVITY_LIMITS = (-5, 50)
COLORMAP = 'NWSRef'

# ============================================================================
# CFAD ANALYSIS SETTINGS
# ============================================================================

CFAD_ENABLED = True
CFAD_TARGET_INDICES = [0]
CFAD_KDP_CALC = True
CFAD_NORM_OPT = 2

# Variable limits and bin sizes
CFAD_Z_LIMITS = [-20, 65]       # Reflectivity (dBZ)
CFAD_ZDR_LIMITS = [-1, 5]       # Differential Reflectivity (dB)
CFAD_RHO_LIMITS = [0.87, 1.03]  # Correlation Coefficient
CFAD_KDP_LIMITS = [-1, 3]       # Specific Differential Phase (deg/km)

CFAD_DZ = 5
CFAD_DZDR = 0.5
CFAD_DRHO = 0.0025
CFAD_DKDP = 0.1

# Output directories
CFAD_OUTPUT_DIR = f"{OUTPUT_DIR}/CFAD_Results"
CFAD_PLOT_OUTPUT = f"{CFAD_OUTPUT_DIR}/cfad_plots"
CFAD_PROFILE_OUTPUT = f"{CFAD_OUTPUT_DIR}/profile_plots"
CFAD_YMAX = 20  # Maximum altitude for plots (km)
CFAD_INCLUDE_IQR_ON_MEAN = True

# Statistical analysis
CFAD_PERCENTILES = [0, 10, 25, 50, 75, 90, 100]
CFAD_PERCENTILE_DISPLAY = [10, 25, 50, 75, 90]

# Single-day analysis (mirrors target date)
CFAD_TARGET_YEAR = TARGET_YEAR
CFAD_TARGET_MONTH = TARGET_MONTH
CFAD_TARGET_DAY = TARGET_DAY

# Multi-temporal analysis
CFAD_MULTI_TEMPORAL = {
    'enabled': True,
    'base_data_path': ARRAY_OUTPUT_DIR,
    'years': [str(y) for y in range(YEAR_START, YEAR_END + 1)],
    'months': VALID_MONTHS,
    'days': ['all'],
    'aggregation_method': 'average',  # 'sum' or 'average'
    'output_suffix': 'multi_temporal',
}

CFAD_PROFILE_COLORS = {
    'Z': 'blue',
    'ZDR': 'green',
    'rho': 'red',
    'kdp': 'purple',
}

# ============================================================================
# REGIONAL FILTERING SETTINGS
# ============================================================================

REGIONAL_CFAD_ENABLED = True
REGIONAL_USE_PARALLEL = True
REGIONAL_MAX_WORKERS = None  # None = use CPU count

REGIONAL_OUTPUT_DIR = f"{OUTPUT_DIR}/Arrays_Regional"
REGIONAL_CFAD_OUTPUT_DIR = f"{OUTPUT_DIR}/CFAD_Regional_Results"

REGIONS = ['urban', 'upwind', 'downwind', 'left', 'right']

REGIONAL_CONFIG = {
    # Reference coordinates (Houston KHGX radar)
    'city_center_lat': 29.4719,
    'city_center_lon': -95.0787,

    # Wind configuration
    'wind_data_path': f'{OUTPUT_DIR}/wind_data/wind_data_1.nc',
    'default_wind_direction': 138.37,  # SE wind direction (degrees)
    'temporal_wind': 138.37,
    'pressure_level': 850,  # hPa
    'temporal_avg': 'overall',  # 'overall', 'daily', or 'hourly'

    # Distance parameters
    'downwind_start': 100,  # km
    'downwind_end': 150,    # km
    'sector_angle': 160,    # degrees (total sector width)

    # Geographic bounds and shapefile (update shapefile path to your data)
    'bounding_box_limits': (-96.22093200683594, -93.9366226196289, 28.483173370361328, 30.45787239074707),
    'shapefile_path': f'{OUTPUT_DIR}/2022_Shapefile/2022_Developed_Shapefile_1.shp',

    # Processing
    'num_workers': None,  # Auto-detect CPU count
    'regions': REGIONS,

    # CFAD regional output
    'cfad_enabled': True,
    'cfad_output_dir': f'{OUTPUT_DIR}/CFAD_Regional_Results',
    'cfad_plot_output_dir': f'{OUTPUT_DIR}/CFAD_Regional_Results/plots',
    'cfad_profile_output_dir': f'{OUTPUT_DIR}/CFAD_Regional_Results/profiles',
}

# Backward compatibility
REGIONAL_CFAD_CONFIG = REGIONAL_CONFIG
