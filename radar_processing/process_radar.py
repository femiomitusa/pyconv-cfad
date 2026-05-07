import os
import sys
import glob
import xarray as xr
import numpy as np
from tqdm import tqdm
from typing import Tuple, List

# Add project root to Python path for config import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from root config
from config import (
    BASE_DATA_DIR, STATS_FILE, RADAR_FILE_PATTERN,
    GRID_BOUNDS, GRID_POINTS, GRID_SPACING,
    RUN_RADAR_PROCESSING, RADAR_VISUALIZATION,
    SKIP_EXISTING_PROCESSING, YEAR_START, YEAR_END, VALID_MONTHS,
    TARGET_MODE, TARGET_YEAR, TARGET_MONTH, TARGET_DAY,
    QUIET_MODE
)

# Import utility functions
from utils import get_data_directory, get_array_directory, get_figures_directory, get_date_string

# Import from radar processing package - use correct import paths
try:
    from radar_processing.utils import filter_tracked_cells, get_datetime_from_filename, extract_cell_parameters
    from radar_processing.visualization import create_radar_plot
    from radar_processing import (
        setup_radar_grid,
        process_radar_file,
        parallel_process_cells
    )
    RADAR_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Radar processing modules not fully available: {e}")
    print("Some radar processing functions may not work.")
    RADAR_MODULES_AVAILABLE = False
    
    # Define minimal fallback functions
    def setup_radar_grid(*args, **kwargs):
        raise NotImplementedError("Radar processing modules not available")
    
    def process_radar_file(*args, **kwargs):
        raise NotImplementedError("Radar processing modules not available")
    
    def parallel_process_cells(*args, **kwargs):
        raise NotImplementedError("Radar processing modules not available")
    
    def setup_output_directories(*args, **kwargs):
        raise NotImplementedError("Radar processing modules not available")
    
    def get_datetime_from_filename(*args, **kwargs):
        raise NotImplementedError("Radar processing modules not available")
    
    def get_output_paths(*args, **kwargs):
        raise NotImplementedError("Radar processing modules not available")
    
    def filter_tracked_cells(*args, **kwargs):
        raise NotImplementedError("Radar processing modules not available")
    
    def extract_cell_parameters(*args, **kwargs):
        raise NotImplementedError("Radar processing modules not available")
    
    def create_radar_plot(*args, **kwargs):
        raise NotImplementedError("Radar processing modules not available")


def validate_input_data(year: int, month: int, day: int) -> Tuple[bool, List[str], str]:
    """Validate that input radar data exists for the specified date.
    
    Args:
        year: Year (e.g., 2019)
        month: Month as integer (e.g., 6 for June)
        day: Day as integer (e.g., 29)
        
    Returns:
        Tuple[bool, List[str], str]: (data_valid, radar_files, data_directory)
    """
    data_dir = get_data_directory(year, month, day, BASE_DATA_DIR)
    
    if not os.path.exists(data_dir):
        print(f"❌ Input validation failed: Data directory does not exist: {data_dir}")
        return False, [], data_dir
    
    # Check for NEXRAD Level 2 files with KHGX pattern for specific date
    # Format: KHGX20190629_HHMMSS_V06 for June 29, 2019
    date_str = f"{year:04d}{month:02d}{day:02d}"
    radar_pattern = f"{data_dir}/KHGX{date_str}_*_V06"
    radar_files = glob.glob(radar_pattern)
    
    if not radar_files:
        print(f"❌ Input validation failed: No NEXRAD files found for {year}-{month:02d}-{day:02d}")
        print(f"   Searched pattern: {radar_pattern}")
        return False, [], data_dir
    
    # print(f"✅ Input validation passed: Found {len(radar_files)} NEXRAD files for {year}-{month:02d}-{day:02d}")
    return True, radar_files, data_dir


def check_output_exists(year: int, month: int, day: int) -> Tuple[bool, int, str]:
    """Check if processed arrays already exist for the specified date.
    
    Verifies that all input files have been processed by comparing
    the number of input radar files to output array files.
    
    Args:
        year: Year (e.g., 2019)
        month: Month as integer (e.g., 6 for June)
        day: Day as integer (e.g., 29)
        
    Returns:
        Tuple[bool, int, str]: (all_processed, num_arrays, array_directory)
    """
    # Get expected input and output directories
    data_dir = get_data_directory(year, month, day, BASE_DATA_DIR)
    array_dir = get_array_directory(year, month, day, BASE_DATA_DIR)
    
    if not os.path.exists(array_dir):
        return False, 0, array_dir
    
    # Count input radar files for this date
    date_str = f"{year:04d}{month:02d}{day:02d}"
    radar_pattern = f"{data_dir}/KHGX{date_str}_*_V06"
    input_files = glob.glob(radar_pattern)
    num_input_files = len(input_files)
    
    # Count processed .npy array files
    array_files = glob.glob(f"{array_dir}/*.npy")
    num_arrays = len(array_files)
    
    # All files are processed if output count matches input count
    all_processed = num_arrays >= num_input_files and num_input_files > 0
    
    return all_processed, num_arrays, array_dir


def check_date_in_stats(year: int, month: int, day: int) -> bool:
    """Check if a date exists in the stats file.
    
    Returns:
        bool: True if date exists in stats file, False otherwise
    """
    try:
        with xr.open_dataset(STATS_FILE) as stats:
            time_mask = (
                (stats['base_time'].dt.year == year) & 
                (stats['base_time'].dt.month == month) & 
                (stats['base_time'].dt.day == day)
            )
            date_filtered = stats.where(time_mask, drop=True)
            return date_filtered.sizes.get('tracks', 0) > 0
    except Exception:
        return True  # If can't check, proceed with processing


def log_skipped_day(year: int, month: int, day: int, reason: str):
    """Log a skipped day to the skipped days file."""
    log_file = os.path.join(BASE_DATA_DIR, "skipped_days.log")
    with open(log_file, 'a') as f:
        f.write(f"{year}-{month:02d}-{day:02d}: {reason}\n")


def process_radar_data_with_safeguards(year: int, month: int, day: int, force: bool = False) -> bool:
    """Process radar data for a specific date with comprehensive safeguards.
    
    Args:
        year: Year (e.g., 2019)
        month: Month as integer (e.g., 6 for June)
        day: Day as integer (e.g., 29)
        force: If True, process even if arrays already exist
        
    Returns:
        bool: True if processing successful or arrays already exist, False on failure
    """
    
    # SAFEGUARD 1: Validate input data exists
    data_valid, radar_files, data_dir = validate_input_data(year, month, day)
    if not data_valid:
        return False
    
    # SAFEGUARD 1.5: Check if date exists in stats file (skip if not)
    if not force and os.path.exists(STATS_FILE):
        if not check_date_in_stats(year, month, day):
            log_skipped_day(year, month, day, "Date not found in stats file")
            return True  # Return success but skip processing
    
    # SAFEGUARD 2: Check if output already exists (unless forced)
    if not force and SKIP_EXISTING_PROCESSING:
        all_processed, num_arrays, array_dir = check_output_exists(year, month, day)
        if all_processed:
            print(f"Skipping {year}-{month:02d}-{day:02d} - all {num_arrays} arrays already exist")
            return True
        if num_arrays > 0:
            print(f"Partial output found for {year}-{month:02d}-{day:02d} ({num_arrays} arrays); processing missing files")
    
    # Create output directories
    array_dir = get_array_directory(year, month, day, BASE_DATA_DIR)
    fig_dir = get_figures_directory(year, month, day, BASE_DATA_DIR)
    os.makedirs(array_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    if not QUIET_MODE:
        print(f"Input directory: {data_dir}")
        print(f"Array output directory: {array_dir}")
        print(f"Figure output directory: {fig_dir}")
        print(f"Processing {len(radar_files)} radar files...")
    
    try:
        # Use the updated date format for processing
        date_str = get_date_string(year, month, day)
        success = process_radar_data_for_date(year, month, day)
        
        # Simple check - if any array files exist, processing succeeded
        array_dir = get_array_directory(year, month, day, BASE_DATA_DIR)
        actual_files = len(glob.glob(f"{array_dir}/*.npy"))
        if actual_files > 0:
            print(f"✅ Processing successful! Generated {actual_files} array files")
            return True
        else:
            print("❌ No arrays generated")
            return False
            
    except Exception as e:
        print(f"❌ Error during radar processing: {e}")
        return False


def process_radar_data_for_date(year: int, month: int, day: int) -> bool:
    """Process radar data for a specific date using parallel processing.
    
    Args:
        year: Year (e.g., 2019)
        month: Month as integer (e.g., 6 for June)
        day: Day as integer (e.g., 29)
        
    Returns:
        bool: True if processing successful, False otherwise
    """
    date_str = get_date_string(year, month, day)
    
    # Setup directories using config helpers
    data_dir = get_data_directory(year, month, day, BASE_DATA_DIR)
    array_dir = get_array_directory(year, month, day, BASE_DATA_DIR)
    fig_dir = get_figures_directory(year, month, day, BASE_DATA_DIR)
    
    # Ensure output directories exist
    os.makedirs(array_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    try:
        # Load tracked cells data
        if not os.path.exists(STATS_FILE):
            print(f"Warning: Stats file not found at {STATS_FILE}")
            print("Proceeding without cell tracking data...")
            tracked_cells = None
        else:
            tracked_cells = xr.open_dataset(STATS_FILE)
        
        # Get list of radar files for specific date
        # Format: KHGX20190629_HHMMSS_V06 for June 29, 2019
        date_str = f"{year:04d}{month:02d}{day:02d}"
        radar_pattern = f"{data_dir}/KHGX{date_str}_*_V06"
        radar_files = glob.glob(radar_pattern)
        
        if not radar_files:
            print(f"No radar files found matching {radar_pattern}")
            return False
        
        # Setup radar grid
        xx, yy, z = setup_radar_grid(GRID_BOUNDS, GRID_POINTS, GRID_SPACING)
        
        processed_count = 0
        
        # Simple progress - just process files without individual progress bar
        for i, filename in enumerate(sorted(radar_files)):
            basename = os.path.basename(filename)
            
            # Generate output paths
            array_filename = basename.replace('_V06', '.npy')
            fig_filename = basename.replace('_V06', '_plot.png')
            
            data_output_path = os.path.join(array_dir, array_filename)
            fig_output_path = os.path.join(fig_dir, fig_filename)
            
            # Skip if already processed (individual file check)
            if os.path.exists(data_output_path):
                if not RADAR_VISUALIZATION or os.path.exists(fig_output_path):
                    continue
            
            try:
                # Get time range for this file
                start_date, end_date = get_datetime_from_filename(basename)
                if not all([start_date, end_date]):
                    print(f"Could not parse datetime from {basename}, skipping...")
                    continue
                
                # Filter tracked cells for this time period (if available)
                if tracked_cells is not None:
                    cells = filter_tracked_cells(tracked_cells, start_date, end_date)
                    if cells.empty:
                        # Skip files with no tracked cells (silent skip)
                        continue
                    cell_params = extract_cell_parameters(cells)
                else:
                    # If no cell tracking data, create dummy parameters for grid processing
                    cell_params = []
                
                # Process radar file
                radar_fields = process_radar_file(filename)
                if not radar_fields:
                    print(f"Failed to process radar file {basename}, skipping...")
                    continue
                
                # Process cells in parallel (or just grid if no cells)
                if cell_params:
                    results = parallel_process_cells(cell_params, radar_fields, xx, yy)
                else:
                    # Process entire grid if no cell tracking
                    results = [{'data': radar_fields, 'metadata': {'file': basename}}]
                
                # Save results
                if results:
                    results_array = np.array(results, dtype=object)
                    np.save(data_output_path, results_array)
                    processed_count += 1
                    
                    # Create visualization if enabled
                    if RADAR_VISUALIZATION:
                        if cell_params:
                            cell_locations = [
                                (res.get('gridlon', 0), res.get('gridlat', 0), res.get('radius', 1000))
                                for res in results if isinstance(res, dict)
                            ]
                        else:
                            cell_locations = []
                        
                        create_radar_plot(
                            xx, yy,
                            radar_fields.get('reflectivity', radar_fields.get('Z', None)),
                            cell_locations,
                            f"Radar {basename} - {date_str}",
                            fig_output_path
                        )
                else:
                    print(f"No results generated for {basename}")
                    
            except Exception as e:
                print(f"Error processing {basename}: {e}")
                continue
        
        if not QUIET_MODE:
            print(f"Successfully processed {processed_count} out of {len(radar_files)} radar files")
        return processed_count > 0
        
    except Exception as e:
        print(f"Error in radar processing: {e}")
        return False

def is_valid_month(month_dir: str, valid_months_list: List[str] = None) -> bool:
    """Check if directory name matches a valid month (case-insensitive)."""
    if valid_months_list is None:
        valid_months_list = VALID_MONTHS
    return month_dir.lower() in [m.lower() for m in valid_months_list]

def process_for_config_targets() -> bool:
    """Process radar data for targets specified in configuration.
    
    Uses CFAD_TARGET_YEAR, CFAD_TARGET_MONTH, CFAD_TARGET_DAY from config.
    
    Returns:
        bool: True if processing successful, False otherwise
    """
    from config import CFAD_TARGET_YEAR, CFAD_TARGET_MONTH, CFAD_TARGET_DAY
    
    year = CFAD_TARGET_YEAR
    month = CFAD_TARGET_MONTH  
    day = CFAD_TARGET_DAY
    
    print(f"Processing radar data for configured target: {year}-{month:02d}-{day:02d}")
    return process_radar_data_with_safeguards(year, month, day)


def main():
    """Process radar data if enabled in config with updated safeguards."""
    if not RUN_RADAR_PROCESSING:
        print("Radar processing is disabled in config.py")
        return
    
    # Check if radar processing modules are available
    if not RADAR_MODULES_AVAILABLE:
        print("❌ Radar processing modules not available")
        print("Please check radar_processing package installation")
        return

    if not QUIET_MODE:
        print()
        print(f"Base data directory: {BASE_DATA_DIR}")
    
    if TARGET_MODE:
        if not QUIET_MODE:
            print(f"🎯 Target date mode: {TARGET_YEAR}-{TARGET_MONTH:02d}-{TARGET_DAY:02d}")
        years_to_process = [TARGET_YEAR]
        import calendar
        target_month_name = calendar.month_abbr[TARGET_MONTH]
        valid_months = [target_month_name]
    else:
        if not QUIET_MODE:
            print(f"Processing years: {YEAR_START} to {YEAR_END}")
            print(f"Valid months: {VALID_MONTHS}")
        years_to_process = list(range(YEAR_START, YEAR_END + 1))
        valid_months = VALID_MONTHS
    
    if not QUIET_MODE:
        print(f"Safeguards enabled: {SKIP_EXISTING_PROCESSING}")

    success_count = 0
    total_count = 0

    # Process each year
    for year in years_to_process:
        year_dir = os.path.join(BASE_DATA_DIR, str(year))
        
        # Skip if year directory doesn't exist
        if not os.path.isdir(year_dir):
            print(f"Warning: Year directory {year} not found at {year_dir}, skipping...")
            continue
            
        if not QUIET_MODE:
            print()
            print(f"PROCESSING YEAR: {year}")
            print()
        
        # Get list of day directories (Jul16, Jun29, etc.)
        day_dirs = []
        for item in os.listdir(year_dir):
            item_path = os.path.join(year_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                # Check if directory starts with valid month
                for month in valid_months:
                    if item.startswith(month):
                        day_dirs.append(item)
                        break
        
        if not day_dirs:
            print(f"No valid day directories found for year {year}")
            continue
        
        day_dirs.sort()
        if not QUIET_MODE:
            print(f"Found {len(day_dirs)} day directories to process: {day_dirs}")
        
        # Month-level progress bar
        month_progress = tqdm(day_dirs, desc=f"{year}", leave=False, ncols=50) if not QUIET_MODE else day_dirs
        
        # Process each day directory
        for day_name in month_progress:
            day_dir = os.path.join(year_dir, day_name)
            
            # Extract month from day directory name (e.g., "Jul16" -> "Jul")
            import calendar
            month_name = day_name[:3]  # First 3 characters
            month_names = [calendar.month_abbr[i] for i in range(1, 13)]
            month_num = month_names.index(month_name) + 1
            
            if not QUIET_MODE:
                print(f"\nProcessing day: {year}/{day_name} (Month {month_num})")
            
            # Get all radar files in this day directory
            radar_files = [f for f in os.listdir(day_dir) 
                          if f.startswith(RADAR_FILE_PATTERN.replace('*', '')) and f.endswith('_V06')]
            
            if not radar_files:
                if not QUIET_MODE:
                    print(f"No radar files found in {day_dir}")
                continue
            
            # All files in this directory are from the same day - no grouping needed
            day_files = [os.path.join(day_dir, filename) for filename in radar_files]
            
            if not QUIET_MODE:
                print(f"Found {len(day_files)} radar files for {day_name}")
            
            # Extract day number from directory name (e.g., "Jul16" -> 16)
            file_day = int(day_name[3:])  # Remove first 3 chars (month name)
            
            # Process this day
            total_count += 1
            if not QUIET_MODE:
                print(f"\nProcessing: {year}-{month_num:02d}-{file_day:02d} ({len(day_files)} files)")
            
            try:
                success = process_radar_data_with_safeguards(year, month_num, file_day)
                if success:
                    success_count += 1
                    if QUIET_MODE:
                        print(f"✅ {year}-{month_num:02d}-{file_day:02d} ({len(day_files)} files)")
                    else:
                        print(f"✅ Successfully processed {year}-{month_num:02d}-{file_day:02d}")
                else:
                    print(f"❌ Failed to process {year}-{month_num:02d}-{file_day:02d}")
            except Exception as e:
                print(f"❌ Error processing {year}-{month_num:02d}-{file_day:02d}: {e}")
                if not QUIET_MODE:
                    import traceback
                    traceback.print_exc()

    print(f"\\n{'='*80}")
    print("RADAR PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total days processed: {total_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {total_count - success_count}")
    if total_count > 0:
        print(f"Success rate: {(success_count/total_count)*100:.1f}%")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
