#!/usr/bin/env python3
"""
CFAD Analysis Runner with Configuration Integration and Comprehensive Validation
Integrates cfad_analysis_latest.py with the main project configuration with safeguards
"""

import sys
import os
import glob
import numpy as np
from typing import Tuple, List, Dict, Any

# Add parent directory to path to import config
sys.path.append('..')
sys.path.append('.')

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, script_dir)

try:
    import config
    from config import (
        RUN_CFAD_ANALYSIS, SKIP_EXISTING_ANALYSIS, REGIONAL_CFAD_ENABLED, REGIONS
    )
    from utils import get_array_directory, get_data_directory
except ImportError:
    print("Error: Could not import config.py. Make sure you're running from the project root or cfad_analysis directory.")
    sys.exit(1)

# Import the CFAD analysis functions
try:
    from cfad_analysis_latest import (
        load_and_filter_data, process_cfad_data, compute_all_stats,
        plot_cfads, plot_profiles, plot_percentiles, process_multi_temporal_cfad,
        bin_centers_dict
    )
except ImportError:
    # If direct import fails, try importing from current directory
    import cfad_analysis_latest
    load_and_filter_data = cfad_analysis_latest.load_and_filter_data
    process_cfad_data = cfad_analysis_latest.process_cfad_data
    compute_all_stats = cfad_analysis_latest.compute_all_stats
    plot_cfads = cfad_analysis_latest.plot_cfads
    plot_profiles = cfad_analysis_latest.plot_profiles
    plot_percentiles = cfad_analysis_latest.plot_percentiles
    process_multi_temporal_cfad = cfad_analysis_latest.process_multi_temporal_cfad
    bin_centers_dict = cfad_analysis_latest.bin_centers_dict


def configure_cfad_bins(cfad_module) -> None:
    """Synchronize CFAD bin edges with the central config.py settings."""
    bin_specs = {
        'Z': (config.CFAD_Z_LIMITS, config.CFAD_DZ, 'z'),
        'ZDR': (config.CFAD_ZDR_LIMITS, config.CFAD_DZDR, 'zdr'),
        'rho': (config.CFAD_RHO_LIMITS, config.CFAD_DRHO, 'rho'),
        'kdp': (config.CFAD_KDP_LIMITS, config.CFAD_DKDP, 'kdp'),
    }

    cfad_module.bin_centers_dict.clear()
    for var, (limits, step, attr_prefix) in bin_specs.items():
        start, stop = limits
        bins = np.arange(start, stop + step * 0.5, step)
        centers = (bins[:-1] + bins[1:]) / 2
        setattr(cfad_module, f'{attr_prefix}_bins', bins)
        setattr(cfad_module, f'bins_{attr_prefix}', centers)
        cfad_module.bin_centers_dict[var] = centers


def validate_cfad_input_arrays(year: int, month: int, day: int, region: str = None) -> Tuple[bool, List[str], str]:
    """Validate that processed radar arrays exist for CFAD analysis.
    
    Args:
        year: Year (e.g., 2019)
        month: Month as integer (e.g., 6 for June)
        day: Day as integer (e.g., 29)
        region: Optional region name for regional analysis
        
    Returns:
        Tuple[bool, List[str], str]: (arrays_valid, array_files, array_directory)
    """
    if region:
        # For regional analysis, use regional array directory with same MonthDay format
        import calendar
        month_name = calendar.month_abbr[month]
        array_dir = os.path.join(config.BASE_DATA_DIR, 'Arrays', region, str(year), f"{month_name}{day:02d}")
    else:
        # For standard analysis, use main array directory
        array_dir = get_array_directory(year, month, day, config.BASE_DATA_DIR)
    
    if not os.path.exists(array_dir):
        print(f"❌ CFAD input validation failed: Array directory does not exist: {array_dir}")
        return False, [], array_dir
    
    # Check for processed .npy array files for specific date
    # Format: KHGX20190629_HHMMSS_V06.npy for June 29, 2019
    date_str = f"{year:04d}{month:02d}{day:02d}"
    array_pattern = f"{array_dir}/KHGX{date_str}_*.npy"
    array_files = glob.glob(array_pattern)
    
    if not array_files:
        print(f"❌ CFAD input validation failed: No .npy arrays found for {year}-{month:02d}-{day:02d}")
        print(f"   Searched pattern: {array_pattern}")
        return False, [], array_dir
    
    # Validate array files are not empty and have reasonable size
    valid_arrays = []
    for array_file in array_files:
        try:
            file_size = os.path.getsize(array_file)
            if file_size > 100:  # At least 100 bytes
                valid_arrays.append(array_file)
            else:
                print(f"Warning: Array file {array_file} is too small ({file_size} bytes)")
        except Exception as e:
            print(f"Warning: Could not check array file {array_file}: {e}")
    
    if not valid_arrays:
        print(f"❌ CFAD input validation failed: No valid .npy arrays found in {array_dir}")
        return False, [], array_dir
    
    print(f"✅ CFAD input validation passed: Found {len(valid_arrays)} valid arrays in {array_dir}")
    return True, valid_arrays, array_dir


def check_cfad_output_exists(year: int, month: int, day: int, region: str = None) -> Tuple[bool, Dict[str, bool], str]:
    """Check if CFAD analysis results already exist for the specified date.
    
    Args:
        year: Year (e.g., 2019)
        month: Month as integer (e.g., 6 for June)
        day: Day as integer (e.g., 29)
        
    Returns:
        Tuple[bool, Dict[str, bool], str]: (results_exist, output_status, output_directory)
    """
    # Check appropriate output directory (regional or main)
    if region:
        output_dir = os.path.join(config.CFAD_OUTPUT_DIR, region.capitalize())
        plot_output = os.path.join(output_dir, "cfad_plots")
        profile_output = os.path.join(output_dir, "profile_plots")
    else:
        output_dir = config.CFAD_OUTPUT_DIR
        plot_output = config.CFAD_PLOT_OUTPUT
        profile_output = config.CFAD_PROFILE_OUTPUT
    
    # Check for various output files
    output_status = {
        'filtered_data': os.path.exists(os.path.join(output_dir, "filtered_cfad_data.npy")),
        'cfad_plots': False,
        'profile_plots': False
    }
    
    # Check for plot files in appropriate directories
    if os.path.exists(plot_output):
        plot_files = glob.glob(f"{plot_output}/*.png")
        output_status['cfad_plots'] = len(plot_files) > 0
    
    if os.path.exists(profile_output):
        profile_files = glob.glob(f"{profile_output}/*.png")
        output_status['profile_plots'] = len(profile_files) > 0
    
    # Also check main output directory for plot files (fallback)
    if not output_status['cfad_plots']:
        cfad_files = glob.glob(f"{output_dir}/cfad_plots*.png")
        output_status['cfad_plots'] = len(cfad_files) > 0
    
    if not output_status['profile_plots']:
        profile_files = glob.glob(f"{output_dir}/profile_plots*.png")
        output_status['profile_plots'] = len(profile_files) > 0
    
    # Consider results to exist if we have filtered data and at least some plots
    results_exist = (output_status['filtered_data'] and 
                    (output_status['cfad_plots'] or output_status['profile_plots']))
    
    return results_exist, output_status, output_dir


def validate_array_data_quality(array_files: List[str]) -> Tuple[bool, Dict[str, Any]]:
    """Validate the quality of radar array data before CFAD processing.
    
    Args:
        array_files: List of paths to .npy array files
        
    Returns:
        Tuple[bool, Dict[str, Any]]: (data_valid, quality_metrics)
    """
    quality_metrics = {
        'total_files': len(array_files),
        'valid_files': 0,
        'total_data_points': 0,
        'non_zero_points': 0,
        'suspicious_files': []
    }
    
    try:
        # Sample a reasonable number of files for efficiency (up to 50 files)
        sample_size = min(50, len(array_files))
        sampled_files = array_files[:sample_size]
        for array_file in sampled_files:
            try:
                data = np.load(array_file, allow_pickle=True)
                quality_metrics['valid_files'] += 1
                
                # Basic data quality checks
                if hasattr(data, 'shape'):
                    quality_metrics['total_data_points'] += np.prod(data.shape)
                    if np.any(data != 0):
                        quality_metrics['non_zero_points'] += np.count_nonzero(data)
                elif isinstance(data, (list, np.ndarray)) and len(data) > 0:
                    quality_metrics['total_data_points'] += len(data)
                    # For object arrays, check if they contain data
                    try:
                        if any(item is not None for item in data.flat if hasattr(data, 'flat')):
                            quality_metrics['non_zero_points'] += 1
                    except:
                        quality_metrics['non_zero_points'] += 1  # Assume valid if can't check
                else:
                    quality_metrics['suspicious_files'].append(os.path.basename(array_file))
                    
            except Exception as e:
                quality_metrics['suspicious_files'].append(
                    f"{os.path.basename(array_file)}: {str(e)}"
                )
    
        # Determine if data quality is acceptable (use sampled files count)
        sampled_count = len(sampled_files)
        valid_ratio = quality_metrics['valid_files'] / max(sampled_count, 1)
        has_data = quality_metrics['non_zero_points'] > 0
        few_suspicious = len(quality_metrics['suspicious_files']) < sampled_count * 0.5
        
        # More realistic thresholds for radar data: at least 20% valid files with some data
        data_valid = valid_ratio > 0.2 and has_data and few_suspicious
        
        if data_valid:
            print(f"✅ Data quality validation passed:")
            print(f"   Valid files: {quality_metrics['valid_files']}/{sampled_count} (sampled from {quality_metrics['total_files']} total)")
            print(f"   Non-zero data points: {quality_metrics['non_zero_points']}")
        else:
            print(f"❌ Data quality validation failed:")
            print(f"   Valid files: {quality_metrics['valid_files']}/{sampled_count} (sampled from {quality_metrics['total_files']} total)")
            print(f"   Suspicious files: {len(quality_metrics['suspicious_files'])}")
            if quality_metrics['suspicious_files']:
                print(f"   Issues: {quality_metrics['suspicious_files'][:3]}...")
        
        return data_valid, quality_metrics
        
    except Exception as e:
        print(f"❌ Error during data quality validation: {e}")
        return False, quality_metrics


def run_cfad_analysis_with_safeguards(year: int, month: int, day: int, force: bool = False, region: str = None) -> bool:
    """Run CFAD analysis for a specific date with comprehensive safeguards.
    
    Args:
        year: Year (e.g., 2019)
        month: Month as integer (e.g., 6 for June)
        day: Day as integer (e.g., 29)
        force: If True, run analysis even if results already exist
        
    Returns:
        bool: True if analysis successful or results already exist, False on failure
    """
    print(f"\n{'='*60}")
    if region:
        print(f"CFAD ANALYSIS ({region.upper()}) - {year}-{month:02d}-{day:02d}")
    else:
        print(f"CFAD ANALYSIS - {year}-{month:02d}-{day:02d}")
    print(f"{'='*60}")
    
    # Check if CFAD analysis is enabled
    if not RUN_CFAD_ANALYSIS:
        print("CFAD analysis is disabled in configuration")
        return True
    
    # SAFEGUARD 1: Validate input arrays exist
    arrays_valid, array_files, array_dir = validate_cfad_input_arrays(year, month, day, region)
    if not arrays_valid:
        return False
    
    # SAFEGUARD 2: Check if output already exists (unless forced)
    if not force and SKIP_EXISTING_ANALYSIS:
        results_exist, output_status, _ = check_cfad_output_exists(year, month, day, region)
        if results_exist:
            print(f"CFAD results already exist for {year}-{month:02d}-{day:02d}")
            print(f"Output status: {output_status}")
            print("Skipping analysis due to SKIP_EXISTING_ANALYSIS=True")
            return True
    
    # SAFEGUARD 3: Validate data quality
    data_valid, quality_metrics = validate_array_data_quality(array_files)
    if not data_valid:
        print("❌ Data quality validation failed - aborting CFAD analysis")
        return False
    
    print()
    print(f"Output directory: {config.CFAD_OUTPUT_DIR}")
    print(f"Processing {len(array_files)} array files...")
    print(f"Data quality: {quality_metrics['valid_files']} valid files with {quality_metrics['non_zero_points']} data points")
    
    try:
        # Update configuration for this specific date
        original_values = {}
        original_values['target_year'] = config.TARGET_YEAR
        original_values['target_month'] = config.TARGET_MONTH
        original_values['target_day'] = config.TARGET_DAY

        # Temporarily update config for this run
        config.TARGET_YEAR = year
        config.TARGET_MONTH = month
        config.TARGET_DAY = day

        # Run the analysis
        success = run_single_day_analysis(region)
        
        # Restore original configuration
        config.TARGET_YEAR = original_values['target_year']
        config.TARGET_MONTH = original_values['target_month']
        config.TARGET_DAY = original_values['target_day']

        if success:
            # Verify output was created
            results_exist, output_status, _ = check_cfad_output_exists(year, month, day, region)
            if results_exist:
                print(f"✅ CFAD analysis successful!")
                print(f"   Results: {output_status}")
                return True
            else:
                print("❌ CFAD analysis completed but no output found")
                return False
        else:
            print("❌ CFAD analysis failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during CFAD analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


def setup_cfad_environment():
    """Setup environment variables based on config.py"""
    
    # Create output directories
    os.makedirs(config.CFAD_OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CFAD_PLOT_OUTPUT, exist_ok=True)
    os.makedirs(config.CFAD_PROFILE_OUTPUT, exist_ok=True)
    
    # Set environment variables for CFAD analysis
    os.environ['CFAD_FOLDER_PATH'] = config.ARRAY_OUTPUT_DIR
    os.environ['CFAD_STATS_PATH'] = config.STATS_FILE
    os.environ['CFAD_OUTPUT_PATH'] = os.path.join(config.CFAD_OUTPUT_DIR, "filtered_cfad_data.npy")
    os.environ['CFAD_PARENT_DIR'] = config.ARRAY_OUTPUT_DIR
    os.environ['CFAD_SAVE_DIR'] = config.CFAD_OUTPUT_DIR
    os.environ['CFAD_PLOT_OUTPUT'] = config.CFAD_PLOT_OUTPUT
    os.environ['CFAD_PROFILE_OUTPUT'] = config.CFAD_PROFILE_OUTPUT
    os.environ['CFAD_TARGET_INDICES'] = ' '.join(map(str, config.CFAD_TARGET_INDICES))
    
    print("CFAD environment configured successfully")
    print(f"Output directory: {config.CFAD_OUTPUT_DIR}")
    print(f"Data source: {config.ARRAY_OUTPUT_DIR}")
    print(f"Stats file: {config.STATS_FILE}")

def run_single_day_analysis(region: str = None):
    """Run single-day CFAD analysis using configuration"""
    
    # Override global variables in cfad_analysis_latest with config values
    import cfad_analysis_latest as cfad
    
    # Update configuration
    configure_cfad_bins(cfad)
    cfad.target_indices = config.CFAD_TARGET_INDICES
    cfad.target_year = config.TARGET_YEAR
    cfad.target_month = config.TARGET_MONTH  
    cfad.target_day = config.TARGET_DAY
    cfad.ymax = config.CFAD_YMAX
    cfad.include_iqr_on_mean = config.CFAD_INCLUDE_IQR_ON_MEAN
    cfad.percentiles = config.CFAD_PERCENTILES
    cfad.zlims = config.CFAD_Z_LIMITS
    cfad.zdrlims = config.CFAD_ZDR_LIMITS
    cfad.rholims = config.CFAD_RHO_LIMITS
    cfad.kdplims = config.CFAD_KDP_LIMITS
    cfad.norm_opt = config.CFAD_NORM_OPT
    cfad.kdp_calc = config.CFAD_KDP_CALC
    cfad.profile_colors = config.CFAD_PROFILE_COLORS
    cfad.percentile_display = config.CFAD_PERCENTILE_DISPLAY
    
    # Set heights from the global variable in cfad module
    cfad.heights = cfad.heights
    
    # Update paths
    cfad.folder_path = config.ARRAY_OUTPUT_DIR
    cfad.stats_path = config.STATS_FILE
    cfad.output_path = os.path.join(config.CFAD_OUTPUT_DIR, "filtered_cfad_data.npy")
    cfad.parent_directory = config.ARRAY_OUTPUT_DIR
    cfad.save_data = config.CFAD_OUTPUT_DIR
    cfad.plot_output = config.CFAD_PLOT_OUTPUT
    cfad.profile_output = config.CFAD_PROFILE_OUTPUT
    
    
    # Find appropriate data directory based on configuration
    year_dir = str(config.TARGET_YEAR)
    month_names = {6: 'Jun', 7: 'Jul', 8: 'Aug'}  # Map month numbers to names
    month_dir = month_names.get(config.TARGET_MONTH, f"Month{config.TARGET_MONTH}")

    # Use the specific directory path for the target date
    if region:
        # For regional analysis, use regional array directory with MonthDay format
        import calendar
        month_name = calendar.month_abbr[config.TARGET_MONTH]
        target_array_dir = os.path.join(config.BASE_DATA_DIR, 'Arrays', region, 
                                       str(config.TARGET_YEAR), f"{month_name}{config.TARGET_DAY:02d}")
    else:
        # For standard analysis, use main array directory
        target_array_dir = get_array_directory(config.TARGET_YEAR, config.TARGET_MONTH, config.TARGET_DAY, config.BASE_DATA_DIR)
    
    if not os.path.exists(target_array_dir):
        print(f"Warning: Target array directory not found: {target_array_dir}")
        print("Available directories:", os.listdir(config.ARRAY_OUTPUT_DIR) if os.path.exists(config.ARRAY_OUTPUT_DIR) else "None")
        return False
    
    # Check if directory contains array files
    npy_files = [f for f in os.listdir(target_array_dir) if f.endswith('.npy')]
    if not npy_files:
        print(f"Warning: No .npy files found in {target_array_dir}")
        return False
    
    potential_dirs = [os.path.relpath(target_array_dir, config.ARRAY_OUTPUT_DIR)]
    
    # Use the specific target directory
    target_dir = potential_dirs[0]  # This is the relative path to our target directory
    print()
    print(f"Using target directory: {target_dir}")
    print(f"Full path: {target_array_dir}")
    print(f"Array files available: {len(npy_files)}")
    
    cfad.directory = target_dir
    cfad.folder_path = os.path.join(config.ARRAY_OUTPUT_DIR, target_dir)
    
    print()
    print(f"Processing data from: {cfad.folder_path}")
    print(f"Target date: {config.TARGET_YEAR}-{config.TARGET_MONTH:02d}-{config.TARGET_DAY:02d}")
    print(f"Target indices: {config.CFAD_TARGET_INDICES}")
    
    # Initialize aggregated_data
    cfad.aggregated_data = {idx: {
        'total_raw_hist': {},
        'final_means': {},
        'stats': {}
    } for idx in config.CFAD_TARGET_INDICES}
    
    # Set up regional paths if region is specified
    if region:
        region_dir = os.path.join(config.CFAD_OUTPUT_DIR, region.capitalize())
        os.makedirs(region_dir, exist_ok=True)
        cfad.output_path = os.path.join(region_dir, "filtered_cfad_data.npy")
        cfad.save_data = region_dir
        plot_dir = os.path.join(region_dir, "cfad_plots")
        profile_dir = os.path.join(region_dir, "profile_plots")
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(profile_dir, exist_ok=True)
        cfad.plot_output = os.path.join(plot_dir, "cfad_plots")
        cfad.profile_output = os.path.join(profile_dir, "profile_plots")
    
    try:
        # Run the analysis for each target index
        for target_time_index in config.CFAD_TARGET_INDICES:
            print(f"Processing time index: {target_time_index}")
            
            # Load and filter data - if this fails, skip CFAD analysis
            try:
                load_and_filter_data(
                    cfad.folder_path, cfad.stats_path, cfad.output_path,
                    config.TARGET_YEAR, config.TARGET_MONTH, config.TARGET_DAY,
                    target_time_index, region=getattr(config, 'CFAD_REGION', None)
                )
                
                # Check if filtered data file was created
                if not os.path.exists(cfad.output_path):
                    print(f"No filtered data available for {config.TARGET_YEAR}-{config.TARGET_MONTH:02d}-{config.TARGET_DAY:02d}")
                    print("Skipping CFAD analysis for this region.")
                    return True  # Return success to continue with other regions
                    
            except Exception as e:
                print(f"Data filtering failed: {e}")
                print("Skipping CFAD analysis for this region.")
                return True  # Return success to continue with other regions
            
            success = process_cfad_data(
                cfad.save_data,
                target_time_index, 
                cfad.heights,
                config.CFAD_NORM_OPT, 
                config.CFAD_KDP_CALC
            )
            
            if not success:
                print(f"❌ CFAD processing failed for time index {target_time_index}")
                return False
        
        # Compute statistics and generate plots
        print("Computing statistics...")
        compute_all_stats(config.CFAD_TARGET_INDICES, cfad.aggregated_data, bin_centers_dict, config.CFAD_PERCENTILES)
        
        print("Generating CFAD plots...")
        plot_cfads(config.CFAD_TARGET_INDICES, cfad.aggregated_data, cfad.plot_output, 
                  cfad.heights, config.CFAD_YMAX, config.CFAD_Z_LIMITS, config.CFAD_ZDR_LIMITS, 
                  config.CFAD_RHO_LIMITS, config.CFAD_KDP_LIMITS, region)
        
        print("Generating profile plots...")
        plot_profiles(config.CFAD_TARGET_INDICES, cfad.aggregated_data, cfad.profile_output,
                     cfad.heights, config.CFAD_YMAX, config.CFAD_INCLUDE_IQR_ON_MEAN, region)
        
        print("Generating percentile plots...")
        plot_percentiles(config.CFAD_TARGET_INDICES, cfad.aggregated_data, cfad.profile_output,
                        cfad.heights, config.CFAD_YMAX, region)
        
        print("Single-day CFAD analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during CFAD analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_multi_temporal_analysis():
    """Run multi-temporal CFAD analysis using configuration"""
    print("=" * 60)
    print("RUNNING CFAD ANALYSIS - MULTI-TEMPORAL MODE")
    print("=" * 60)
    
    # Import and configure
    import cfad_analysis_latest as cfad
    
    # Update multi-temporal configuration
    configure_cfad_bins(cfad)
    cfad.multi_temporal_config = config.CFAD_MULTI_TEMPORAL.copy()
    
    try:
        # Run multi-temporal analysis
        final_hists, final_raw_hists = process_multi_temporal_cfad(cfad.multi_temporal_config)
        
        if final_hists and final_raw_hists:
            # Initialize aggregated_data for multi-temporal results
            cfad.aggregated_data = {0: {}}
            cfad.aggregated_data[0]['final_means'] = {f'hist_{var}': final_hists[var] for var in final_hists}
            cfad.aggregated_data[0]['total_raw_hist'] = final_raw_hists
            cfad.aggregated_data[0]['stats'] = {}
            
            # Compute statistics
            print("Computing statistics from aggregated data...")
            compute_all_stats([0], cfad.aggregated_data, bin_centers_dict, config.CFAD_PERCENTILES)
            
            # Generate plots
            suffix = config.CFAD_MULTI_TEMPORAL['output_suffix']
            print(f"Generating plots with suffix: {suffix}")
            
            plot_cfads([0], cfad.aggregated_data, f"{config.CFAD_PLOT_OUTPUT}_{suffix}", 
                      cfad.heights, config.CFAD_YMAX, config.CFAD_Z_LIMITS, config.CFAD_ZDR_LIMITS,
                      config.CFAD_RHO_LIMITS, config.CFAD_KDP_LIMITS)
            
            plot_profiles([0], cfad.aggregated_data, f"{config.CFAD_PROFILE_OUTPUT}_{suffix}",
                         cfad.heights, config.CFAD_YMAX, config.CFAD_INCLUDE_IQR_ON_MEAN)
            
            plot_percentiles([0], cfad.aggregated_data, f"{config.CFAD_PROFILE_OUTPUT}_{suffix}",
                           cfad.heights, config.CFAD_YMAX)
            
            print("Multi-temporal CFAD analysis completed successfully!")
            return True
        else:
            print("No data was successfully processed. Please check your configuration and data paths.")
            return False
            
    except Exception as e:
        print(f"Error during multi-temporal CFAD analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_cfad_for_config_targets() -> bool:
    """Run CFAD analysis for targets specified in configuration.
    
    Uses CFAD_TARGET_YEAR, CFAD_TARGET_MONTH, CFAD_TARGET_DAY from config.
    
    Returns:
        bool: True if analysis successful, False otherwise
    """
    year = config.TARGET_YEAR
    month = config.TARGET_MONTH
    day = config.TARGET_DAY

    print(f"Running CFAD analysis for configured target: {year}-{month:02d}-{day:02d}")
    
    # Check if regional CFAD is enabled
    if REGIONAL_CFAD_ENABLED:
        print()
        print(f"Regional CFAD analysis enabled - processing {len(REGIONS)} regions")
        all_success = True
        
        # Process each region
        for region in REGIONS:            
            success = run_cfad_analysis_with_safeguards(year, month, day, region=region)
            if not success:
                print(f"❌ Failed to process region: {region}")
                all_success = False
            else:
                print(f"✅ Successfully processed region: {region}")
        
        return all_success
    else:
        # Run standard analysis only
        return run_cfad_analysis_with_safeguards(year, month, day)


def main():
    """Main function to run CFAD analysis based on configuration with comprehensive validation"""
    
    # Check if CFAD analysis is enabled
    if not getattr(config, 'RUN_CFAD_ANALYSIS', False):
        print("CFAD analysis is disabled in config.py")
        return False
    
    print()
    print(f"Base data directory: {config.BASE_DATA_DIR}")
    print(f"Array directory: {config.ARRAY_OUTPUT_DIR}")
    print(f"Output directory: {config.CFAD_OUTPUT_DIR}")
    print(f"Safeguards enabled: {getattr(config, 'SKIP_EXISTING_ANALYSIS', True)}")
    print()

    try:
        # Setup environment
        setup_cfad_environment()
        print()
        
        # Check if multi-temporal analysis is enabled (can be overridden by environment variable)
        multi_temporal_enabled = config.CFAD_MULTI_TEMPORAL.get('enabled', False)
        if os.environ.get('CFAD_MULTI_TEMPORAL_ENABLED', '').lower() == 'false':
            multi_temporal_enabled = False
            
        if multi_temporal_enabled:
            print("Running multi-temporal CFAD analysis...")
            success = run_multi_temporal_analysis()
        else:
            print("Running single-day CFAD analysis with safeguards...")
            success = run_cfad_for_config_targets()

        print(f"\n{'='*80}")
        if success:
            print("CFAD ANALYSIS COMPLETED SUCCESSFULLY!")
            print(f"{'='*80}")
            print(f"Results saved to: {config.CFAD_OUTPUT_DIR}")
            if os.path.exists(config.CFAD_PLOT_OUTPUT):
                plot_count = len(glob.glob(f"{config.CFAD_PLOT_OUTPUT}/*.png"))
                print(f"CFAD plots: {plot_count} files in {config.CFAD_PLOT_OUTPUT}")
            if os.path.exists(config.CFAD_PROFILE_OUTPUT):
                profile_count = len(glob.glob(f"{config.CFAD_PROFILE_OUTPUT}/*.png"))
                print(f"Profile plots: {profile_count} files in {config.CFAD_PROFILE_OUTPUT}")
        else:
            print("CFAD ANALYSIS FAILED!")
        print(f"{'='*80}")
        
        return success
        
    except Exception as e:
        print(f"\\n{'='*80}")
        print("CFAD ANALYSIS PIPELINE ERROR!")
        print(f"{'='*80}")
        print(f"Unexpected error in main pipeline: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*80}")
        return False

if __name__ == "__main__":
    main()