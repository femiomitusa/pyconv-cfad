# ============================================================================
# IMPORTS
# ============================================================================
import numpy as np
import pandas as pd
import xarray as xr
import os
import pyart
import warnings
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob

pyart.load_config()

from myutils.sort import latlon2cart, filter_data_for_stage
from myutils.cfad import process_data, cfad_calc, vert_stats

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
# All paths can be overridden by environment variables
env_paths = {
    'folder_path': ('CFAD_FOLDER_PATH', "/Users/paulomits/Documents/pyconv/Data/Arrays/Jun29"),
    'stats_path': ('CFAD_STATS_PATH', '/Users/paulomits/Documents/pyconv/Data/stats_all.nc'),
    'output_path': ('CFAD_OUTPUT_PATH', "/Users/paulomits/Documents/pyconv/Data/Arrays/filtered_cfad_data.npy"),
    'parent_directory': ('CFAD_PARENT_DIR', '/Users/paulomits/Documents/pyconv/Data/Arrays'),
    'directory': ('CFAD_DIR', 'Jun29'),
    'base_dir': ('CFAD_BASE_DIR', '/Users/paulomits/Documents/pyconv/Data/Results/results_Jun29_1/'),
    'plot_output': ('CFAD_PLOT_OUTPUT', '/Users/paulomits/Documents/pyconv/Data/Results/results_Jun29_1/cfad'),
    'profile_output': ('CFAD_PROFILE_OUTPUT', '/Users/paulomits/Documents/pyconv/Data/Results/results_Jun29_1/profiles')
}

# Load paths from environment
for var, (env_key, default) in env_paths.items():
    globals()[var] = os.getenv(env_key, default)

save_data = os.path.join(base_dir, directory)

# ============================================================================
# ANALYSIS MODE CONFIGURATION
# ============================================================================

# --- Single-Day Analysis Settings ---
target_indices = [int(i) for i in os.getenv('CFAD_TARGET_INDICES', '0').split()]
target_year, target_month, target_day = 2019, 6, 29

# --- Multi-Temporal Analysis Settings ---
multi_temporal_config = {
    'enabled': False,  # Set to True to enable multi-temporal mode
    'base_data_path': '/Users/paulomits/Documents/pyconv/Data/Arrays', 
    'years': ['2018'],    # List of years or 'all'
    'months': ['Jun'], # List of months or 'all'
    'days': 'all',      # List of days or 'all'
    'aggregation_method': 'average',  # 'sum' or 'average'; For multi-year analysis, average is often better because it accounts for years with different amounts of data.
    'output_suffix': 'multi_temporal'
}

# ============================================================================
# DATA PROCESSING CONFIGURATION
# ============================================================================

# --- CFAD Processing Parameters ---
norm_opt = 2          # Normalization option
kdp_calc = True       # Enable KDP calculations
dzgrid = 500          # Grid spacing
percentiles = [0, 10, 25, 50, 75, 90, 100]  # Statistical percentiles to compute

# --- Spatial and Vertical Grid Configuration ---
grid_config = {
    'spatial': {'bounds': (-120000, 120000), 'points': 401},
    'vertical': {'start': 0, 'end': 20000, 'points': 41, 'step': 500}
}

# --- Variable Bin Configuration ---
bin_config = {
    'Z':   {'lims': (-20, 65),    'step': 5},      # Reflectivity
    'ZDR': {'lims': (-1, 5),      'step': 0.5},    # Differential Reflectivity
    'rho': {'lims': (0.87, 1.03), 'step': 0.0025}, # Correlation Coefficient
    'kdp': {'lims': (-1, 3),      'step': 0.1}     # Specific Differential Phase
}

# ============================================================================
# PLOTTING CONFIGURATION
# ============================================================================

# --- General Plot Settings ---
ymax = 11                  # Maximum altitude for plots (km)
include_iqr_on_mean = True # Show IQR bands on mean plots

# --- Profile Plot Colors ---
profile_colors = {
    'Z': 'blue',      # Reflectivity
    'ZDR': 'green',   # Differential Reflectivity
    'rho': 'red',     # Correlation Coefficient
    'kdp': 'purple'   # Specific Differential Phase
}

# --- Percentile Plot Configuration ---
percentile_display = [10, 25, 50, 75, 90]  # Which percentiles to show

percentile_styles = {
    10: (':', 1.5),                    # Dotted line, thin
    25: ('--', 2.0),                   # Dashed line, medium
    50: ('-', 3.0),                    # Solid line, thick (median)
    75: ('-.', 2.0),                   # Dash-dot line, medium
    90: ((0, (3, 1, 1, 1)), 1.5)      # Custom dash pattern, thin
}

percentile_labels = {
    10: '10th percentile',
    25: '25th percentile (Q1)',
    50: '50th percentile (Median)',
    75: '75th percentile (Q3)',
    90: '90th percentile'
}

# ============================================================================
# MATPLOTLIB SETTINGS
# ============================================================================
mpl.rcParams.update({
    'font.size': 13,
    'font.family': 'Arial',
    'figure.facecolor': 'white'
})

warnings.filterwarnings("ignore")

# ============================================================================
# AUTO-GENERATED VARIABLES (Do not modify)
# ============================================================================

# Generate spatial grids
xx, yy = np.meshgrid(*[np.linspace(*grid_config['spatial']['bounds'], 
                                   grid_config['spatial']['points'])] * 2)

# Generate vertical grids
z = np.linspace(*[grid_config['vertical'][k] for k in ['start', 'end', 'points']])
heights = np.arange(0, 20500, grid_config['vertical']['step'])
heights = heights[heights <= grid_config['vertical']['end']]

# Generate bins and bin centers for each variable
bin_centers_dict = {}
for var, config in bin_config.items():
    start, stop = config['lims']
    step = config['step']
    bins = np.arange(start, stop + step * 0.5, step)
    bin_centers_dict[var] = (bins[:-1] + bins[1:]) / 2
    
    # Create global variables for backward compatibility
    globals()[f'{var.lower()}_bins'] = bins
    globals()[f'bins_{var.lower()}'] = bin_centers_dict[var]
    globals()[f'{var.lower()}lims'] = config['lims']

# ============================================================================
# END OF CONFIGURATION
# ============================================================================

def load_and_filter_data(folder_path, stats_path, output_path, target_year, target_month, target_day, target_time_index, region=None):
    # Create target date string for filename filtering
    target_date_str = f"{target_year}{target_month:02d}{target_day:02d}"
    
    # Load only dictionary data from .npy files matching the target date
    # If region is specified, use the regional folder path
    if region:
        base_dir = os.path.dirname(os.path.dirname(folder_path))
        year_dir = os.path.basename(os.path.dirname(folder_path))
        day_dir = os.path.basename(folder_path)
        folder_path = os.path.join(base_dir, region, year_dir, day_dir)
    
    all_data = []
    matching_files = []
    for file in os.listdir(folder_path):
        if file.endswith('.npy') and not file.startswith('._') and target_date_str in file:
            matching_files.append(file)
            data = np.load(os.path.join(folder_path, file), allow_pickle=True)
            if isinstance(data, np.ndarray) and data.dtype == object:
                all_data.extend(data.flatten())
    
    print(f"Loaded {len(matching_files)} files for date {target_year}-{target_month:02d}-{target_day:02d}")
    
    cfad_df = pd.DataFrame(all_data)
    
    # Filter stats data
    with xr.open_dataset(stats_path) as stats:
        # First filter by date to avoid index errors
        time_mask = (
            (stats['base_time'].dt.year == target_year) & 
            (stats['base_time'].dt.month == target_month) & 
            (stats['base_time'].dt.day == target_day)
        )
        
        # Check if any data exists for target date
        date_filtered = stats.where(time_mask, drop=True)
        if date_filtered.sizes.get('tracks', 0) == 0:
            print(f"No storm cell tracks found in stats file for date {target_year}-{target_month:02d}-{target_day:02d}")
            print("Skipping CFAD analysis - stats data required for this date.")
            return
        
        # Now apply other filters on the date-filtered data
        time_count = date_filtered.sizes.get('times', 0)
        safe_time_index = target_time_index if target_time_index < time_count else 0
        mask = (
            np.isnan(date_filtered['start_split_tracknumber']) &
            (date_filtered['maxrange_flag'].isel(times=safe_time_index) == 1)
        )
        stats_filtered = date_filtered.isel(tracks=np.where(mask)[0])
        stats_filtered = stats_filtered.isel(times=safe_time_index)
            
        stats_df = stats_filtered[['base_time', 'meanlat', 'meanlon', 'tracks']].to_dataframe().reset_index()
    
    # Convert to numeric and merge
    # Handle both singular and plural forms of coordinate columns
    lat_col = 'latitudes' if 'latitudes' in cfad_df.columns else 'latitude'
    lon_col = 'longitudes' if 'longitudes' in cfad_df.columns else 'longitude'
    
    for col in [lat_col, lon_col]:
        cfad_df[col] = pd.to_numeric(cfad_df[col], errors='coerce')
    for col in ['meanlat', 'meanlon']:
        stats_df[col] = pd.to_numeric(stats_df[col], errors='coerce')
    
    if 'tracks' in cfad_df.columns and 'tracks' in stats_df.columns:
        cfad_df['tracks'] = pd.to_numeric(cfad_df['tracks'], errors='coerce')
        stats_df['tracks'] = pd.to_numeric(stats_df['tracks'], errors='coerce')
        merge_keys = ['tracks']
        if 'basetime' in cfad_df.columns and 'base_time' in stats_df.columns:
            cfad_df['_merge_time'] = pd.to_datetime(cfad_df['basetime'], errors='coerce')
            stats_df['_merge_time'] = pd.to_datetime(stats_df['base_time'], errors='coerce')
            merge_keys.append('_merge_time')
        filtered_cfad_df = cfad_df.merge(
            stats_df[merge_keys].drop_duplicates(),
            on=merge_keys,
            how='inner'
        ).drop(columns=['_merge_time'], errors='ignore')
    else:
        filtered_cfad_df = cfad_df.merge(
            stats_df[['meanlat', 'meanlon']],
            left_on=[lat_col, lon_col],
            right_on=['meanlat', 'meanlon'],
            how='inner'
        ).drop(columns=['meanlat', 'meanlon'])
    
    # Convert to list of dicts using itertuples (faster than iterrows)
    array_of_dicts = []
    for row in filtered_cfad_df.itertuples(index=False, name=None):
        row_dict = dict(zip(filtered_cfad_df.columns, row))
        for key, val in row_dict.items():
            if isinstance(val, np.ndarray):
                row_dict[key] = val.tolist()
        array_of_dicts.append(row_dict)
    
    np.save(output_path, np.array(array_of_dicts, dtype=object), allow_pickle=True)

def process_cfad_data(save_data, target_time_index, heights, norm_opt, kdp_calc):
    """
    Clean CFAD processing - works for both single/multi-temporal modes
    
    Args:
        save_data: Directory containing filtered data file
        target_time_index: Time index for processing
        heights: Height array for CFAD calculations
        norm_opt, kdp_calc: Processing options
        
    Returns:
        bool: True if successful, False otherwise
    """
    global aggregated_data
    os.makedirs(save_data, exist_ok=True)
    
    # 1. Load filtered data (single file)
    filtered_file = os.path.join(save_data, "filtered_cfad_data.npy")
    if not os.path.exists(filtered_file):
        print(f"No filtered data found at {filtered_file}")
        return False
    
    data = np.load(filtered_file, allow_pickle=True)
    
    # 2. Group by basetime (for unique time counting)
    grouped = {}
    for d in data:
        if basetime := d.get('basetime'):
            grouped.setdefault(basetime, []).append(d)
    
    print(f"Found {len(grouped)} unique time periods")
    
    # Check if we have any valid data
    if len(grouped) == 0:
        print("No valid time periods found - arrays may be empty or missing basetime data")
        return False
    
    # 3. Process each time group
    all_processed = []
    all_lengths = []
    for basetime in sorted(grouped):
        time_data = np.array(grouped[basetime], dtype=object)
        all_processed.append(process_data(time_data))
        all_lengths.append(len(time_data))
    
    # Save lengths and total
    np.savez(f'{save_data}/total_and_lengths_index_{target_time_index}.npz', 
             lengths=all_lengths, total=sum(all_lengths))
    
    # 4. Generate histograms and aggregate
    vars_to_process = ['Z', 'ZDR', 'rho'] + (['kdp'] if kdp_calc else [])
    var_indices = {'Z': 0, 'ZDR': 1, 'rho': 2, 'kdp': 3}
    bins = {'Z': z_bins, 'ZDR': zdr_bins, 'rho': rho_bins, 'kdp': kdp_bins}
    
    hist_data = {var: [] for var in vars_to_process}
    raw_hist_data = {var: [] for var in vars_to_process}
    bin_centers = {var: [] for var in vars_to_process}
    
    # Process histograms
    for processed in all_processed:
        for var in vars_to_process:
            try:
                data_var = processed[var_indices[var]].T[:len(heights), :]
                hist, ctr, raw_hist = cfad_calc(data_var, bins[var], norm_opt)
                hist_data[var].append(hist)
                raw_hist_data[var].append(raw_hist)
                bin_centers[var].append(ctr)
            except Exception as e:
                print(f"Warning: Skipping {var} for time index {target_time_index}: {e}")
    
    # Save histogram data
    save_dict = {}
    for var in vars_to_process:
        save_dict.update({
            f'hist_{var}': hist_data[var],
            f'raw_hist_{var}': raw_hist_data[var],
            f'{var}ctr': bin_centers[var]
        })
    np.savez(f'{save_data}/hist_mean_index_{target_time_index}.npz', **save_dict)
    
    # 5. Aggregate results
    for var in vars_to_process:
        if hist_data[var]:  # Only if we have data
            # Use actual dimensions from first histogram
            shape = hist_data[var][0].shape
            aggregated_data[target_time_index]['total_raw_hist'][var] = np.zeros(shape)
            aggregated_data[target_time_index]['final_means'][f'hist_{var}'] = np.zeros(shape)
            
            # Sum all histograms
            for raw_h, h in zip(raw_hist_data[var], hist_data[var]):
                aggregated_data[target_time_index]['total_raw_hist'][var] += raw_h
                aggregated_data[target_time_index]['final_means'][f'hist_{var}'] += h
        else:
            print(f"Warning: No data for variable {var}")
    
    return True

def compute_stats_from_hist(hist, bin_centers, percentiles):
    """Compute statistics from histogram with efficient edge case handling"""
    num_heights = hist.shape[1]
    stats = {stat: np.full(num_heights, np.nan) for stat in ['mean', 'mode'] + [f'p{p}' for p in percentiles] + ['IQR']}
    
    for h in range(num_heights):
        hist_level = hist[:, h]
        total_count = np.sum(hist_level)
        if total_count == 0:
            continue
            
        normalized_hist = hist_level / total_count
        nonzero_mask = hist_level > 0
        nonzero_count = np.count_nonzero(nonzero_mask)
        
        # Mean and mode
        stats['mean'][h] = np.sum(bin_centers * normalized_hist)
        stats['mode'][h] = bin_centers[np.argmax(hist_level)]
        
        # Handle special cases
        if nonzero_count == 1:
            # Single value case
            value = bin_centers[nonzero_mask][0]
            for p in percentiles:
                stats[f'p{p}'][h] = value
            stats['IQR'][h] = 0
            
        elif nonzero_count == 2 and len(np.unique(hist_level[nonzero_mask])) == 1:
            # Two values with equal weights
            values = bin_centers[nonzero_mask]
            for p in percentiles:
                stats[f'p{p}'][h] = values[0] if p <= 50 else values[1]
            stats['IQR'][h] = values[1] - values[0]
            
        else:
            # General case - compute percentiles
            cumsum = np.cumsum(normalized_hist)
            
            for p in percentiles:
                target = p / 100.0
                idx = np.searchsorted(cumsum, target)
                
                if idx == 0:
                    stats[f'p{p}'][h] = bin_centers[0]
                elif idx >= len(bin_centers):
                    stats[f'p{p}'][h] = bin_centers[-1]
                elif idx > 0 and cumsum[idx-1] < cumsum[idx]:
                    # Linear interpolation
                    frac = (target - cumsum[idx-1]) / (cumsum[idx] - cumsum[idx-1])
                    stats[f'p{p}'][h] = bin_centers[idx-1] + frac * (bin_centers[idx] - bin_centers[idx-1])
                else:
                    stats[f'p{p}'][h] = bin_centers[idx-1 if idx > 0 else idx]
            
            # Calculate IQR if percentiles include 25 and 75
            if 25 in percentiles and 75 in percentiles:
                stats['IQR'][h] = stats['p75'][h] - stats['p25'][h]
    
    return stats

def compute_all_stats(target_indices, aggregated_data, bin_centers_dict, percentiles):
    variables = ['Z', 'ZDR', 'rho', 'kdp']
    for idx in target_indices:
        aggregated_data[idx]['stats'] = {
            var: compute_stats_from_hist(
                aggregated_data[idx]['total_raw_hist'][var],
                bin_centers_dict[var],
                percentiles
            )
            for var in variables
            if var in aggregated_data[idx]['total_raw_hist']
        }

def plot_cfads(target_indices, aggregated_data, plot_output, heights, ymax, zlims, zdrlims, rholims, kdplims, region=None):
    # Define plot configuration
    plot_config = [
        ('hist_Z', 'Z (dBZ)', '(a) Reflectivity (Z)', zlims),
        ('hist_ZDR', r'$Z_{DR}$ (dB)', r'(b) Differential Reflectivity ($Z_{DR}$)', zdrlims),
        ('hist_rho', r'$\rho_{HV}$', r'(c) Correlation Coefficient ($\rho_{HV}$)', rholims),
        ('hist_kdp', r'K$_{DP}$ (deg/km)', r'(d) Specific Differential Phase ($K_{DP}$)', kdplims)
    ]
    
    for idx in target_indices:
        data = aggregated_data[idx]
        fig, axs = plt.subplots(2, 2, figsize=(15, 10), dpi=150)
        axs = axs.flatten()
        
        # Create descriptive name for the analysis
        descriptive_name = "Cell Initiation"
        
        for i, (hist_key, xlabel, title, xlims) in enumerate(plot_config):
            # Get histogram data and create meshgrid
            hist_data = data['final_means'][hist_key]
            x_vals = np.linspace(xlims[0], xlims[1], hist_data.shape[0])
            X, Y = np.meshgrid(heights, x_vals)
            
            # Create contour plot
            pcm = axs[i].contourf(Y, X/1000, hist_data, levels=20, cmap='HomeyerRainbow')
            
            # Set labels and properties
            axs[i].set_xlabel(xlabel)
            axs[i].set_ylabel('Altitude (km)')
            axs[i].set_title(title)  # Remove region and index from subplot titles
            axs[i].set_xlim(xlims)
            axs[i].set_ylim(0, ymax)
            fig.colorbar(pcm, ax=axs[i])
        
        # Add main title with region information
        region_suffix = f" ({region.capitalize()})" if region else ""
        fig.suptitle(f'CFAD Analysis - {descriptive_name}{region_suffix}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)  # Make room for suptitle
        plt.savefig(f'{plot_output}_index_{idx}.png', dpi=300)
        plt.close()

def plot_profiles(target_indices, aggregated_data, profile_output, heights, ymax, include_iqr_on_mean, region=None):
    # Configuration
    plot_config = {
        'Z': ('Z (dBZ)', 'Reflectivity (Z)'),
        'ZDR': ('$Z_{DR}$ (dB)', r'Differential Reflectivity ($Z_{DR}$)'),
        'rho': (r'$\rho_{HV}$', r'Correlation Coefficient ($\rho_{HV}$)'),
        'kdp': ('K$_{DP}$ (deg/km)', r'Specific Differential Phase ($K_{DP}$)')
    }
    
    stat_types = [
        ('mean', 'Mean', include_iqr_on_mean),
        ('p50', 'Median', False),
        ('mode', 'Mode', False)
    ]
    
    for stat_key, stat_name, show_iqr in stat_types:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=150)
        axes = axes.flatten()
        
        for idx_var, (var, (xlabel, title)) in enumerate(plot_config.items()):
            ax = axes[idx_var]
            color = profile_colors[var]
            
            # Plot all time indices
            for i, idx in enumerate(target_indices):
                stats = aggregated_data[idx]['stats'][var]
                label = f'Index {idx}' if len(target_indices) > 1 else f'{stat_name} Profile'
                
                # Main line
                ax.plot(stats[stat_key], heights/1000, color=color, label=label, linewidth=2.5)
                
                # IQR bands for mean plots
                if show_iqr and stat_key == 'mean':
                    ax.fill_betweenx(heights/1000, stats['p25'], stats['p75'],
                                   color=color, alpha=0.2, label='IQR' if i == 0 else None)
            
            # Customize subplot
            ax.set(title=f'({chr(97 + idx_var)}) {title}', 
                   xlabel=xlabel, ylabel='Altitude (km)', ylim=(0, ymax))
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.title.set_fontsize(14)
            
            if len(target_indices) > 1 or show_iqr:
                ax.legend(fontsize=10)
        
        # Overall formatting
        region_suffix = f" ({region.capitalize()})" if region else ""
        fig.suptitle(f'{stat_name} Vertical Profiles of Dual-Polarization Variables{region_suffix}',
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.savefig(f'{profile_output}_{stat_name.lower()}_2x2.png', dpi=300, bbox_inches='tight')
        plt.close()
        
def plot_percentiles(target_indices, aggregated_data, profile_output, heights, ymax, region=None):
    """Create 2x2 percentile family plots showing multiple percentiles per variable"""
    plot_config = {
        'Z': ('Z (dBZ)', 'Reflectivity (Z)'),
        'ZDR': ('$Z_{DR}$ (dB)', r'Differential Reflectivity ($Z_{DR})'),
        'rho': (r'$\rho_{HV}$', r'Correlation Coefficient ($\rho_{HV}$)'),
        'kdp': ('K$_{DP}$ (deg/km)', r'Specific Differential Phase ($K_{DP}$)')
    }
    
    
    # Alpha mapping for visual hierarchy
    alpha_map = {50: 1.0, 25: 0.8, 75: 0.8}  # Others default to 0.6
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=150)
    axes = axes.flatten()
    
    for idx_var, (var, (xlabel, title)) in enumerate(plot_config.items()):
        ax = axes[idx_var]
        color = profile_colors[var]
        
        # Plot percentiles for all time indices
        for i, idx in enumerate(target_indices):
            stats = aggregated_data[idx]['stats'][var]
            
            for percentile in percentile_display:
                percentile_key = f'p{percentile}'
                if percentile_key not in stats:
                    continue
                
                # Get style configuration
                line_style, line_width = percentile_styles[percentile]
                alpha = alpha_map.get(percentile, 0.6)
                
                # Create label (only for first index to avoid legend duplication)
                label = None
                if i == 0:
                    label = (f'Index {idx} - {percentile_labels[percentile]}' 
                            if len(target_indices) > 1 else percentile_labels[percentile])
                
                # Plot line
                ax.plot(stats[percentile_key], heights/1000,
                       color=color, linestyle=line_style, linewidth=line_width,
                       alpha=alpha, label=label)
        
        # Customize subplot
        ax.set(title=f'({chr(97 + idx_var)}) {title}',
               xlabel=xlabel, ylabel='Altitude (km)', ylim=(0, ymax))
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.title.set_fontsize(14)
        ax.legend(fontsize=10, loc='best')
    
    # Overall formatting
    region_suffix = f" ({region.capitalize()})" if region else ""
    fig.suptitle(f'Percentile Vertical Profiles of Dual-Polarization Variables{region_suffix}',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(f'{profile_output}_percentiles_2x2.png', dpi=300, bbox_inches='tight')
    plt.close()

def discover_temporal_data(config):
    """Auto-discover data structure and collect all valid paths"""
    base_path = config.get('base_data_path')
    if not base_path or not os.path.exists(base_path):
        print(f"Warning: Base path is invalid or does not exist: {base_path}")
        return []
    
    data_paths = []
    
    # Get directories based on config
    def get_dirs(path, config_key, parent_name):
        if config[config_key] == 'all':
            dirs = [d for d in os.listdir(path) 
                   if os.path.isdir(os.path.join(path, d)) and 
                   (config_key != 'years' or d.isdigit())]
        else:
            dirs = [str(item) for item in config[config_key]]
        
        return [(d, os.path.join(path, d)) for d in dirs 
                if os.path.exists(os.path.join(path, d))]
    
    # Iterate through year/month hierarchy (no day subdirectories)
    for year, year_path in get_dirs(base_path, 'years', 'Base'):
        for month, month_path in get_dirs(year_path, 'months', f'Year {year}'):
            # Check if there are .npy files in this month directory
            npy_files = glob.glob(os.path.join(month_path, '*.npy'))
            if npy_files:
                data_paths.append({
                    'year': year,
                    'month': month,
                    'day': None,  # No day level in this structure
                    'path': month_path
                })
    
    # Log missing paths if any were specified but not found
    if not data_paths and config['years'] != 'all':
        print(f"Warning: No valid data paths found for specified criteria")
    
    return data_paths

def process_single_period(folder_path, data_info):
    """Process a single temporal period and return histograms"""
    if data_info['day'] is not None:
        period = f"{data_info['year']}/{data_info['month']}/{data_info['day']}"
    else:
        period = f"{data_info['year']}/{data_info['month']}"
    print(f"  Processing {period}")
    
    npy_files = glob.glob(os.path.join(folder_path, '*.npy'))
    if not npy_files:
        print(f"    No .npy files found in {folder_path}")
        return {}, {}
    
    all_processed = []
    
    # Process all .npy files
    for file in npy_files:
        try:
            loaded_array = np.load(file, allow_pickle=True)
            if not (isinstance(loaded_array, np.ndarray) and loaded_array.dtype == object):
                continue
                
            # Ensure array format
            if loaded_array.ndim == 0:
                loaded_array = np.array([loaded_array.item()], dtype=object)
            
            # Group by basetime
            grouped = {}
            for d in loaded_array:
                if basetime := d.get('basetime'):
                    grouped.setdefault(basetime, []).append(d)
            
            if not grouped:
                continue
            
            # Process each basetime group
            for basetime in sorted(grouped):
                data = np.array(grouped[basetime], dtype=object)
                all_processed.append(process_data(data))
                
        except Exception as e:
            print(f"    Warning: Error processing {file}: {e}")
    
    if not all_processed:
        print(f"    No valid data found for {period}")
        return {}, {}
    
    # Calculate histograms
    vars_config = {
        'Z': (0, z_bins),
        'ZDR': (1, zdr_bins),
        'rho': (2, rho_bins),
        'kdp': (3, kdp_bins)
    }
    
    vars_to_process = [var for var in vars_config if var != 'kdp' or kdp_calc]
    hist_data = {var: [] for var in vars_to_process}
    raw_hist_data = {var: [] for var in vars_to_process}
    
    for processed in all_processed:
        for var in vars_to_process:
            try:
                var_index, bin_var = vars_config[var]
                data_var = processed[var_index].T[:len(heights), :]
                hist, ctr, raw_hist = cfad_calc(data_var, bin_var, norm_opt)
                hist_data[var].append(hist)
                raw_hist_data[var].append(raw_hist)
            except:
                continue
    
    return hist_data, raw_hist_data

def aggregate_histograms(hist_collection, method='sum'):
    """Aggregate histograms using specified method"""
    agg_func = {'sum': np.sum, 'average': np.mean}.get(method)
    if not agg_func:
        raise ValueError(f"Unknown aggregation method: {method}")
    
    return {var: agg_func(np.stack(hists), axis=0) 
            for var, hists in hist_collection.items() if hists}

def process_multi_temporal_cfad(config):
    """Process multiple temporal periods and aggregate results"""
    print(f"Starting multi-temporal CFAD analysis...")
    print(f"Aggregation method: {config['aggregation_method']}")
    
    # Discover all data paths
    data_paths = discover_temporal_data(config)
    print(f"Found {len(data_paths)} temporal periods to process")
    
    if not data_paths:
        print("No valid data paths found. Check your configuration.")
        return {}, {}
    
    # Initialize storage
    vars_list = ['Z', 'ZDR', 'rho', 'kdp']
    all_histograms = {var: [] for var in vars_list}
    all_raw_histograms = {var: [] for var in vars_list}
    
    # Process each temporal period
    for data_info in data_paths:
        period_hists, period_raw_hists = process_single_period(data_info['path'], data_info)
        
        # Accumulate histograms
        for var in vars_list:
            if period_hists.get(var):
                all_histograms[var].extend(period_hists[var])
                all_raw_histograms[var].extend(period_raw_hists[var])
    
    # Validate data collection
    total_histograms = sum(len(hists) for hists in all_histograms.values())
    print(f"Total histograms collected: {total_histograms}")
    
    if not total_histograms:
        print("No histograms were successfully collected. Check your data.")
        return {}, {}
    
    # Aggregate histograms
    print(f"Aggregating histograms using {config['aggregation_method']} method...")
    final_histograms = aggregate_histograms(all_histograms, config['aggregation_method'])
    final_raw_histograms = aggregate_histograms(all_raw_histograms, config['aggregation_method'])
    
    print("Multi-temporal aggregation complete!")
    return final_histograms, final_raw_histograms

# Main Execution
if __name__ == "__main__":
    mode = "MULTI-TEMPORAL" if multi_temporal_config['enabled'] else "SINGLE-DAY"
    print(f"{'=' * 60}\n{mode} CFAD ANALYSIS MODE\n{'=' * 60}")
    
    if multi_temporal_config['enabled']:
        # Multi-temporal analysis
        final_hists, final_raw_hists = process_multi_temporal_cfad(multi_temporal_config)
        
        if not (final_hists and final_raw_hists):
            print("No data was successfully processed. Please check your configuration and data paths.")
            exit(1)
        
        # Setup aggregated data
        aggregated_data = {
            0: {
                'final_means': {f'hist_{var}': hist for var, hist in final_hists.items()},
                'total_raw_hist': final_raw_hists,
                'stats': {}
            }
        }
        target_indices = [0]
        suffix = multi_temporal_config['output_suffix']
        output_prefix = f"{suffix}"
        
    else:
        # Single-day analysis
        aggregated_data = {idx: {
            'total_raw_hist': {},
            'final_means': {},
            'stats': {}
        } for idx in target_indices}
        
        for idx in target_indices:
            load_and_filter_data(folder_path, stats_path, output_path, target_year, target_month, target_day, idx)
            process_cfad_data(save_data, idx, heights, norm_opt, kdp_calc)
        
        suffix = ""
        output_prefix = ""
    
    # Common processing for both modes
    print("Computing statistics from aggregated data...")
    compute_all_stats(target_indices, aggregated_data, bin_centers_dict, percentiles)
    
    # Generate plots
    print(f"Generating plots{f' with suffix: {suffix}' if suffix else ''}...")
    plot_base = f"{plot_output}{f'_{suffix}' if suffix else ''}"
    profile_base = f"{profile_output}{f'_{suffix}' if suffix else ''}"
    
    plot_cfads(target_indices, aggregated_data, plot_base, heights, ymax, zlims, zdrlims, rholims, kdplims)
    plot_profiles(target_indices, aggregated_data, profile_base, heights, ymax, include_iqr_on_mean)
    plot_percentiles(target_indices, aggregated_data, profile_base, heights, ymax)
    
    print(f"{mode.title()} analysis complete!")
    if suffix:
        print(f"Output files saved with '{suffix}' suffix")
