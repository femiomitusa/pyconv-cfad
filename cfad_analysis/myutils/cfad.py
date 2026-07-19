import warnings
import numpy as np


def cfad_calc(data, 
              bins, 
              norm_opt =2):
    """Calculate CFAD (Contoured Frequency by Altitude Diagram).
    
    Args:
        data: Input data array (heights x points)
        bins: Bin edges for histogram
        norm_opt: Normalization option
        
    Returns:
        Tuple of (normalized_histogram, bin_centers, raw_histogram)
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        if data.size == 0:
            hist2d = np.zeros((len(bins) - 1, data.shape[0]))
            bin_ctrs = (bins[:-1] + bins[1:]) / 2
            return hist2d, bin_ctrs, hist2d.copy()
        
        # Initialize histogram array (bins x heights)
        hist2d = np.zeros((len(bins) - 1, data.shape[0]))
        
        # Calculate histogram for each height level
        for lev in range(data.shape[0]):
            # Convert to numeric and filter out non-numeric values
            level_data_raw = data[lev, :]
            
            # Convert to numeric, replacing non-numeric with NaN
            try:
                level_data_numeric = np.array([
                    float(x) if x is not None and str(x).replace('.', '').replace('-', '').isdigit()
                    else np.nan for x in level_data_raw
                ])
            except (ValueError, TypeError):
                # If conversion fails, try direct conversion
                try:
                    level_data_numeric = np.array(level_data_raw, dtype=float)
                except (ValueError, TypeError):
                    # Last resort: skip this level
                    continue
            
            # Remove invalid values (NaN, inf, etc.)
            level_data = level_data_numeric[np.isfinite(level_data_numeric)]
            
            if len(level_data) > 0:
                histogram, _ = np.histogram(level_data, bins)
                hist2d[:, lev] = histogram
        
        # Store raw histogram for statistics
        raw_hist2d = hist2d.copy()
        
        # Apply normalization
        if norm_opt == 1:
            # Normalize by sum at each level
            level_sums = np.sum(hist2d, axis=0)
            valid_levels = level_sums > 0
            hist2d[:, valid_levels] /= level_sums[valid_levels]
        elif norm_opt == 2:
            # Normalize by maximum value
            max_val = np.max(hist2d)
            if max_val > 0:
                hist2d /= max_val
        
        # Calculate bin centers
        bin_ctrs = (bins[:-1] + bins[1:]) / 2
        
        return hist2d, bin_ctrs, raw_hist2d

def vert_stats(data, percentiles=[0, 25, 50, 75, 90, 100]):
    """Compute vertical statistics for each height level.
    
    Args:
        data: Input data array (heights x points)
        percentiles: List of percentiles to calculate
        
    Returns:
        Dictionary containing statistics for each height level
    """
    if data.size == 0:
        return {
            stat: np.full(data.shape[0], np.nan)
            for stat in ['mean'] + [f'p{p}' for p in percentiles] + ['IQR']
        }
    
    # Apply mask to invalid values
    data = np.ma.masked_invalid(data)
    num_heights = data.shape[0]
    
    # Initialize statistics dictionary
    stats = {
        stat: np.zeros(num_heights) 
        for stat in ['mean'] + [f'p{p}' for p in percentiles] + ['IQR']
    }
    
    # Calculate statistics for each height level
    for h in range(num_heights):
        level_data = np.ma.compressed(data[h, :])
        if len(level_data) > 0:
            stats['mean'][h] = np.mean(level_data)
            for p in percentiles:
                stats[f'p{p}'][h] = np.percentile(level_data, p)
            stats['IQR'][h] = stats['p75'][h] - stats['p25'][h]
        else:
            stats['mean'][h] = np.nan
            for p in percentiles:
                stats[f'p{p}'][h] = np.nan
            stats['IQR'][h] = np.nan
    
    return stats

def process_data(data_list):
    zdr_data = []
    rhv_data = []
    kdp_data = []
    lon_list = []
    lat_list = []
    Z_data = []

    for d in data_list:
        # zdr_data.extend(d['differential_reflectivity_save'])
        zdr_data.extend([val for val in d['differential_reflectivity_save'] if isinstance(val, list)])
        Z_data.extend([val for val in d['reflectivity_save'] if isinstance(val, list)])
        rhv_data.extend([val for val in d['cross_correlation_ratio_save'] if isinstance(val, list)])
        kdp_data.extend([val for val in d['kdp_save'] if isinstance(val, list)])

        lon_val = d.get('gridlon', None)
        lat_val = d.get('gridlat', None)

        if lon_val is not None:
            lon_list.append(lon_val)

        if lat_val is not None:
            lat_list.append(lat_val)

    # Use object arrays for variable-length data
    combined_zdr_data = np.array(zdr_data, dtype=object)
    combined_rhv_data = np.array(rhv_data, dtype=object)
    combined_Z_data = np.array(Z_data, dtype=object)
    combined_kdp_data = np.array(kdp_data, dtype=object)
    combined_lon = np.array(lon_list)
    combined_lat = np.array(lat_list)
    
    return combined_Z_data, combined_zdr_data, combined_rhv_data, combined_kdp_data, combined_lon, combined_lat
   
