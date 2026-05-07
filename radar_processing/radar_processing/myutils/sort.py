# Convert to Cartesian coordinates
import numpy as np

def latlon2cart(latcoords, loncoords, latradar=29.4719, lonradar=-95.0787):
    # Ensure latcoords and loncoords are numpy arrays to handle element-wise operations
    latcoords = np.array(latcoords)
    loncoords = np.array(loncoords)
    
    # Earth's radius in kilometers
    r_earth = 6378.1
    
    # Convert latitude and longitude from degrees to radians
    phi_s = np.radians(latcoords)
    lambda_s = np.radians(loncoords)
    phi_f = np.radians(latradar)
    lambda_f = np.radians(lonradar)
    
    # Calculate differences in coordinates
    dlat = phi_f - phi_s
    dlon = lambda_f - lambda_s
    
    # Haversine formula to calculate the great-circle distance
    a = np.sin(dlat / 2)**2 + np.cos(phi_s) * np.cos(phi_f) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = r_earth * c
    
    # Calculate intermediate values for Cartesian conversion
    y1 = np.sin(dlon) * np.cos(phi_f)
    x1 = np.cos(phi_s) * np.sin(phi_f) - np.sin(phi_s) * np.cos(phi_f) * np.cos(dlon)
    
    # Calculate bearing (theta)
    theta = np.arctan2(y1, x1)
    
    # Convert to radar Cartesian coordinates and convert distance to meters
    x = d * np.sin(theta + np.pi) * 1000  # Convert to meters
    y = d * np.cos(theta + np.pi) * 1000  # Convert to meters
    
    # Round to the nearest 500 meters and ensure it does not exceed the bounds
    # x_rounded = np.clip(500 * np.round(x / 500), -100000, 100000)
    # y_rounded = np.clip(500 * np.round(y / 500), -100000, 100000)
    
    # # Convert numpy arrays back to lists
    # x_list = x_rounded.tolist()
    # y_list = y_rounded.tolist()
    
    return x, y


def extract_and_save(data, arr_save, storm_loc, j):
    data_extracted = np.squeeze(data[storm_loc])
    arr_save[:, j] = data_extracted
    return arr_save



def extract_and_save_parallel(i, gridlon, gridlat, radius, xx, yy, Z, Zdr, rhv, kdp, base_time, area):
    xctr, yctr, rad_lim = gridlon[i], gridlat[i], radius[i]
    rad = np.sqrt((xx - xctr) ** 2 + (yy - yctr) ** 2)
    storm_loc = rad <= rad_lim
    num_grid_pts = np.sum(storm_loc)

    Zdata_save = np.full((num_grid_pts, Z.shape[0]), np.nan)
    Zdr_save = np.full((num_grid_pts, Zdr.shape[0]), np.nan)
    rhv_save = np.full((num_grid_pts, rhv.shape[0]), np.nan)
    kdp_save = np.full((num_grid_pts, kdp.shape[0]), np.nan)

    for j in range(Z.shape[0]):
        Zdata_save = extract_and_save(np.squeeze(Z[j, :, :]), Zdata_save, storm_loc, j)
        Zdr_save = extract_and_save(np.squeeze(Zdr[j, :, :]), Zdr_save, storm_loc, j)
        rhv_save = extract_and_save(np.squeeze(rhv[j, :, :]), rhv_save, storm_loc, j)
        kdp_save = extract_and_save(np.squeeze(kdp[j, :, :]), kdp_save, storm_loc, j)

    return {
        'basetime': base_time[i],
        'radius (m)': radius[i],
        'area (km)': area[i],
        'gridlon (m)': gridlon[i],
        'gridlat (m)': gridlat[i],
        'Zdata_save': Zdata_save.tolist(),
        'Zdr_save': Zdr_save.tolist(),
        'rhv_save': rhv_save.tolist(),
        'kdp_save': kdp_save.tolist(),
    }

def filter_data_for_stage(datas, basetimes):
    # Filter data for non-empty numpy arrays matching each basetime
    return [np.array([d for d in datas if d['basetime'] == basetime], dtype=object) for basetime in basetimes if len([d for d in datas if d['basetime'] == basetime]) > 0]
