import numpy as np


def latlon2cart(latcoords, loncoords, latradar=29.4719, lonradar=-95.0787):
    """Convert lat/lon to radar-centred Cartesian coordinates (metres)."""
    latcoords = np.array(latcoords)
    loncoords = np.array(loncoords)

    r_earth = 6378.1  # km

    phi_s = np.radians(latcoords)
    lambda_s = np.radians(loncoords)
    phi_f = np.radians(latradar)
    lambda_f = np.radians(lonradar)

    dlat = phi_f - phi_s
    dlon = lambda_f - lambda_s

    a = np.sin(dlat / 2) ** 2 + np.cos(phi_s) * np.cos(phi_f) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = r_earth * c

    y1 = np.sin(dlon) * np.cos(phi_f)
    x1 = np.cos(phi_s) * np.sin(phi_f) - np.sin(phi_s) * np.cos(phi_f) * np.cos(dlon)
    theta = np.arctan2(y1, x1)

    x = d * np.sin(theta + np.pi) * 1000  # metres
    y = d * np.cos(theta + np.pi) * 1000
    return x, y


def extract_and_save(data, arr_save, storm_loc, j):
    arr_save[:, j] = np.squeeze(data[storm_loc])
    return arr_save


def filter_data_for_stage(datas, basetimes):
    by_time = {}
    for d in datas:
        by_time.setdefault(d['basetime'], []).append(d)
    return [np.array(by_time[t], dtype=object) for t in basetimes if t in by_time]
