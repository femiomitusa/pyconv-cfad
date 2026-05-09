#!/usr/bin/env python3
import io
import json
import os
import sys
import tempfile
import urllib.request
import warnings
import zipfile
from pathlib import Path

# PROJ database must be in env before any CRS operations
for _p in (
    Path('/home/oomitusa/miniforge3/envs/metstat/lib/python3.12/site-packages/pyproj/proj_dir/share/proj'),
    Path('/home/oomitusa/miniforge3/envs/metstat/share/proj'),
):
    if _p.exists():
        os.environ['PROJ_DATA'] = str(_p)
        os.environ['PROJ_LIB']  = str(_p)
        break

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib import patheffects
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

sys.path.append(str(Path(__file__).parent))
from config import REGIONAL_CONFIG, OUTPUT_DIR

CENSUS_UAC_URL     = 'https://www2.census.gov/geo/tiger/TIGER2022/UAC/tl_2022_us_uac20.zip'
HOUSTON_GEOID      = '40429'
SHAPEFILE_DIR      = Path(OUTPUT_DIR) / 'urban_shapefile'
SHAPEFILE_PATH     = SHAPEFILE_DIR / 'houston_uac20.shp'
REGIONS_OUTPUT_DIR = Path(OUTPUT_DIR) / 'regions'
DOMAIN_RADIUS_KM   = 125
URBAN_RADIUS_WARN  = 70  # warn if urban hull exceeds this — Stage 3 sectors get compressed


def download_houston_shapefile(dest: Path = SHAPEFILE_PATH) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  Shapefile already present: {dest}")
        return dest

    print("  Downloading Census TIGER/Line Urban Areas...")
    with urllib.request.urlopen(CENSUS_UAC_URL) as resp:
        raw = resp.read()

    print("  Extracting and filtering to Houston (GEOID20=40429)...")
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        with tempfile.TemporaryDirectory() as tmpdir:
            zf.extractall(tmpdir)
            shp_files = list(Path(tmpdir).glob('*.shp'))
            if not shp_files:
                sys.exit("No .shp found inside the downloaded zip.")
            gdf = gpd.read_file(shp_files[0])

    houston = gdf[gdf['GEOID20'] == HOUSTON_GEOID]
    if houston.empty:
        sys.exit(f"GEOID20 '{HOUSTON_GEOID}' not found. Sample IDs: {gdf['GEOID20'].head().tolist()}")

    houston.to_crs('EPSG:4326').to_file(dest)
    print(f"  Saved: {dest}  ({houston['NAME20'].iloc[0]})")
    return dest


def _utm_crs(lon: float, lat: float) -> str:
    zone = int((lon + 180) / 6) + 1
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"


def build_urban_boundary(shapefile_path: str, city_lon: float, city_lat: float):
    utm = _utm_crs(city_lon, city_lat)
    gdf = gpd.read_file(shapefile_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    merged = unary_union(gdf.to_crs(utm).geometry)
    if merged.geom_type == 'MultiPolygon':
        merged = unary_union(sorted(merged.geoms, key=lambda g: g.area, reverse=True)[:2])

    hull_utm   = merged.convex_hull
    hull_wgs84 = gpd.GeoSeries([hull_utm], crs=utm).to_crs("EPSG:4326").iloc[0]

    radius_km = np.sqrt(hull_utm.area / np.pi) / 1000
    print(f"  Urban convex hull equivalent radius: {radius_km:.1f} km")
    if radius_km > URBAN_RADIUS_WARN:
        warnings.warn(
            f"Urban hull radius ({radius_km:.0f} km) exceeds {URBAN_RADIUS_WARN} km — "
            "the urban polygon is large relative to the analysis domain.",
            stacklevel=2,
        )
    return hull_wgs84


def save_urban_geojson(hull_wgs84, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'urban.geojson'
    with open(path, 'w') as f:
        json.dump(
            {'type': 'Feature', 'geometry': mapping(hull_wgs84), 'properties': {'region': 'urban'}},
            f,
        )
    print(f"  Saved: {path}")
    return path


def _circle_lonlat(cx, cy, radius_km, n=360):
    a    = np.linspace(0, 2 * np.pi, n)
    lons = cx + (radius_km / (111.320 * np.cos(np.radians(cy)))) * np.sin(a)
    lats = cy + (radius_km / 110.574) * np.cos(a)
    return lons, lats


def plot_urban_domain(
    hull_wgs84,
    city_lon: float,
    city_lat: float,
    domain_radius_km: float,
    output_path: Path,
) -> None:
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(11, 10), subplot_kw={'projection': proj})

    ax.add_feature(cfeature.OCEAN.with_scale('10m'),     facecolor='#cde6f5', zorder=0)
    ax.add_feature(cfeature.LAND.with_scale('10m'),      facecolor='#f2efe9', zorder=0)
    ax.add_feature(cfeature.LAKES.with_scale('10m'),     facecolor='#cde6f5', zorder=0)
    ax.add_feature(cfeature.STATES.with_scale('10m'),    edgecolor='#aaaaaa', linewidth=0.7, zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='#555555', linewidth=1.0, zorder=2)

    gdf = gpd.GeoDataFrame(geometry=[hull_wgs84], crs='EPSG:4326')
    gdf.plot(ax=ax, color='#d62728', alpha=0.30, transform=proj, zorder=3)
    gdf.boundary.plot(ax=ax, color='#d62728', linewidth=2.0, transform=proj, zorder=4)

    pe = [patheffects.withStroke(linewidth=3, foreground='white')]
    c  = hull_wgs84.centroid
    ax.text(c.x, c.y, 'Urban', fontsize=12, ha='center', va='center',
            fontweight='bold', color='#d62728', transform=proj, zorder=6, path_effects=pe)

    dlons, dlats = _circle_lonlat(city_lon, city_lat, domain_radius_km)
    ax.plot(dlons, dlats, color='#333333', linewidth=1.4, linestyle='--', transform=proj, zorder=5)

    ax.plot(city_lon, city_lat, marker='^', color='black', markersize=10, transform=proj, zorder=8)
    ax.text(city_lon + 0.06, city_lat + 0.06, 'KHGX\nradar',
            fontsize=9, ha='left', va='bottom', fontweight='bold', color='black',
            transform=proj, zorder=8, path_effects=pe)

    ax.text(-94.6, 28.5, 'Gulf of\nMexico', fontsize=10, ha='center',
            color='#1a5a8a', style='italic', transform=proj, zorder=6, alpha=0.8)

    x0, y0 = 0.96, 0.18
    ax.annotate('N', xy=(x0, y0 + 0.05), xytext=(x0, y0),
                xycoords='axes fraction', textcoords='axes fraction',
                ha='center', fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.8))

    sb_lat  = city_lat - domain_radius_km / 110.574 * 0.88
    sb_lon0 = city_lon - 0.1
    sb_lon1 = sb_lon0 + 50 / (111.320 * np.cos(np.radians(sb_lat)))
    ax.plot([sb_lon0, sb_lon1], [sb_lat, sb_lat], 'k-', lw=3, transform=proj, zorder=7)
    ax.text((sb_lon0 + sb_lon1) / 2, sb_lat + 0.04, '50 km',
            ha='center', fontsize=8, transform=proj, zorder=7)

    handles = [
        mpatches.Patch(facecolor='#d62728', alpha=0.55, label='Urban convex hull'),
        plt.Line2D([0], [0], color='#333333', lw=1.4, linestyle='--',
                   label=f'KHGX domain ({domain_radius_km} km)'),
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=9, framealpha=0.92, edgecolor='#cccccc')

    pad_lon = domain_radius_km / (111.320 * np.cos(np.radians(city_lat))) + 0.4
    pad_lat = domain_radius_km / 110.574 + 0.4
    ax.set_extent(
        [city_lon - pad_lon, city_lon + pad_lon, city_lat - pad_lat, city_lat + pad_lat],
        crs=proj,
    )

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.4, linestyle='--')
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlocator     = mticker.MultipleLocator(0.5)
    gl.ylocator     = mticker.MultipleLocator(0.5)
    gl.xlabel_style = {'size': 9}
    gl.ylabel_style = {'size': 9}

    ax.set_title(
        'PyMOOSAIC — Urban Boundary  |  KHGX Houston\n'
        f'Domain {domain_radius_km} km  ·  '
        'Directional regions classified per-storm in Stage 3',
        fontsize=11, pad=10,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  Map saved: {output_path}")


def main():
    output_dir = REGIONS_OUTPUT_DIR
    city_lat   = REGIONAL_CONFIG['city_center_lat']
    city_lon   = REGIONAL_CONFIG['city_center_lon']

    shapefile = download_houston_shapefile()

    print(f"  City center   : {city_lat:.4f}N  {city_lon:.4f}E")
    print(f"  Domain radius : {DOMAIN_RADIUS_KM} km")
    print(f"  Shapefile     : {shapefile}")
    print()

    print("Building urban boundary...")
    urban_wgs84 = build_urban_boundary(str(shapefile), city_lon, city_lat)

    print(f"\nSaving urban.geojson to {output_dir}...")
    save_urban_geojson(urban_wgs84, output_dir)

    print("\nGenerating diagnostic map...")
    plot_urban_domain(
        urban_wgs84, city_lon, city_lat,
        domain_radius_km=DOMAIN_RADIUS_KM,
        output_path=output_dir / 'region_map.png',
    )


if __name__ == '__main__':
    main()
