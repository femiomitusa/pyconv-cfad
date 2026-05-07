#!/usr/bin/env python3
"""
NEXRAD Row-Based Regional Filtering using pycellstats approach

Filters individual storm cell rows from .npy files by geographic regions
using the exact polygon definitions from pycellstats.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import calendar
import numpy as np
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
import geopandas as gpd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import BASE_DATA_DIR, ARRAY_OUTPUT_DIR, REGIONAL_CONFIG, REGIONAL_USE_PARALLEL, REGIONAL_MAX_WORKERS, YEAR_START, YEAR_END, VALID_MONTHS, SKIP_EXISTING_FILTERING

# pycellstats configuration (from config.py)
CITY_CENTER_LAT = REGIONAL_CONFIG['city_center_lat']
CITY_CENTER_LON = REGIONAL_CONFIG['city_center_lon']
DOWNWIND_START = REGIONAL_CONFIG['downwind_start']
DOWNWIND_END = REGIONAL_CONFIG['downwind_end']
SECTOR_ANGLE = REGIONAL_CONFIG['sector_angle']
BOUNDING_BOX_LIMITS = REGIONAL_CONFIG['bounding_box_limits']
TEMPORAL_WIND = REGIONAL_CONFIG['temporal_wind']
SHAPEFILE_PATH = REGIONAL_CONFIG['shapefile_path']

def define_urban_region(shapefile_path: str, circle_scale: float = 1.0):
    """
    Define urban region from EPSG:4326 shapefile (exact pycellstats implementation).
    """
    try:
        # Read shapefile and verify CRS
        gdf = gpd.read_file(shapefile_path)
        if gdf.crs.to_epsg() != 4326:
            raise ValueError("Shapefile must be in EPSG:4326")
        
        # Get center point and UTM projection
        center_approx = gdf.geometry.union_all().centroid
        utm_zone = int((center_approx.x + 180) / 6) + 1
        utm_crs = f"EPSG:{32600 + utm_zone if center_approx.y >= 0 else 32700 + utm_zone}"
        
        # Project to UTM, merge polygons, and select largest areas
        gdf_utm = gdf.to_crs(utm_crs)
        merged_utm = gdf_utm.geometry.union_all()
        
        if hasattr(merged_utm, 'geoms'):
            # MultiPolygon - select the two largest polygons
            selected_polygons_utm = merged_utm.__class__(
                sorted(merged_utm.geoms, key=lambda x: x.area, reverse=True)[:2]
            )
        else:
            selected_polygons_utm = merged_utm
        
        # Calculate center and radius in UTM
        center_utm = selected_polygons_utm.centroid
        radius_meters = np.sqrt(selected_polygons_utm.area / np.pi) * circle_scale
        
        # Convert center back to WGS84
        center_wgs84 = gpd.GeoDataFrame(geometry=[center_utm], crs=utm_crs).to_crs(epsg=4326).geometry[0]
        radius_degrees = radius_meters / (111319.5 * np.cos(np.radians(center_wgs84.y)))
        
        return {
            'center': (center_wgs84.y, center_wgs84.x),
            'radius': radius_degrees
        }
    except Exception as e:
        print(f"Error reading shapefile: {e}")
        # Use default values
        return {
            'center': (CITY_CENTER_LAT, CITY_CENTER_LON),
            'radius': 0.45  # Default radius in degrees
        }

def create_extended_wedge(center: Tuple[float, float], inner_radius: float, outer_radius: float, 
                         angle_start: float, angle_end: float, num_points: int = 100) -> Polygon:
    """
    Create a wedge polygon in projected meters.

    The angles are bearings in degrees clockwise from north, matching the wind
    direction convention used in the configuration. The center coordinate must
    be supplied as projected ``(x, y)`` coordinates, and radii are in meters.
    """
    inner_angles = np.radians(np.linspace(angle_start, angle_end, num_points))
    outer_angles = np.radians(np.linspace(angle_end, angle_start, num_points))
    inner_points = [
        (center[0] + inner_radius * np.sin(a), center[1] + inner_radius * np.cos(a))
        for a in inner_angles
    ]
    outer_points = [
        (center[0] + outer_radius * np.sin(a), center[1] + outer_radius * np.cos(a))
        for a in outer_angles
    ]
    return Polygon(inner_points + outer_points + [inner_points[0]])


def _utm_crs_for_lonlat(lon: float, lat: float) -> str:
    utm_zone = int((lon + 180) / 6) + 1
    epsg = 32600 + utm_zone if lat >= 0 else 32700 + utm_zone
    return f"EPSG:{epsg}"


def _project_geometry(geometry, source_crs: str, target_crs: str):
    return gpd.GeoSeries([geometry], crs=source_crs).to_crs(target_crs).iloc[0]

def create_regions(shapefile_path: str = SHAPEFILE_PATH) -> Dict[str, Polygon]:
    """
    Create the 5 regions using exact pycellstats approach.
    """
    # Get urban region parameters
    urban_region = define_urban_region(shapefile_path)
    city_center = urban_region['center']
    urban_radius = urban_region['radius']
    
    # Convert city center to (lon, lat) for geometric operations
    city_center_lonlat = (city_center[1], city_center[0])
    utm_crs = _utm_crs_for_lonlat(city_center_lonlat[0], city_center_lonlat[1])
    center_utm = _project_geometry(Point(city_center_lonlat), "EPSG:4326", utm_crs)
    center_xy = (center_utm.x, center_utm.y)
    
    # Define the bounding box for clipping
    minx, maxx, miny, maxy = BOUNDING_BOX_LIMITS
    bounding_box = box(minx, miny, maxx, maxy)
    bounding_box_utm = _project_geometry(bounding_box, "EPSG:4326", utm_crs)
    
    # Create the urban circle as a polygon. The urban radius is returned in
    # degrees by define_urban_region(), so keep this geometry in WGS84.
    urban_circle_poly = Point(city_center_lonlat).buffer(urban_radius)
    
    # Create the extended wedges using configured kilometer distances.
    inner_radius_m = DOWNWIND_START * 1000.0
    outer_radius_m = DOWNWIND_END * 1000.0
    wind_angle = TEMPORAL_WIND
    
    downwind_wedge_utm = create_extended_wedge(center_xy, inner_radius_m, outer_radius_m, 
                                             wind_angle-(SECTOR_ANGLE/2), wind_angle+(SECTOR_ANGLE/2))
    upwind_wedge_utm = create_extended_wedge(center_xy, inner_radius_m, outer_radius_m, 
                                           wind_angle+180-(SECTOR_ANGLE/2), wind_angle+180+(SECTOR_ANGLE/2))
    right_wedge_utm = create_extended_wedge(center_xy, inner_radius_m, outer_radius_m, 
                                          wind_angle-180+(SECTOR_ANGLE/2), wind_angle-(SECTOR_ANGLE/2))
    left_wedge_utm = create_extended_wedge(center_xy, inner_radius_m, outer_radius_m, 
                                         wind_angle+(SECTOR_ANGLE/2), wind_angle+180-(SECTOR_ANGLE/2))
    
    # Clip in projected coordinates, then convert back to WGS84 for point tests.
    downwind_wedge_poly = _project_geometry(downwind_wedge_utm.intersection(bounding_box_utm), utm_crs, "EPSG:4326")
    upwind_wedge_poly = _project_geometry(upwind_wedge_utm.intersection(bounding_box_utm), utm_crs, "EPSG:4326")
    right_wedge_poly = _project_geometry(right_wedge_utm.intersection(bounding_box_utm), utm_crs, "EPSG:4326")
    left_wedge_poly = _project_geometry(left_wedge_utm.intersection(bounding_box_utm), utm_crs, "EPSG:4326")
    
    # Clip the regions to the bounding box
    regions = {
        'urban': urban_circle_poly.intersection(bounding_box),
        'downwind': downwind_wedge_poly.intersection(bounding_box),
        'upwind': upwind_wedge_poly.intersection(bounding_box),
        'right': right_wedge_poly.intersection(bounding_box),
        'left': left_wedge_poly.intersection(bounding_box)
    }
    
    return regions

def classify_row_by_region(row_data: dict, regions: Dict[str, Polygon]) -> Optional[str]:
    """
    Classify a single storm cell row by region based on coordinates.
    """
    # Extract coordinates from row data
    lat = None
    lon = None
    
    # Try different coordinate field names
    if 'latitude' in row_data and 'longitude' in row_data:
        lat = float(row_data['latitude'])
        lon = float(row_data['longitude'])
    elif 'lat' in row_data and 'lon' in row_data:
        lat = float(row_data['lat'])
        lon = float(row_data['lon'])
    
    if lat is None or lon is None:
        return None
    
    # Create point and check which region it belongs to
    point = Point(lon, lat)
    
    # Check regions in order (urban first, then others)
    for region_name, region_polygon in regions.items():
        if region_polygon.contains(point):
            return region_name
    
    return None

def check_regional_output_exists(file_path: Path, output_base: str, regions: Dict[str, Polygon]) -> bool:
    """
    Check if regional output files already exist for this input file.
    
    Args:
        file_path: Path to input .npy file
        output_base: Base output directory
        regions: Dictionary of region polygons
        
    Returns:
        bool: True if all regional outputs exist, False otherwise
    """
    # Skip metadata files
    if file_path.name.startswith('._') or file_path.name.startswith('.'):
        return True  # Skip processing metadata files
    
    # Extract year/month from file path
    parts = file_path.parts
    arrays_idx = None
    for i, part in enumerate(parts):
        if 'Arrays' in part:
            arrays_idx = i
            break
    
    if arrays_idx is None or arrays_idx + 2 >= len(parts):
        return False
    
    year = parts[arrays_idx + 1]
    month = parts[arrays_idx + 2]
    
    # Check if output files exist for each region
    for region in regions.keys():
        output_dir = Path(output_base) / 'Arrays' / region / year / month
        output_file = output_dir / file_path.name
        if not output_file.exists():
            return False
    
    return True

def process_npy_file(file_path: Path, regions: Dict[str, Polygon], output_base: str, force: bool = False) -> Dict[str, int]:
    """
    Process a single .npy file and sort rows by region.
    """
    region_counts = {region: 0 for region in regions.keys()}
    
    # SAFEGUARD: Check if output already exists (unless forced)
    if not force and SKIP_EXISTING_FILTERING:
        if check_regional_output_exists(file_path, output_base, regions):
            # Count existing files to return accurate counts
            parts = file_path.parts
            arrays_idx = None
            for i, part in enumerate(parts):
                if 'Arrays' in part:
                    arrays_idx = i
                    break
            
            if arrays_idx is not None and arrays_idx + 2 < len(parts):
                year = parts[arrays_idx + 1]
                month = parts[arrays_idx + 2]
                
                # Count rows in existing regional files
                for region in regions.keys():
                    output_dir = Path(output_base) / 'Arrays' / region / year / month
                    output_file = output_dir / file_path.name
                    if output_file.exists():
                        try:
                            existing_data = np.load(output_file, allow_pickle=True)
                            region_counts[region] = len(existing_data) if existing_data.dtype == 'object' else 0
                        except:
                            region_counts[region] = 0
            
            return region_counts
    
    try:
        # Load the .npy file
        data = np.load(file_path, allow_pickle=True)
        
        # Skip if not object array or empty
        if data.dtype != 'object' or len(data) == 0:
            return region_counts
        
        # Initialize regional data containers
        regional_data = {region: [] for region in regions.keys()}
        
        # Process each row (storm cell)
        for row in data:
            if not isinstance(row, dict):
                continue
                
            region = classify_row_by_region(row, regions)
            if region:
                regional_data[region].append(row)
                region_counts[region] += 1
        
        # Save regional data files
        parts = file_path.parts
        arrays_idx = None
        for i, part in enumerate(parts):
            if 'Arrays' in part:
                arrays_idx = i
                break
        
        if arrays_idx is not None and arrays_idx + 2 < len(parts):
            year = parts[arrays_idx + 1]
            month = parts[arrays_idx + 2]
            
            # Save each region's data
            for region, rows in regional_data.items():
                if len(rows) > 0:
                    # Create output directory
                    output_dir = Path(output_base) / 'Arrays' / region / year / month
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save filtered data
                    output_file = output_dir / file_path.name
                    regional_array = np.array(rows, dtype=object)
                    np.save(output_file, regional_array)
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return region_counts

def process_file_wrapper(args):
    """
    Wrapper function for parallel processing of .npy files.
    """
    file_path, regions, output_base, force = args
    return process_npy_file(file_path, regions, output_base, force)

def filter_month_directory(month_dir: Path, regions: Dict[str, Polygon], output_base: str, 
                          use_parallel: bool = True, max_workers: int = None, force: bool = False) -> Dict[str, int]:
    """
    Process all .npy files in a month directory with optional parallel processing.
    
    Args:
        month_dir: Path to month directory containing .npy files
        regions: Dictionary of region polygons
        output_base: Base output directory
        use_parallel: Whether to use parallel processing (default: True)
        max_workers: Maximum number of parallel workers (default: CPU count)
    """
    total_counts = {region: 0 for region in regions.keys()}
    
    # Process all .npy files (exclude macOS metadata files)
    npy_files = [f for f in month_dir.glob('*.npy') if not f.name.startswith('._') and not f.name.startswith('.')]
    
    if not npy_files:
        return total_counts
    
    
    if use_parallel and len(npy_files) > 1:
        # Use parallel processing for multiple files
        if max_workers is None:
            max_workers = min(mp.cpu_count(), len(npy_files))
        
        # Prepare arguments for parallel processing
        file_args = [(file_path, regions, output_base, force) for file_path in npy_files]
        
        print(f"Using {max_workers} parallel workers...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {executor.submit(process_file_wrapper, args): args[0] 
                            for args in file_args}
            
            # Process completed tasks
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    file_counts = future.result()
                    # Add to totals
                    for region, count in file_counts.items():
                        total_counts[region] += count
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    else:
        # Sequential processing for single file or when parallel is disabled
        for file_path in npy_files:
            file_counts = process_npy_file(file_path, regions, output_base, force)
            
            # Add to totals
            for region, count in file_counts.items():
                total_counts[region] += count
    
    return total_counts

def filter_nexrad_by_region(source_dir: str, target_base: str, 
                          date_filter: Optional[str] = None, 
                          use_parallel: bool = True, max_workers: int = None, force: bool = False) -> Dict[str, int]:
    """
    Main function to filter NEXRAD data by regions with parallel processing.
    
    Args:
        source_dir: Source directory containing Arrays/YEAR/MONTH structure
        target_base: Target base directory for regional output
        date_filter: Optional date filter in YYYY-MM or YYYY-MM-DD format
        use_parallel: Whether to use parallel processing (default: True)
        max_workers: Maximum number of parallel workers (default: CPU count)
    """
    print()
    regions = create_regions()
    
    source_path = Path(source_dir)
    total_counts = {region: 0 for region in regions.keys()}
    
    if not source_path.exists():
        print(f"Source directory does not exist: {source_path}")
        return total_counts
    
    # Scan directory structure
    for year_dir in source_path.iterdir():
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
            
        year = int(year_dir.name)
        
        # Apply date filter if specified, otherwise use config year range
        if date_filter:
            date_parts = date_filter.split('-')
            if len(date_parts) >= 2:
                filter_year = int(date_parts[0])
                filter_month = int(date_parts[1])
                filter_day = int(date_parts[2]) if len(date_parts) == 3 else None
                if year != filter_year:
                    continue
        else:
            # If no date filter, only process years in config range
            if year < YEAR_START or year > YEAR_END:
                continue
        
        for month_dir in year_dir.iterdir():
            if not month_dir.is_dir():
                continue
                
            # Apply month/day filter if specified, otherwise use config valid months
            if date_filter and len(date_parts) >= 2:
                if filter_day is not None:
                    # Match specific day directory (e.g., Jul16)
                    target_dir_name = f"{calendar.month_abbr[filter_month]}{filter_day:02d}"
                    if month_dir.name != target_dir_name:
                        continue
                else:
                    # Match any day in the month (e.g., Jul*)
                    month_name = calendar.month_abbr[filter_month]
                    if not month_dir.name.startswith(month_name):
                        continue
            elif not date_filter:
                # If no date filter, only process valid month/day directories
                if not any(month_dir.name.startswith(month) for month in VALID_MONTHS):
                    continue
            
            print(f"Processing {month_dir}")
            
            # Process month data with parallel processing
            month_counts = filter_month_directory(month_dir, regions, target_base, 
                                                use_parallel, max_workers, force)
            
            # Add to totals
            for region, count in month_counts.items():
                total_counts[region] += count
    
    return total_counts

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Filter NEXRAD storm cell rows by regions (pycellstats approach)")
    
    parser.add_argument('--source', '-s', type=str, 
                       default=ARRAY_OUTPUT_DIR,
                       help='Source directory path')
    parser.add_argument('--target', '-t', type=str,
                       default=BASE_DATA_DIR,
                       help='Target base directory path')
    parser.add_argument('--date', '-d', type=str,
                       help='Date filter in YYYY-MM or YYYY-MM-DD format')
    parser.add_argument('--parallel', '-p', action='store_true', default=REGIONAL_USE_PARALLEL,
                       help=f'Use parallel processing (default: {REGIONAL_USE_PARALLEL})')
    parser.add_argument('--no-parallel', action='store_false', dest='parallel',
                       help='Disable parallel processing')
    parser.add_argument('--workers', '-w', type=int, default=REGIONAL_MAX_WORKERS,
                       help=f'Number of parallel workers (default: {REGIONAL_MAX_WORKERS or "CPU count"})')
    parser.add_argument('--force', '-f', action='store_true',
                       help='Force processing even if regional output already exists')
    
    args = parser.parse_args()
    
    print(f"Source: {args.source}")
    print(f"Target: {args.target}")
    print(f"Date filter: {args.date or 'None'}")
    print(f"Parallel processing: {args.parallel}")
    if args.parallel:
        workers = args.workers or mp.cpu_count()
        print(f"Max workers: {workers}")
    print(f"Safeguards enabled: {SKIP_EXISTING_FILTERING}")
    print(f"Force processing: {args.force}")
    
    # Run filtering
    region_counts = filter_nexrad_by_region(args.source, args.target, args.date, 
                                          args.parallel, args.workers, args.force)
    
    # Print results
    print("\nResults:")
    print("-" * 20)
    total_rows = sum(region_counts.values())
    
    for region, count in region_counts.items():
        percentage = (count / total_rows * 100) if total_rows > 0 else 0
        print(f"{region:>10}: {count:>6} rows ({percentage:5.1f}%)")
    
    print(f"{'Total':>10}: {total_rows:>6} rows")
    print(f"\nFiltering complete - regional data saved to {args.target}/Arrays/[region]/")

if __name__ == "__main__":
    main()