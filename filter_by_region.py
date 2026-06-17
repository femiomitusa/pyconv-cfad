#!/usr/bin/env python3

from __future__ import annotations

import argparse
import calendar
import json
import math
import multiprocessing as mp
import shutil
import sys
import warnings
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# macOS writes ._<name> resource-fork files into Zarr stores; Zarr v3 warns about them harmlessly.
warnings.filterwarnings(
    'ignore',
    message=r'Object at \._.*is not recognized as a component of a Zarr hierarchy',
    category=UserWarning,
    module='zarr',
)

import numpy as np
import xarray as xr
from shapely.geometry import Point, shape

sys.path.append(str(Path(__file__).parent))

from config import (
    ARRAY_OUTPUT_DIR,
    BASE_DATA_DIR,
    OUTPUT_DIR,
    REGIONAL_CONFIG,
    REGIONAL_MAX_WORKERS,
    REGIONAL_USE_PARALLEL,
    SKIP_EXISTING_FILTERING,
    TARGET_DAY,
    TARGET_MODE,
    TARGET_MONTH,
    TARGET_YEAR,
    VALID_MONTHS,
    YEAR_END,
    YEAR_START,
)


CITY_CENTER_LAT = REGIONAL_CONFIG['city_center_lat']
CITY_CENTER_LON = REGIONAL_CONFIG['city_center_lon']
URBAN_GEOJSON = Path(OUTPUT_DIR) / 'regions' / 'urban.geojson'
HALF_ANGLE = 45.0

REGIONS = ('urban', 'downwind', 'right', 'upwind', 'left')
OUTPUT_GROUPS = ('all', *REGIONS, 'unclassified')
REGION_TO_CODE = {
    'unclassified': -1,
    'urban': 0,
    'downwind': 1,
    'right': 2,
    'upwind': 3,
    'left': 4,
}
CODE_TO_REGION = {v: k for k, v in REGION_TO_CODE.items()}
UNCLASSIFIED_REASON_TO_CODE = {
    'classified': 0,
    'missing_coordinate': 1,
    'missing_motion_bearing': 2,
}


def load_urban_polygon():
    """Load the Stage 0 urban boundary from a GeoJSON geometry or Feature."""
    if not URBAN_GEOJSON.exists():
        sys.exit(
            f"Urban polygon not found: {URBAN_GEOJSON}\n"
            "Run setup_regions.py first."
        )

    with open(URBAN_GEOJSON) as f:
        geojson = json.load(f)

    geojson_type = geojson.get('type')
    if geojson_type == 'Feature':
        geom = geojson['geometry']
    elif geojson_type in {'Polygon', 'MultiPolygon'}:
        geom = geojson
    else:
        raise ValueError(
            f"{URBAN_GEOJSON} must be a GeoJSON Feature, Polygon, or MultiPolygon; "
            f"found {geojson_type!r}."
        )

    polygon = shape(geom)
    if polygon.is_empty:
        raise ValueError(f"Urban polygon is empty: {URBAN_GEOJSON}")
    if not polygon.is_valid:
        raise ValueError(f"Urban polygon is invalid: {URBAN_GEOJSON}")
    return polygon


def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Geodetic bearing in degrees clockwise from north, from point 1 to point 2."""
    dlon = math.radians(lon2 - lon1)
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    x = math.sin(dlon) * math.cos(lat2r)
    y = (
        math.cos(lat1r) * math.sin(lat2r)
        - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    )
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _is_missing(value: float) -> bool:
    try:
        return not np.isfinite(float(value))
    except (TypeError, ValueError):
        return True


def _wind_from_direction(motion_to_bearing_deg: float) -> float:
    """Convert a tracked-cell motion-to bearing into meteorological from-direction."""
    return (float(motion_to_bearing_deg) + 180.0) % 360.0


def _downwind_bearing(wind_from_direction_deg: float) -> float:
    """Return the direction toward which the meteorological wind is blowing."""
    return (float(wind_from_direction_deg) + 180.0) % 360.0


def classify_cell_details(row_data: dict, urban_polygon, half_angle: float = HALF_ANGLE) -> dict:
    """Return region plus diagnostic bearing metadata for one observation."""
    lat = row_data.get('centroid_lat')
    lon = row_data.get('centroid_lon')
    motion_to = row_data.get('motion_bearing_deg')

    if _is_missing(lat) or _is_missing(lon):
        return {
            'region': 'unclassified',
            'region_code': REGION_TO_CODE['unclassified'],
            'city_bearing_deg': np.nan,
            'downwind_delta_deg': np.nan,
            'wind_from_direction_deg': np.nan,
            'downwind_bearing_deg': np.nan,
            'unclassified_reason_code': UNCLASSIFIED_REASON_TO_CODE['missing_coordinate'],
        }

    lat = float(lat)
    lon = float(lon)
    city_bearing = _bearing(CITY_CENTER_LAT, CITY_CENTER_LON, lat, lon)
    wind_from = np.nan
    downwind_bearing = np.nan
    downwind_delta = np.nan
    if not _is_missing(motion_to):
        wind_from = _wind_from_direction(motion_to)
        downwind_bearing = _downwind_bearing(wind_from)
        downwind_delta = (city_bearing - downwind_bearing) % 360

    if urban_polygon.covers(Point(lon, lat)):
        return {
            'region': 'urban',
            'region_code': REGION_TO_CODE['urban'],
            'city_bearing_deg': city_bearing,
            'downwind_delta_deg': downwind_delta,
            'wind_from_direction_deg': wind_from,
            'downwind_bearing_deg': downwind_bearing,
            'unclassified_reason_code': UNCLASSIFIED_REASON_TO_CODE['classified'],
        }

    if _is_missing(motion_to):
        return {
            'region': 'unclassified',
            'region_code': REGION_TO_CODE['unclassified'],
            'city_bearing_deg': city_bearing,
            'downwind_delta_deg': np.nan,
            'wind_from_direction_deg': np.nan,
            'downwind_bearing_deg': np.nan,
            'unclassified_reason_code': UNCLASSIFIED_REASON_TO_CODE['missing_motion_bearing'],
        }

    if downwind_delta <= half_angle or downwind_delta >= 360 - half_angle:
        region = 'downwind'
    elif downwind_delta < 180 - half_angle:
        region = 'right'
    elif downwind_delta <= 180 + half_angle:
        region = 'upwind'
    else:
        region = 'left'

    return {
        'region': region,
        'region_code': REGION_TO_CODE[region],
        'city_bearing_deg': city_bearing,
        'downwind_delta_deg': downwind_delta,
        'wind_from_direction_deg': wind_from,
        'downwind_bearing_deg': downwind_bearing,
        'unclassified_reason_code': UNCLASSIFIED_REASON_TO_CODE['classified'],
    }


def _require_tracking_vars(ds, tracking_path: Path) -> None:
    required = ('centroid_lat', 'centroid_lon')
    missing = [name for name in required if name not in ds]
    if missing:
        raise ValueError(f"{tracking_path.name} missing required variable(s): {', '.join(missing)}")


def classify_dataset(ds, urban_polygon) -> dict:
    """Classify all obs in an open xarray Dataset and return vector metadata."""
    n_obs = ds.sizes.get('obs', 0)
    regions = np.full(n_obs, 'unclassified', dtype=object)
    region_codes = np.full(n_obs, REGION_TO_CODE['unclassified'], dtype=np.int16)
    city_bearings = np.full(n_obs, np.nan, dtype=np.float32)
    downwind_deltas = np.full(n_obs, np.nan, dtype=np.float32)
    wind_from_directions = np.full(n_obs, np.nan, dtype=np.float32)
    downwind_bearings = np.full(n_obs, np.nan, dtype=np.float32)
    reason_codes = np.full(
        n_obs,
        UNCLASSIFIED_REASON_TO_CODE['missing_motion_bearing'],
        dtype=np.int16,
    )

    if n_obs == 0:
        return {
            'regions': regions,
            'region_codes': region_codes,
            'city_bearings': city_bearings,
            'downwind_deltas': downwind_deltas,
            'wind_from_directions': wind_from_directions,
            'downwind_bearings': downwind_bearings,
            'reason_codes': reason_codes,
        }

    lats = ds['centroid_lat'].values.astype(float)
    lons = ds['centroid_lon'].values.astype(float)
    bearings = (
        ds['motion_bearing_deg'].values.astype(float)
        if 'motion_bearing_deg' in ds
        else np.full(n_obs, np.nan, dtype=float)
    )

    for i in range(n_obs):
        details = classify_cell_details(
            {
                'centroid_lat': lats[i],
                'centroid_lon': lons[i],
                'motion_bearing_deg': bearings[i],
            },
            urban_polygon,
        )
        regions[i] = details['region']
        region_codes[i] = details['region_code']
        city_bearings[i] = details['city_bearing_deg']
        downwind_deltas[i] = details['downwind_delta_deg']
        wind_from_directions[i] = details['wind_from_direction_deg']
        downwind_bearings[i] = details['downwind_bearing_deg']
        reason_codes[i] = details['unclassified_reason_code']

    return {
        'regions': regions,
        'region_codes': region_codes,
        'city_bearings': city_bearings,
        'downwind_deltas': downwind_deltas,
        'wind_from_directions': wind_from_directions,
        'downwind_bearings': downwind_bearings,
        'reason_codes': reason_codes,
    }


def annotate_dataset(ds, classification: dict):
    """Return the source dataset with obs-level Stage 3 metadata attached."""
    annotated = ds.copy()
    annotated['region_code'] = ('obs', classification['region_codes'])
    annotated['city_bearing_deg'] = ('obs', classification['city_bearings'])
    annotated['wind_from_direction_deg'] = ('obs', classification['wind_from_directions'])
    annotated['downwind_bearing_deg'] = ('obs', classification['downwind_bearings'])
    annotated['downwind_delta_deg'] = ('obs', classification['downwind_deltas'])
    annotated['unclassified_reason_code'] = ('obs', classification['reason_codes'])

    annotated['region_code'].attrs.update({
        'long_name': 'Stage 3 regional classification code',
        'flag_values': np.array(sorted(CODE_TO_REGION), dtype=np.int16),
        'flag_meanings': ' '.join(CODE_TO_REGION[k] for k in sorted(CODE_TO_REGION)),
    })
    annotated['city_bearing_deg'].attrs.update({
        'units': 'degree',
        'long_name': 'bearing from Houston reference point to cell centroid',
    })
    annotated['wind_from_direction_deg'].attrs.update({
        'units': 'degree',
        'long_name': 'meteorological wind direction associated with the tracked storm-cell motion',
        'comment': 'Meteorological from-direction convention: 0 from north, 90 from east, clockwise positive.',
    })
    annotated['downwind_bearing_deg'].attrs.update({
        'units': 'degree',
        'long_name': 'direction toward which the meteorological wind is blowing',
        'comment': 'Computed as (wind_from_direction_deg + 180) modulo 360.',
    })
    annotated['downwind_delta_deg'].attrs.update({
        'units': 'degree',
        'long_name': 'city_bearing_deg minus downwind_bearing_deg modulo 360',
    })
    annotated['unclassified_reason_code'].attrs.update({
        'flag_values': np.array(list(UNCLASSIFIED_REASON_TO_CODE.values()), dtype=np.int16),
        'flag_meanings': ' '.join(UNCLASSIFIED_REASON_TO_CODE),
    })
    annotated.attrs.update({
        'stage3_region_codes': json.dumps(CODE_TO_REGION, sort_keys=True),
        'stage3_unclassified_reason_codes': json.dumps(
            {v: k for k, v in UNCLASSIFIED_REASON_TO_CODE.items()},
            sort_keys=True,
        ),
        'stage3_city_center_lat': CITY_CENTER_LAT,
        'stage3_city_center_lon': CITY_CENTER_LON,
        'stage3_sector_half_angle_deg': HALF_ANGLE,
        'stage3_urban_geojson': str(URBAN_GEOJSON),
        'stage3_direction_convention': (
            'classification uses meteorological wind FROM direction derived from tracked storm-cell motion; '
            'downwind is centered on (wind_from_direction_deg + 180) modulo 360 and upwind is centered on wind_from_direction_deg'
        ),
    })
    return annotated


def _output_path(tracking_path: Path, output_base: str, group: str) -> Path:
    year = tracking_path.parent.parent.name
    day_dir = tracking_path.parent.name
    stem = tracking_path.stem.replace('_tracking', '')
    return Path(output_base) / 'Arrays_Regional' / group / year / day_dir / f'{stem}_regional.zarr'


def _regional_outputs_exist(tracking_path: Path, output_base: str) -> bool:
    """True if every Stage 3 output group already exists for this tracking Zarr store."""
    return all(_output_path(tracking_path, output_base, group).exists() for group in OUTPUT_GROUPS)


def _existing_output_counts(tracking_path: Path, output_base: str) -> dict[str, int]:
    """Read counts from existing Stage 3 Zarr stores when processing is skipped."""
    counts: dict[str, int] = {r: 0 for r in REGIONS}
    counts['unclassified'] = 0

    for group in (*REGIONS, 'unclassified'):
        out_path = _output_path(tracking_path, output_base, group)
        try:
            with xr.open_zarr(out_path, consolidated=False) as ds:
                counts[group] = int(ds.attrs.get('region_obs_count', ds.sizes.get('obs', 0)))
        except Exception:
            counts[group] = 0

    return counts


def _write_dataset(ds, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        shutil.rmtree(out_path)
    ds.to_zarr(out_path, mode='w', consolidated=False)


def _subset_region_dataset(annotated: xr.Dataset, obs_idx: np.ndarray) -> xr.Dataset:
    """Return a region subset and keep ragged samples consistent with selected obs."""
    region_ds = annotated.isel(obs=obs_idx)
    if 'sample_obs' not in annotated or 'sample' not in annotated.sizes:
        return region_ds

    sample_obs = annotated['sample_obs'].values.astype(np.int64)
    keep_sample = np.isin(sample_obs, obs_idx)
    sample_idx = np.where(keep_sample)[0]
    region_ds = region_ds.isel(sample=sample_idx)

    # Remap original obs indices to the compact obs dimension in this regional view.
    remap = {int(old): int(new) for new, old in enumerate(obs_idx)}
    remapped = np.array([remap[int(v)] for v in sample_obs[sample_idx]], dtype=np.int32)
    region_ds['sample_obs'] = ('sample', remapped)
    return region_ds


def _write_region_view(annotated, region_codes: np.ndarray, tracking_path: Path, output_base: str, group: str) -> int:
    if group == 'unclassified':
        idx = np.where(region_codes == REGION_TO_CODE['unclassified'])[0]
    else:
        idx = np.where(region_codes == REGION_TO_CODE[group])[0]

    if len(idx) == 0:
        region_ds = xr.Dataset(attrs={
            'region': group,
            'region_obs_count': 0,
            'source_file': tracking_path.name,
            'empty_region_file': 'true',
        })
    else:
        region_ds = _subset_region_dataset(annotated, idx)
        region_ds.attrs.update({
            'region': group,
            'region_obs_count': int(len(idx)),
        })
    _write_dataset(region_ds, _output_path(tracking_path, output_base, group))
    return int(len(idx))


def process_tracking_zarr(
    tracking_path: Path,
    urban_polygon,
    output_base: str,
    force: bool = False,
) -> dict[str, int]:
    """Classify one Stage 2 tracking Zarr store and write annotated + regional Zarr outputs."""
    counts: dict[str, int] = {r: 0 for r in REGIONS}
    counts['unclassified'] = 0

    if not force and SKIP_EXISTING_FILTERING and _regional_outputs_exist(tracking_path, output_base):
        return _existing_output_counts(tracking_path, output_base)

    try:
        with xr.open_zarr(tracking_path, consolidated=False) as ds:
            _require_tracking_vars(ds, tracking_path)
            classification = classify_dataset(ds, urban_polygon)
            annotated = annotate_dataset(ds.load(), classification)
    except Exception as exc:
        print(f"  Cannot process {tracking_path.name}: {exc}")
        return counts

    region_codes = classification['region_codes']

    all_path = _output_path(tracking_path, output_base, 'all')
    annotated.attrs.update({
        'region': 'all',
        'region_obs_count': int(annotated.sizes.get('obs', 0)),
    })
    _write_dataset(annotated, all_path)

    for region in REGIONS:
        counts[region] = _write_region_view(annotated, region_codes, tracking_path, output_base, region)
    counts['unclassified'] = _write_region_view(
        annotated, region_codes, tracking_path, output_base, 'unclassified'
    )

    return counts


def _process_zarr_wrapper(args: tuple) -> dict[str, int]:
    tracking_path, urban_polygon, output_base, force = args
    return process_tracking_zarr(tracking_path, urban_polygon, output_base, force)


def filter_day_directory(
    day_dir: Path,
    urban_polygon,
    output_base: str,
    use_parallel: bool = True,
    max_workers: int | None = None,
    force: bool = False,
) -> dict[str, int]:
    """Process all *_tracking.zarr stores in one day directory."""
    tracking_stores = sorted(day_dir.glob('*_tracking.zarr'))
    total: dict[str, int] = {r: 0 for r in REGIONS}
    total['unclassified'] = 0
    if not tracking_stores:
        return total

    if use_parallel and len(tracking_stores) > 1:
        n_workers = max_workers or min(mp.cpu_count(), len(tracking_stores))
        file_args = [(f, urban_polygon, output_base, force) for f in tracking_stores]
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_process_zarr_wrapper, a): a[0] for a in file_args}
            for fut in as_completed(futures):
                try:
                    for r, c in fut.result().items():
                        total[r] += c
                except Exception as exc:
                    print(f"  Error: {futures[fut].name}: {exc}")
    else:
        for tracking_store in tracking_stores:
            for r, c in process_tracking_zarr(tracking_store, urban_polygon, output_base, force).items():
                total[r] += c

    return total


def _parse_date_filter(date_filter: str | None) -> tuple[int | None, int | None, int | None]:
    if not date_filter:
        return None, None, None

    parts = date_filter.split('-')
    if len(parts) > 3:
        raise ValueError("Date filter must be YYYY, YYYY-MM, or YYYY-MM-DD")

    year = int(parts[0])
    month = int(parts[1]) if len(parts) >= 2 else None
    day = int(parts[2]) if len(parts) == 3 else None
    return year, month, day


def _iter_day_dirs(source_path: Path, date_filter: str | None) -> Iterable[Path]:
    filter_year, filter_month, filter_day = _parse_date_filter(date_filter)

    for year_dir in sorted(source_path.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue

        year = int(year_dir.name)
        if filter_year is not None:
            if year != filter_year:
                continue
        elif year < YEAR_START or year > YEAR_END:
            continue

        for day_dir in sorted(year_dir.iterdir()):
            if not day_dir.is_dir():
                continue

            if filter_month is not None:
                expected_prefix = calendar.month_abbr[filter_month]
                if not day_dir.name.startswith(expected_prefix):
                    continue
                if filter_day is not None and day_dir.name != f'{expected_prefix}{filter_day:02d}':
                    continue
            elif not any(day_dir.name.startswith(m) for m in VALID_MONTHS):
                continue

            yield day_dir


def filter_nexrad_by_region(
    source_dir: str,
    target_base: str,
    date_filter: str | None = None,
    use_parallel: bool = True,
    max_workers: int | None = None,
    force: bool = False,
) -> dict[str, int]:
    """Walk Stage 2 tracking outputs and write Stage 3 regional products."""
    urban_polygon = load_urban_polygon()
    source_path = Path(source_dir)
    total: dict[str, int] = {r: 0 for r in REGIONS}
    total['unclassified'] = 0

    if not source_path.exists():
        print(f"Source directory does not exist: {source_path}")
        return total

    for day_dir in _iter_day_dirs(source_path, date_filter):
        print(f"Processing {day_dir}")
        for r, c in filter_day_directory(
            day_dir,
            urban_polygon,
            target_base,
            use_parallel,
            max_workers,
            force,
        ).items():
            total[r] += c

    return total


def _date_filter_from_config() -> str | None:
    if not TARGET_MODE:
        return None
    return f'{TARGET_YEAR}-{TARGET_MONTH:02d}-{TARGET_DAY:02d}'


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Classify Stage 2 tracking Zarr stores by region.')
    parser.add_argument('--date', help='Optional date filter: YYYY, YYYY-MM, or YYYY-MM-DD')
    parser.add_argument('--force', action='store_true', help='Overwrite existing regional Zarr outputs')
    parser.add_argument('--serial', action='store_true', help='Disable file-level parallelism')
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source = ARRAY_OUTPUT_DIR
    target = BASE_DATA_DIR
    date_filter = args.date or _date_filter_from_config()
    use_parallel = REGIONAL_USE_PARALLEL and not args.serial
    workers = REGIONAL_MAX_WORKERS
    force = args.force

    print(f"Source  : {source}")
    print(f"Target  : {target}")
    print(f"Date    : {date_filter or 'all'}")
    print(f"Parallel: {use_parallel}  Workers: {workers or 'auto'}")
    print(f"Force   : {force}  Skip-existing: {SKIP_EXISTING_FILTERING}")
    print()

    counts = filter_nexrad_by_region(
        source,
        target,
        date_filter,
        use_parallel,
        workers,
        force,
    )

    classified_total = sum(counts[r] for r in REGIONS)
    total = classified_total + counts['unclassified']

    print("\nResults")
    print('-' * 29)
    for r in REGIONS:
        pct = counts[r] / total * 100 if total else 0
        print(f"  {r:>12}: {counts[r]:>6} obs  ({pct:5.1f}%)")
    pct = counts['unclassified'] / total * 100 if total else 0
    print(f"  {'unclassified':>12}: {counts['unclassified']:>6} obs  ({pct:5.1f}%)")
    print(f"  {'total':>12}: {total:>6} obs")
    print(f"\nOutput -> {target}/Arrays_Regional/")


if __name__ == '__main__':
    main()