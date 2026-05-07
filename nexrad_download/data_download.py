#!/usr/bin/env python3
"""Download NEXRAD Level 2 radar data from AWS S3 with parallel threads."""

import sys
import boto3
import calendar
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from botocore import UNSIGNED
from botocore.client import Config
from tqdm import tqdm
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config import (
        TARGET_MODE, BASE_DATA_DIR, TARGET_YEAR, TARGET_MONTH, TARGET_DAY,
        YEAR_START, YEAR_END, VALID_MONTHS, SKIP_EXISTING_DOWNLOADS,
    )
except ImportError as e:
    print(f"Error: Could not import config.py: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logging.getLogger('urllib3.connectionpool').setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

S3_BUCKET = 'unidata-nexrad-level2'
RADAR_STATION = 'KHGX'
MAX_WORKERS = 40

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED, max_pool_connections=MAX_WORKERS))


def output_dir(year: int, month: int, day: int) -> Path:
    """Return the local directory for a given date: BASE_DATA_DIR/YEAR/MonthDD/"""
    return Path(BASE_DATA_DIR) / str(year) / f"{calendar.month_abbr[month]}{day:02d}"


def list_files(year: int, month: int, day: int) -> list[str]:
    """Return S3 keys of all V06 (non-MDM) NEXRAD files for the given date."""
    prefix = f"{year}/{month:02d}/{day:02d}/{RADAR_STATION}/"
    try:
        pages = s3.get_paginator('list_objects_v2').paginate(Bucket=S3_BUCKET, Prefix=prefix)
        return [
            obj['Key']
            for page in pages
            for obj in page.get('Contents', [])
            if obj['Key'].endswith('_V06') and '_MDM' not in obj['Key']
        ]
    except Exception as e:
        logger.error(f"Could not list S3 files for {year}-{month:02d}-{day:02d}: {e}")
        return []


def download_file(s3_key: str, dest_dir: Path) -> tuple[str, bool, str]:
    """Download one file from S3. Returns (filename, success, status)."""
    filename = Path(s3_key).name
    dest = dest_dir / filename

    if SKIP_EXISTING_DOWNLOADS and dest.exists():
        return filename, True, "exists"

    try:
        s3.download_file(S3_BUCKET, s3_key, str(dest))
        return filename, True, "downloaded"
    except Exception as e:
        return filename, False, str(e)


def download_day(year: int, month: int, day: int) -> dict:
    """Download all NEXRAD files for one day and return a stats dictionary."""
    dest_dir = output_dir(year, month, day)
    dest_dir.mkdir(parents=True, exist_ok=True)

    date_str = f"{year}-{month:02d}-{day:02d}"
    keys = list_files(year, month, day)
    if not keys:
        return {'date': date_str, 'total': 0, 'downloaded': 0, 'skipped': 0, 'failed': 0}

    stats = {'downloaded': 0, 'skipped': 0, 'failed': 0}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_file, key, dest_dir): key for key in keys}
        with tqdm(total=len(keys), desc=f"  {date_str}", unit="file", leave=False, ncols=80) as bar:
            for future in as_completed(futures):
                filename, success, status = future.result()
                if not success:
                    stats['failed'] += 1
                    logger.error(f"  Failed: {filename} — {status}")
                elif status == "exists":
                    stats['skipped'] += 1
                else:
                    stats['downloaded'] += 1
                bar.update(1)

    if stats['failed']:
        logger.warning(f"  {date_str}: {stats['failed']} files failed")

    return {'date': date_str, 'total': len(keys), **stats}


def download_month(year: int, month: int) -> list[dict]:
    """Download all days in a given month."""
    _, num_days = calendar.monthrange(year, month)
    return [download_day(year, month, day) for day in range(1, num_days + 1)]


def download_date_range() -> list[dict]:
    """Download all years and months configured in YEAR_START/YEAR_END/VALID_MONTHS."""
    month_map = {calendar.month_abbr[i]: i for i in range(1, 13)}
    months = [month_map[m] for m in VALID_MONTHS]
    years = list(range(YEAR_START, YEAR_END + 1))

    all_stats = []
    total = len(years) * len(months)
    for i, year in enumerate(years):
        for j, month in enumerate(months):
            op = i * len(months) + j + 1
            logger.info(f"[{op}/{total}] {calendar.month_abbr[month]} {year}")
            all_stats.extend(download_month(year, month))
    return all_stats


def print_summary(stats: list[dict] | dict, duration: float) -> None:
    """Print aggregate download statistics."""
    if isinstance(stats, dict):
        stats = [stats]

    total = sum(s['total'] for s in stats)
    downloaded = sum(s['downloaded'] for s in stats)
    skipped = sum(s['skipped'] for s in stats)
    failed = sum(s['failed'] for s in stats)

    print("\n" + "=" * 50)
    print("DOWNLOAD COMPLETE")
    print("=" * 50)
    print(f"Total files found:    {total:>8}")
    print(f"Downloaded:           {downloaded:>8}")
    print(f"Skipped (existing):   {skipped:>8}")
    print(f"Failed:               {failed:>8}")
    print(f"Duration:             {int(duration // 60)}m {int(duration % 60)}s")
    print("=" * 50)


def main() -> int:
    logger.info(f"Output directory:  {BASE_DATA_DIR}")
    logger.info(f"Mode:              {'target date' if TARGET_MODE else 'date range'}")
    logger.info(f"Skip existing:     {SKIP_EXISTING_DOWNLOADS}")

    start = datetime.now()

    try:
        if TARGET_MODE:
            logger.info(f"Target: {TARGET_YEAR}-{TARGET_MONTH:02d}-{TARGET_DAY:02d}\n")
            stats = download_day(TARGET_YEAR, TARGET_MONTH, TARGET_DAY)
        else:
            stats = download_date_range()

        print_summary(stats, (datetime.now() - start).total_seconds())
        return 0

    except KeyboardInterrupt:
        logger.info("\nDownload interrupted by user.")
        return 1

    except Exception as e:
        logger.error(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
