#!/bin/bash

# ==============================================================================
# OPTIMIZED NEXRAD DATA DOWNLOAD SCRIPT
# Algorithmic improvements: Batched S3 operations, eliminated subprocess overhead,
# hash-based file checking, and configurable parallelism
# ==============================================================================

set -euo pipefail

# ------------------------------------------------------------------------------
# Configuration Loading (Optimized: Single Python call)
# ------------------------------------------------------------------------------

load_configuration() {
    local config_path="./config.py"
    if [[ ! -f "$config_path" ]]; then
        config_path="../config.py"
        if [[ ! -f "$config_path" ]]; then
            echo "Error: config.py not found in current directory or parent directory!"
            exit 1
        fi
    fi

    # Single Python call to extract all config at once (eliminates 6+ subprocess calls)
    eval "$(python3 - <<'PYTHON'
import sys
sys.path.insert(0, '.')
from config import (
    TARGET_MODE, BASE_DATA_DIR, TARGET_YEAR, TARGET_MONTH, TARGET_DAY,
    YEAR_START, YEAR_END, VALID_MONTHS
)
import calendar

print(f"TARGET_MODE='{TARGET_MODE}'")
print(f"BASE_DATA_DIR='{BASE_DATA_DIR}'")
print(f"TARGET_YEAR={TARGET_YEAR}")
print(f"TARGET_MONTH={TARGET_MONTH}")
print(f"TARGET_DAY={TARGET_DAY}")
print(f"YEAR_START={YEAR_START}")
print(f"YEAR_END={YEAR_END}")

if TARGET_MODE:
    month_name = calendar.month_abbr[TARGET_MONTH]
    print(f"VALID_MONTHS='{month_name}'")
    print(f"years=({TARGET_YEAR})")
else:
    months_str = ' '.join(VALID_MONTHS)
    print(f"VALID_MONTHS='{months_str}'")
    years_list = ' '.join(str(y) for y in range(YEAR_START, YEAR_END + 1))
    print(f"years=({years_list})")
PYTHON
)"

    if [[ "$TARGET_MODE" == "False" ]]; then
        echo "📅 Processing years: ${years[@]}"
        echo "📅 Valid months: $VALID_MONTHS"
    fi
}

# ------------------------------------------------------------------------------
# Optimized Date Functions (Pure Bash - eliminates Perl subprocess calls)
# ------------------------------------------------------------------------------

# Extract month number from date string (YYYY-MM-DD)
get_month() {
    echo "${1:5:2}"
}

# Extract day number from date string (YYYY-MM-DD)
get_day() {
    echo "${1:8:2}"
}

# Get month abbreviation (1=Jan, 2=Feb, etc.)
get_month_abbr() {
    local months=(Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec)
    echo "${months[$1-1]}"
}

# Increment date by one day (pure bash using date command)
increment_date() {
    local current="$1"
    if date -v +1d > /dev/null 2>&1; then
        # macOS/BSD date
        date -j -v+1d -f "%Y-%m-%d" "$current" "+%Y-%m-%d"
    else
        # GNU date
        date -d "$current + 1 day" "+%Y-%m-%d"
    fi
}

# ------------------------------------------------------------------------------
# Optimized S3 Operations (Batch downloads with AWS CLI sync)
# ------------------------------------------------------------------------------

# Configure AWS CLI for optimal parallelism
export AWS_MAX_CONCURRENT_REQUESTS=20
export AWS_MAX_QUEUE_SIZE=10000

# Download files for a specific month using AWS S3 sync (single API call)
download_month_batch() {
    local year="$1"
    local month_num="$2"
    local month_name="$3"
    local base_dir="$4"

    local s3_path="s3://unidata-nexrad-level2/${year}/$(printf '%02d' $month_num)/"
    local year_path="${base_dir}/${year}"

    echo "📥 Syncing ${month_name} ${year} from S3 (batch operation)..."

    # Use aws s3 sync for efficient batch download
    # S3 structure: s3://.../2025/06/01/KHGX/files, 02/KHGX/files, etc.
    # Syncing from s3://.../2025/06/ to local creates: YEAR/01/KHGX/files, YEAR/02/KHGX/files
    # (The "06" is stripped as it's the sync starting point)
    aws s3 sync "$s3_path" "$year_path" \
        --exclude "*" \
        --include "*/KHGX/KHGX*_V06" \
        --exclude "*_MDM" \
        --no-sign-request \
        --quiet \
        2>/dev/null || {
            echo "⚠️  Warning: Could not sync ${month_name} ${year}"
            return 1
        }

    # Reorganize files from AWS sync structure to MonthDay format
    # AWS creates: YEAR/01/KHGX/files, YEAR/02/KHGX/files, etc.
    # We want: YEAR/Jun01/files, YEAR/Jun02/files, etc.

    echo "🔄 Reorganizing files to ${month_name}DD format..."

    # Process ONLY numeric day directories (01-31) created by this sync
    local reorganized_count=0
    for day_num in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31; do
        local day_path="${year_path}/${day_num}"

        # Skip if directory doesn't exist (not all months have 31 days)
        if [[ ! -d "$day_path" ]]; then
            continue
        fi

        # Check if KHGX subdirectory exists
        local khgx_dir="${day_path}/KHGX"
        if [[ -d "$khgx_dir" ]]; then
            # Create target directory: YEAR/Jun01, YEAR/Jun02, etc.
            local target_dir="${year_path}/${month_name}${day_num}"
            mkdir -p "$target_dir"

            # Count files to move
            local file_count
            file_count=$(ls -1 "$khgx_dir" 2>/dev/null | wc -l)

            if [[ $file_count -gt 0 ]]; then
                # Move all files from KHGX directory to target
                mv "$khgx_dir"/* "$target_dir/" 2>/dev/null && {
                    ((reorganized_count++))
                    echo "  ✓ Moved ${day_num} → ${month_name}${day_num} (${file_count} files)"
                }
            fi

            # Clean up empty directories
            rmdir "$khgx_dir" 2>/dev/null || true
            rmdir "$day_path" 2>/dev/null || true
        fi
    done

    if [[ $reorganized_count -gt 0 ]]; then
        echo "✅ Reorganized $reorganized_count days for ${month_name} ${year}"
    else
        echo "⚠️  Warning: No files were reorganized for ${month_name} ${year}"
        echo "   Looking in: ${year_path}"
        echo "   Expected pattern: ${year_path}/DD/KHGX/"
        ls -la "${year_path}" 2>/dev/null | head -10
    fi

    return 0
}

# Download specific day (for target mode)
download_day_targeted() {
    local year="$1"
    local month="$2"
    local day="$3"
    local month_name="$4"
    local base_dir="$5"

    local s3_base_path="s3://unidata-nexrad-level2/${year}/${month}/${day}/KHGX"
    local download_dir="${base_dir}/${year}/${month_name}${day}"

    mkdir -p "$download_dir"

    echo "🎯 Downloading ${year}-${month}-${day}..."

    # Get file list
    local s3_files
    s3_files=$(aws s3 ls "${s3_base_path}/" --recursive --no-sign-request 2>/dev/null | \
               awk '{print $4}' | grep -v '/$' | grep -v '_MDM$')

    if [[ -z "$s3_files" ]]; then
        echo "No files found in ${s3_base_path}"
        return 1
    fi

    # Build hash table of existing files for O(1) lookup
    declare -A existing_files
    for file in "$download_dir"/*; do
        if [[ -f "$file" ]]; then
            existing_files["$(basename "$file")"]=1
        fi
    done

    # Count and download missing files
    local missing_count=0
    local downloaded=0

    while IFS= read -r s3_file; do
        local filename
        filename=$(basename "$s3_file")

        if [[ ! ${existing_files[$filename]+_} ]]; then
            ((missing_count++))
        fi
    done <<< "$s3_files"

    if [[ $missing_count -eq 0 ]]; then
        echo "✅ All files already exist for ${month_name}${day}"
        return 0
    fi

    echo "📥 Downloading $missing_count missing files..."

    # Download missing files with controlled parallelism
    local max_parallel=20

    while IFS= read -r s3_file; do
        local filename
        filename=$(basename "$s3_file")

        if [[ ! ${existing_files[$filename]+_} ]]; then
            # Wait for free slot
            while (( $(jobs -r | wc -l) >= max_parallel )); do
                sleep 0.1
            done

            # Download in background
            (aws s3 cp "${s3_base_path}/${filename}" "${download_dir}/" \
                --no-sign-request --quiet && ((downloaded++))) &
        fi
    done <<< "$s3_files"

    # Wait for all downloads
    wait

    echo "✅ Downloaded ${downloaded} files for ${month_name}${day}"
    return 0
}

# ------------------------------------------------------------------------------
# Main Processing Logic
# ------------------------------------------------------------------------------

process_year_month_batch() {
    local year="$1"
    local month_name="$2"

    # Convert month name to number
    local month_num
    case "$month_name" in
        Jan) month_num=1 ;;
        Feb) month_num=2 ;;
        Mar) month_num=3 ;;
        Apr) month_num=4 ;;
        May) month_num=5 ;;
        Jun) month_num=6 ;;
        Jul) month_num=7 ;;
        Aug) month_num=8 ;;
        Sep) month_num=9 ;;
        Oct) month_num=10 ;;
        Nov) month_num=11 ;;
        Dec) month_num=12 ;;
    esac

    download_month_batch "$year" "$month_num" "$month_name" "$BASE_DATA_DIR"
}

main() {
    # Load configuration
    load_configuration

    local start_time
    start_time=$(date +%s)

    if [[ "$TARGET_MODE" == "True" ]]; then
        # Target mode: Download specific day
        echo "🎯 Target mode: ${TARGET_YEAR}-$(printf '%02d' $TARGET_MONTH)-$(printf '%02d' $TARGET_DAY)"

        local month_name
        month_name=$(get_month_abbr $TARGET_MONTH)

        download_day_targeted \
            "$TARGET_YEAR" \
            "$(printf '%02d' $TARGET_MONTH)" \
            "$(printf '%02d' $TARGET_DAY)" \
            "$month_name" \
            "$BASE_DATA_DIR"

    else
        # Range mode: Batch download by month
        echo "📊 Processing ${#years[@]} years, months: $VALID_MONTHS"

        local total_operations=$((${#years[@]} * $(echo $VALID_MONTHS | wc -w)))
        local current_op=0

        for year in ${years[@]}; do
            echo ""
            echo "📅 Year: $year"

            for month_name in $VALID_MONTHS; do
                ((current_op++))
                echo "[$current_op/$total_operations] Processing ${month_name} ${year}..."

                process_year_month_batch "$year" "$month_name"

                # Clean up any MDM files
                find "${BASE_DATA_DIR}/${year}" -type f -name '*_MDM' -delete 2>/dev/null || true
            done
        done
    fi

    # Final cleanup
    echo ""
    echo "🧹 Cleaning up _MDM files..."
    for year in ${years[@]}; do
        find "${BASE_DATA_DIR}/${year}" -type f -name '*_MDM' -delete 2>/dev/null || true
    done

    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))

    echo ""
    echo "================================================"
    echo "✅ DOWNLOAD COMPLETE"
    echo "Total time: ${duration}s ($((duration / 60))m $((duration % 60))s)"
    echo "================================================"
}

# Run main function
main "$@"
