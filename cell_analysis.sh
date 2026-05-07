#!/bin/bash
# Weather Radar Analysis Pipeline - End-to-End Integration

set -Eeuo pipefail

# --- Constants and Globals ---

readonly COLOR_RED='\033[0;31m'
readonly COLOR_GREEN='\033[0;32m'
readonly COLOR_YELLOW='\033[1;33m'
readonly COLOR_BLUE='\033[0;34m'
readonly COLOR_NC='\033[0m'

readonly NEXRAD_DOWNLOAD_SCRIPT="nexrad_download/data_download.py"
readonly RADAR_PROCESSING_SCRIPT="radar_processing/process_radar.py"
readonly REGIONAL_FILTERING_SCRIPT="filter_by_region.py"
readonly CFAD_RUNNER_SCRIPT="cfad_analysis/run_cfad_with_config.py"

PIPELINE_START_TIME=$(date +%s)
STAGE_START_TIME=0
STAGE_COUNT=0
TOTAL_STAGES=2  # prerequisite validation + pipeline completion; incremented per enabled stage
CURRENT_STAGE=""


# --- Utility Functions ---

print_colored() {
    local color="$1"
    local message="$2"
    echo -e "${color}${message}${COLOR_NC}"
}

print_header() {
    print_colored "$COLOR_BLUE" "================================================================================"
    print_colored "$COLOR_BLUE" "$1"
    print_colored "$COLOR_BLUE" "================================================================================"
}

print_section() {
    echo
    print_colored "$COLOR_YELLOW" "----------------------------------------"
    print_colored "$COLOR_YELLOW" "$1"
    print_colored "$COLOR_YELLOW" "----------------------------------------"
}

print_success() {
    print_colored "$COLOR_GREEN" "✅ $1"
}

print_error() {
    print_colored "$COLOR_RED" "❌ $1"
}

print_warning() {
    print_colored "$COLOR_YELLOW" "⚠️  $1"
}

fatal_error() {
    print_error "FATAL: $1"
    exit 1
}

# --- Pipeline Stage Management ---

stage_start() {
    STAGE_COUNT=$((STAGE_COUNT + 1))
    STAGE_START_TIME=$(date +%s)
    CURRENT_STAGE="$1"

    print_section "STAGE $STAGE_COUNT/$TOTAL_STAGES: $CURRENT_STAGE"
    trap 'stage_end "$CURRENT_STAGE" "false"; exit 1' ERR
}

stage_end() {
    local stage_name="$1"
    local success="$2"

    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - STAGE_START_TIME))

    if [ "$success" = "true" ]; then
        print_success "$stage_name completed in ${duration}s."
    else
        print_error "$stage_name failed after ${duration}s."
    fi
    trap - ERR
}

# --- Core Logic Functions ---

load_config() {
    print_section "LOADING CONFIGURATION"
    if [ ! -f "config.py" ]; then fatal_error "Configuration file 'config.py' not found."; fi

    while IFS= read -r -d '' k && IFS= read -r -d '' v; do
        export "$k=$v"
    done < <(python3 - <<'PY'
import config, sys

KEYS = [
    'BASE_DATA_DIR','YEAR_START','YEAR_END','VALID_MONTHS',
    'RUN_NEXRAD_DOWNLOAD','RUN_RADAR_PROCESSING','RUN_CFAD_ANALYSIS',
    'RUN_REGIONAL_FILTERING','TARGET_YEAR','TARGET_MONTH','TARGET_DAY',
    'TARGET_MODE','CFAD_OUTPUT_DIR'
]

def emit(k, v):
    sys.stdout.write(k); sys.stdout.write("\0")
    sys.stdout.write("" if v is None else str(v)); sys.stdout.write("\0")

for k in KEYS:
    val = getattr(config, k, None)
    if isinstance(val, list):
        val = " ".join(map(str, val))
    emit(k, val)
PY
)

    print_success "Configuration loaded successfully."
    print_loaded_configuration

    # Count how many pipeline stages are actually enabled
    if [ "$RUN_NEXRAD_DOWNLOAD"    = "True" ]; then TOTAL_STAGES=$((TOTAL_STAGES + 1)); fi
    if [ "$RUN_RADAR_PROCESSING"   = "True" ]; then TOTAL_STAGES=$((TOTAL_STAGES + 1)); fi
    if [ "$RUN_REGIONAL_FILTERING" = "True" ]; then TOTAL_STAGES=$((TOTAL_STAGES + 1)); fi
    if [ "$RUN_CFAD_ANALYSIS"      = "True" ]; then TOTAL_STAGES=$((TOTAL_STAGES + 1)); fi
}

print_loaded_configuration() {
    echo "  Base data directory: ${BASE_DATA_DIR}"
    echo "  Processing years: ${YEAR_START} to ${YEAR_END}"
    echo "  Valid months: ${VALID_MONTHS}"
    echo "  Download enabled: ${RUN_NEXRAD_DOWNLOAD}"
    echo "  Radar processing enabled: ${RUN_RADAR_PROCESSING}"
    echo "  Regional filtering enabled: ${RUN_REGIONAL_FILTERING}"
    echo "  CFAD analysis enabled: ${RUN_CFAD_ANALYSIS}"
    echo "  Target mode: ${TARGET_MODE}"
}

check_python_modules() {
    local modules_csv="$1"
    MODULES_TO_CHECK="$modules_csv" python3 - <<'PY'
import importlib.util
import sys
import os

modules = os.environ.get('MODULES_TO_CHECK', '').split(',')
missing = [m for m in modules if m and importlib.util.find_spec(m) is None]
if missing:
    print(','.join(missing))
PY
}

validate_core_packages() {
    local missing
    missing=$(check_python_modules "numpy,xarray,matplotlib,tqdm")

    if [ -n "$missing" ]; then
        fatal_error "Required Python packages not found: $missing"
    fi
    print_success "Core Python packages are installed."
}

validate_radar_packages() {
    if [ "$RUN_RADAR_PROCESSING" != "True" ]; then return; fi

    local missing
    missing=$(check_python_modules "pyart,netCDF4,scipy")

    if [ -n "$missing" ]; then
        fatal_error "Radar processing packages not found: $missing"
    fi
    print_success "Radar processing packages are installed."
}

validate_regional_packages() {
    if [ "$RUN_REGIONAL_FILTERING" != "True" ]; then return; fi

    local missing
    missing=$(check_python_modules "shapely,geopandas")

    if [ -n "$missing" ]; then
        print_warning "Regional filtering packages not found: $missing (disabling stage)"
        RUN_REGIONAL_FILTERING="False"
        export RUN_REGIONAL_FILTERING
    else
        print_success "Regional filtering packages are installed."
    fi
}

validate_cfad_packages() {
    if [ "$RUN_CFAD_ANALYSIS" != "True" ]; then return; fi

    local missing
    missing=$(check_python_modules "numpy,xarray,matplotlib,netCDF4")

    if [ -n "$missing" ]; then
        fatal_error "CFAD analysis packages not found: $missing"
    fi
    print_success "CFAD analysis packages are installed."
}

validate_download_packages() {
    if [ "$RUN_NEXRAD_DOWNLOAD" != "True" ]; then return; fi

    local missing
    missing=$(check_python_modules "boto3,tqdm")

    if [ -n "$missing" ]; then
        print_error "NEXRAD download packages not found: $missing"
        echo ""
        echo "Please install dependencies:"
        echo "  pip install -r nexrad_download/requirements.txt"
        echo ""
        echo "Or manually:"
        echo "  pip install boto3 tqdm"
        fatal_error "Missing Python dependencies for NEXRAD download"
    fi
    print_success "NEXRAD download packages are installed."
}

ensure_directory_exists() {
    local dir_path="$1"
    local dir_description="$2"

    if [ ! -d "$dir_path" ]; then
        print_warning "$dir_description does not exist: $dir_path. Creating it now."
        mkdir -p "$dir_path" || fatal_error "Failed to create $dir_description: $dir_path"
        print_success "Created $dir_description: $dir_path"
    else
        print_success "$dir_description exists: $dir_path"
    fi
}

validate_directories() {
    ensure_directory_exists "$BASE_DATA_DIR" "Data directory"

    if [ "$RUN_CFAD_ANALYSIS" = "True" ] && [ -n "$CFAD_OUTPUT_DIR" ]; then
        ensure_directory_exists "$CFAD_OUTPUT_DIR" "CFAD output directory"
    fi
}

validate_required_scripts() {
    if [ "$RUN_NEXRAD_DOWNLOAD" = "True" ] && [ ! -f "$NEXRAD_DOWNLOAD_SCRIPT" ]; then
        fatal_error "NEXRAD download script not found at $NEXRAD_DOWNLOAD_SCRIPT"
    fi

    if [ "$RUN_RADAR_PROCESSING" = "True" ] && [ ! -f "$RADAR_PROCESSING_SCRIPT" ]; then
        fatal_error "Radar processing script not found at $RADAR_PROCESSING_SCRIPT"
    fi

    if [ "$RUN_CFAD_ANALYSIS" = "True" ] && [ ! -f "$CFAD_RUNNER_SCRIPT" ]; then
        fatal_error "CFAD analysis runner not found at $CFAD_RUNNER_SCRIPT"
    fi

    if [ "$RUN_REGIONAL_FILTERING" = "True" ] && [ ! -f "$REGIONAL_FILTERING_SCRIPT" ]; then
        print_warning "Regional filtering script not found at $REGIONAL_FILTERING_SCRIPT (stage will be skipped)"
    fi
}

validate_prerequisites() {
    stage_start "PREREQUISITE VALIDATION"

    validate_core_packages
    validate_download_packages
    validate_radar_packages
    validate_regional_packages
    validate_cfad_packages
    validate_directories
    validate_required_scripts

    stage_end "PREREQUISITE VALIDATION" "true"
}

format_target_date() {
    echo "$TARGET_YEAR-$(printf '%02d' "$TARGET_MONTH")-$(printf '%02d' "$TARGET_DAY")"
}

verify_existing_nexrad_files() {
    local file_count
    file_count=$(find "$BASE_DATA_DIR" -name "KHGX*_V06" -type f | wc -l)
    if [ "${file_count// /}" -eq 0 ]; then
         fatal_error "No NEXRAD files found in $BASE_DATA_DIR. Enable download or ensure data exists."
    fi
    print_success "Found ${file_count// /} existing NEXRAD files."
}

run_nexrad_download() {
    if [ "$RUN_NEXRAD_DOWNLOAD" != "True" ]; then
        print_section "NEXRAD DOWNLOAD"
        print_warning "Download disabled in config.py. Verifying existing data..."
        verify_existing_nexrad_files
        return
    fi

    stage_start "NEXRAD DATA DOWNLOAD"

    if [ "$TARGET_MODE" == "True" ]; then
        echo "Downloading data for target date: $(format_target_date)"
    else
        echo "Downloading data for years $YEAR_START-$YEAR_END, months: $VALID_MONTHS"
    fi

    python3 "$NEXRAD_DOWNLOAD_SCRIPT"

    stage_end "NEXRAD DATA DOWNLOAD" "true"
}

verify_existing_array_files() {
    local array_count
    array_count=$(find "$BASE_DATA_DIR/Arrays" -name "*.npy" -type f 2>/dev/null | wc -l)
    if [ "${array_count// /}" -eq 0 ]; then
        fatal_error "No processed array files found. Enable processing or ensure arrays exist."
    fi
    print_success "Found ${array_count// /} existing array files."
}

run_radar_processing() {
    if [ "$RUN_RADAR_PROCESSING" != "True" ]; then
        print_section "RADAR PROCESSING"
        print_warning "Radar processing is disabled in config.py. Assuming processed arrays exist."
        verify_existing_array_files
        return
    fi

    stage_start "RADAR DATA PROCESSING"
    echo "Processing radar data..."
    python3 "$RADAR_PROCESSING_SCRIPT"
    stage_end "RADAR DATA PROCESSING" "true"
}

has_regional_filtering_dependencies() {
    python3 -c "import shapely, geopandas" 2>/dev/null
}

run_regional_filtering() {
    if [ "$RUN_REGIONAL_FILTERING" != "True" ]; then
        print_section "REGIONAL FILTERING"
        print_warning "Regional filtering is disabled in config.py."
        return
    fi

    stage_start "REGIONAL FILTERING"

    if ! has_regional_filtering_dependencies; then
        print_warning "Optional dependencies (shapely, geopandas) not found. Skipping regional filtering."
        stage_end "REGIONAL FILTERING" "true"
        return
    fi

    if [ "$TARGET_MODE" == "True" ]; then
        echo "Running regional filtering for target date: $(format_target_date)"
        python3 "$REGIONAL_FILTERING_SCRIPT" --date "$(format_target_date)"
    else
        echo "Running regional filtering for years $YEAR_START to $YEAR_END..."
        python3 "$REGIONAL_FILTERING_SCRIPT"
    fi

    stage_end "REGIONAL FILTERING" "true"
}

run_cfad_analysis() {
    if [ "$RUN_CFAD_ANALYSIS" != "True" ]; then
        print_section "CFAD ANALYSIS"
        print_warning "CFAD analysis is disabled in config.py."
        return
    fi

    stage_start "CFAD ANALYSIS"

    if [ "$TARGET_MODE" == "True" ]; then
        echo "Running CFAD analysis for target date: $(format_target_date)"
        CFAD_MULTI_TEMPORAL_ENABLED=False PYART_QUIET=1 python3 "$CFAD_RUNNER_SCRIPT"
    else
        echo "Running CFAD analysis with safeguards..."
        PYART_QUIET=1 python3 "$CFAD_RUNNER_SCRIPT"
    fi

    stage_end "CFAD ANALYSIS" "true"
}

format_duration() {
    local total_seconds="$1"
    echo "${total_seconds}s ($((total_seconds / 60))m $((total_seconds % 60))s)"
}

print_processing_summary() {
    echo "PROCESSING SUMMARY:"
    echo "  Download enabled: $RUN_NEXRAD_DOWNLOAD"
    echo "  Radar processing enabled: $RUN_RADAR_PROCESSING"
    echo "  Regional filtering enabled: $RUN_REGIONAL_FILTERING"
    echo "  CFAD analysis enabled: $RUN_CFAD_ANALYSIS"
}

print_output_locations() {
    echo "OUTPUT LOCATIONS:"
    echo "  Raw data: $BASE_DATA_DIR"
    echo "  Processed arrays: $BASE_DATA_DIR/Arrays"
    if [ "$RUN_REGIONAL_FILTERING" = "True" ]; then
        echo "  Regional data: $BASE_DATA_DIR/Arrays/[region]/"
    fi
    if [ "$RUN_CFAD_ANALYSIS" = "True" ]; then
        echo "  CFAD results: ${CFAD_OUTPUT_DIR:-Not Specified}"
    fi
}

summarize_pipeline() {
    stage_start "PIPELINE COMPLETION"

    local pipeline_end_time
    pipeline_end_time=$(date +%s)
    local duration=$((pipeline_end_time - PIPELINE_START_TIME))

    print_header "WEATHER RADAR ANALYSIS PIPELINE COMPLETED"
    echo "Pipeline finished at: $(date)"
    echo "Total duration: $(format_duration "$duration")"
    echo

    print_processing_summary
    echo

    print_success "All enabled pipeline stages reported completion."
    echo

    print_output_locations

    stage_end "PIPELINE COMPLETION" "true"
    print_header "END OF PIPELINE"
}


# --- Main Execution ---

main() {
    print_header "WEATHER RADAR ANALYSIS PIPELINE"
    echo "Pipeline started at: $(date)"
    echo "Working directory: $(pwd)"

    load_config
    validate_prerequisites
    run_nexrad_download
    run_radar_processing
    run_regional_filtering
    run_cfad_analysis
    summarize_pipeline
}

main "$@"
