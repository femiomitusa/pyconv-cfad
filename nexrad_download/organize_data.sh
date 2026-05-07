#!/bin/bash

# Root directory
ROOT_DIR="/Volumes/One Touch/Oklahoma Research/Data/NEXRAD_DATA"

# Function to process a single year
process_year() {
    local YEAR=$1
    local YEAR_DIR="$ROOT_DIR/$YEAR"
    
    # Skip if year directory doesn't exist
    if [ ! -d "$YEAR_DIR" ]; then
        return
    fi
    
    # Create Jun, Jul, Aug folders
    mkdir -p "$YEAR_DIR/Jun" "$YEAR_DIR/Jul" "$YEAR_DIR/Aug"
    
    # Move files for June (06)
    find "$YEAR_DIR" -maxdepth 2 -type f -name 'KHGX????06??_*' -exec mv {} "$YEAR_DIR/Jun/" \;
    
    # Move files for July (07)
    find "$YEAR_DIR" -maxdepth 2 -type f -name 'KHGX????07??_*' -exec mv {} "$YEAR_DIR/Jul/" \;
    
    # Move files for August (08)
    find "$YEAR_DIR" -maxdepth 2 -type f -name 'KHGX????08??_*' -exec mv {} "$YEAR_DIR/Aug/" \;
    
    # Delete all daily folders (anything not Jun, Jul, Aug)
    find "$YEAR_DIR" -maxdepth 1 -type d -not -path "$YEAR_DIR" -not -name "Jun" -not -name "Jul" -not -name "Aug" -exec rm -rf {} \;
}

# Export the function so it can be used in parallel
export -f process_year

# Process years 2017-2019 in parallel
for YEAR in {2017..2019}; do
    process_year "$YEAR" &
done

# Wait for all background jobs to complete
wait