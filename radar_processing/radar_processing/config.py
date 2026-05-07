# Import all settings from root config
import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import *  # Import all configuration parameters

# Convert relative paths to absolute paths
DATA_DIR = os.path.abspath(DATA_DIR)
STATS_FILE = os.path.abspath(STATS_FILE)
FIGURE_OUTPUT_DIR = os.path.abspath(FIGURE_OUTPUT_DIR)
ARRAY_OUTPUT_DIR = os.path.abspath(ARRAY_OUTPUT_DIR)
