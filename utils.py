"""
Utility functions for the Weather Radar Analysis Suite.
Path construction and date formatting utilities.
"""
import calendar


def get_data_directory(year: int, month: int, day: int, base_data_dir: str) -> str:
    """Get data directory path for specific date.
    
    Args:
        year: Year (e.g., 2019)
        month: Month as integer (e.g., 6 for June)
        day: Day as integer (e.g., 29)
        base_data_dir: Base data directory path
        
    Returns:
        str: Path to data directory (e.g., '/Volumes/My Book/NEXRAD/2019/Jul16')
    """
    if not (1 <= month <= 12):
        raise ValueError(f"Month must be between 1 and 12, got {month}")
    
    month_name = calendar.month_abbr[month]
    return f"{base_data_dir}/{year}/{month_name}{day:02d}"


def get_array_directory(year: int, month: int, day: int, base_data_dir: str) -> str:
    """Get array output directory path for specific date.
    
    Args:
        year: Year (e.g., 2019)
        month: Month as integer (e.g., 6 for June)
        day: Day as integer (e.g., 29)
        base_data_dir: Base data directory path
        
    Returns:
        str: Path to array directory (e.g., '/Volumes/My Book/NEXRAD/Arrays/2019/Jul16')
    """
    if not (1 <= month <= 12):
        raise ValueError(f"Month must be between 1 and 12, got {month}")
    
    month_name = calendar.month_abbr[month]
    return f"{base_data_dir}/Arrays/{year}/{month_name}{day:02d}"


def get_figures_directory(year: int, month: int, day: int, base_data_dir: str) -> str:
    """Get figures output directory path for specific date.
    
    Args:
        year: Year (e.g., 2019)
        month: Month as integer (e.g., 6 for June)
        day: Day as integer (e.g., 29)
        base_data_dir: Base data directory path
        
    Returns:
        str: Path to figures directory (e.g., '/Volumes/My Book/NEXRAD/Figures/2019/Jul16')
    """
    if not (1 <= month <= 12):
        raise ValueError(f"Month must be between 1 and 12, got {month}")
    
    month_name = calendar.month_abbr[month]
    return f"{base_data_dir}/Figures/{year}/{month_name}{day:02d}"


def get_date_string(year: int, month: int, day: int) -> str:
    """Get formatted date string for file operations.
    
    Args:
        year: Year (e.g., 2019)
        month: Month as integer (e.g., 6 for June)
        day: Day as integer (e.g., 29)
        
    Returns:
        str: Formatted date string (e.g., 'Jun29')
    """
    if not (1 <= month <= 12):
        raise ValueError(f"Month must be between 1 and 12, got {month}")
    
    month_name = calendar.month_abbr[month]
    return f"{month_name}{day:02d}"