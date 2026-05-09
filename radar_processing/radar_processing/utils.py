import re
import pandas as pd


def get_datetime_from_filename(filename: str) -> pd.Timestamp | None:
    """Parse scan timestamp from a NEXRAD filename. Returns pd.Timestamp or None."""
    match = re.search(r"(\d{8})_(\d{6})", filename)
    if not match:
        return None
    return pd.to_datetime(f"{match.group(1)}{match.group(2)}", format="%Y%m%d%H%M%S")
