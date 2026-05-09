import argparse
import os
import sys

os.environ["PYART_QUIET"] = "1"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from pathlib import Path
import pyart  # noqa: F401  # registers Py-ART colormaps such as NWSRef
from skimage import measure

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    BASE_DATA_DIR,
    GRID_BOUNDS,
    GRID_POINTS,
    PLOT_FIGSIZE,
    REFLECTIVITY_LIMITS,
    TARGET_DAY,
    TARGET_MONTH,
    TARGET_YEAR,
    COLORMAP,
)
from utils import get_array_directory, get_figures_directory

_CELL_OUTLINE_COLOR = "black"
_CELL_LABEL_COLOR = "black"


def create_radar_plot(xx, yy, Z, track_mask, title, output_path, *, show_ids: bool = False):
    """Save a reflectivity map with tracked-cell contours overlaid.

    Z         : (ny, nx) float — column-max composite reflectivity.
    track_mask: (ny, nx) int32 — persistent track IDs (0 = background).
                Pass None or an all-zero array for scans with no cells.
    """
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

    if Z.ndim == 3:
        Z_safe = np.ma.filled(Z.astype(float), np.nan) if isinstance(Z, np.ma.MaskedArray) else Z
        Z_composite = np.nanmax(Z_safe, axis=0)
    else:
        Z_composite = Z

    z_plot = np.ma.masked_invalid(np.asarray(Z_composite, dtype=float))
    z_plot = np.ma.masked_less(z_plot, REFLECTIVITY_LIMITS[0])

    mappable = ax.pcolormesh(
        xx, yy, z_plot,
        cmap=COLORMAP, vmin=REFLECTIVITY_LIMITS[0], vmax=REFLECTIVITY_LIMITS[1],
    )
    cbar = plt.colorbar(mappable)
    cbar.set_label('Reflectivity (dBZ)', rotation=90, labelpad=15)

    if track_mask is not None:
        cell_ids = [int(c) for c in np.unique(track_mask) if c > 0]
        for cid in cell_ids:
            binary = track_mask == cid
            for contour in measure.find_contours(binary.astype(float), 0.5):
                rows = np.clip(np.rint(contour[:, 0]).astype(int), 0, yy.shape[0] - 1)
                cols = np.clip(np.rint(contour[:, 1]).astype(int), 0, xx.shape[1] - 1)
                ax.plot(xx[rows, cols], yy[rows, cols], color=_CELL_OUTLINE_COLOR, linewidth=1.5)
            if not show_ids:
                continue
            ys, xs = np.where(track_mask == cid)
            cx_m = float(xx[int(np.mean(ys)), int(np.mean(xs))])
            cy_m = float(yy[int(np.mean(ys)), int(np.mean(xs))])
            label = ax.text(
                cx_m, cy_m, str(cid),
                color=_CELL_LABEL_COLOR, fontsize=8, ha="center", va="center", fontweight="bold",
            )
            label.set_path_effects([
                path_effects.Stroke(linewidth=2.5, foreground="white"),
                path_effects.Normal(),
            ])

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([xx.min(), xx.max()])
    ax.set_ylim([yy.min(), yy.max()])
    ax.set_xlabel('East-West Distance from Radar (m)')
    ax.set_ylabel('North-South Distance from Radar (m)')
    plt.title(title)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def regenerate_plots_from_tracking(
    year: int,
    month: int,
    day: int,
    *,
    base_data_dir: str = BASE_DATA_DIR,
    overwrite: bool = True,
    show_ids: bool = False,
) -> int:
    """Regenerate radar plots from an existing daily tracking.nc file."""
    import pandas as pd
    import xarray as xr

    date_str = f"{year}{month:02d}{day:02d}"
    tracking_nc = Path(get_array_directory(year, month, day, base_data_dir)) / f"KHGX{date_str}_tracking.nc"
    fig_dir = Path(get_figures_directory(year, month, day, base_data_dir))

    if not tracking_nc.exists():
        raise FileNotFoundError(f"Tracking file not found: {tracking_nc}")

    fig_dir.mkdir(parents=True, exist_ok=True)
    x = np.linspace(GRID_BOUNDS[0], GRID_BOUNDS[1], GRID_POINTS)
    xx, yy = np.meshgrid(x, x)

    with xr.open_dataset(tracking_nc) as ds:
        if "z_composite" not in ds or "mask" not in ds:
            raise ValueError(f"{tracking_nc} does not contain z_composite and mask")
        z_composites = ds["z_composite"].values
        masks = ds["mask"].values
        scan_times = pd.DatetimeIndex(ds["scan_time"].values)

    count = 0
    for i, scan_time in enumerate(scan_times):
        time_str = pd.Timestamp(scan_time).strftime("%H%M%S")
        fig_name = f"KHGX{date_str}_{time_str}_plot.png"
        fig_path = fig_dir / fig_name
        if fig_path.exists() and not overwrite:
            continue
        create_radar_plot(
            xx,
            yy,
            z_composites[i],
            masks[i],
            f"KHGX{date_str}_{time_str}_V06",
            fig_path,
            show_ids=show_ids,
        )
        count += 1

    return count


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate pyconv-cfad radar plots from an existing tracking.nc file."
    )
    parser.add_argument("--year", type=int, default=TARGET_YEAR)
    parser.add_argument("--month", type=int, default=TARGET_MONTH)
    parser.add_argument("--day", type=int, default=TARGET_DAY)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Only create missing plot files instead of overwriting existing PNGs.",
    )
    parser.add_argument(
        "--show-ids",
        action="store_true",
        help="Draw cell ID numbers inside cell outlines.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    n_plots = regenerate_plots_from_tracking(
        args.year,
        args.month,
        args.day,
        overwrite=not args.skip_existing,
        show_ids=args.show_ids,
    )
    print(f"Regenerated {n_plots} radar plot(s).")
