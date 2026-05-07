import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from config import PLOT_FIGSIZE, REFLECTIVITY_LIMITS, COLORMAP

def create_radar_plot(
    xx: np.ndarray,
    yy: np.ndarray,
    Z: np.ndarray,
    cell_locations: List[Tuple[float, float, float]],
    title: str,
    output_path: str
) -> None:
    """
    Create and save a radar plot with cell locations.
    
    Args:
        xx: X-coordinates grid
        yy: Y-coordinates grid
        Z: Reflectivity data
        cell_locations: List of (x, y, radius) tuples for each cell
        title: Plot title
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    
    # Plot the reflectivity data
    lev = 0  # Plot first vertical level
    mappable = ax.pcolormesh(
        xx, yy,
        np.squeeze(Z[lev, :, :]),
        cmap=COLORMAP,
        vmin=REFLECTIVITY_LIMITS[0],
        vmax=REFLECTIVITY_LIMITS[1]
    )
    
    # Add colorbar
    cbar = plt.colorbar(mappable)
    cbar.set_label('Reflectivity (dBZ)', rotation=90, labelpad=15)
    
    # Plot cell circles
    for xctr, yctr, rad_meters in cell_locations:
        circle = plt.Circle(
            (xctr, yctr),
            rad_meters,
            color='k',
            fill=False,
            linewidth=2
        )
        ax.add_artist(circle)
    
    # Set plot attributes
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([xx.min(), xx.max()])
    ax.set_ylim([yy.min(), yy.max()])
    ax.set_xlabel('North-South Distance from Radar (m)')
    ax.set_ylabel('East-West Distance from Radar (m)')
    plt.title(title)
    
    # Save and close
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
