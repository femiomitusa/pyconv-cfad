"""Tests for detection.py: ETH maps and TOBAC detection."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "radar_processing"))

from radar_processing.detection import compute_eth_maps, detect_cells, DetectionConfig


# ---------------------------------------------------------------------------
# compute_eth_maps
# ---------------------------------------------------------------------------

class TestComputeEthMaps:
    def _make_field(self, shape, fill=np.nan):
        return np.full(shape, fill, dtype=np.float32)

    def test_returns_all_three_thresholds(self):
        Z = self._make_field((5, 3, 3))
        z_levels = np.array([0, 500, 1000, 2000, 5000], dtype=float)
        result = compute_eth_maps(Z, z_levels)
        assert set(result.keys()) == {20, 30, 40}

    def test_all_nan_input_gives_nan_output(self):
        Z = self._make_field((5, 4, 4))
        z_levels = np.linspace(0, 5000, 5)
        result = compute_eth_maps(Z, z_levels)
        for thr in (20, 30, 40):
            assert np.all(np.isnan(result[thr]))

    def test_known_eth_value(self):
        # Z=35 dBZ at level index 2 (altitude 1000 m) for pixel (1,1)
        # ETH_20 and ETH_30 should be 1000 m; ETH_40 should be NaN
        z_levels = np.array([0.0, 500.0, 1000.0, 2000.0, 5000.0])
        Z = np.full((5, 3, 3), np.nan, dtype=np.float32)
        Z[2, 1, 1] = 35.0

        result = compute_eth_maps(Z, z_levels)

        assert result[20][1, 1] == pytest.approx(1000.0)
        assert result[30][1, 1] == pytest.approx(1000.0)
        assert np.isnan(result[40][1, 1])

    def test_eth_is_highest_level_exceeding_threshold(self):
        # Z ≥ 20 dBZ at levels 1 (500 m) and 3 (2000 m) — ETH_20 should be 2000 m
        z_levels = np.array([0.0, 500.0, 1000.0, 2000.0, 5000.0])
        Z = np.full((5, 3, 3), np.nan, dtype=np.float32)
        Z[1, 0, 0] = 25.0   # 500 m
        Z[3, 0, 0] = 22.0   # 2000 m (higher, should win)

        result = compute_eth_maps(Z, z_levels)
        assert result[20][0, 0] == pytest.approx(2000.0)

    def test_output_shape_matches_input_horizontal(self):
        ny, nx = 7, 9
        Z = self._make_field((5, ny, nx))
        z_levels = np.linspace(0, 5000, 5)
        result = compute_eth_maps(Z, z_levels)
        for thr in (20, 30, 40):
            assert result[thr].shape == (ny, nx)

    def test_no_pixel_below_threshold_gives_nan(self):
        z_levels = np.array([0.0, 1000.0, 5000.0])
        Z = np.full((3, 2, 2), 10.0, dtype=np.float32)  # all 10 dBZ < 20
        result = compute_eth_maps(Z, z_levels)
        assert np.all(np.isnan(result[20]))


# ---------------------------------------------------------------------------
# detect_cells — requires TOBAC; skip gracefully if not installed
# ---------------------------------------------------------------------------

tobac = pytest.importorskip("tobac", reason="tobac not installed")


class TestDetectCells:
    def _make_radar_fields(self, ny: int = 40, nx: int = 40, n_levels: int = 5) -> dict:
        """Synthetic radar_fields dict with a masked reflectivity array."""
        import numpy.ma as ma
        ref = ma.array(
            np.full((n_levels, ny, nx), np.nan, dtype=np.float32),
            mask=np.ones((n_levels, ny, nx), dtype=bool),
        )
        return {"reflectivity": ref}

    def _insert_cell(self, fields: dict, cy: int, cx: int, radius: int, dbz: float) -> None:
        """Place a synthetic convective cell at (cy, cx) with given radius."""
        import numpy.ma as ma
        ref = fields["reflectivity"]
        ny, nx = ref.shape[1], ref.shape[2]
        for y in range(max(0, cy - radius), min(ny, cy + radius + 1)):
            for x in range(max(0, cx - radius), min(nx, cx + radius + 1)):
                if (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2:
                    ref[:, y, x] = dbz
                    ref.mask[:, y, x] = False

    def test_no_cell_returns_empty_mask(self):
        fields = self._make_radar_fields()
        cfg = DetectionConfig(thresholds=(30.0, 35.0), segmentation_threshold=25.0,
                              min_pixels=4, grid_spacing_m=500.0)
        z_comp, mask, features = detect_cells(fields, pd.Timestamp("2022-06-08 12:00"), cfg)
        assert mask.shape == (40, 40)
        assert np.all(mask == 0)
        assert features is None

    def test_strong_cell_produces_nonzero_mask(self):
        fields = self._make_radar_fields(ny=60, nx=60)
        self._insert_cell(fields, cy=30, cx=30, radius=8, dbz=45.0)
        cfg = DetectionConfig(thresholds=(30.0, 35.0, 40.0), segmentation_threshold=25.0,
                              min_pixels=4, grid_spacing_m=500.0)
        z_comp, mask, features = detect_cells(fields, pd.Timestamp("2022-06-08 12:00"), cfg)
        assert np.any(mask > 0), "Expected at least one detected cell pixel"

    def test_z_composite_is_column_max(self):
        fields = self._make_radar_fields(ny=20, nx=20, n_levels=3)
        import numpy.ma as ma
        # Level 0: 10 dBZ, level 1: 40 dBZ, level 2: 20 dBZ at pixel (5,5)
        fields["reflectivity"][:, 5, 5] = ma.array([10.0, 40.0, 20.0], mask=False)
        cfg = DetectionConfig(thresholds=(30.0,), segmentation_threshold=25.0,
                              min_pixels=1, grid_spacing_m=500.0)
        z_comp, _, _ = detect_cells(fields, pd.Timestamp("2022-06-08 12:00"), cfg)
        assert z_comp[5, 5] == pytest.approx(40.0)

    def test_z_composite_shape(self):
        fields = self._make_radar_fields(ny=15, nx=20)
        cfg = DetectionConfig(thresholds=(30.0,), segmentation_threshold=25.0,
                              min_pixels=4, grid_spacing_m=500.0)
        z_comp, mask, _ = detect_cells(fields, pd.Timestamp("2022-06-08 12:00"), cfg)
        assert z_comp.shape == (15, 20)
        assert mask.shape == (15, 20)
