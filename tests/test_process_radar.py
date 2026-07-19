"""Tests for process_radar.py: coordinate conversion, obs collection, NetCDF output."""

import tempfile
from pathlib import Path

import numpy as np
import numpy.ma as ma
import pandas as pd
import pytest
import xarray as xr

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "radar_processing"))

# Import private helpers directly
from process_radar import (
    _cart2latlon,
    _collect_frame_obs,
    _write_tracking_netcdf,
)

_RADAR_LAT = 29.4719
_RADAR_LON = -95.0787


# ---------------------------------------------------------------------------
# _cart2latlon
# ---------------------------------------------------------------------------

class TestCart2Latlon:
    def test_origin_returns_radar_location(self):
        xx = np.array([[0.0]])
        yy = np.array([[0.0]])
        lat, lon = _cart2latlon(xx, yy)
        assert lat[0, 0] == pytest.approx(_RADAR_LAT, abs=1e-3)
        assert lon[0, 0] == pytest.approx(_RADAR_LON, abs=1e-3)

    def test_north_displacement_increases_latitude(self):
        xx = np.array([[0.0, 0.0]])
        yy = np.array([[0.0, 50000.0]])    # 50 km North
        lat, lon = _cart2latlon(xx, yy)
        assert lat[0, 1] > lat[0, 0]
        assert lon[0, 1] == pytest.approx(lon[0, 0], abs=0.01)

    def test_east_displacement_increases_longitude(self):
        xx = np.array([[0.0, 50000.0]])    # 50 km East
        yy = np.array([[0.0, 0.0]])
        lat, lon = _cart2latlon(xx, yy)
        assert lon[0, 1] > lon[0, 0]
        assert lat[0, 1] == pytest.approx(lat[0, 0], abs=0.01)

    def test_50km_north_correct_magnitude(self):
        # 50 km North ≈ 0.45° latitude increase
        xx = np.array([[0.0]])
        yy = np.array([[50000.0]])
        lat, lon = _cart2latlon(xx, yy)
        assert lat[0, 0] == pytest.approx(_RADAR_LAT + 0.45, abs=0.05)

    def test_output_dtype_float32(self):
        xx = np.zeros((3, 3))
        yy = np.zeros((3, 3))
        lat, lon = _cart2latlon(xx, yy)
        assert lat.dtype == np.float32
        assert lon.dtype == np.float32

    def test_output_shape_matches_input(self):
        xx = np.zeros((5, 7))
        yy = np.zeros((5, 7))
        lat, lon = _cart2latlon(xx, yy)
        assert lat.shape == (5, 7)
        assert lon.shape == (5, 7)


# ---------------------------------------------------------------------------
# _collect_frame_obs
# ---------------------------------------------------------------------------

def _synthetic_radar_fields(n_levels=5, ny=10, nx=10, dbz=40.0) -> dict:
    """Minimal radar_fields dict with masked arrays shaped (n_levels, ny, nx)."""
    def _field(val):
        return ma.array(
            np.full((n_levels, ny, nx), val, dtype=np.float32),
            mask=False,
        )
    return {
        "reflectivity":              _field(dbz),
        "differential_reflectivity": _field(1.5),
        "cross_correlation_ratio":   _field(0.97),
        "kdp":                       _field(0.5),
        "reflectivity_raw":          _field(dbz),
    }


class TestCollectFrameObs:
    def _base_grids(self, ny=10, nx=10):
        lat = np.full((ny, nx), _RADAR_LAT, dtype=np.float32)
        lon = np.full((ny, nx), _RADAR_LON, dtype=np.float32)
        return lat, lon

    def test_empty_mask_returns_empty_list(self):
        n_levels, ny, nx = 5, 10, 10
        fields = _synthetic_radar_fields(n_levels, ny, nx)
        track_mask = np.zeros((ny, nx), dtype=np.int32)
        z_composite = np.zeros((ny, nx), dtype=np.float32)
        Z_filled = np.full((n_levels, ny, nx), np.nan, dtype=np.float32)
        z_levels = np.linspace(0, 5000, n_levels)
        eth_maps = {20: np.full((ny, nx), np.nan), 30: np.full((ny, nx), np.nan),
                    40: np.full((ny, nx), np.nan)}
        lat, lon = self._base_grids(ny, nx)

        result = _collect_frame_obs(
            track_mask, fields, Z_filled, z_composite,
            lat, lon, eth_maps, {}, pixel_area_km2=0.25, scan_idx=0,
        )
        assert result == []

    def test_single_cell_returns_one_obs(self):
        n_levels, ny, nx = 5, 10, 10
        fields = _synthetic_radar_fields(n_levels, ny, nx, dbz=40.0)
        track_mask = np.zeros((ny, nx), dtype=np.int32)
        track_mask[3:6, 3:6] = 1    # 9-pixel cell
        z_composite = np.full((ny, nx), 40.0, dtype=np.float32)
        Z_filled = np.full((n_levels, ny, nx), 40.0, dtype=np.float32)
        z_levels = np.linspace(0, 5000, n_levels)
        eth_maps = {20: np.full((ny, nx), 2000.0),
                    30: np.full((ny, nx), 1000.0),
                    40: np.full((ny, nx), 500.0)}
        lat, lon = self._base_grids(ny, nx)

        result = _collect_frame_obs(
            track_mask, fields, Z_filled, z_composite,
            lat, lon, eth_maps, {}, pixel_area_km2=0.25, scan_idx=3,
        )
        assert len(result) == 1
        obs = result[0]
        assert obs["track_id"] == 1
        assert obs["scan_idx"] == 3
        assert obs["n_points"] == 9
        assert obs["area_km2"] == pytest.approx(9 * 0.25)

    def test_obs_contains_required_keys(self):
        n_levels, ny, nx = 5, 10, 10
        fields = _synthetic_radar_fields(n_levels, ny, nx)
        track_mask = np.zeros((ny, nx), dtype=np.int32)
        track_mask[4:7, 4:7] = 2
        z_composite = np.full((ny, nx), 35.0, dtype=np.float32)
        Z_filled = np.full((n_levels, ny, nx), 35.0, dtype=np.float32)
        z_levels = np.linspace(0, 5000, n_levels)
        eth_maps = {k: np.full((ny, nx), np.nan) for k in (20, 30, 40)}
        lat, lon = self._base_grids(ny, nx)

        result = _collect_frame_obs(
            track_mask, fields, Z_filled, z_composite,
            lat, lon, eth_maps, {}, pixel_area_km2=0.25, scan_idx=0,
        )
        obs = result[0]
        required = {
            "scan_idx", "track_id", "centroid_y", "centroid_x",
            "centroid_lat", "centroid_lon", "area_km2", "n_points",
            "ref_max_dbz", "ref_mean_dbz", "ref_p75_dbz",
            "eth_20dbz_m", "eth_30dbz_m", "eth_40dbz_m",
            "core_area_45dbz_km2", "eccentricity", "orientation_rad",
            "zdr_mean", "rhohv_mean", "kdp_mean",
        }
        assert required.issubset(obs.keys())

    def test_profile_shape(self):
        n_levels, ny, nx = 5, 10, 10
        fields = _synthetic_radar_fields(n_levels, ny, nx)
        track_mask = np.zeros((ny, nx), dtype=np.int32)
        track_mask[2:5, 2:5] = 1   # 9 pixels
        z_composite = np.full((ny, nx), 40.0, dtype=np.float32)
        Z_filled = np.full((n_levels, ny, nx), 40.0, dtype=np.float32)
        z_levels = np.linspace(0, 5000, n_levels)
        eth_maps = {k: np.full((ny, nx), np.nan) for k in (20, 30, 40)}
        lat, lon = self._base_grids(ny, nx)

        result = _collect_frame_obs(
            track_mask, fields, Z_filled, z_composite,
            lat, lon, eth_maps, {}, pixel_area_km2=0.25, scan_idx=0,
        )
        obs = result[0]
        assert obs["reflectivity_profile"].shape == (9, n_levels)  # (n_pts, n_levels)

    def test_two_cells_returns_two_obs(self):
        n_levels, ny, nx = 3, 15, 15
        fields = _synthetic_radar_fields(n_levels, ny, nx)
        track_mask = np.zeros((ny, nx), dtype=np.int32)
        track_mask[1:4, 1:4] = 1
        track_mask[9:12, 9:12] = 2
        z_composite = np.full((ny, nx), 38.0, dtype=np.float32)
        Z_filled = np.full((n_levels, ny, nx), 38.0, dtype=np.float32)
        z_levels = np.linspace(0, 3000, n_levels)
        eth_maps = {k: np.full((ny, nx), np.nan) for k in (20, 30, 40)}
        lat, lon = self._base_grids(ny, nx)

        result = _collect_frame_obs(
            track_mask, fields, Z_filled, z_composite,
            lat, lon, eth_maps, {}, pixel_area_km2=0.25, scan_idx=0,
        )
        assert len(result) == 2
        track_ids = {o["track_id"] for o in result}
        assert track_ids == {1, 2}


# ---------------------------------------------------------------------------
# _write_tracking_netcdf
# ---------------------------------------------------------------------------

class TestWriteTrackingNetcdf:
    def _make_valid_scans(self, n=2):
        base = pd.Timestamp("2022-06-08 12:00")
        return [("dummy_file.nc", base + pd.Timedelta(minutes=5 * i)) for i in range(n)]

    def _make_track_masks(self, n_scans, ny=8, nx=8):
        masks = []
        for i in range(n_scans):
            m = np.zeros((ny, nx), dtype=np.int32)
            m[2:5, 2:5] = i + 1   # cell i+1
            masks.append(m)
        return masks

    def _make_obs(self, n_scans=2, n_pts=9, n_levels=3):
        obs_list = []
        for i in range(n_scans):
            prof = np.full((n_pts, n_levels), float(i + 30), dtype=np.float32)
            obs_list.append({
                "scan_idx": i, "track_id": i + 1,
                "centroid_y": 3.0, "centroid_x": 3.0,
                "centroid_lat": _RADAR_LAT, "centroid_lon": _RADAR_LON,
                "area_km2": n_pts * 0.25, "n_points": n_pts,
                "ref_max_dbz": 40.0, "ref_mean_dbz": 38.0, "ref_p75_dbz": 39.0,
                "eth_20dbz_m": 5000.0, "eth_30dbz_m": 3000.0, "eth_40dbz_m": np.nan,
                "core_area_45dbz_km2": 0.0,
                "eccentricity": 0.1, "orientation_rad": 0.0,
                "zdr_mean": 1.5, "rhohv_mean": 0.97, "kdp_mean": 0.3,
                "reflectivity_profile": prof,
                "differential_reflectivity_profile": prof * 0.1,
                "cross_correlation_ratio_profile": np.full_like(prof, 0.97),
                "kdp_profile": np.full_like(prof, 0.3),
            })
        return obs_list

    def test_output_file_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            nc_path = Path(tmp) / "test_tracking.nc"
            n_levels = 3
            valid_scans = self._make_valid_scans(2)
            track_masks = self._make_track_masks(2)
            lat = np.full((8, 8), _RADAR_LAT, dtype=np.float32)
            lon = np.full((8, 8), _RADAR_LON, dtype=np.float32)
            z_levels = np.linspace(0, 3000, n_levels)
            obs = self._make_obs(2, n_levels=n_levels)
            _write_tracking_netcdf(nc_path, valid_scans, track_masks, obs, lat, lon, z_levels)
            assert nc_path.exists()

    def test_output_dimensions_correct(self):
        with tempfile.TemporaryDirectory() as tmp:
            nc_path = Path(tmp) / "tracking.nc"
            n_scans, ny, nx, n_levels = 3, 8, 8, 4
            valid_scans = self._make_valid_scans(n_scans)
            track_masks = self._make_track_masks(n_scans, ny, nx)
            lat = np.full((ny, nx), _RADAR_LAT, dtype=np.float32)
            lon = np.full((ny, nx), _RADAR_LON, dtype=np.float32)
            z_levels = np.linspace(0, 4000, n_levels)
            obs = self._make_obs(n_scans, n_levels=n_levels)
            _write_tracking_netcdf(nc_path, valid_scans, track_masks, obs, lat, lon, z_levels)
            with xr.open_dataset(nc_path) as ds:
                assert ds["mask"].shape == (n_scans, ny, nx)
                assert ds.sizes["scan"] == n_scans
                assert ds.sizes["obs"] == n_scans
                assert ds.sizes["level"] == n_levels

    def test_required_variables_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            nc_path = Path(tmp) / "tracking.nc"
            valid_scans = self._make_valid_scans(1)
            track_masks = self._make_track_masks(1)
            lat = np.full((8, 8), _RADAR_LAT, dtype=np.float32)
            lon = np.full((8, 8), _RADAR_LON, dtype=np.float32)
            z_levels = np.linspace(0, 3000, 3)
            obs = self._make_obs(1, n_levels=3)
            _write_tracking_netcdf(nc_path, valid_scans, track_masks, obs, lat, lon, z_levels)
            with xr.open_dataset(nc_path) as ds:
                for var in ("mask", "scan_time", "lat", "lon", "z_levels",
                            "track_id", "scan_idx", "area_km2", "n_points",
                            "ref_max_dbz", "ref_mean_dbz", "ref_p75_dbz",
                            "eth_20dbz_m", "eth_30dbz_m", "eth_40dbz_m",
                            "core_area_45dbz_km2", "eccentricity", "orientation_rad",
                            "zdr_mean", "rhohv_mean", "kdp_mean",
                            "reflectivity", "differential_reflectivity",
                            "cross_correlation_ratio", "kdp"):
                    assert var in ds, f"Missing variable: {var}"

    def test_track_ids_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            nc_path = Path(tmp) / "tracking.nc"
            valid_scans = self._make_valid_scans(2)
            track_masks = self._make_track_masks(2)
            lat = np.full((8, 8), _RADAR_LAT, dtype=np.float32)
            lon = np.full((8, 8), _RADAR_LON, dtype=np.float32)
            z_levels = np.linspace(0, 3000, 3)
            obs = self._make_obs(2, n_levels=3)
            _write_tracking_netcdf(nc_path, valid_scans, track_masks, obs, lat, lon, z_levels)
            with xr.open_dataset(nc_path) as ds:
                ids = ds["track_id"].values.tolist()
                assert ids == [1, 2]

    def test_no_obs_writes_scan_level_variables_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            nc_path = Path(tmp) / "tracking_empty.nc"
            valid_scans = self._make_valid_scans(2)
            track_masks = [np.zeros((8, 8), dtype=np.int32)] * 2
            lat = np.full((8, 8), _RADAR_LAT, dtype=np.float32)
            lon = np.full((8, 8), _RADAR_LON, dtype=np.float32)
            z_levels = np.linspace(0, 3000, 3)
            _write_tracking_netcdf(nc_path, valid_scans, track_masks, [], lat, lon, z_levels)
            assert nc_path.exists()
            with xr.open_dataset(nc_path) as ds:
                assert "mask" in ds
                assert "scan_time" in ds
