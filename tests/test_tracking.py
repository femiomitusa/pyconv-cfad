"""Tests for tracking.py: relabeling, shape stats, and TOBAC linking wrapper."""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "radar_processing"))

from radar_processing.tracking import relabel_mask, build_track_masks, collect_shape_stats


# ---------------------------------------------------------------------------
# relabel_mask
# ---------------------------------------------------------------------------

class TestRelabelMask:
    def test_empty_mapping_returns_zeros(self):
        mask = np.array([[0, 1], [2, 0]], dtype=np.int32)
        result = relabel_mask(mask, {})
        assert np.all(result == 0)
        assert result.shape == mask.shape

    def test_basic_relabeling(self):
        mask = np.array([[0, 1, 2], [1, 0, 2]], dtype=np.int32)
        result = relabel_mask(mask, {1: 10, 2: 20})
        expected = np.array([[0, 10, 20], [10, 0, 20]], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_background_zero_preserved(self):
        mask = np.array([[0, 0, 1]], dtype=np.int32)
        result = relabel_mask(mask, {1: 99})
        assert result[0, 0] == 0
        assert result[0, 1] == 0
        assert result[0, 2] == 99

    def test_noncontiguous_feature_ids(self):
        # Feature IDs 3 and 7 (non-consecutive, as TOBAC assigns globally unique IDs)
        mask = np.array([[3, 0], [0, 7]], dtype=np.int32)
        result = relabel_mask(mask, {3: 1, 7: 2})
        assert result[0, 0] == 1
        assert result[1, 1] == 2
        assert result[0, 1] == 0

    def test_unlinked_features_become_background(self):
        # Feature 2 is in the mask but was not linked by TOBAC (cell == -1)
        # and therefore absent from feature_to_cell — must become 0, not a wrong cell.
        mask = np.array([[1, 2], [0, 0]], dtype=np.int32)
        result = relabel_mask(mask, {1: 10})
        assert result[0, 0] == 10
        assert result[0, 1] == 0   # unlinked feature → background

    def test_large_unlinked_id_does_not_corrupt_linked_id(self):
        # Feature 5 is unlinked. max(feature_to_cell) == 1. Old clip approach
        # would silently map 5 → 1 → cell_id 10. Fixed version maps it to 0.
        mask = np.array([[1, 5]], dtype=np.int32)
        result = relabel_mask(mask, {1: 10})
        assert result[0, 0] == 10
        assert result[0, 1] == 0

    def test_output_dtype_is_int32(self):
        mask = np.array([[1]], dtype=np.int32)
        result = relabel_mask(mask, {1: 7})
        assert result.dtype == np.int32


# ---------------------------------------------------------------------------
# build_track_masks
# ---------------------------------------------------------------------------

class TestBuildTrackMasks:
    def _make_tracks(self, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def test_none_tracks_returns_zero_masks(self):
        masks = [np.ones((3, 3), dtype=np.int32), np.ones((3, 3), dtype=np.int32)]
        times = [pd.Timestamp("2022-06-08 12:00"), pd.Timestamp("2022-06-08 12:05")]
        result = build_track_masks(None, masks, times)
        for r in result:
            assert np.all(r == 0)

    def test_all_unassigned_returns_zero_masks(self):
        t0 = pd.Timestamp("2022-06-08 12:00")
        mask = np.array([[0, 1], [0, 0]], dtype=np.int32)
        tracks = self._make_tracks([
            {"time": t0.to_datetime64(), "feature": 1, "cell": -1},
        ])
        result = build_track_masks(tracks, [mask], [t0])
        assert np.all(result[0] == 0)

    def test_single_scan_single_cell(self):
        t0 = pd.Timestamp("2022-06-08 12:00")
        mask = np.array([[0, 1, 1], [0, 0, 0]], dtype=np.int32)
        tracks = self._make_tracks([
            {"time": t0.to_datetime64(), "feature": 1, "cell": 42},
        ])
        result = build_track_masks(tracks, [mask], [t0])
        assert result[0][0, 1] == 42
        assert result[0][0, 2] == 42
        assert result[0][0, 0] == 0

    def test_persistent_id_across_two_scans(self):
        t0 = pd.Timestamp("2022-06-08 12:00")
        t1 = pd.Timestamp("2022-06-08 12:05")
        # Feature 1 at t0 and feature 2 at t1 both belong to cell 7
        mask0 = np.array([[0, 1]], dtype=np.int32)
        mask1 = np.array([[2, 0]], dtype=np.int32)
        tracks = self._make_tracks([
            {"time": t0.to_datetime64(), "feature": 1, "cell": 7},
            {"time": t1.to_datetime64(), "feature": 2, "cell": 7},
        ])
        result = build_track_masks(tracks, [mask0, mask1], [t0, t1])
        assert result[0][0, 1] == 7
        assert result[1][0, 0] == 7


# ---------------------------------------------------------------------------
# collect_shape_stats
# ---------------------------------------------------------------------------

class TestCollectShapeStats:
    def test_circular_cell_low_eccentricity(self):
        # A filled square is close to circular — eccentricity should be < 0.9
        mask = np.zeros((20, 20), dtype=np.int32)
        mask[7:13, 7:13] = 1      # 6×6 square ≈ circular
        stats = collect_shape_stats(mask)
        assert 1 in stats
        assert stats[1]["eccentricity"] < 0.9

    def test_elongated_cell_high_eccentricity(self):
        mask = np.zeros((20, 20), dtype=np.int32)
        mask[9, 2:18] = 1          # single-row strip — very elongated
        stats = collect_shape_stats(mask)
        assert 1 in stats
        assert stats[1]["eccentricity"] > 0.9

    def test_multiple_cells(self):
        mask = np.zeros((20, 20), dtype=np.int32)
        mask[2:5, 2:5] = 1
        mask[12:15, 12:15] = 2
        stats = collect_shape_stats(mask)
        assert set(stats.keys()) == {1, 2}

    def test_empty_mask_returns_empty_dict(self):
        mask = np.zeros((10, 10), dtype=np.int32)
        assert collect_shape_stats(mask) == {}

    def test_orientation_rad_in_range(self):
        mask = np.zeros((20, 20), dtype=np.int32)
        mask[8:12, 5:15] = 1
        stats = collect_shape_stats(mask)
        assert -np.pi / 2 <= stats[1]["orientation_rad"] <= np.pi / 2
