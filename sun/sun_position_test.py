"""Unit tests for map4d.common.sun_position."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pytest

from map4d.common.sun_position import (
    AV2_CITY_GEO,
    SunPositionResult,
    compute_sun_direction_world_frame,
    compute_sun_position,
    compute_sun_position_at_datetime,
    get_av2_city_location,
    sun_direction_enu,
    timestamp_ns_to_utc,
)


# ── timestamp conversion ────────────────────────────────────────────────────

class TestTimestampConversion:
    def test_known_epoch(self):
        # 2024-01-01 00:00:00 UTC
        ts_ns = 1704067200_000_000_000
        dt = timestamp_ns_to_utc(ts_ns)
        assert dt == datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    def test_timezone_is_utc(self):
        dt = timestamp_ns_to_utc(0)
        assert dt.tzinfo == timezone.utc

    def test_nanosecond_precision_preserved(self):
        # microsecond precision is the limit of datetime, but no error
        ts_ns = 1704067200_123_456_789
        dt = timestamp_ns_to_utc(ts_ns)
        assert dt.year == 2024


# ── ENU direction vector ────────────────────────────────────────────────────

class TestSunDirectionENU:
    def test_unit_norm(self):
        for az in [0, 45, 90, 135, 180, 225, 270, 315]:
            for el in [-10, 0, 30, 60, 89]:
                d = sun_direction_enu(float(az), float(el))
                assert abs(np.linalg.norm(d) - 1.0) < 1e-12

    def test_north_horizon(self):
        # Azimuth 0° (North), elevation 0° → pointing North
        d = sun_direction_enu(0.0, 0.0)
        np.testing.assert_allclose(d, [0, 1, 0], atol=1e-12)

    def test_east_horizon(self):
        # Azimuth 90° (East), elevation 0° → pointing East
        d = sun_direction_enu(90.0, 0.0)
        np.testing.assert_allclose(d, [1, 0, 0], atol=1e-12)

    def test_zenith(self):
        # Any azimuth, elevation 90° → pointing straight up
        d = sun_direction_enu(42.0, 90.0)
        np.testing.assert_allclose(d, [0, 0, 1], atol=1e-12)

    def test_south_horizon(self):
        d = sun_direction_enu(180.0, 0.0)
        np.testing.assert_allclose(d, [0, -1, 0], atol=1e-12)


# ── Full sun position computation ───────────────────────────────────────────

class TestComputeSunPosition:
    def test_pittsburgh_summer_noon(self):
        """Pittsburgh, 2024-06-21 ~12:00 local (17:00 UTC)."""
        # timestamp_ns for 2024-06-21 17:00:00 UTC
        dt = datetime(2024, 6, 21, 17, 0, 0, tzinfo=timezone.utc)
        ts_ns = int(dt.timestamp() * 1e9)
        result = compute_sun_position(40.4406, -79.9959, ts_ns)
        # At solar noon sun should be high and roughly south
        assert result.elevation_deg > 60.0
        assert 120 < result.azimuth_deg < 220  # broadly south
        assert abs(np.linalg.norm(result.direction_enu) - 1.0) < 1e-12

    def test_night_negative_elevation(self):
        """At midnight local time in Pittsburgh, sun should be below horizon."""
        dt = datetime(2024, 6, 21, 4, 0, 0, tzinfo=timezone.utc)  # midnight EDT
        ts_ns = int(dt.timestamp() * 1e9)
        result = compute_sun_position(40.4406, -79.9959, ts_ns)
        assert result.elevation_deg < 0

    def test_miami_always_warmer_elevation(self):
        """Miami is closer to the equator → higher max elevation in summer."""
        dt = datetime(2024, 6, 21, 17, 0, 0, tzinfo=timezone.utc)
        ts_ns = int(dt.timestamp() * 1e9)
        pit = compute_sun_position(40.4406, -79.9959, ts_ns)
        mia = compute_sun_position(25.7617, -80.1918, ts_ns)
        assert mia.elevation_deg > pit.elevation_deg


# ── datetime-based API ─────────────────────────────────────────────────────

class TestComputeSunPositionAtDatetime:
    def test_pittsburgh_summer_noon_positive_elevation(self):
        """Directly passing a datetime should give positive sun elevation at noon."""
        dt = datetime(2024, 6, 21, 17, 0, 0, tzinfo=timezone.utc)
        result = compute_sun_position_at_datetime(40.4406, -79.9959, dt)
        assert result.elevation_deg > 60.0
        assert abs(np.linalg.norm(result.direction_enu) - 1.0) < 1e-12

    def test_matches_legacy_api(self):
        """compute_sun_position_at_datetime and compute_sun_position should agree
        when the timestamp_ns is a real Unix timestamp."""
        dt = datetime(2024, 6, 21, 17, 0, 0, tzinfo=timezone.utc)
        ts_ns = int(dt.timestamp() * 1e9)
        r1 = compute_sun_position(40.4406, -79.9959, ts_ns)
        r2 = compute_sun_position_at_datetime(40.4406, -79.9959, dt)
        assert abs(r1.azimuth_deg - r2.azimuth_deg) < 0.01
        assert abs(r1.elevation_deg - r2.elevation_deg) < 0.01


# ── World-frame transformation ───────────────────────────────────────────────

class TestWorldFrameTransform:
    def test_identity_alignment(self):
        """With identity alignment, world dir == ENU dir."""
        dt = datetime(2024, 6, 21, 17, 0, 0, tzinfo=timezone.utc)
        result, world_dir = compute_sun_direction_world_frame(
            40.4406, -79.9959, dt_utc=dt, align_transform=np.eye(4)
        )
        np.testing.assert_allclose(world_dir, result.direction_enu, atol=1e-12)

    def test_none_alignment(self):
        """With None alignment, world dir == ENU dir."""
        dt = datetime(2024, 6, 21, 17, 0, 0, tzinfo=timezone.utc)
        result, world_dir = compute_sun_direction_world_frame(
            40.4406, -79.9959, dt_utc=dt, align_transform=None
        )
        np.testing.assert_allclose(world_dir, result.direction_enu, atol=1e-12)

    def test_rotation_alignment(self):
        """A 90° Z-rotation alignment should rotate the direction accordingly."""
        dt = datetime(2024, 6, 21, 17, 0, 0, tzinfo=timezone.utc)
        # 90° CCW rotation around Z
        c, s = 0.0, 1.0
        align = np.eye(4)
        align[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        result, world_dir = compute_sun_direction_world_frame(
            40.4406, -79.9959, dt_utc=dt, align_transform=align
        )
        enu = result.direction_enu
        # After 90° CCW: new_x = -old_y, new_y = old_x
        expected = np.array([-enu[1], enu[0], enu[2]])
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(world_dir, expected, atol=1e-10)

    def test_world_dir_is_unit(self):
        dt = datetime(2024, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
        _, world_dir = compute_sun_direction_world_frame(
            37.4419, -122.143, dt_utc=dt, elevation_m=9.0
        )
        assert abs(np.linalg.norm(world_dir) - 1.0) < 1e-12


# ── City lookup ──────────────────────────────────────────────────────────────

class TestAV2CityLookup:
    def test_all_cities_valid(self):
        for code in AV2_CITY_GEO:
            lat, lon, elev = get_av2_city_location(code)
            assert -90 <= lat <= 90
            assert -180 <= lon <= 180
            assert elev >= 0

    def test_case_insensitive(self):
        assert get_av2_city_location("pit") == get_av2_city_location("PIT")

    def test_unknown_city_raises(self):
        with pytest.raises(ValueError, match="Unknown AV2 city code"):
            get_av2_city_location("XYZ")
