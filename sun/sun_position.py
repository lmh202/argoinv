"""Compute the sun direction vector given geographic location and UTC timestamp.

This module is independent of the map4d training/rendering pipeline.
It uses the `astral` library for high-precision solar position calculation
and outputs a unit direction vector pointing from the observer toward the sun
in the ENU (East-North-Up) local geographic coordinate system.

Coordinate convention (ENU):
    +X = East
    +Y = North
    +Z = Up

The world frame used by this project (see docs/DOCS.md) is Z-up with XY
parallel to the ground, which matches ENU up to a rotation around Z (the
heading offset between the city frame X-axis and geographic East).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from astral import Observer
from astral import sun as astral_sun

# ── Argoverse 2 city geographic references ──────────────────────────────────
# Approximate city-center coordinates used when no per-log GPS is available.
# These are sufficient for sun-direction computation because sun azimuth/
# elevation change by < 0.01° per km of horizontal displacement.
#
# City codes: ATX=Austin, DTW=Detroit, MIA=Miami, PAO=Palo Alto, PIT=Pittsburgh, WDC=Washington DC
AV2_CITY_GEO: dict[str, tuple[float, float, float]] = {
    "ATX": (30.2672, -97.7431, 149.0),   # Austin, TX
    "DTW": (42.3314, -83.0458, 189.0),   # Detroit, MI
    "MIA": (25.7617, -80.1918, 2.0),     # Miami, FL
    "PAO": (37.4419, -122.1430, 9.0),    # Palo Alto, CA
    "PIT": (40.4406, -79.9959, 367.0),   # Pittsburgh, PA
    "WDC": (38.9072, -77.0369, 22.0),    # Washington, DC
}

# nuScenes map geographic references (approximate map centers).
# Location keys from v1.0 log.json.
NUSCENES_LOCATION_GEO: dict[str, tuple[float, float, float]] = {
    "boston-seaport": (42.3368, -71.0578, 6.0),
    "singapore-onenorth": (1.2882, 103.7848, 15.0),
    "singapore-hollandvillage": (1.2994, 103.7822, 15.0),
    "singapore-queenstown": (1.2783, 103.7674, 15.0),
}


@dataclass
class SunPositionResult:
    """Result of a sun position computation for a single instant."""

    azimuth_deg: float
    """Sun azimuth in degrees, measured clockwise from geographic North."""
    elevation_deg: float
    """Sun elevation in degrees above the horizon (negative = below)."""
    direction_enu: np.ndarray
    """Unit vector pointing toward the sun in ENU coordinates (3,)."""
    timestamp_utc: datetime
    """The UTC datetime used for the computation."""
    latitude: float
    longitude: float
    elevation_m: float


def timestamp_ns_to_utc(timestamp_ns: int) -> datetime:
    """Convert a nanosecond-precision Unix timestamp to a UTC datetime."""
    seconds = timestamp_ns / 1e9
    return datetime.fromtimestamp(seconds, tz=timezone.utc)


def sun_direction_enu(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Convert solar azimuth & elevation to a unit vector in ENU frame.

    Azimuth is measured clockwise from North (0°=N, 90°=E, 180°=S, 270°=W).
    Elevation is the angle above the horizon.

    Returns:
        np.ndarray of shape (3,): [East, North, Up] unit vector pointing
        toward the sun.
    """
    az_rad = math.radians(azimuth_deg)
    el_rad = math.radians(elevation_deg)
    east = math.sin(az_rad) * math.cos(el_rad)
    north = math.cos(az_rad) * math.cos(el_rad)
    up = math.sin(el_rad)
    return np.array([east, north, up], dtype=np.float64)


def compute_sun_position_at_datetime(
    latitude: float,
    longitude: float,
    dt_utc: datetime,
    elevation_m: float = 0.0,
) -> SunPositionResult:
    """Compute the sun direction for a given location and UTC datetime.

    Args:
        latitude: Observer latitude in degrees.
        longitude: Observer longitude in degrees.
        dt_utc: UTC datetime (must be timezone-aware).
        elevation_m: Observer elevation in meters above sea level.

    Returns:
        SunPositionResult with azimuth, elevation, and ENU direction vector.
    """
    observer = Observer(latitude=latitude, longitude=longitude, elevation=elevation_m)
    azimuth = astral_sun.azimuth(observer, dt_utc)
    elevation = astral_sun.elevation(observer, dt_utc)
    direction = sun_direction_enu(azimuth, elevation)
    return SunPositionResult(
        azimuth_deg=azimuth,
        elevation_deg=elevation,
        direction_enu=direction,
        timestamp_utc=dt_utc,
        latitude=latitude,
        longitude=longitude,
        elevation_m=elevation_m,
    )


def compute_sun_position(
    latitude: float,
    longitude: float,
    timestamp_ns: int,
    elevation_m: float = 0.0,
) -> SunPositionResult:
    """Compute sun direction from a Unix nanosecond timestamp.

    .. warning::
       Argoverse 2 ``timestamp_ns`` values are NOT Unix timestamps.
       Use :func:`compute_sun_position_at_datetime` with an explicit
       UTC datetime instead.
    """
    dt_utc = timestamp_ns_to_utc(timestamp_ns)
    return compute_sun_position_at_datetime(latitude, longitude, dt_utc, elevation_m)


def compute_sun_direction_world_frame(
    latitude: float,
    longitude: float,
    dt_utc: datetime,
    elevation_m: float = 0.0,
    city_rotation_from_enu: np.ndarray | None = None,
    align_transform: np.ndarray | None = None,
) -> tuple[SunPositionResult, np.ndarray]:
    """Compute the sun direction in the project world frame.

    The project world frame is Z-up and aligned with the Argoverse city frame,
    optionally post-multiplied by an alignment transform for multi-sequence
    setups.

    If ``city_rotation_from_enu`` is None, we assume the city frame axes are
    aligned with ENU (East→+X, North→+Y, Up→+Z).  This is a reasonable
    approximation for Argoverse 2 whose city frames are UTM-derived local
    Cartesian coordinates.

    Args:
        latitude: Observer latitude in degrees.
        longitude: Observer longitude in degrees.
        dt_utc: Timezone-aware UTC datetime.
        elevation_m: Elevation in metres above sea level.
        city_rotation_from_enu: Optional 3×3 rotation matrix that maps ENU
            vectors into city-frame vectors.  If None, identity is assumed.
        align_transform: Optional 4×4 alignment matrix applied *after* the
            city-frame transform (rotation part only is used for directions).

    Returns:
        A tuple of (SunPositionResult, world_direction) where
        ``world_direction`` is a (3,) unit vector in the project world frame
        pointing toward the sun.
    """
    result = compute_sun_position_at_datetime(latitude, longitude, dt_utc, elevation_m)
    direction = result.direction_enu.copy()

    # ENU → city frame
    if city_rotation_from_enu is not None:
        direction = city_rotation_from_enu @ direction

    # city frame → world frame (apply rotation part of alignment only)
    if align_transform is not None:
        R = align_transform[:3, :3]
        direction = R @ direction

    # Re-normalise to guard against accumulated floating-point error
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm

    return result, direction


def get_av2_city_location(city_code: str) -> tuple[float, float, float]:
    """Return (latitude, longitude, elevation_m) for an Argoverse 2 city code.

    Raises:
        ValueError: If the city code is not recognised.
    """
    city_code = city_code.upper()
    if city_code not in AV2_CITY_GEO:
        raise ValueError(
            f"Unknown AV2 city code '{city_code}'. "
            f"Valid codes: {sorted(AV2_CITY_GEO.keys())}"
        )
    return AV2_CITY_GEO[city_code]


def get_nuscenes_location(location: str) -> tuple[float, float, float]:
    """Return (latitude, longitude, elevation_m) for a nuScenes location key."""
    key = location.lower()
    if key not in NUSCENES_LOCATION_GEO:
        raise ValueError(
            f"Unknown nuScenes location '{location}'. "
            f"Valid keys: {sorted(NUSCENES_LOCATION_GEO.keys())}"
        )
    return NUSCENES_LOCATION_GEO[key]
