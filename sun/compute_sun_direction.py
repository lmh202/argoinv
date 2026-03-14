"""Compute sun direction for Argoverse 2 sequences.

This script reads AV2 log data (timestamps + ego poses) and outputs, for each
lidar frame, the sun direction vector in the project world coordinate frame.
It is completely independent of the map4d training / rendering pipeline.

Usage (after registering the entry-point, see pyproject.toml):
    mp-sun av2 --log-id <LOG_ID> --data data/Argoverse2/

Or run directly:
    python -m map4d.scripts.compute_sun_direction av2 --log-id <LOG_ID>
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Tuple, Union

import numpy as np
from nerfstudio.utils.rich_utils import CONSOLE
from typing_extensions import Annotated

try:
    import av2.utils.io as io_utils
    from av2.datasets.sensor.av2_sensor_dataloader import (
        AV2SensorDataLoader,
        convert_pose_dataframe_to_SE3,
    )
except ImportError:
    av2 = None  # type: ignore

import tyro

from map4d.common.sun_position import (
    SunPositionResult,
    compute_sun_direction_world_frame,
    get_av2_city_location,
)


@dataclass
class ComputeSunDirectionAV2:
    """Compute per-frame sun direction for an Argoverse 2 sequence."""

    data: Path = Path("data/Argoverse2/")
    """Root directory of the Argoverse 2 dataset."""
    city: str = "PIT"
    """AV2 city code: one of ATX, DTW, MIA, PAO, PIT, WDC."""
    log_id: str | None = "0c61aea3-3cba-35f3-8971-df42cd5b9b1a"
    """Log ID to process. Required when location_aabb is not set."""
    location_aabb: tuple[int, int, int, int] | None = None
    """Optional location AABB (xmin, ymin, xmax, ymax) for multi-log filtering."""
    split: Literal["train", "val", "test"] = "train"
    """Dataset split."""
    apply_alignment: bool = True
    """Whether to apply multi-sequence alignment (rotation only) to the sun direction."""
    alignment_path: Path = Path("assets/")
    """Path to alignment transform files."""
    capture_utc: str = "2024-06-21 17:00"
    """Approximate capture time in UTC, format 'YYYY-MM-DD HH:MM'.
    AV2 timestamps are hardware clock values, NOT Unix timestamps.
    Provide the real-world UTC time when this log was captured.
    Within a log (~16 s), relative offsets between frames are applied
    automatically, so ±30 min accuracy is sufficient."""
    output: Path | None = None
    """Output JSON path. Defaults to <data>/sun_direction_<seq_name>.json"""
    cameras: Tuple[str, ...] = (
        "ring_front_center",
        "ring_front_left",
        "ring_front_right",
        "ring_rear_left",
        "ring_rear_right",
        "ring_side_left",
        "ring_side_right",
    )
    """Cameras to include when generating per-camera-frame results."""
    per_camera: bool = False
    """If True, also output sun direction for each camera timestamp (in
    addition to the lidar timestamps)."""

    def main(self) -> None:
        if av2 is None:
            CONSOLE.log(
                "[bold red]AV2 API is not installed. "
                "Please install it with `pip install git+https://github.com/tobiasfshr/av2-api.git`."
            )
            return

        # ── Resolve sequence name & log ids ──────────────────────────────
        if self.location_aabb is not None:
            seq_name = self.city + "_" + "_".join(map(str, self.location_aabb))
        elif self.log_id is not None:
            seq_name = self.log_id
        else:
            CONSOLE.log("[bold red]Either --log-id or --location-aabb must be provided.")
            return

        loader = AV2SensorDataLoader(data_dir=self.data / self.split, labels_dir=self.data)

        # ── Get log ids & timestamps ─────────────────────────────────────
        CONSOLE.log("Collecting log ids and timestamps …")
        split_log_ids, timestamps_per_log = self._get_logs_timestamps(loader)

        if not split_log_ids:
            CONSOLE.log("[bold red]No matching logs found.")
            return

        # ── Resolve alignment transforms ─────────────────────────────────
        transforms_per_log = self._get_alignment_transforms(seq_name, split_log_ids)

        # ── Geographic reference ─────────────────────────────────────────
        lat, lon, elev = get_av2_city_location(self.city)
        CONSOLE.log(
            f"City [bold]{self.city}[/bold]: lat={lat:.4f}, lon={lon:.4f}, "
            f"elev={elev:.0f} m  (city-center approximation)"
        )

        # ── Parse capture UTC ────────────────────────────────────────────
        base_utc = datetime.strptime(self.capture_utc, "%Y-%m-%d %H:%M").replace(
            tzinfo=timezone.utc
        )
        CONSOLE.log(f"Using capture base UTC: [bold]{base_utc.isoformat()}[/bold]")

        # ── Per-frame computation ────────────────────────────────────────
        results: list[dict] = []
        for log, timestamps, align_transform in zip(
            split_log_ids, timestamps_per_log, transforms_per_log
        ):
            log_city = loader.get_city_name(log)
            CONSOLE.log(f"Processing log [bold]{log}[/bold] (city={log_city}, frames={len(timestamps)})")

            align_rot = align_transform if self.apply_alignment else None

            # Use relative offsets from the first timestamp within this log
            t0_ns = timestamps[0]

            for frame_id, ts_ns in enumerate(timestamps):
                dt_utc = base_utc + timedelta(seconds=(ts_ns - t0_ns) / 1e9)
                result, world_dir = compute_sun_direction_world_frame(
                    latitude=lat,
                    longitude=lon,
                    dt_utc=dt_utc,
                    elevation_m=elev,
                    city_rotation_from_enu=None,  # AV2 city frame ≈ ENU
                    align_transform=align_rot,
                )
                entry = self._make_entry(
                    log_id=log,
                    frame_id=frame_id,
                    timestamp_ns=ts_ns,
                    result=result,
                    world_dir=world_dir,
                    cam_name=None,
                )
                results.append(entry)

                # Optionally include per-camera timestamps
                if self.per_camera:
                    for camera in self.cameras:
                        img_fpath = loader.get_closest_img_fpath(log, camera, ts_ns)
                        if img_fpath is None:
                            continue
                        cam_ts_ns = int(img_fpath.parts[-1].replace(".jpg", ""))
                        if cam_ts_ns == ts_ns:
                            # Same timestamp → same direction, skip duplicate
                            continue
                        cam_dt_utc = base_utc + timedelta(seconds=(cam_ts_ns - t0_ns) / 1e9)
                        cam_result, cam_world_dir = compute_sun_direction_world_frame(
                            latitude=lat,
                            longitude=lon,
                            dt_utc=cam_dt_utc,
                            elevation_m=elev,
                            city_rotation_from_enu=None,
                            align_transform=align_rot,
                        )
                        results.append(
                            self._make_entry(
                                log_id=log,
                                frame_id=frame_id,
                                timestamp_ns=cam_ts_ns,
                                result=cam_result,
                                world_dir=cam_world_dir,
                                cam_name=camera,
                            )
                        )

        # ── Save ─────────────────────────────────────────────────────────
        out_path = self.output or (self.data / f"sun_direction_{seq_name}.json")
        os.makedirs(out_path.parent, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        CONSOLE.log(f"[bold green]Saved {len(results)} sun direction entries to {out_path}")

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _make_entry(
        log_id: str,
        frame_id: int,
        timestamp_ns: int,
        result: SunPositionResult,
        world_dir: np.ndarray,
        cam_name: str | None,
    ) -> dict:
        return {
            "log_id": log_id,
            "frame_id": frame_id,
            "timestamp_ns": timestamp_ns,
            "timestamp_utc": result.timestamp_utc.isoformat(),
            "azimuth_deg": round(result.azimuth_deg, 4),
            "elevation_deg": round(result.elevation_deg, 4),
            "sun_direction_enu": [round(float(v), 6) for v in result.direction_enu],
            "sun_direction_world": [round(float(v), 6) for v in world_dir],
            "camera": cam_name,
        }

    def _get_logs_timestamps(
        self, loader: "AV2SensorDataLoader"
    ) -> tuple[list[str], list[list[int]]]:
        """Collect log ids and per-log timestamps, filtered by city / AABB."""
        log_ids = loader.get_log_ids()
        split_log_ids: list[str] = []
        timestamps_per_log: list[list[int]] = []

        for log in log_ids:
            log_city = loader.get_city_name(log)
            if self.city not in ("None", log_city):
                continue
            if self.log_id is not None and self.location_aabb is None and log != self.log_id:
                continue

            log_poses_df = io_utils.read_feather(
                self.data / self.split / log / "city_SE3_egovehicle.feather"
            )
            timestamps = loader.get_ordered_log_lidar_timestamps(log)
            used: list[int] = []

            for ts in timestamps:
                if self.location_aabb is not None:
                    pose_df = log_poses_df.loc[log_poses_df["timestamp_ns"] == ts]
                    if len(pose_df) == 0:
                        continue
                    pose = convert_pose_dataframe_to_SE3(pose_df)
                    tx, ty, _ = pose.translation
                    if not (
                        self.location_aabb[0] < tx < self.location_aabb[2]
                        and self.location_aabb[1] < ty < self.location_aabb[3]
                    ):
                        continue
                used.append(ts)

            if used:
                split_log_ids.append(log)
                timestamps_per_log.append(used)

        return split_log_ids, timestamps_per_log

    def _get_alignment_transforms(
        self, seq_name: str, log_ids: list[str]
    ) -> list[np.ndarray]:
        """Load or generate identity alignment transforms."""
        if len(log_ids) > 1 and self.apply_alignment:
            tf_path = self.alignment_path / f"{seq_name}_transforms.json"
            if tf_path.exists():
                with open(tf_path, "r") as f:
                    tfd = json.load(f)
                return [np.array(tfd[log]) for log in log_ids]
            else:
                CONSOLE.log(
                    f"[yellow]Alignment file {tf_path} not found; using identity transforms."
                )
        return [np.eye(4) for _ in log_ids]


# ── CLI ──────────────────────────────────────────────────────────────────────

Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[ComputeSunDirectionAV2, tyro.conf.subcommand(name="av2")],
    ]
]


def entrypoint() -> None:
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()
