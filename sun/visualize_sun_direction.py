"""Visualize sun direction as an arrow overlaid on Argoverse 2 camera images.

Given a metadata pickle (produced by mp-process av2) and a city code, this
script:
  1. Loads image paths, camera poses (cam2world, OpenGL convention), and
     intrinsics from the metadata.
  2. Computes the sun direction vector for each frame using the original
     nanosecond timestamp extracted from the image filename.
  3. Projects the sun direction into each camera to obtain a 2D vanishing
     point, then draws an arrow from the image center toward that point.

Usage:
    python -m map4d.scripts.visualize_sun_direction \
        --metadata data/Argoverse2/metadata_<seq>.pkl \
        --city PIT \
        --output-dir outputs/sun_vis \
        --max-images 20
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
from nerfstudio.utils.rich_utils import CONSOLE
from tqdm import tqdm

import tyro

from map4d.common.sun_position import (
    compute_sun_direction_world_frame,
    get_av2_city_location,
)
from map4d.data.parser.typing import ImageInfo

# OpenGL ↔ OpenCV conversion (flip Y and Z)
_OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0])


def _project_sun_direction(
    sun_dir_world: np.ndarray,
    cam2world: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray | None, bool]:
    """Project the sun direction into pixel coordinates.

    Args:
        sun_dir_world: (3,) unit vector in world frame pointing toward the sun.
        cam2world: (4, 4) camera-to-world matrix (OpenGL convention).
        intrinsics: (3, 3) camera intrinsic matrix.

    Returns:
        (pixel_xy, in_front): pixel_xy is (2,) projected pixel position of the
        sun vanishing point, or None if degenerate.  in_front is True when the
        sun is in front of the camera (positive depth in OpenCV convention).
    """
    # world2cam rotation (OpenGL)
    R_w2c_gl = cam2world[:3, :3].T  # transpose = inverse for rotation

    # Direction in camera space (OpenGL: +X right, +Y up, -Z forward)
    d_gl = R_w2c_gl @ sun_dir_world

    # Convert to OpenCV camera convention (+X right, +Y down, +Z forward)
    d_cv = _OPENGL_TO_OPENCV @ d_gl

    in_front = d_cv[2] > 0

    # Avoid division by near-zero
    if abs(d_cv[2]) < 1e-8:
        return None, in_front

    # Project direction (vanishing point = K @ d / d_z)
    p_homo = intrinsics @ d_cv
    px = p_homo[0] / p_homo[2]
    py = p_homo[1] / p_homo[2]

    return np.array([px, py]), in_front


def _draw_sun_arrow(
    image: np.ndarray,
    center: tuple[int, int],
    target: np.ndarray,
    in_front: bool,
    elevation_deg: float,
    azimuth_deg: float,
    arrow_length: int = 200,
) -> np.ndarray:
    """Draw a sun-direction arrow and annotation on the image.

    Args:
        image: HWC uint8 image (BGR for OpenCV).
        center: (cx, cy) arrow start in pixels.
        target: (2,) vanishing point in pixels (may be outside image).
        in_front: Whether the sun is in front of the camera.
        elevation_deg: Sun elevation for annotation.
        azimuth_deg: Sun azimuth for annotation.
        arrow_length: Length of the drawn arrow in pixels.

    Returns:
        Annotated image.
    """
    img = image.copy()
    cx, cy = center

    # Direction from center to vanishing point
    dx = target[0] - cx
    dy = target[1] - cy
    norm = np.sqrt(dx * dx + dy * dy)
    if norm < 1e-3:
        return img

    # Normalise and scale to fixed arrow length
    dx_n = dx / norm * arrow_length
    dy_n = dy / norm * arrow_length

    if not in_front:
        # Sun is behind camera → flip arrow direction, use dashed style
        dx_n = -dx_n
        dy_n = -dy_n

    end_x = int(cx + dx_n)
    end_y = int(cy + dy_n)

    # Colours: yellow if in front, dim grey if behind
    colour = (0, 220, 255) if in_front else (120, 120, 120)
    thickness = 3 if in_front else 2

    cv2.arrowedLine(img, (cx, cy), (end_x, end_y), colour, thickness, tipLength=0.25)

    # Small circle at center
    cv2.circle(img, (cx, cy), 6, colour, -1)

    # Text annotation
    label = f"Az {azimuth_deg:.1f}  El {elevation_deg:.1f}"
    if not in_front:
        label += " (behind)"
    # Position text near the arrow tip
    txt_x = max(10, min(end_x + 10, img.shape[1] - 250))
    txt_y = max(30, min(end_y - 10, img.shape[0] - 10))
    cv2.putText(
        img,
        label,
        (txt_x, txt_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        colour,
        2,
        cv2.LINE_AA,
    )

    # Sun symbol (small circle) at arrow tip
    cv2.circle(img, (end_x, end_y), 10, colour, 2)

    return img


@dataclass
class VisualizeSunDirection:
    """Overlay sun-direction arrows on Argoverse 2 camera images."""

    metadata: Path
    """Path to the metadata pickle (e.g. data/Argoverse2/metadata_<seq>.pkl)."""
    capture_utc: str = "2024-06-21 17:00"
    """Approximate capture time in UTC, format 'YYYY-MM-DD HH:MM'.
    AV2 timestamps are hardware clock values, NOT Unix timestamps.
    Provide the real-world UTC time when this log was captured.
    Within a log (~16 s), relative offsets between frames are applied
    automatically, so ±30 min accuracy is sufficient."""
    city: str = "PIT"
    """AV2 city code for geographic reference (ATX/DTW/MIA/PAO/PIT/WDC)."""
    output_dir: Path = Path("outputs/sun_vis")
    """Directory to save annotated images."""
    max_images: int = 20
    """Maximum number of images to process (0 = all)."""
    cam_ids: tuple[int, ...] | None = None
    """Restrict to these camera IDs.  None = all cameras."""
    arrow_length: int = 200
    """Arrow length in pixels."""

    def main(self) -> None:
        # ── Load metadata ────────────────────────────────────────────────
        CONSOLE.log(f"Loading metadata from {self.metadata}")
        with open(self.metadata, "rb") as f:
            images, _annotations, _bounds, _pcs, _scale = pickle.load(f)
        images: list[ImageInfo]

        # ── Geographic reference ─────────────────────────────────────────
        lat, lon, elev = get_av2_city_location(self.city)
        CONSOLE.log(f"City {self.city}: lat={lat:.4f}, lon={lon:.4f}, elev={elev:.0f} m")

        # ── Parse capture UTC ────────────────────────────────────────────
        base_utc = datetime.strptime(self.capture_utc, "%Y-%m-%d %H:%M").replace(
            tzinfo=timezone.utc
        )
        CONSOLE.log(f"Using capture base UTC: [bold]{base_utc.isoformat()}[/bold]")

        # ── Filter images ────────────────────────────────────────────
        if self.cam_ids is not None:
            images = [im for im in images if im.cam_id in self.cam_ids]
        if self.max_images > 0:
            images = images[: self.max_images]

        CONSOLE.log(f"Processing {len(images)} images …")
        os.makedirs(self.output_dir, exist_ok=True)

        # Collect all AV2 timestamps to compute relative offsets
        all_ts = []
        for im_info in images:
            fname = Path(str(im_info.image_path)).stem
            try:
                all_ts.append(int(fname))
            except ValueError:
                all_ts.append(None)

        valid_ts = [t for t in all_ts if t is not None]
        t0_ns = min(valid_ts) if valid_ts else 0

        for im_info, ts_val in zip(images, all_ts):
            img_path = str(im_info.image_path)
            if not os.path.exists(img_path):
                CONSOLE.log(f"[yellow]Image not found: {img_path}")
                continue

            if ts_val is None:
                CONSOLE.log(f"[yellow]Cannot parse timestamp from {Path(img_path).stem}")
                continue
            timestamp_ns = ts_val

            # Compute UTC datetime from relative offset
            dt_utc = base_utc + timedelta(seconds=(timestamp_ns - t0_ns) / 1e9)

            # Compute sun direction in world frame
            result, sun_dir_world = compute_sun_direction_world_frame(
                latitude=lat,
                longitude=lon,
                dt_utc=dt_utc,
                elevation_m=elev,
            )

            # Project into camera
            proj, in_front = _project_sun_direction(
                sun_dir_world, im_info.pose, im_info.intrinsics
            )
            if proj is None:
                continue

            # Load and annotate image
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            cx, cy = w // 2, h // 2

            img_annotated = _draw_sun_arrow(
                img,
                center=(cx, cy),
                target=proj,
                in_front=in_front,
                elevation_deg=result.elevation_deg,
                azimuth_deg=result.azimuth_deg,
                arrow_length=self.arrow_length,
            )

            # Save
            out_name = f"frame{im_info.frame_id:04d}_cam{im_info.cam_id}_{fname}.jpg"
            out_path = self.output_dir / out_name
            cv2.imwrite(str(out_path), img_annotated)

        CONSOLE.log(f"[bold green]Saved annotated images to {self.output_dir}")


def entrypoint() -> None:
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(VisualizeSunDirection).main()


if __name__ == "__main__":
    entrypoint()
