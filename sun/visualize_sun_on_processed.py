"""Visualize scene-level sun direction on processed dataset images."""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import cv2
import numpy as np


def project_direction_to_image(
    sun_dir_world: np.ndarray,
    cam2world: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray | None, bool]:
    # world -> camera
    d_cam = cam2world[:3, :3].T @ sun_dir_world
    in_front = d_cam[2] > 1e-8
    if abs(d_cam[2]) < 1e-8:
        return None, in_front
    px = intrinsics[0, 0] * (d_cam[0] / d_cam[2]) + intrinsics[0, 2]
    py = intrinsics[1, 1] * (d_cam[1] / d_cam[2]) + intrinsics[1, 2]
    return np.array([px, py], dtype=np.float64), in_front


def draw_arrow(image: np.ndarray, target: np.ndarray, in_front: bool) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    cx, cy = w // 2, h // 2
    dx = float(target[0] - cx)
    dy = float(target[1] - cy)
    norm = np.sqrt(dx * dx + dy * dy)
    if norm < 1e-6:
        return out
    arrow_len = min(h, w) * 0.35
    dx = dx / norm * arrow_len
    dy = dy / norm * arrow_len
    if not in_front:
        dx, dy = -dx, -dy
    ex, ey = int(cx + dx), int(cy + dy)
    color = (0, 220, 255) if in_front else (120, 120, 120)
    cv2.arrowedLine(out, (cx, cy), (ex, ey), color, 3 if in_front else 2, tipLength=0.22)
    cv2.circle(out, (cx, cy), 5, color, -1)
    return out


def parse_frames(frames_str: str) -> List[int]:
    vals = []
    for s in frames_str.split(","):
        s = s.strip()
        if not s:
            continue
        vals.append(int(s))
    return vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", type=str, required=True, help="processed scene dir, e.g. .../mini_training/000")
    parser.add_argument("--frames", type=str, default="0,5", help="comma-separated frame indices")
    parser.add_argument("--cam_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    scene_dir = args.scene_dir
    output_dir = args.output_dir or os.path.join(scene_dir, "sun_vis")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(scene_dir, "scene_meta.json"), "r") as f:
        meta = json.load(f)
    sun_dir = meta.get("sun", {}).get("sun_direction_world", None)
    if sun_dir is None:
        raise RuntimeError("scene_meta.json has no sun.sun_direction_world")
    sun_dir = np.asarray(sun_dir, dtype=np.float64)
    sun_dir = sun_dir / (np.linalg.norm(sun_dir) + 1e-8)

    intrinsic = np.loadtxt(os.path.join(scene_dir, "intrinsics", f"{args.cam_id}.txt"))
    K = np.array(
        [
            [intrinsic[0], 0.0, intrinsic[2]],
            [0.0, intrinsic[1], intrinsic[3]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    cam_to_ego = np.loadtxt(os.path.join(scene_dir, "extrinsics", f"{args.cam_id}.txt"))
    frames = parse_frames(args.frames)
    ego_start = np.loadtxt(os.path.join(scene_dir, "ego_pose", f"{frames[0]:03d}.txt"))

    for fi in frames:
        img_path = os.path.join(scene_dir, "images", f"{fi:03d}_{args.cam_id}.jpg")
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        ego_cur = np.loadtxt(os.path.join(scene_dir, "ego_pose", f"{fi:03d}.txt"))
        ego_rel = np.linalg.inv(ego_start) @ ego_cur
        cam2world = ego_rel @ cam_to_ego
        target, in_front = project_direction_to_image(sun_dir, cam2world, K)
        if target is None:
            continue
        vis = draw_arrow(img, target, in_front)
        cv2.imwrite(os.path.join(output_dir, f"{fi:03d}_{args.cam_id}_sun.jpg"), vis)

    print(f"saved visualizations to {output_dir}")


if __name__ == "__main__":
    main()

