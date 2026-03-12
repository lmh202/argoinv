import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from datasets.tools.multiprocess_utils import track_parallel_progress


RING_CAMERAS = [
    "ring_front_center",
    "ring_front_left",
    "ring_front_right",
    "ring_side_left",
    "ring_side_right",
    "ring_rear_left",
    "ring_rear_right",
]

CAM_NAME_TO_ID = {name: idx for idx, name in enumerate(RING_CAMERAS)}

# opencv: x right, y down, z front
# dataset(ego): x front, y left, z up
OPENCV2DATASET = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)

FORMAT_VERSION = "argoverse_v2_dynamic_fix"
SCENE_COORD_FRAME = "city"
LIDAR_POINT_FRAME = "ego"
OBJECT_POSE_FRAME = "city"

AV2_VEHICLE_CATEGORIES = {
    "REGULAR_VEHICLE",
    "LARGE_VEHICLE",
    "BUS",
    "BOX_TRUCK",
    "TRUCK",
    "TRUCK_CAB",
    "SCHOOL_BUS",
    "ARTICULATED_BUS",
    "RAILED_VEHICLE",
    "VEHICULAR_TRAILER",
}

AV2_PEDESTRIAN_CATEGORIES = {
    "PEDESTRIAN",
    "STROLLER",
    "WHEELCHAIR",
    "OFFICIAL_SIGNALER",
}

AV2_CYCLIST_CATEGORIES = {
    "BICYCLIST",
    "MOTORCYCLIST",
    "WHEELED_RIDER",
    "BICYCLE",
    "MOTORCYCLE",
    "WHEELED_DEVICE",
}

MOTION_SPEED_THRESHOLD_MPS = 1.0


@dataclass
class PoseSE3:
    mat: np.ndarray

    @classmethod
    def from_quat_trans(
        cls, qw: float, qx: float, qy: float, qz: float, tx: float, ty: float, tz: float
    ) -> "PoseSE3":
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
        pose[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
        return cls(mat=pose)


def _nearest_idx(sorted_array: np.ndarray, target: int) -> int:
    idx = int(np.searchsorted(sorted_array, target))
    if idx <= 0:
        return 0
    if idx >= len(sorted_array):
        return len(sorted_array) - 1
    left = sorted_array[idx - 1]
    right = sorted_array[idx]
    return idx if abs(right - target) < abs(target - left) else idx - 1


class ArgoVerseProcessor:
    """
    Convert Argoverse2 sensor raw logs into the repository's unified processed format.
    The output directory layout is intentionally aligned with Waymo preprocessing output.
    """

    def __init__(
        self,
        load_dir: str,
        save_dir: str,
        process_keys: List[str] = None,
        process_id_list: List[int] = None,
        workers: int = 8,
    ) -> None:
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.workers = int(workers)
        self.process_id_list = process_id_list
        self.process_keys = process_keys or [
            "images",
            "lidar",
            "calib",
            "pose",
            "dynamic_masks",
            "objects",
        ]
        self.scene_names = sorted(
            [
                d
                for d in os.listdir(self.load_dir)
                if os.path.isdir(os.path.join(self.load_dir, d))
            ]
        )
        os.makedirs(self.save_dir, exist_ok=True)
        scene_map = {f"{i:03d}": name for i, name in enumerate(self.scene_names)}
        with open(os.path.join(self.save_dir, "scene_index_map.json"), "w") as f:
            json.dump(scene_map, f, indent=2)
        self.create_folder()

    def __len__(self) -> int:
        return len(self.scene_names)

    def convert(self) -> None:
        if self.process_id_list is None:
            id_list = list(range(len(self)))
        else:
            id_list = list(self.process_id_list)
        track_parallel_progress(self.convert_one, id_list, self.workers)

    def convert_one(self, scene_idx: int) -> None:
        scene_name = self.scene_names[scene_idx]
        src_scene = os.path.join(self.load_dir, scene_name)
        dst_scene = os.path.join(self.save_dir, f"{scene_idx:03d}")

        intrinsics_df = pd.read_feather(os.path.join(src_scene, "calibration", "intrinsics.feather"))
        sensor_df = pd.read_feather(
            os.path.join(src_scene, "calibration", "egovehicle_SE3_sensor.feather")
        )
        ego_df = pd.read_feather(os.path.join(src_scene, "city_SE3_egovehicle.feather"))
        anno_df = pd.read_feather(os.path.join(src_scene, "annotations.feather"))

        cam_ts: Dict[str, np.ndarray] = {}
        cam_path_by_ts: Dict[str, Dict[int, str]] = {}
        for cam_name in RING_CAMERAS:
            cam_dir = os.path.join(src_scene, "sensors", "cameras", cam_name)
            files = sorted([f for f in os.listdir(cam_dir) if f.endswith(".jpg")])
            ts = np.array([int(f.split(".")[0]) for f in files], dtype=np.int64)
            cam_ts[cam_name] = ts
            cam_path_by_ts[cam_name] = {int(t): os.path.join(cam_dir, f"{t}.jpg") for t in ts}

        lidar_dir = os.path.join(src_scene, "sensors", "lidar")
        lidar_files = sorted([f for f in os.listdir(lidar_dir) if f.endswith(".feather")])
        lidar_ts = np.array([int(f.split(".")[0]) for f in lidar_files], dtype=np.int64)

        # Use front-center camera timeline as canonical frame index.
        frame_ts = cam_ts["ring_front_center"]
        frame_count = len(frame_ts)

        scene_meta = {
            "scene_name": scene_name,
            "num_frames": frame_count,
            "ring_cameras": RING_CAMERAS,
            "format_version": FORMAT_VERSION,
            "scene_coord_frame": SCENE_COORD_FRAME,
            "lidar_point_frame": LIDAR_POINT_FRAME,
            "object_pose_frame": OBJECT_POSE_FRAME,
        }
        with open(os.path.join(dst_scene, "scene_meta.json"), "w") as f:
            json.dump(scene_meta, f, indent=2)

        frame_rows = self._build_frame_annotation_rows(frame_ts, anno_df)
        track_speed_map = self._build_track_speed_map(anno_df, ego_df)

        if "calib" in self.process_keys:
            self._save_calib(dst_scene, intrinsics_df, sensor_df)
        if "pose" in self.process_keys:
            self._save_pose(dst_scene, frame_ts, ego_df)
        if "images" in self.process_keys:
            self._save_images_and_sky_masks(dst_scene, frame_ts, cam_ts, cam_path_by_ts)
        if "lidar" in self.process_keys:
            self._save_lidar(dst_scene, frame_ts, lidar_ts, lidar_dir)
        if "dynamic_masks" in self.process_keys:
            self._save_dynamic_masks(
                dst_scene=dst_scene,
                frame_ts=frame_ts,
                frame_rows=frame_rows,
                cam_ts=cam_ts,
                intrinsics_df=intrinsics_df,
                sensor_df=sensor_df,
                ego_df=ego_df,
                track_speed_map=track_speed_map,
            )
        if "objects" in self.process_keys:
            self._save_objects(dst_scene, frame_rows, ego_df)

    def _build_frame_annotation_rows(
        self, frame_ts: np.ndarray, anno_df: pd.DataFrame
    ) -> List[Optional[pd.DataFrame]]:
        anno_df = anno_df.copy()
        anno_df["timestamp_ns"] = anno_df["timestamp_ns"].astype(np.int64)
        anno_ts = np.sort(anno_df["timestamp_ns"].unique())
        grouped = {int(ts): g for ts, g in anno_df.groupby("timestamp_ns")}

        frame_rows: List[Optional[pd.DataFrame]] = []
        for ts in frame_ts:
            if len(anno_ts) == 0:
                frame_rows.append(None)
                continue
            idx = _nearest_idx(anno_ts, int(ts))
            frame_rows.append(grouped.get(int(anno_ts[idx])))
        return frame_rows

    def _build_track_speed_map(self, anno_df: pd.DataFrame, ego_df: pd.DataFrame) -> Dict[Tuple[str, int], float]:
        """Estimate per-track speed (m/s) in city frame at each annotation timestamp."""
        speed_map: Dict[Tuple[str, int], float] = {}
        if anno_df is None or len(anno_df) == 0:
            return speed_map

        ego_ts = ego_df["timestamp_ns"].to_numpy(dtype=np.int64)
        recs = []
        for _, row in anno_df.iterrows():
            obj_pose_city = self._object_pose_in_city(row, ego_ts, ego_df)
            recs.append(
                {
                    "track_uuid": str(row["track_uuid"]),
                    "timestamp_ns": int(row["timestamp_ns"]),
                    "x": float(obj_pose_city[0, 3]),
                    "y": float(obj_pose_city[1, 3]),
                    "z": float(obj_pose_city[2, 3]),
                }
            )
        traj_df = pd.DataFrame(recs).sort_values(["track_uuid", "timestamp_ns"])

        for track_uuid, g in traj_df.groupby("track_uuid"):
            ts = g["timestamp_ns"].to_numpy(dtype=np.int64)
            xyz = g[["x", "y", "z"]].to_numpy(dtype=np.float64)
            n = len(g)
            if n == 1:
                speed_map[(track_uuid, int(ts[0]))] = 0.0
                continue
            for i in range(n):
                if i == 0:
                    dt = float(ts[1] - ts[0]) * 1e-9
                    dist = float(np.linalg.norm(xyz[1] - xyz[0]))
                elif i == n - 1:
                    dt = float(ts[-1] - ts[-2]) * 1e-9
                    dist = float(np.linalg.norm(xyz[-1] - xyz[-2]))
                else:
                    dt = float(ts[i + 1] - ts[i - 1]) * 1e-9
                    dist = float(np.linalg.norm(xyz[i + 1] - xyz[i - 1]))
                speed_map[(track_uuid, int(ts[i]))] = dist / dt if dt > 1e-6 else 0.0
        return speed_map

    @staticmethod
    def _pose_at_timestamp(ts_array: np.ndarray, pose_df: pd.DataFrame, ts: int) -> np.ndarray:
        idx = _nearest_idx(ts_array, int(ts))
        row = pose_df.iloc[idx]
        return PoseSE3.from_quat_trans(
            float(row["qw"]),
            float(row["qx"]),
            float(row["qy"]),
            float(row["qz"]),
            float(row["tx_m"]),
            float(row["ty_m"]),
            float(row["tz_m"]),
        ).mat

    @staticmethod
    def _box_corners(size_xyz: np.ndarray) -> np.ndarray:
        lx, ly, lz = float(size_xyz[0]), float(size_xyz[1]), float(size_xyz[2])
        hx, hy, hz = lx / 2.0, ly / 2.0, lz / 2.0
        return np.array(
            [
                [hx, hy, hz],
                [hx, -hy, hz],
                [hx, hy, -hz],
                [hx, -hy, -hz],
                [-hx, hy, hz],
                [-hx, -hy, hz],
                [-hx, hy, -hz],
                [-hx, -hy, -hz],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _estimate_sky_mask(img_bgr: np.ndarray) -> np.ndarray:
        """
        Estimate sky mask from RGB image using color priors + geometric component filtering.
        Returns uint8 mask in {0,255}.
        """
        h, w = img_bgr.shape[:2]
        upper = np.zeros((h, w), dtype=np.uint8)
        upper[: int(0.78 * h), :] = 1

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        # blue sky + bright cloud priors
        blue_like = ((h_ch >= 85) & (h_ch <= 130) & (s_ch >= 45) & (v_ch >= 80))
        cloud_like = ((s_ch <= 35) & (v_ch >= 200))
        candidate = ((blue_like | cloud_like) & (upper > 0)).astype(np.uint8) * 255

        # Clean small blobs and connect nearby regions.
        k3 = np.ones((3, 3), np.uint8)
        k5 = np.ones((5, 5), np.uint8)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, k3, iterations=1)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, k5, iterations=2)

        # Component-level geometric filtering to suppress sign poles / low objects.
        num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(
            (candidate > 0).astype(np.uint8), connectivity=8
        )
        keep = np.zeros((h, w), dtype=np.uint8)
        min_area = max(64, int(0.0005 * h * w))
        for lid in range(1, num_labels):
            x, y, ww, hh, area = stats[lid]
            _, cy = centers[lid]
            if area < min_area:
                continue
            if y > int(0.45 * h):
                continue
            if (y + hh) > int(0.62 * h):
                continue
            if cy > 0.32 * h:
                continue
            keep[labels == lid] = 255

        # Fill small holes inside sky regions.
        keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, k5, iterations=1)
        return keep

    def _object_pose_in_city(self, row: pd.Series, ego_ts: np.ndarray, ego_df: pd.DataFrame) -> np.ndarray:
        # AV2 annotations are in ego frame at the annotation timestamp.
        ego_pose = PoseSE3.from_quat_trans(
            float(row["qw"]),
            float(row["qx"]),
            float(row["qy"]),
            float(row["qz"]),
            float(row["tx_m"]),
            float(row["ty_m"]),
            float(row["tz_m"]),
        ).mat
        anno_ts = int(row["timestamp_ns"]) if "timestamp_ns" in row else 0
        city_T_ego = self._pose_at_timestamp(ego_ts, ego_df, anno_ts)
        return city_T_ego @ ego_pose

    def _save_calib(self, dst_scene: str, intrinsics_df: pd.DataFrame, sensor_df: pd.DataFrame) -> None:
        for cam_name in RING_CAMERAS:
            cam_id = CAM_NAME_TO_ID[cam_name]
            intr = intrinsics_df[intrinsics_df["sensor_name"] == cam_name].iloc[0]
            fx, fy, cx, cy = float(intr["fx_px"]), float(intr["fy_px"]), float(intr["cx_px"]), float(intr["cy_px"])
            k1, k2, k3 = float(intr["k1"]), float(intr["k2"]), float(intr["k3"])
            # Keep Waymo-compatible 9 coeffs: fx fy cx cy k1 k2 p1 p2 k3
            np.savetxt(
                os.path.join(dst_scene, "intrinsics", f"{cam_id}.txt"),
                np.array([fx, fy, cx, cy, k1, k2, 0.0, 0.0, k3], dtype=np.float64),
            )

            extr = sensor_df[sensor_df["sensor_name"] == cam_name].iloc[0]
            ego_T_cam = PoseSE3.from_quat_trans(
                float(extr["qw"]),
                float(extr["qx"]),
                float(extr["qy"]),
                float(extr["qz"]),
                float(extr["tx_m"]),
                float(extr["ty_m"]),
                float(extr["tz_m"]),
            ).mat
            np.savetxt(os.path.join(dst_scene, "extrinsics", f"{cam_id}.txt"), ego_T_cam)

    def _save_pose(self, dst_scene: str, frame_ts: np.ndarray, ego_df: pd.DataFrame) -> None:
        ego_ts = ego_df["timestamp_ns"].to_numpy(dtype=np.int64)
        for fi, ts in enumerate(frame_ts):
            city_T_ego = self._pose_at_timestamp(ego_ts, ego_df, int(ts))
            np.savetxt(os.path.join(dst_scene, "ego_pose", f"{fi:03d}.txt"), city_T_ego)

    def _save_images_and_sky_masks(
        self,
        dst_scene: str,
        frame_ts: np.ndarray,
        cam_ts: Dict[str, np.ndarray],
        cam_path_by_ts: Dict[str, Dict[int, str]],
    ) -> None:
        for fi, ref_ts in enumerate(tqdm(frame_ts, desc="Saving images", dynamic_ncols=True)):
            for cam_name in RING_CAMERAS:
                cam_id = CAM_NAME_TO_ID[cam_name]
                idx = _nearest_idx(cam_ts[cam_name], int(ref_ts))
                ts = int(cam_ts[cam_name][idx])
                src_img = cam_path_by_ts[cam_name][ts]
                dst_img = os.path.join(dst_scene, "images", f"{fi:03d}_{cam_id}.jpg")
                img = cv2.imread(src_img, cv2.IMREAD_COLOR)
                cv2.imwrite(dst_img, img)
                sky_mask = self._estimate_sky_mask(img)
                cv2.imwrite(os.path.join(dst_scene, "sky_masks", f"{fi:03d}_{cam_id}.png"), sky_mask)

    def _save_lidar(
        self,
        dst_scene: str,
        frame_ts: np.ndarray,
        lidar_ts: np.ndarray,
        lidar_dir: str,
    ) -> None:
        for fi, ts in enumerate(tqdm(frame_ts, desc="Saving lidar", dynamic_ncols=True)):
            idx = _nearest_idx(lidar_ts, int(ts))
            lidar_file = os.path.join(lidar_dir, f"{int(lidar_ts[idx])}.feather")
            lidar_df = pd.read_feather(lidar_file)

            # AV2 lidar points are already in ego frame.
            pts = lidar_df[["x", "y", "z"]].to_numpy(dtype=np.float32)
            n = pts.shape[0]
            origins = np.zeros_like(pts, dtype=np.float32)
            flows = np.zeros_like(pts, dtype=np.float32)
            flow_classes = np.zeros((n, 1), dtype=np.float32)
            grounds = np.zeros((n, 1), dtype=np.float32)
            intensity = lidar_df[["intensity"]].to_numpy(dtype=np.float32)
            elongation = np.zeros((n, 1), dtype=np.float32)
            laser_ids = lidar_df[["laser_number"]].to_numpy(dtype=np.float32)
            cloud = np.concatenate(
                [origins, pts, flows, flow_classes, grounds, intensity, elongation, laser_ids], axis=1
            )
            cloud.astype(np.float32).tofile(os.path.join(dst_scene, "lidar", f"{fi:03d}.bin"))

    def _save_dynamic_masks(
        self,
        dst_scene: str,
        frame_ts: np.ndarray,
        frame_rows: List[Optional[pd.DataFrame]],
        cam_ts: Dict[str, np.ndarray],
        intrinsics_df: pd.DataFrame,
        sensor_df: pd.DataFrame,
        ego_df: pd.DataFrame,
        track_speed_map: Dict[Tuple[str, int], float],
    ) -> None:
        ego_ts = ego_df["timestamp_ns"].to_numpy(dtype=np.int64)
        cam_intr: Dict[str, Tuple[float, float, float, float]] = {}
        cam_to_ego: Dict[str, np.ndarray] = {}

        for cam_name in RING_CAMERAS:
            intr = intrinsics_df[intrinsics_df["sensor_name"] == cam_name].iloc[0]
            cam_intr[cam_name] = (
                float(intr["fx_px"]),
                float(intr["fy_px"]),
                float(intr["cx_px"]),
                float(intr["cy_px"]),
            )

            extr = sensor_df[sensor_df["sensor_name"] == cam_name].iloc[0]
            ego_T_cam = PoseSE3.from_quat_trans(
                float(extr["qw"]),
                float(extr["qx"]),
                float(extr["qy"]),
                float(extr["qz"]),
                float(extr["tx_m"]),
                float(extr["ty_m"]),
                float(extr["tz_m"]),
            ).mat
            cam_to_ego[cam_name] = ego_T_cam

        for fi, ref_ts in enumerate(tqdm(frame_ts, desc="Saving dynamic masks", dynamic_ncols=True)):
            rows = frame_rows[fi]

            for cam_name in RING_CAMERAS:
                cam_id = CAM_NAME_TO_ID[cam_name]
                img_path = os.path.join(dst_scene, "images", f"{fi:03d}_{cam_id}.jpg")
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                h, w = img.shape[:2]
                ts_idx = _nearest_idx(cam_ts[cam_name], int(ref_ts))
                cam_capture_ts = int(cam_ts[cam_name][ts_idx])
                city_T_ego = self._pose_at_timestamp(ego_ts, ego_df, cam_capture_ts)

                mask_all = np.zeros((h, w), dtype=np.float32)
                mask_human = np.zeros((h, w), dtype=np.float32)
                mask_vehicle = np.zeros((h, w), dtype=np.float32)

                if rows is not None and len(rows) > 0:
                    cam2world = city_T_ego @ cam_to_ego[cam_name]
                    world2cam = np.linalg.inv(cam2world)
                    fx, fy, cx, cy = cam_intr[cam_name]
                    for _, row in rows.iterrows():
                        class_name = self._category_to_class(str(row["category"]))
                        if class_name is None:
                            continue
                        if int(row.get("num_interior_pts", 0)) <= 0:
                            continue
                        track_uuid = str(row["track_uuid"])
                        anno_ts = int(row["timestamp_ns"])
                        speed_mps = float(track_speed_map.get((track_uuid, anno_ts), 0.0))
                        if speed_mps <= MOTION_SPEED_THRESHOLD_MPS:
                            continue
                        obj_pose = self._object_pose_in_city(row, ego_ts, ego_df)
                        size_xyz = np.array(
                            [float(row["length_m"]), float(row["width_m"]), float(row["height_m"])],
                            dtype=np.float64,
                        )
                        corners_obj = self._box_corners(size_xyz)
                        corners_obj_h = np.concatenate(
                            [corners_obj, np.ones((corners_obj.shape[0], 1), dtype=np.float64)], axis=1
                        )
                        corners_world = (obj_pose @ corners_obj_h.T).T
                        corners_cam = (world2cam @ corners_world.T).T[:, :3]

                        # Waymo-style strict projection checks.
                        depth = corners_cam[:, 2]
                        if np.any(depth <= 0.5):
                            continue

                        u = fx * corners_cam[:, 0] / depth + cx
                        v = fy * corners_cam[:, 1] / depth + cy
                        u = np.clip(u, 0, w - 1)
                        v = np.clip(v, 0, h - 1)
                        umin, umax = float(np.min(u)), float(np.max(u))
                        vmin, vmax = float(np.min(v)), float(np.max(v))
                        if (umax - umin) < 1.0 or (vmax - vmin) < 1.0:
                            continue

                        x0, x1 = int(np.floor(umin)), int(np.ceil(umax))
                        y0, y1 = int(np.floor(vmin)), int(np.ceil(vmax))
                        x0, x1 = max(0, x0), min(w, x1)
                        y0, y1 = max(0, y0), min(h, y1)
                        if x1 <= x0 or y1 <= y0:
                            continue

                        mask_all[y0:y1, x0:x1] = np.maximum(mask_all[y0:y1, x0:x1], speed_mps)
                        if class_name == "Vehicle":
                            mask_vehicle[y0:y1, x0:x1] = np.maximum(mask_vehicle[y0:y1, x0:x1], speed_mps)
                        elif class_name in ("Pedestrian", "Cyclist"):
                            mask_human[y0:y1, x0:x1] = np.maximum(mask_human[y0:y1, x0:x1], speed_mps)

                mask_all_u8 = ((mask_all > MOTION_SPEED_THRESHOLD_MPS) * 255).astype(np.uint8)
                mask_human_u8 = ((mask_human > MOTION_SPEED_THRESHOLD_MPS) * 255).astype(np.uint8)
                mask_vehicle_u8 = ((mask_vehicle > MOTION_SPEED_THRESHOLD_MPS) * 255).astype(np.uint8)

                cv2.imwrite(os.path.join(dst_scene, "dynamic_masks", "all", f"{fi:03d}_{cam_id}.png"), mask_all_u8)
                cv2.imwrite(
                    os.path.join(dst_scene, "dynamic_masks", "human", f"{fi:03d}_{cam_id}.png"), mask_human_u8
                )
                cv2.imwrite(
                    os.path.join(dst_scene, "dynamic_masks", "vehicle", f"{fi:03d}_{cam_id}.png"), mask_vehicle_u8
                )

    def _category_to_class(self, cat: str) -> Optional[str]:
        c = cat.upper()
        if c in AV2_PEDESTRIAN_CATEGORIES:
            return "Pedestrian"
        if c in AV2_CYCLIST_CATEGORIES:
            return "Cyclist"
        if c in AV2_VEHICLE_CATEGORIES:
            return "Vehicle"
        # Ignore static map objects and unsupported classes for dynamic-object modeling.
        return None

    def _save_objects(
        self, dst_scene: str, frame_rows: List[Optional[pd.DataFrame]], ego_df: pd.DataFrame
    ) -> None:
        frame_instances: Dict[str, List[int]] = {str(i): [] for i in range(len(frame_rows))}
        instances_info: Dict[int, Dict] = {}
        track_to_id: Dict[str, int] = {}
        ego_ts = ego_df["timestamp_ns"].to_numpy(dtype=np.int64)
        for fi, rows in enumerate(frame_rows):
            if rows is None:
                continue
            for _, row in rows.iterrows():
                class_name = self._category_to_class(str(row["category"]))
                if class_name is None:
                    continue
                track = str(row["track_uuid"])
                if track not in track_to_id:
                    track_to_id[track] = len(track_to_id)
                    instances_info[track_to_id[track]] = {
                        "id": track,
                        "class_name": class_name,
                        "frame_annotations": {"frame_idx": [], "obj_to_world": [], "box_size": []},
                    }
                ins_id = track_to_id[track]
                frame_instances[str(fi)].append(ins_id)

                obj_pose = self._object_pose_in_city(row, ego_ts, ego_df)
                instances_info[ins_id]["frame_annotations"]["frame_idx"].append(fi)
                instances_info[ins_id]["frame_annotations"]["obj_to_world"].append(obj_pose.tolist())
                instances_info[ins_id]["frame_annotations"]["box_size"].append(
                    [float(row["length_m"]), float(row["width_m"]), float(row["height_m"])]
                )

        with open(os.path.join(dst_scene, "instances", "instances_info.json"), "w") as fp:
            json.dump(instances_info, fp, indent=2)
        with open(os.path.join(dst_scene, "instances", "frame_instances.json"), "w") as fp:
            json.dump(frame_instances, fp, indent=2)

    def create_folder(self) -> None:
        if self.process_id_list is None:
            id_list = range(len(self))
        else:
            id_list = self.process_id_list
        for i in id_list:
            scene_root = os.path.join(self.save_dir, f"{int(i):03d}")
            if "images" in self.process_keys:
                os.makedirs(os.path.join(scene_root, "images"), exist_ok=True)
                os.makedirs(os.path.join(scene_root, "sky_masks"), exist_ok=True)
            if "calib" in self.process_keys:
                os.makedirs(os.path.join(scene_root, "extrinsics"), exist_ok=True)
                os.makedirs(os.path.join(scene_root, "intrinsics"), exist_ok=True)
            if "pose" in self.process_keys:
                os.makedirs(os.path.join(scene_root, "ego_pose"), exist_ok=True)
            if "lidar" in self.process_keys:
                os.makedirs(os.path.join(scene_root, "lidar"), exist_ok=True)
            if "dynamic_masks" in self.process_keys:
                os.makedirs(os.path.join(scene_root, "dynamic_masks", "all"), exist_ok=True)
                os.makedirs(os.path.join(scene_root, "dynamic_masks", "human"), exist_ok=True)
                os.makedirs(os.path.join(scene_root, "dynamic_masks", "vehicle"), exist_ok=True)
            if "objects" in self.process_keys:
                os.makedirs(os.path.join(scene_root, "instances"), exist_ok=True)
