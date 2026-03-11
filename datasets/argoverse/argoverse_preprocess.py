import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

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
        }
        with open(os.path.join(dst_scene, "scene_meta.json"), "w") as f:
            json.dump(scene_meta, f, indent=2)

        if "calib" in self.process_keys:
            self._save_calib(dst_scene, intrinsics_df, sensor_df)
        if "pose" in self.process_keys:
            self._save_pose(dst_scene, frame_ts, ego_df)
        if "images" in self.process_keys:
            self._save_images_and_sky_masks(dst_scene, frame_ts, cam_ts, cam_path_by_ts)
        if "lidar" in self.process_keys:
            self._save_lidar(dst_scene, frame_ts, lidar_ts, lidar_dir)
        if "dynamic_masks" in self.process_keys:
            self._save_empty_dynamic_masks(dst_scene, frame_count)
        if "objects" in self.process_keys:
            self._save_objects(dst_scene, frame_ts, anno_df)

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
            idx = _nearest_idx(ego_ts, int(ts))
            row = ego_df.iloc[idx]
            city_T_ego = PoseSE3.from_quat_trans(
                float(row["qw"]),
                float(row["qx"]),
                float(row["qy"]),
                float(row["qz"]),
                float(row["tx_m"]),
                float(row["ty_m"]),
                float(row["tz_m"]),
            ).mat
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
                # Required by current training loss path.
                sky_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                cv2.imwrite(os.path.join(dst_scene, "sky_masks", f"{fi:03d}_{cam_id}.png"), sky_mask)

    def _save_lidar(
        self, dst_scene: str, frame_ts: np.ndarray, lidar_ts: np.ndarray, lidar_dir: str
    ) -> None:
        for fi, ts in enumerate(tqdm(frame_ts, desc="Saving lidar", dynamic_ncols=True)):
            idx = _nearest_idx(lidar_ts, int(ts))
            lidar_file = os.path.join(lidar_dir, f"{int(lidar_ts[idx])}.feather")
            lidar_df = pd.read_feather(lidar_file)

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

    def _save_empty_dynamic_masks(self, dst_scene: str, frame_count: int) -> None:
        # Keep compatibility with configs that enable dynamic masks.
        for fi in range(frame_count):
            for cam_id in CAM_NAME_TO_ID.values():
                img_path = os.path.join(dst_scene, "images", f"{fi:03d}_{cam_id}.jpg")
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                h, w = img.shape[:2]
                empty = np.zeros((h, w), dtype=np.uint8)
                for cls in ["all", "human", "vehicle"]:
                    cv2.imwrite(
                        os.path.join(dst_scene, "dynamic_masks", cls, f"{fi:03d}_{cam_id}.png"), empty
                    )

    def _category_to_class(self, cat: str) -> str:
        c = cat.upper()
        if "PEDESTRIAN" in c:
            return "Pedestrian"
        if "CYCLIST" in c or "BICYCLE" in c or "MOTORCYCLE" in c:
            return "Cyclist"
        return "Vehicle"

    def _save_objects(self, dst_scene: str, frame_ts: np.ndarray, anno_df: pd.DataFrame) -> None:
        frame_instances: Dict[str, List[int]] = {str(i): [] for i in range(len(frame_ts))}
        instances_info: Dict[int, Dict] = {}
        track_to_id: Dict[str, int] = {}
        anno_df = anno_df.copy()
        anno_df["timestamp_ns"] = anno_df["timestamp_ns"].astype(np.int64)
        anno_ts = np.sort(anno_df["timestamp_ns"].unique())

        # Build fast lookup by timestamp.
        grouped = {int(ts): g for ts, g in anno_df.groupby("timestamp_ns")}

        for fi, ts in enumerate(frame_ts):
            if len(anno_ts) == 0:
                continue
            idx = _nearest_idx(anno_ts, int(ts))
            rows = grouped.get(int(anno_ts[idx]))
            if rows is None:
                continue
            for _, row in rows.iterrows():
                track = str(row["track_uuid"])
                if track not in track_to_id:
                    track_to_id[track] = len(track_to_id)
                    instances_info[track_to_id[track]] = {
                        "id": track,
                        "class_name": self._category_to_class(str(row["category"])),
                        "frame_annotations": {"frame_idx": [], "obj_to_world": [], "box_size": []},
                    }
                ins_id = track_to_id[track]
                frame_instances[str(fi)].append(ins_id)

                obj_pose = PoseSE3.from_quat_trans(
                    float(row["qw"]),
                    float(row["qx"]),
                    float(row["qy"]),
                    float(row["qz"]),
                    float(row["tx_m"]),
                    float(row["ty_m"]),
                    float(row["tz_m"]),
                ).mat
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
