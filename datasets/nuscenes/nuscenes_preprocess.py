import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from datasets.tools.multiprocess_utils import track_parallel_progress


NUSC_CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
]
CAM_NAME_TO_ID = {name: idx for idx, name in enumerate(NUSC_CAMERAS)}

FORMAT_VERSION = "nuscenes_v1_dynamic_fix"
SCENE_COORD_FRAME = "global"
LIDAR_POINT_FRAME = "ego"
OBJECT_POSE_FRAME = "global"

MOTION_SPEED_THRESHOLD_MPS = 1.0


def _nearest_idx(sorted_array: np.ndarray, target: int) -> int:
    idx = int(np.searchsorted(sorted_array, target))
    if idx <= 0:
        return 0
    if idx >= len(sorted_array):
        return len(sorted_array) - 1
    left = sorted_array[idx - 1]
    right = sorted_array[idx]
    return idx if abs(right - target) < abs(target - left) else idx - 1


def _to_matrix(qw: float, qx: float, qy: float, qz: float, tx: float, ty: float, tz: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    pose[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    return pose


def _compute_box_corners(length: float, width: float, height: float) -> np.ndarray:
    hx, hy, hz = length / 2.0, width / 2.0, height / 2.0
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


def _estimate_sky_mask(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    upper = np.zeros((h, w), dtype=np.uint8)
    upper[: int(0.78 * h), :] = 1

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    blue_like = ((h_ch >= 85) & (h_ch <= 130) & (s_ch >= 45) & (v_ch >= 80))
    cloud_like = ((s_ch <= 35) & (v_ch >= 200))
    candidate = ((blue_like | cloud_like) & (upper > 0)).astype(np.uint8) * 255

    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, k3, iterations=1)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, k5, iterations=2)

    num_labels, labels, stats, centers = cv2.connectedComponentsWithStats((candidate > 0).astype(np.uint8), 8)
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
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, k5, iterations=1)
    return keep


class SkyMaskEstimator:
    """Sky mask extractor with configurable backends."""

    def __init__(
        self,
        method: str = "color",
        segformer_model_id: str = "nvidia/segformer-b5-finetuned-ade-640-640",
        segformer_device: str = "auto",
    ) -> None:
        method = method.lower()
        if method not in {"color", "segformer"}:
            raise ValueError(f"Unsupported sky mask method '{method}', choose from: color, segformer")
        self.method = method
        self.segformer_model_id = segformer_model_id
        self.segformer_device = segformer_device
        self._segformer_initialized = False
        self._segformer_available = False
        self._processor = None
        self._model = None
        self._torch = None
        self._sky_label_ids: List[int] = []
        self._init_error: Optional[str] = None
        self._fallback_warned = False

    def estimate(self, img_bgr: np.ndarray) -> np.ndarray:
        if self.method == "segformer":
            if self._ensure_segformer():
                return self._estimate_segformer(img_bgr)
            if self._init_error and (not self._fallback_warned):
                print(f"[SkyMaskEstimator] SegFormer unavailable, fallback to color: {self._init_error}")
                self._fallback_warned = True
        return _estimate_sky_mask(img_bgr)

    def _ensure_segformer(self) -> bool:
        if self._segformer_initialized:
            return self._segformer_available
        self._segformer_initialized = True
        try:
            import torch  # type: ignore
            from transformers import AutoImageProcessor, SegformerForSemanticSegmentation  # type: ignore

            device = self.segformer_device.lower()
            if device == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            elif device in {"cpu", "cuda"}:
                self._device = device
            else:
                raise ValueError(f"Invalid segformer_device '{self.segformer_device}', choose from auto/cpu/cuda")
            if self._device == "cuda" and not torch.cuda.is_available():
                self._device = "cpu"

            self._processor = AutoImageProcessor.from_pretrained(self.segformer_model_id)
            self._model = SegformerForSemanticSegmentation.from_pretrained(self.segformer_model_id)
            self._model.to(self._device).eval()
            self._torch = torch

            id2label = getattr(self._model.config, "id2label", {}) or {}
            sky_ids: List[int] = []
            for k, v in id2label.items():
                label = str(v).lower()
                if "sky" in label:
                    sky_ids.append(int(k))
            if not sky_ids:
                sky_ids = [2]  # ADE20K common sky label id
            self._sky_label_ids = sorted(set(sky_ids))
            self._segformer_available = True
            print(
                f"[SkyMaskEstimator] Loaded SegFormer model='{self.segformer_model_id}', "
                f"device='{self._device}', sky_label_ids={self._sky_label_ids}"
            )
        except Exception as exc:
            self._init_error = str(exc)
            self._segformer_available = False
        return self._segformer_available

    def _estimate_segformer(self, img_bgr: np.ndarray) -> np.ndarray:
        assert self._processor is not None and self._model is not None and self._torch is not None
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        inputs = self._processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with self._torch.inference_mode():
            logits = self._model(**inputs).logits
            logits = self._torch.nn.functional.interpolate(
                logits,
                size=img_bgr.shape[:2],
                mode="bilinear",
                align_corners=False,
            )
            pred = logits.argmax(dim=1)[0].detach().cpu().numpy()
        sky = np.isin(pred, self._sky_label_ids).astype(np.uint8) * 255
        k3 = np.ones((3, 3), np.uint8)
        sky = cv2.morphologyEx(sky, cv2.MORPH_OPEN, k3, iterations=1)
        sky = cv2.morphologyEx(sky, cv2.MORPH_CLOSE, k3, iterations=1)
        return sky


def _category_to_class(category: str) -> Optional[str]:
    c = category.lower()
    if c.startswith("human.pedestrian"):
        return "Pedestrian"
    if c in {"vehicle.bicycle", "vehicle.motorcycle"}:
        return "Cyclist"
    if c.startswith("vehicle."):
        return "Vehicle"
    return None


@dataclass
class FrameRecord:
    sample_token: str
    timestamp_us: int
    cam_tokens: Dict[str, str]
    lidar_token: str


class NuScenesProcessor:
    def __init__(
        self,
        load_dir: str,
        save_dir: str,
        split: str = "train",
        version: str = "v1.0-trainval",
        process_keys: List[str] = None,
        process_id_list: List[int] = None,
        workers: int = 8,
        sky_mask_method: str = "color",
        segformer_model_id: str = "nvidia/segformer-b5-finetuned-ade-640-640",
        segformer_device: str = "auto",
        **_: dict,
    ) -> None:
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.split = split
        self.version = version
        self.workers = int(workers)
        self.process_id_list = process_id_list
        self.sky_mask_method = sky_mask_method
        self.segformer_model_id = segformer_model_id
        self.segformer_device = segformer_device
        self.process_keys = process_keys or [
            "images",
            "lidar",
            "calib",
            "pose",
            "dynamic_masks",
            "objects",
        ]
        self.sky_mask_estimator = SkyMaskEstimator(
            method=self.sky_mask_method,
            segformer_model_id=self.segformer_model_id,
            segformer_device=self.segformer_device,
        )

        self.nusc = NuScenes(version=self.version, dataroot=self.load_dir, verbose=False)
        self.scene_names = self._get_split_scene_names()
        self.scenes = sorted([s for s in self.nusc.scene if s["name"] in self.scene_names], key=lambda x: x["name"])

        os.makedirs(self.save_dir, exist_ok=True)
        scene_map = {f"{i:03d}": s["name"] for i, s in enumerate(self.scenes)}
        with open(os.path.join(self.save_dir, "scene_index_map.json"), "w") as f:
            json.dump(scene_map, f, indent=2)
        self.create_folder()

    def _get_split_scene_names(self) -> set:
        split_alias = {
            "training": "train",
            "validation": "val",
            "testing": "test",
        }
        split = split_alias.get(self.split, self.split)
        splits = create_splits_scenes()

        if self.version == "v1.0-mini":
            if split == "train":
                split = "mini_train"
            elif split == "val":
                split = "mini_val"
            elif split == "test":
                raise ValueError("v1.0-mini does not provide a test split")

        if split not in splits:
            raise ValueError(f"Unsupported NuScenes split '{self.split}' for version '{self.version}'")
        return set(splits[split])

    def __len__(self) -> int:
        return len(self.scenes)

    def convert(self) -> None:
        if self.process_id_list is None:
            id_list = list(range(len(self)))
        else:
            id_list = list(self.process_id_list)
        track_parallel_progress(self.convert_one, id_list, self.workers)

    def convert_one(self, scene_idx: int) -> None:
        scene = self.scenes[scene_idx]
        dst_scene = os.path.join(self.save_dir, f"{scene_idx:03d}")

        frames = self._collect_scene_frames(scene)
        if len(frames) == 0:
            return

        annos_per_frame, speed_map = self._collect_annotations_and_speed(frames)
        sun_info = self._compute_scene_sun(scene, frames)

        log = self.nusc.get("log", scene["log_token"])
        scene_meta = {
            "scene_token": scene["token"],
            "scene_name": scene["name"],
            "log_token": scene["log_token"],
            "location": log["location"],
            "date_captured": log.get("date_captured"),
            "num_frames": len(frames),
            "cameras": NUSC_CAMERAS,
            "format_version": FORMAT_VERSION,
            "scene_coord_frame": SCENE_COORD_FRAME,
            "lidar_point_frame": LIDAR_POINT_FRAME,
            "object_pose_frame": OBJECT_POSE_FRAME,
            "timestamp_first_utc": datetime.fromtimestamp(frames[0].timestamp_us / 1e6, tz=timezone.utc).isoformat(),
            "timestamp_last_utc": datetime.fromtimestamp(frames[-1].timestamp_us / 1e6, tz=timezone.utc).isoformat(),
            "sun": sun_info,
        }
        with open(os.path.join(dst_scene, "scene_meta.json"), "w") as f:
            json.dump(scene_meta, f, indent=2)

        if "calib" in self.process_keys:
            self._save_calib(dst_scene, frames[0])
        if "pose" in self.process_keys:
            self._save_pose(dst_scene, frames)
        if "images" in self.process_keys:
            self._save_images_and_sky_masks(dst_scene, frames)
        if "lidar" in self.process_keys:
            self._save_lidar(dst_scene, frames)
        if "dynamic_masks" in self.process_keys:
            self._save_dynamic_masks(dst_scene, frames, annos_per_frame, speed_map)
        if "objects" in self.process_keys:
            self._save_objects(dst_scene, annos_per_frame)

    def _collect_scene_frames(self, scene: Dict) -> List[FrameRecord]:
        frames: List[FrameRecord] = []
        cur = scene["first_sample_token"]
        while cur:
            sample = self.nusc.get("sample", cur)
            cam_tokens = {}
            valid = True
            for cam in NUSC_CAMERAS:
                tok = sample["data"].get(cam)
                if tok is None:
                    valid = False
                    break
                cam_tokens[cam] = tok
            lidar_token = sample["data"].get("LIDAR_TOP")
            if valid and lidar_token is not None:
                ts = int(self.nusc.get("sample_data", cam_tokens["CAM_FRONT"])["timestamp"])
                frames.append(
                    FrameRecord(
                        sample_token=sample["token"],
                        timestamp_us=ts,
                        cam_tokens=cam_tokens,
                        lidar_token=lidar_token,
                    )
                )
            cur = sample["next"]
        return frames

    def _collect_annotations_and_speed(self, frames: List[FrameRecord]) -> Tuple[List[List[Dict]], Dict[Tuple[str, int], float]]:
        annos_per_frame: List[List[Dict]] = []
        traj: Dict[str, List[Tuple[int, int, np.ndarray]]] = {}

        for fi, frame in enumerate(frames):
            sample = self.nusc.get("sample", frame.sample_token)
            frame_annos: List[Dict] = []
            for anno_token in sample["anns"]:
                anno = self.nusc.get("sample_annotation", anno_token)
                class_name = _category_to_class(anno["category_name"])
                if class_name is None:
                    continue
                if int(anno.get("num_lidar_pts", 0)) <= 0:
                    continue

                qw, qx, qy, qz = anno["rotation"]
                tx, ty, tz = anno["translation"]
                # nuScenes size is [w, l, h]
                w, l, h = anno["size"]
                length, width, height = float(l), float(w), float(h)
                obj_pose = _to_matrix(float(qw), float(qx), float(qy), float(qz), float(tx), float(ty), float(tz))
                rec = {
                    "instance_token": anno["instance_token"],
                    "class_name": class_name,
                    "obj_to_world": obj_pose,
                    "box_size": [length, width, height],
                    "timestamp_us": frame.timestamp_us,
                }
                frame_annos.append(rec)
                traj.setdefault(anno["instance_token"], []).append((fi, frame.timestamp_us, np.array([tx, ty, tz], dtype=np.float64)))
            annos_per_frame.append(frame_annos)

        speed_map: Dict[Tuple[str, int], float] = {}
        for inst, seq in traj.items():
            seq = sorted(seq, key=lambda x: x[1])
            n = len(seq)
            for i in range(n):
                if n == 1:
                    speed = 0.0
                elif i == 0:
                    dt = (seq[1][1] - seq[0][1]) * 1e-6
                    dist = float(np.linalg.norm(seq[1][2] - seq[0][2]))
                    speed = dist / dt if dt > 1e-6 else 0.0
                elif i == n - 1:
                    dt = (seq[-1][1] - seq[-2][1]) * 1e-6
                    dist = float(np.linalg.norm(seq[-1][2] - seq[-2][2]))
                    speed = dist / dt if dt > 1e-6 else 0.0
                else:
                    dt = (seq[i + 1][1] - seq[i - 1][1]) * 1e-6
                    dist = float(np.linalg.norm(seq[i + 1][2] - seq[i - 1][2]))
                    speed = dist / dt if dt > 1e-6 else 0.0
                speed_map[(inst, seq[i][0])] = speed
        return annos_per_frame, speed_map

    def _compute_scene_sun(self, scene: Dict, frames: List[FrameRecord]) -> Dict:
        log = self.nusc.get("log", scene["log_token"])
        location = log["location"]
        mid = frames[len(frames) // 2]
        dt_utc = datetime.fromtimestamp(mid.timestamp_us / 1e6, tz=timezone.utc)
        try:
            from sun.sun_position import compute_sun_position_at_datetime, get_nuscenes_location

            lat, lon, elev = get_nuscenes_location(location)
            result = compute_sun_position_at_datetime(lat, lon, dt_utc, elev)
            direction = [float(v) for v in result.direction_enu]
            norm = float(np.linalg.norm(direction))
            if norm > 0:
                direction = [v / norm for v in direction]
            return {
                "location": location,
                "timestamp_us": int(mid.timestamp_us),
                "timestamp_utc": dt_utc.isoformat(),
                "latitude": float(lat),
                "longitude": float(lon),
                "elevation_m": float(elev),
                "azimuth_deg": float(result.azimuth_deg),
                "elevation_deg": float(result.elevation_deg),
                "sun_direction_world": direction,
            }
        except Exception as exc:
            return {
                "location": location,
                "timestamp_us": int(mid.timestamp_us),
                "timestamp_utc": dt_utc.isoformat(),
                "error": f"sun computation failed: {exc}",
                "sun_direction_world": None,
            }

    def _save_calib(self, dst_scene: str, frame0: FrameRecord) -> None:
        for cam in NUSC_CAMERAS:
            cam_id = CAM_NAME_TO_ID[cam]
            sd = self.nusc.get("sample_data", frame0.cam_tokens[cam])
            cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])

            k = np.array(cs["camera_intrinsic"], dtype=np.float64)
            fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
            intr = np.array([fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
            np.savetxt(os.path.join(dst_scene, "intrinsics", f"{cam_id}.txt"), intr)

            qw, qx, qy, qz = cs["rotation"]
            tx, ty, tz = cs["translation"]
            ego_T_cam = _to_matrix(float(qw), float(qx), float(qy), float(qz), float(tx), float(ty), float(tz))
            np.savetxt(os.path.join(dst_scene, "extrinsics", f"{cam_id}.txt"), ego_T_cam)

    def _save_pose(self, dst_scene: str, frames: List[FrameRecord]) -> None:
        for fi, frame in enumerate(frames):
            sd = self.nusc.get("sample_data", frame.cam_tokens["CAM_FRONT"])
            pose = self.nusc.get("ego_pose", sd["ego_pose_token"])
            qw, qx, qy, qz = pose["rotation"]
            tx, ty, tz = pose["translation"]
            global_T_ego = _to_matrix(float(qw), float(qx), float(qy), float(qz), float(tx), float(ty), float(tz))
            np.savetxt(os.path.join(dst_scene, "ego_pose", f"{fi:03d}.txt"), global_T_ego)

    def _save_images_and_sky_masks(self, dst_scene: str, frames: List[FrameRecord]) -> None:
        for fi, frame in enumerate(tqdm(frames, desc="Saving images", dynamic_ncols=True)):
            for cam in NUSC_CAMERAS:
                cam_id = CAM_NAME_TO_ID[cam]
                sd = self.nusc.get("sample_data", frame.cam_tokens[cam])
                src_img = os.path.join(self.load_dir, sd["filename"])
                dst_img = os.path.join(dst_scene, "images", f"{fi:03d}_{cam_id}.jpg")
                img = cv2.imread(src_img, cv2.IMREAD_COLOR)
                cv2.imwrite(dst_img, img)
                sky_mask = self.sky_mask_estimator.estimate(img)
                cv2.imwrite(os.path.join(dst_scene, "sky_masks", f"{fi:03d}_{cam_id}.png"), sky_mask)

    def _save_lidar(self, dst_scene: str, frames: List[FrameRecord]) -> None:
        for fi, frame in enumerate(tqdm(frames, desc="Saving lidar", dynamic_ncols=True)):
            lidar_sd = self.nusc.get("sample_data", frame.lidar_token)
            cs = self.nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
            qw, qx, qy, qz = cs["rotation"]
            tx, ty, tz = cs["translation"]
            ego_T_lidar = _to_matrix(float(qw), float(qx), float(qy), float(qz), float(tx), float(ty), float(tz))

            lidar_path = os.path.join(self.load_dir, lidar_sd["filename"])
            raw = np.fromfile(lidar_path, dtype=np.float32)
            if raw.size % 5 != 0:
                raw = raw[: (raw.size // 5) * 5]
            points = raw.reshape(-1, 5)
            pts_lidar = points[:, :3].astype(np.float32)
            pts_ego = (ego_T_lidar[:3, :3] @ pts_lidar.T + ego_T_lidar[:3, 3:4]).T

            n = pts_ego.shape[0]
            origins = np.zeros_like(pts_ego, dtype=np.float32)
            flows = np.zeros_like(pts_ego, dtype=np.float32)
            flow_classes = np.zeros((n, 1), dtype=np.float32)
            grounds = np.zeros((n, 1), dtype=np.float32)
            intensity = points[:, 3:4].astype(np.float32)
            elongation = np.zeros((n, 1), dtype=np.float32)
            laser_ids = points[:, 4:5].astype(np.float32)
            cloud = np.concatenate(
                [origins, pts_ego.astype(np.float32), flows, flow_classes, grounds, intensity, elongation, laser_ids],
                axis=1,
            )
            cloud.astype(np.float32).tofile(os.path.join(dst_scene, "lidar", f"{fi:03d}.bin"))

    def _save_dynamic_masks(
        self,
        dst_scene: str,
        frames: List[FrameRecord],
        annos_per_frame: List[List[Dict]],
        speed_map: Dict[Tuple[str, int], float],
    ) -> None:
        for fi, frame in enumerate(tqdm(frames, desc="Saving dynamic masks", dynamic_ncols=True)):
            frame_annos = annos_per_frame[fi]
            for cam in NUSC_CAMERAS:
                cam_id = CAM_NAME_TO_ID[cam]
                img_path = os.path.join(dst_scene, "images", f"{fi:03d}_{cam_id}.jpg")
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                h, w = img.shape[:2]

                sd = self.nusc.get("sample_data", frame.cam_tokens[cam])
                cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
                pose = self.nusc.get("ego_pose", sd["ego_pose_token"])

                K = np.array(cs["camera_intrinsic"], dtype=np.float64)
                fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

                ego_T_cam = _to_matrix(
                    float(cs["rotation"][0]),
                    float(cs["rotation"][1]),
                    float(cs["rotation"][2]),
                    float(cs["rotation"][3]),
                    float(cs["translation"][0]),
                    float(cs["translation"][1]),
                    float(cs["translation"][2]),
                )
                global_T_ego = _to_matrix(
                    float(pose["rotation"][0]),
                    float(pose["rotation"][1]),
                    float(pose["rotation"][2]),
                    float(pose["rotation"][3]),
                    float(pose["translation"][0]),
                    float(pose["translation"][1]),
                    float(pose["translation"][2]),
                )
                global_T_cam = global_T_ego @ ego_T_cam
                cam_T_global = np.linalg.inv(global_T_cam)

                mask_all = np.zeros((h, w), dtype=np.float32)
                mask_human = np.zeros((h, w), dtype=np.float32)
                mask_vehicle = np.zeros((h, w), dtype=np.float32)

                for anno in frame_annos:
                    speed = float(speed_map.get((anno["instance_token"], fi), 0.0))
                    if speed <= MOTION_SPEED_THRESHOLD_MPS:
                        continue
                    cls = anno["class_name"]
                    l, bw, bh = anno["box_size"]
                    corners = _compute_box_corners(l, bw, bh)
                    corners_h = np.concatenate([corners, np.ones((corners.shape[0], 1), dtype=np.float64)], axis=1)
                    corners_global = (anno["obj_to_world"] @ corners_h.T).T
                    corners_cam = (cam_T_global @ corners_global.T).T[:, :3]
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

                    mask_all[y0:y1, x0:x1] = np.maximum(mask_all[y0:y1, x0:x1], speed)
                    if cls == "Vehicle":
                        mask_vehicle[y0:y1, x0:x1] = np.maximum(mask_vehicle[y0:y1, x0:x1], speed)
                    elif cls in ("Pedestrian", "Cyclist"):
                        mask_human[y0:y1, x0:x1] = np.maximum(mask_human[y0:y1, x0:x1], speed)

                mask_all_u8 = ((mask_all > MOTION_SPEED_THRESHOLD_MPS) * 255).astype(np.uint8)
                mask_human_u8 = ((mask_human > MOTION_SPEED_THRESHOLD_MPS) * 255).astype(np.uint8)
                mask_vehicle_u8 = ((mask_vehicle > MOTION_SPEED_THRESHOLD_MPS) * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(dst_scene, "dynamic_masks", "all", f"{fi:03d}_{cam_id}.png"), mask_all_u8)
                cv2.imwrite(os.path.join(dst_scene, "dynamic_masks", "human", f"{fi:03d}_{cam_id}.png"), mask_human_u8)
                cv2.imwrite(os.path.join(dst_scene, "dynamic_masks", "vehicle", f"{fi:03d}_{cam_id}.png"), mask_vehicle_u8)

    def _save_objects(self, dst_scene: str, annos_per_frame: List[List[Dict]]) -> None:
        frame_instances: Dict[str, List[int]] = {str(i): [] for i in range(len(annos_per_frame))}
        instances_info: Dict[int, Dict] = {}
        track_to_id: Dict[str, int] = {}

        for fi, annos in enumerate(annos_per_frame):
            for anno in annos:
                track = anno["instance_token"]
                if track not in track_to_id:
                    ins_id = len(track_to_id)
                    track_to_id[track] = ins_id
                    instances_info[ins_id] = {
                        "id": track,
                        "class_name": anno["class_name"],
                        "frame_annotations": {"frame_idx": [], "obj_to_world": [], "box_size": []},
                    }
                ins_id = track_to_id[track]
                frame_instances[str(fi)].append(ins_id)
                instances_info[ins_id]["frame_annotations"]["frame_idx"].append(fi)
                instances_info[ins_id]["frame_annotations"]["obj_to_world"].append(anno["obj_to_world"].tolist())
                instances_info[ins_id]["frame_annotations"]["box_size"].append(anno["box_size"])

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
