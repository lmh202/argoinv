import json
import logging
import os
from typing import Dict

import cv2
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

from datasets.base.lidar_source import SceneLidarSource
from datasets.base.pixel_source import CameraData, ScenePixelSource
from datasets.base.scene_dataset import ModelType

logger = logging.getLogger()

EXPECTED_FORMAT_VERSION = "nuscenes_v1_dynamic_fix"

OBJECT_CLASS_NODE_MAPPING = {
    "Vehicle": ModelType.RigidNodes,
    "Pedestrian": ModelType.SMPLNodes,
    "Cyclist": ModelType.DeformableNodes,
}


def _load_checked_scene_meta(data_path: str) -> Dict:
    meta_path = os.path.join(data_path, "scene_meta.json")
    if not os.path.exists(meta_path):
        raise RuntimeError(f"Missing scene_meta.json at {meta_path}. Please re-run NuScenes preprocessing.")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    if meta.get("format_version") != EXPECTED_FORMAT_VERSION:
        raise RuntimeError(
            "Unsupported NuScenes preprocessed format. "
            f"Expected format_version={EXPECTED_FORMAT_VERSION}, got {meta.get('format_version')}."
        )
    if meta.get("lidar_point_frame") != "ego":
        raise RuntimeError(f"Expected lidar_point_frame=ego, got {meta.get('lidar_point_frame')}.")
    if meta.get("object_pose_frame") != "global":
        raise RuntimeError(f"Expected object_pose_frame=global, got {meta.get('object_pose_frame')}.")
    return meta


class NuScenesCameraData(CameraData):
    def __init__(self, **kwargs):
        self.dynamic_masks = None
        self.human_masks = None
        self.vehicle_masks = None
        super().__init__(**kwargs)

    def create_all_filelist(self):
        img_filepaths = []
        dynamic_mask_filepaths, sky_mask_filepaths = [], []
        human_mask_filepaths, vehicle_mask_filepaths = [], []

        fine_mask_path = os.path.join(self.data_path, "fine_dynamic_masks")
        dynamic_mask_dir = "fine_dynamic_masks" if os.path.exists(fine_mask_path) else "dynamic_masks"
        for t in range(self.start_timestep, self.end_timestep):
            img_filepaths.append(os.path.join(self.data_path, "images", f"{t:03d}_{self.cam_id}.jpg"))
            dynamic_mask_filepaths.append(
                os.path.join(self.data_path, dynamic_mask_dir, "all", f"{t:03d}_{self.cam_id}.png")
            )
            human_mask_filepaths.append(
                os.path.join(self.data_path, dynamic_mask_dir, "human", f"{t:03d}_{self.cam_id}.png")
            )
            vehicle_mask_filepaths.append(
                os.path.join(self.data_path, dynamic_mask_dir, "vehicle", f"{t:03d}_{self.cam_id}.png")
            )
            sky_mask_filepaths.append(os.path.join(self.data_path, "sky_masks", f"{t:03d}_{self.cam_id}.png"))

        self.img_filepaths = np.array(img_filepaths)
        self.dynamic_mask_filepaths = np.array(dynamic_mask_filepaths)
        self.human_mask_filepaths = np.array(human_mask_filepaths)
        self.vehicle_mask_filepaths = np.array(vehicle_mask_filepaths)
        self.sky_mask_filepaths = np.array(sky_mask_filepaths)

        self.intensity_filepaths = np.array([])
        self.albedo_filepaths = np.array([])
        self.normal_filepaths = np.array([])
        self.roughness_filepaths = np.array([])
        self.shading_filepaths = np.array([])
        self.load_shading = False

    def load_calibrations(self):
        intrinsic = np.loadtxt(os.path.join(self.data_path, "intrinsics", f"{self.cam_id}.txt"))
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        k1, k2, p1, p2, k3 = intrinsic[4], intrinsic[5], intrinsic[6], intrinsic[7], intrinsic[8]
        fx, fy = fx * self.load_size[1] / self.original_size[1], fy * self.load_size[0] / self.original_size[0]
        cx, cy = cx * self.load_size[1] / self.original_size[1], cy * self.load_size[0] / self.original_size[0]
        _intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        _distortions = np.array([k1, k2, p1, p2, k3])

        cam_to_ego = np.loadtxt(os.path.join(self.data_path, "extrinsics", f"{self.cam_id}.txt"))

        cam_to_worlds, intrinsics, distortions = [], [], []
        ego_to_world_start = np.loadtxt(os.path.join(self.data_path, "ego_pose", f"{self.start_timestep:03d}.txt"))
        for t in range(self.start_timestep, self.end_timestep):
            ego_to_world_cur = np.loadtxt(os.path.join(self.data_path, "ego_pose", f"{t:03d}.txt"))
            ego_to_world = np.linalg.inv(ego_to_world_start) @ ego_to_world_cur
            cam2world = ego_to_world @ cam_to_ego
            cam_to_worlds.append(cam2world)
            intrinsics.append(_intrinsics)
            distortions.append(_distortions)

        self.intrinsics = torch.from_numpy(np.stack(intrinsics, axis=0)).float()
        self.distortions = torch.from_numpy(np.stack(distortions, axis=0)).float()
        self.cam_to_worlds = torch.from_numpy(np.stack(cam_to_worlds, axis=0)).float()

    def load_images(self, load_intensity=True):
        images = []
        for ix, fname in enumerate(self.img_filepaths):
            rgb = Image.open(fname).convert("RGB")
            rgb = rgb.resize((self.load_size[1], self.load_size[0]), Image.BILINEAR)
            if self.undistort:
                rgb = cv2.undistort(np.array(rgb), self.intrinsics[ix].numpy(), self.distortions[ix].numpy())
            images.append(rgb)
        self.images = torch.from_numpy(np.stack(images, axis=0)).float() / 255.0

    @classmethod
    def get_camera2worlds(cls, data_path: str, cam_id: str, start_timestep: int, end_timestep: int) -> torch.Tensor:
        cam_to_ego = np.loadtxt(os.path.join(data_path, "extrinsics", f"{cam_id}.txt"))
        cam_to_worlds = []
        ego_to_world_start = np.loadtxt(os.path.join(data_path, "ego_pose", f"{start_timestep:03d}.txt"))
        for t in range(start_timestep, end_timestep):
            ego_to_world_cur = np.loadtxt(os.path.join(data_path, "ego_pose", f"{t:03d}.txt"))
            ego_to_world = np.linalg.inv(ego_to_world_start) @ ego_to_world_cur
            cam2world = ego_to_world @ cam_to_ego
            cam_to_worlds.append(cam2world)
        return torch.from_numpy(np.stack(cam_to_worlds, axis=0)).float()


class NuScenesPixelSource(ScenePixelSource):
    def __init__(
        self,
        dataset_name: str,
        pixel_data_config: OmegaConf,
        data_path: str,
        start_timestep: int,
        end_timestep: int,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(dataset_name, pixel_data_config, device=device)
        self.data_path = data_path
        self.scene_meta = _load_checked_scene_meta(data_path)
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.load_data()

    def load_cameras(self):
        self._timesteps = torch.arange(self.start_timestep, self.end_timestep)
        self.register_normalized_timestamps()
        for idx, cam_id in enumerate(self.camera_list):
            camera = NuScenesCameraData(
                dataset_name=self.dataset_name,
                data_path=self.data_path,
                cam_id=cam_id,
                start_timestep=self.start_timestep,
                end_timestep=self.end_timestep,
                load_dynamic_mask=self.data_cfg.load_dynamic_mask,
                load_sky_mask=self.data_cfg.load_sky_mask,
                downscale_when_loading=self.data_cfg.downscale_when_loading[idx],
                undistort=self.data_cfg.undistort,
                buffer_downscale=self.buffer_downscale,
                device=self.device,
            )
            camera.load_time(self.normalized_time)
            unique_img_idx = torch.arange(len(camera), device=self.device) * len(self.camera_list) + idx
            camera.set_unique_ids(unique_cam_idx=idx, unique_img_idx=unique_img_idx)
            self.camera_data[cam_id] = camera

    def load_objects(self):
        instances_info_path = os.path.join(self.data_path, "instances", "instances_info.json")
        frame_instances_path = os.path.join(self.data_path, "instances", "frame_instances.json")
        with open(instances_info_path, "r") as f:
            instances_info = json.load(f)
        with open(frame_instances_path, "r") as f:
            frame_instances = json.load(f)

        num_instances = len(instances_info)
        num_full_frames = len(frame_instances)
        instances_pose = np.zeros((num_full_frames, num_instances, 4, 4))
        instances_size = np.zeros((num_full_frames, num_instances, 3))
        instances_true_id = np.arange(num_instances)
        instances_model_types = np.ones(num_instances) * -1

        ego_to_world_start = np.loadtxt(os.path.join(self.data_path, "ego_pose", f"{self.start_timestep:03d}.txt"))

        for k, v in instances_info.items():
            k_i = int(k)
            instances_model_types[k_i] = OBJECT_CLASS_NODE_MAPPING[v["class_name"]]
            for frame_idx, obj_to_world, box_size in zip(
                v["frame_annotations"]["frame_idx"],
                v["frame_annotations"]["obj_to_world"],
                v["frame_annotations"]["box_size"],
            ):
                obj_to_world = np.array(obj_to_world).reshape(4, 4)
                obj_to_world = np.linalg.inv(ego_to_world_start) @ obj_to_world
                instances_pose[frame_idx, k_i] = obj_to_world
                instances_size[frame_idx, k_i] = np.array(box_size)

        per_frame_instance_mask = np.zeros((num_full_frames, num_instances))
        for frame_idx, valid_instances in frame_instances.items():
            per_frame_instance_mask[int(frame_idx), valid_instances] = 1

        instances_pose = torch.from_numpy(instances_pose[self.start_timestep : self.end_timestep]).float()
        instances_size = torch.from_numpy(instances_size[self.start_timestep : self.end_timestep]).float()
        instances_true_id = torch.from_numpy(instances_true_id).long()
        instances_model_types = torch.from_numpy(instances_model_types).long()
        per_frame_instance_mask = torch.from_numpy(
            per_frame_instance_mask[self.start_timestep : self.end_timestep]
        ).bool()

        ins_frame_cnt = per_frame_instance_mask.sum(dim=0)
        instances_pose = instances_pose[:, ins_frame_cnt > 0]
        instances_size = instances_size[:, ins_frame_cnt > 0]
        instances_true_id = instances_true_id[ins_frame_cnt > 0]
        instances_model_types = instances_model_types[ins_frame_cnt > 0]
        per_frame_instance_mask = per_frame_instance_mask[:, ins_frame_cnt > 0]

        self.instances_pose = instances_pose
        self.instances_size = instances_size.sum(0) / per_frame_instance_mask.sum(0).unsqueeze(-1)
        self.per_frame_instance_mask = per_frame_instance_mask
        self.instances_true_id = instances_true_id
        self.instances_model_types = instances_model_types


class NuScenesLiDARSource(SceneLidarSource):
    def __init__(
        self,
        lidar_data_config: OmegaConf,
        data_path: str,
        start_timestep: int,
        end_timestep: int,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(lidar_data_config, device=device)
        self.data_path = data_path
        self.scene_meta = _load_checked_scene_meta(data_path)
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.create_all_filelist()
        self.load_data()

    def create_all_filelist(self):
        self.lidar_filepaths = np.array(
            [os.path.join(self.data_path, "lidar", f"{t:03d}.bin") for t in range(self.start_timestep, self.end_timestep)]
        )

    def load_calibrations(self):
        lidar_to_worlds = []
        ego_to_world_start = np.loadtxt(os.path.join(self.data_path, "ego_pose", f"{self.start_timestep:03d}.txt"))
        for t in range(self.start_timestep, self.end_timestep):
            ego_to_world_cur = np.loadtxt(os.path.join(self.data_path, "ego_pose", f"{t:03d}.txt"))
            lidar_to_world = np.linalg.inv(ego_to_world_start) @ ego_to_world_cur
            lidar_to_worlds.append(lidar_to_world)
        self.lidar_to_worlds = torch.from_numpy(np.stack(lidar_to_worlds, axis=0)).float()

    def load_lidar(self):
        origins, directions, ranges, laser_ids = [], [], [], []
        flows, flow_classes, grounds, timesteps = [], [], [], []
        for t in range(len(self.lidar_filepaths)):
            lidar_info = np.memmap(self.lidar_filepaths[t], dtype=np.float32, mode="r").reshape(-1, 14)
            lidar_origins = torch.from_numpy(np.array(lidar_info[:, :3], copy=True)).float()
            lidar_points = torch.from_numpy(np.array(lidar_info[:, 3:6], copy=True)).float()
            lidar_ids = torch.from_numpy(np.array(lidar_info[:, 13], copy=True)).float()
            lidar_flows = torch.from_numpy(np.array(lidar_info[:, 6:9], copy=True)).float()
            lidar_flow_classes = torch.from_numpy(np.array(lidar_info[:, 9], copy=True)).long()
            ground_labels = torch.from_numpy(np.array(lidar_info[:, 10], copy=True)).long()

            valid_mask = torch.ones_like(lidar_origins[:, 0]).bool()
            if self.data_cfg.truncated_max_range is not None:
                valid_mask = lidar_points[:, 0] < self.data_cfg.truncated_max_range
            if self.data_cfg.truncated_min_range is not None:
                valid_mask = valid_mask & (lidar_points[:, 0] > self.data_cfg.truncated_min_range)
            lidar_origins = lidar_origins[valid_mask]
            lidar_points = lidar_points[valid_mask]
            lidar_ids = lidar_ids[valid_mask]
            lidar_flows = lidar_flows[valid_mask]
            lidar_flow_classes = lidar_flow_classes[valid_mask]
            ground_labels = ground_labels[valid_mask]

            lidar_origins = (self.lidar_to_worlds[t][:3, :3] @ lidar_origins.T + self.lidar_to_worlds[t][:3, 3:4]).T
            lidar_points = (self.lidar_to_worlds[t][:3, :3] @ lidar_points.T + self.lidar_to_worlds[t][:3, 3:4]).T
            lidar_flows = (self.lidar_to_worlds[t][:3, :3] @ lidar_flows.T).T

            lidar_directions = lidar_points - lidar_origins
            lidar_ranges = torch.norm(lidar_directions, dim=-1, keepdim=True)
            lidar_directions = lidar_directions / (lidar_ranges + 1e-8)
            lidar_timestamp = torch.ones_like(lidar_ranges).squeeze(-1) * t

            origins.append(lidar_origins)
            directions.append(lidar_directions)
            ranges.append(lidar_ranges)
            laser_ids.append(lidar_ids)
            flows.append(lidar_flows)
            flow_classes.append(lidar_flow_classes)
            grounds.append(ground_labels)
            timesteps.append(lidar_timestamp)

        self.origins = torch.cat(origins, dim=0)
        self.directions = torch.cat(directions, dim=0)
        self.ranges = torch.cat(ranges, dim=0)
        self.laser_ids = torch.cat(laser_ids, dim=0)
        self.visible_masks = torch.zeros_like(self.ranges).squeeze().bool()
        self.colors = torch.ones_like(self.directions)
        self.flows = torch.cat(flows, dim=0)
        self.flow_classes = torch.cat(flow_classes, dim=0)
        self.grounds = torch.cat(grounds, dim=0).bool()
        self._timesteps = torch.cat(timesteps, dim=0)
        self.register_normalized_timestamps()

