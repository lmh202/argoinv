# Argoverse Phase 1: Data Contract and Sample Validation

This report captures the Phase 1 outcome for Argoverse2 raw data under:

- `/data0/dataset/av2/sensor/train`

and processed sample output:

- `/data0/dataset/preprocessed/argoverse/000`

## 1) Raw schema (validated on one scene)

Sample scene:

- `00a6ffc1-6ce9-3bc3-a060-6006e9893a1a`

Validated files and key fields:

- `calibration/intrinsics.feather`
  - `sensor_name, fx_px, fy_px, cx_px, cy_px, k1, k2, k3, height_px, width_px`
- `calibration/egovehicle_SE3_sensor.feather`
  - `sensor_name, qw, qx, qy, qz, tx_m, ty_m, tz_m`
- `city_SE3_egovehicle.feather`
  - `timestamp_ns, qw, qx, qy, qz, tx_m, ty_m, tz_m`
- `annotations.feather`
  - `timestamp_ns, track_uuid, category, length_m, width_m, height_m, qw, qx, qy, qz, tx_m, ty_m, tz_m, num_interior_pts`
- `sensors/lidar/*.feather`
  - `x, y, z, intensity, laser_number, offset_ns`

Counts observed:

- Front center images: 319
- Lidar sweeps: 157

## 2) Contract mapping to repository unified format

Target format follows existing Waymo-style processed directory:

- `images/{frame:03d}_{cam_id}.jpg`
- `intrinsics/{cam_id}.txt`
- `extrinsics/{cam_id}.txt`
- `ego_pose/{frame:03d}.txt`
- `lidar/{frame:03d}.bin`
- `instances/instances_info.json`
- `instances/frame_instances.json`
- `sky_masks/{frame:03d}_{cam_id}.png`
- `dynamic_masks/{all,human,vehicle}/{frame:03d}_{cam_id}.png`

Current mapping strategy:

- Canonical timeline: `ring_front_center` image timestamps
- Camera sync: nearest timestamp per ring camera
- Lidar sync: nearest timestamp per canonical frame
- Pose sync: nearest `city_SE3_egovehicle` per canonical frame
- Object sync: nearest annotation timestamp per canonical frame

Coordinate transforms used:

- `ego_T_cam` (raw calibration) written as extrinsics
- loader applies `OPENCV2DATASET` to align camera ray convention
- per-frame world alignment uses first-frame ego pose as origin

## 3) Phase 1 validation result

Preprocessing for one scene completed successfully:

- command completed with exit code 0
- output tree exists at `/data0/dataset/preprocessed/argoverse/000`

## 4) Critical compatibility gap discovered

When attempting to instantiate `DrivingDataset` with the current pipeline, load fails at:

- missing `intensity/{frame}_{cam}.npy`

Reason:

- `datasets/base/pixel_source.py` currently always tries to load:
  - `intensity`
  - `normal`
  - optional material priors

Argoverse preprocess currently does not produce those supervision files.

Impact:

- The dataset cannot pass full runtime initialization yet.

This is an expected Phase 2+ integration item (loader/base compatibility adjustments or generating these priors).
