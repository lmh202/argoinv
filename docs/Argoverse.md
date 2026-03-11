# Preparing Argoverse2 Dataset

This document describes how to run InvRGB+L on Argoverse2 using the native
pipeline added in this repository.

## 1) Raw data layout

The expected raw root is:

`/data0/dataset/av2/sensor`

with split subfolders:

- `train/`
- `val/`
- `test/`

Each scene should contain:

- `calibration/intrinsics.feather`
- `calibration/egovehicle_SE3_sensor.feather`
- `city_SE3_egovehicle.feather`
- `annotations.feather`
- `sensors/cameras/<ring_camera>/*.jpg`
- `sensors/lidar/*.feather`

## 2) Convert raw Argoverse2 to processed format

```bash
export PYTHONPATH=$(pwd)

python datasets/preprocess.py \
  --data_root /data0/dataset/av2/sensor \
  --dataset argoverse \
  --split training \
  --target_dir /data0/dataset/preprocessed/argoverse \
  --workers 8 \
  --scene_ids 0 \
  --process_keys images lidar calib pose dynamic_masks objects
```

Notes:

- `--split training` maps to raw folder `train`.
- For smoke test, use a single `scene_id`.
- Processed output is saved under
  `/data0/dataset/preprocessed/argoverse/training/<scene_idx_3digits>/`.

## 3) Train with Argoverse config

Single camera quick run:

```bash
export PYTHONPATH=$(pwd)

python tools/train.py \
  --config_file configs/invrgbl.yaml \
  --output_root ./work_dirs \
  --project invrgbl_argo \
  --run_name argo_smoke \
  dataset=argoverse/1cams \
  data.data_root=/data0/dataset/preprocessed/argoverse/training \
  data.scene_idx=0 \
  data.start_timestep=0 \
  data.end_timestep=50 \
  trainer.optim.num_iters=200
```

More cameras:

- `dataset=argoverse/3cams`
- `dataset=argoverse/7cams`

## 4) 5-minute smoke test

Use the command below for a bounded short run (typically ~2 minutes, clearly
under 5 minutes on this environment):

```bash
export PYTHONPATH=$(pwd)

python tools/train.py \
  --config_file configs/invrgbl.yaml \
  --output_root /data0/invrgbl/work_dirs \
  --project invrgbl_argo \
  --run_name phase4_smoke \
  dataset=argoverse/1cams \
  data.data_root=/data0/dataset/preprocessed/argoverse \
  data.scene_idx=0 \
  data.start_timestep=0 \
  data.end_timestep=20 \
  data.preload_device=cpu \
  trainer.optim.num_iters=20 \
  logging.vis_freq=10 \
  logging.print_freq=5 \
  logging.saveckpt_freq=10 \
  render.render_full=False \
  render.render_test=False
```

Expected smoke-test outputs:

- checkpoint around step 10:
  `work_dirs/invrgbl_argo/phase4_smoke/checkpoint_00010.pth`
- final checkpoint:
  `work_dirs/invrgbl_argo/phase4_smoke/checkpoint_final.pth`
- visualization images under:
  `work_dirs/invrgbl_argo/phase4_smoke/images/`

## 5) Important compatibility notes

- Current Argoverse preprocessing generates **empty sky masks** to satisfy
  existing mask/depth loss paths.
- Dynamic masks are currently emitted as empty masks unless you implement
  projection-based dynamic mask extraction.
- Object boxes are read from `annotations.feather` and mapped to
  `Vehicle/Pedestrian/Cyclist` for node initialization.
- `intensity_images` supervision is optional in the Argoverse path; therefore
  LiDAR-intensity RMSE can show as `-1` in metric logs when those priors are not
  provided.
