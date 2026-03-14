# InvRGBL（NuScenes + 固定太阳方向）

本仓库当前重点流程：  
1. 将 nuScenes raw 转成 InvRGBL 可训练格式。  
2. 在预处理阶段利用 nuScenes UTC 时间计算场景固定太阳方向。  
3. 训练时默认读取 `scene_meta.json` 的太阳方向，只优化太阳/天空强度。

## 1. 环境安装

```bash
conda create -n InvRGBL python=3.9 -y
conda activate InvRGBL

pip install -r requirements.txt
pip install nuscenes-devkit
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@v1.3.0
pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git
pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast
pip install --no-build-isolation ./bvh
```

## 2. nuScenes 数据准备

假设你已上传 `v1.0-mini.tgz` 到 `/data0`：

```bash
mkdir -p /data0/dataset/nuscenes
tar -xzf /data0/v1.0-mini.tgz -C /data0/dataset/nuscenes
```

解压后应包含：

- `/data0/dataset/nuscenes/v1.0-mini`
- `/data0/dataset/nuscenes/samples`
- `/data0/dataset/nuscenes/sweeps`
- `/data0/dataset/nuscenes/maps`

## 3. 预处理（含太阳方向）

### 3.1 mini_train 单场景 smoke test

```bash
cd /data0/invrgbl
export PYTHONPATH=$(pwd)
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"

python datasets/preprocess.py \
  --data_root /data0/dataset/nuscenes \
  --dataset nuscenes \
  --version v1.0-mini \
  --split mini_train \
  --target_dir /data0/dataset/preprocessed/nuscenes \
  --workers 1 \
  --scene_ids 0 \
  --sky_mask_method segformer \
  --segformer_device cpu \
  --process_keys images lidar calib pose dynamic_masks objects
```

输出路径示例：

- `/data0/dataset/preprocessed/nuscenes/mini_training/000`

其中 `scene_meta.json` 内包含：

- `sun.sun_direction_world`
- `sun.timestamp_utc`
- `sun.location`

### 3.2 mini_train 全场景

```bash
python datasets/preprocess.py \
  --data_root /data0/dataset/nuscenes \
  --dataset nuscenes \
  --version v1.0-mini \
  --split mini_train \
  --target_dir /data0/dataset/preprocessed/nuscenes \
  --workers 8 \
  --sky_mask_method segformer \
  --segformer_device cpu \
  --process_keys images lidar calib pose dynamic_masks objects
```

## 4. 太阳方向可视化检查（在原图标注太阳方向）

```bash
cd /data0/invrgbl
export PYTHONPATH=$(pwd)

python sun/visualize_sun_on_processed.py \
  --scene_dir /data0/dataset/preprocessed/nuscenes/mini_training/000 \
  --frames 0,3,6,9,12,15,18,21,24,27,30,33,36 \
  --cam_id 0 \
  --output_dir /data0/dataset/preprocessed/nuscenes/mini_training/000/sun_vis_more
```

输出目录：

- `/data0/dataset/preprocessed/nuscenes/mini_training/000/sun_vis`

## 5. 训练命令（nuScenes）

### 5.1 正式训练（推荐）

```bash
cd /data0/invrgbl
export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=0
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
export TORCH_HOME=/tmp/torch_home
mkdir -p /tmp/torch_extensions /tmp/torch_home

python tools/train.py \
  --device cuda \
  --config_file configs/invrgbl.yaml \
  --output_root /data0/invrgbl/work_dirs \
  --project invrgbl_nusc \
  --run_name nusc_full_scene000 \
  dataset=nuscenes/1cams \
  data.data_root=/data0/dataset/preprocessed/nuscenes/mini_training \
  data.scene_idx=0 \
  data.start_timestep=0 \
  data.end_timestep=38 \
  data.preload_device=cpu \
  trainer.optim.num_iters=30000 \
  logging.vis_freq=5000 \
  logging.print_freq=200 \
  logging.saveckpt_freq=5000 \
  render.render_full=False \
  render.render_test=False
```

### 5.2 Smoke Test（联调）

```bash
python tools/train.py \
  --device cuda \
  --config_file configs/invrgbl.yaml \
  --output_root /data0/invrgbl/work_dirs \
  --project invrgbl_nusc \
  --run_name nusc_smoke \
  dataset=nuscenes/1cams \
  data.data_root=/data0/dataset/preprocessed/nuscenes/mini_training \
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

说明：

- 若 `model.Sky.params.fixed_sun_direction` 未手动设置，训练会优先读取场景 `scene_meta.json` 的太阳方向。
- 当前 nuScenes 预处理使用 keyframe 作为时间轴。
- `start_timestep` / `end_timestep` 都是闭区间；scene `000` 当前最大可用索引是 `038`。
- 首次运行会下载 LPIPS/AlexNet 权重到 `$TORCH_HOME`；`bvh` 扩展会编译到 `$TORCH_EXTENSIONS_DIR`。
