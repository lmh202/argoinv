# InvRGB+L 本地复现说明（中文）

本仓库是 InvRGB+L 的本地复现与扩展版本，基于 3D Gaussian Splatting 实现 RGB + LiDAR 联合逆渲染。  
当前已支持：

- Waymo 数据流程（原始支持）
- Argoverse2 数据流程（本地扩展：预处理、训练、评估链路已跑通）

---

## 1. 项目结构速览

- `tools/train.py`：训练入口
- `tools/eval.py`：评估与渲染入口
- `datasets/preprocess.py`：数据预处理统一入口
- `configs/invrgbl.yaml`：动态场景训练配置
- `configs/invrgbl_static.yaml`：静态场景训练配置
- `configs/datasets/*`：数据集配置（Waymo / Argoverse）

---

## 2. 环境安装

```bash
conda create -n InvRGBL python=3.9 -y
conda activate InvRGBL

pip install -r requirements.txt
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@v1.3.0
pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git
pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast
pip install --no-build-isolation ./bvh
```

常见问题：

- `No module named git`：先安装 `git`
- `No module named torch`（安装 gsplat 时）：加 `--no-build-isolation`
- CUDA 版本不匹配：确保编译 CUDA 与 `torch.version.cuda` 对齐

---

## 3. 数据准备

## Waymo

参考：`docs/Waymo.md`

## Argoverse2

参考：`docs/Argoverse.md`

原始目录示例：

- `/data0/dataset/av2/sensor/train`
- `/data0/dataset/av2/sensor/val`
- `/data0/dataset/av2/sensor/test`

预处理命令（以 training 为例）：

注意：

- 2026-03 起 Argoverse 预处理格式已更新（含坐标修复与动态 mask 生成）。
- 旧的预处理结果需要重跑，否则训练会因 `scene_meta.format_version` 不匹配或对象初始化异常而失败。

```bash
cd /data0/invrgbl
export PYTHONPATH=$(pwd)

python datasets/preprocess.py \
  --data_root /data0/dataset/av2/sensor \
  --dataset argoverse \
  --split training \
  --target_dir /data0/dataset/preprocessed/argoverse \
  --workers 8 \
  --process_keys images lidar calib pose dynamic_masks objects
```

单场景调试（可选）：

```bash
cd /data0/invrgbl
export PYTHONPATH=$(pwd)

python datasets/preprocess.py \
  --data_root /data0/dataset/av2/sensor \
  --dataset argoverse \
  --split training \
  --target_dir /data0/dataset/preprocessed/argoverse \
  --workers 1 \
  --scene_ids 0 \
  --process_keys images lidar calib pose dynamic_masks objects
```

输出路径示例：

- `/data0/dataset/preprocessed/argoverse/training/000`

---

## 4. 训练命令

推荐使用下面这条命令进行 Argoverse 动态场景完整训练（30k iter，带显存友好参数）：

```bash
cd /data0/invrgbl
export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

nohup python tools/train.py \
  --config_file configs/invrgbl.yaml \
  --output_root /data0/invrgbl/work_dirs \
  --project invrgbl_argo \
  --run_name argo_train_v1 \
  dataset=argoverse/1cams \
  data.data_root=/data0/dataset/preprocessed/argoverse/training \
  data.scene_idx=0 \
  data.start_timestep=0 \
  data.end_timestep=-1 \
  data.preload_device=cpu \
  data.pixel_source.downscale_when_loading=[1] \
  trainer.optim.num_iters=30000 \
  trainer.optim.grad_clip_norm=1.0 \
  trainer.optim.nan_guard=true \
  trainer.render.pbr=true \
  trainer.render.pbr_incident_sample_num=40 \
  trainer.render.pbr_visibility_chunk_size=120000 \
  trainer.render.pbr_shading_chunk_size=120000 \
  trainer.render.pbr_cache_device=cpu \
  trainer.render.pbr_cache_dtype=float16 \
  trainer.render.pbr_compute_dtype=float16 \
  model.Background.init.from_lidar.num_samples=500000 \
  logging.vis_freq=5000 \
  logging.print_freq=500 \
  logging.saveckpt_freq=5000 > /data0/invrgbl/out.log 2>&1 &
```

如果使用 Waymo，把 `dataset` 改为 `waymo/1cams`，并替换对应 `data.data_root`。

---

## 5. 五分钟内 Smoke Test（推荐先跑）

```bash
cd /data0/invrgbl
export PYTHONPATH=$(pwd)

nohup python tools/train.py \
  --config_file configs/invrgbl.yaml \
  --output_root /data0/invrgbl/work_dirs \
  --project invrgbl_argo \
  --run_name argo_smoke \
  dataset=argoverse/1cams \
  data.data_root=/data0/dataset/preprocessed/argoverse/training \
  data.scene_idx=0 \
  data.start_timestep=0 \
  data.end_timestep=20 \
  data.preload_device=cpu \
  trainer.optim.num_iters=20 \
  logging.vis_freq=10 \
  logging.print_freq=5 \
  logging.saveckpt_freq=10 \
  render.render_full=False \
  render.render_test=False >> phase4_smoke.log 2>&1 &
```

---

## 6. 如何查看运行效果

训练输出目录：

- `work_dirs/<project>/<run_name>/`

重点文件：

- `config.yaml`：实际生效配置
- `metrics.json`：训练指标
- `images/`：中间可视化
- `videos/`：渲染视频
- `checkpoint_*.pth`：阶段权重
- `checkpoint_final.pth`：最终权重

评估命令：

```bash
export PYTHONPATH=$(pwd)

python tools/eval.py \
  --resume_from work_dirs/<project>/<run_name>/checkpoint_final.pth
```

评估结果：

- `work_dirs/<project>/<run_name>/metrics_eval/`
- `work_dirs/<project>/<run_name>/videos_eval/`

---

## 7. Argoverse2 当前说明

- 当前可在无 `intensity/normal` 先验文件时运行
- 若无 intensity 监督，日志中的 LiDAR intensity RMSE 可能为 `-1`
- 动态 mask 由预处理阶段自动生成，建议保持开启（Argoverse 配置默认开启）

---

## 8. 致谢

- [DriveStudio](https://github.com/ziyc/drivestudio)
- [InvRGB+L 原始项目](https://github.com/cxx226/InvRGBL)
