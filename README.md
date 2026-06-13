# CasMVSNet_pl
Unofficial implementation of [Cascade Cost Volume for High-Resolution Multi-View Stereo and Stereo Matching](https://arxiv.org/pdf/1912.06378.pdf) using [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)

Official implementation: [CasMVSNet](https://github.com/alibaba/cascade-stereo/tree/master/CasMVSNet)

Reference MVSNet implementation: [MVSNet_pl](https://github.com/kwea123/MVSNet_pl)

### Update

1.  Implement groupwise correlation in [Learning Inverse Depth Regression for Multi-View Stereowith Correlation Cost Volume](https://arxiv.org/abs/1912.11746). It achieves almost the same result as original variance-based cost volume, but with fewer parameters and consumes lower memory, so it is highly recommended to use (in contrast, the inverse depth sampling in that paper turns out to have no effect in my experiments, maybe because DTU is indoor dataset, and inverse depth improves outdoor dataset better). To activate, set `--num_groups 8` in training.
2.  2020/03/06: Add [Tanks and temples](https://www.tanksandtemples.org/) [evaluation](evaluations/tanks)!
3.  2020/03/07: Add [BlendedMVS](https://github.com/YoYo000/BlendedMVS) [evaluation](evaluations/blendedmvs)!
4.  2020/03/31: Add [BlendedMVS](https://github.com/YoYo000/BlendedMVS) [training](#blendedmvs)!
5.  2020/04/30: Add point cloud to mesh [guideline](#point-cloud-to-mesh)!

# Installation

## Hardware

* OS: Ubuntu 16.04 or 18.04
* NVIDIA GPU with **CUDA>=10.0** (tested with 1 RTX 2080Ti)

## Software

* Python==3.7 (installation via [anaconda](https://www.anaconda.com/distribution/) is recommended, use `conda create -n casmvsnet_pl python=3.7` to create a conda environment and activate it by `conda activate casmvsnet_pl`)
* Python libraries
    * Install core requirements by `pip install -r requirements.txt`
    * Install [Inplace-ABN](https://github.com/mapillary/inplace_abn) by `pip install inplace-abn`

# Training

Please see each subsection for training on different datasets.
Available training datasets:
*  [DTU dataset](#dtu-dataset)
*  [BlendedMVS](#blendedmvs)

## Underwater refractive UwMVS training

This fork adds a differentiable flat-port refractive camera layer for underwater MVS. The original CasMVSNet pinhole homography warp is still the default. Add `--use_refractive` to train with refractive geometry.

The refractive path uses two depth meanings:

* `depth_l`: camera-coordinate z-depth. This is the public output used by the existing loss, metrics, validation, and evaluation code.
* `depth_ray_l`: underwater ray distance. This is used internally by the refractive warp.

This keeps the dataset depth maps compatible with the original DTU-style z-depth supervision while using ray-distance geometry inside the cost volume.

### Dataset layout

Use the UwMVS training/validation root as `--root_dir`. In WSL, the Windows path

```text
D:\hurry\UwMVS\UwMVS_Training_Validation
```

is usually mounted as

```text
/mnt/d/hurry/UwMVS/UwMVS_Training_Validation
```

The expected directory structure is:

```text
UwMVS_Training_Validation/
  Cameras/
    pair.txt
    train/
      00000000_cam.txt
      ...
  Depths_raw/
    scanXX/
      depth_map_0000.pfm
      depth_visual_0000.png
      ...
  Rectified_UW/
    Bluish/
      scanXX_train/
        rect_001_0_r5000.png
        ...
    Greenish/
    Hazy/
    Lowlight/
  train.txt
  val.txt
  test.txt
```

`datasets/dtu.py` first looks for `train.txt`, `val.txt`, and `test.txt` directly under `--root_dir`. If they are present, you do not need to copy them into `datasets/lists/dtu/`. Use `--scan_list_dir` only if you want to keep the split files somewhere else.

### Underwater degradation selection

The UwMVS underwater images are stored under four degradation types:

* `Bluish`
* `Greenish`
* `Hazy`
* `Lowlight`

Choose them with `--uw_degradations`.

Train only Bluish:

```bash
--uw_degradations Bluish
```

Train several types:

```bash
--uw_degradations Bluish Greenish Hazy
```

Train all four types:

```bash
--uw_degradations all
```

`all` expands every scan/view/light sample into four underwater variants, so one epoch is roughly four times longer than single-type training. Use all four types if the final model should be robust to different underwater appearances. Use one type only if the target test domain is known and fixed.

### Recommended training commands

Memory-safe fixed-refractive training for a 6 GB GPU:

```bash
python train.py \
  --dataset_name dtu \
  --root_dir /mnt/d/hurry/UwMVS/UwMVS_Training_Validation \
  --uw_degradations all \
  --use_refractive \
  --freeze_refractive_params \
  --refractive_depth_chunk 1 \
  --cost_reg_checkpoint \
  --num_epochs 16 --batch_size 1 \
  --depth_interval 2.65 --n_depths 8 32 48 \
  --interval_ratios 1.0 2.0 4.0 \
  --optimizer adam --lr 1e-3 --lr_scheduler cosine \
  --exp_name uw_refractive_fixed
```

This trains the MVS network with fixed refractive parameters. It is the most stable way to start on small GPUs.

Fine-tune learnable refractive parameters after the fixed-geometry model runs:

```bash
python train.py \
  --dataset_name dtu \
  --root_dir /mnt/d/hurry/UwMVS/UwMVS_Training_Validation \
  --uw_degradations all \
  --use_refractive \
  --refractive_depth_chunk 1 \
  --cost_reg_checkpoint \
  --num_epochs 4 --batch_size 1 \
  --depth_interval 2.65 --n_depths 4 16 32 \
  --interval_ratios 1.0 2.0 4.0 \
  --optimizer adam --lr 1e-4 --lr_scheduler cosine \
  --ckpt_path ckpts/uw_refractive_fixed/<checkpoint_name>.ckpt \
  --exp_name uw_refractive_learnable
```

Remove `--freeze_refractive_params` to make `z_inner`, `glass_thickness`, and `n_glass` learnable. This needs more GPU memory because the Newton projection graph must be kept for backpropagation.

### Important training options

`--use_refractive`: Enables refractive cost-volume warping. Without this flag the model uses the original pinhole homography warp.

`--z_inner`: Distance from camera center to the inner glass surface. Default: `10.0`.

`--glass_thickness`: Flat-port glass thickness. Default: `5.0`.

`--n_air`, `--n_glass`, `--n_water`: Refractive indices. Defaults: `1.0`, `1.52`, `1.333`.

`--freeze_refractive_params`: Keeps `z_inner`, `glass_thickness`, and `n_glass` fixed. This greatly reduces memory because the refractive projection grid is computed without storing the Newton backprop graph. The network still trains normally, but the refractive physical parameters are not learned.

`--refractive_newton_iters`: Number of Newton iterations for forward refractive projection. Default: `4`. Larger values may improve projection accuracy but cost more time and memory.

`--refractive_depth_chunk`: Number of depth hypotheses projected at once in refractive warp. Smaller values reduce peak memory but slow training. On a 6 GB GPU use `1`; on larger GPUs try `2` or `4`.

`--cost_reg_checkpoint`: Uses PyTorch checkpointing for the 3D cost regularization network. This saves activation memory during training by recomputing the 3D CNN in backward pass. It makes training slower but is useful on small GPUs.

`--uw_degradations`: Underwater image type selection. Use one or more of `Bluish`, `Greenish`, `Hazy`, `Lowlight`, or use `all`.

`--scan_list_dir`: Optional directory containing `train.txt`, `val.txt`, and `test.txt`. If not set, the dataset uses the split files under `--root_dir` when present.

`--n_depths`: Number of depth hypotheses at each cascade level. Default: `8 32 48`. Reducing this, for example to `4 16 32`, is the most direct way to lower memory and speed up debugging.

`--interval_ratios`: Multiplier for `--depth_interval` at each cascade level. Default: `1.0 2.0 4.0`.

`--depth_interval`: Finest-level z-depth interval in dataset units. UwMVS/DTU-style training uses `2.65` by default.

`--n_views`: Number of views including the reference view. Default: `3`. Increasing it usually improves MVS matching but increases memory and runtime.

### Memory notes

Refractive training is heavier than the original homography warp because it needs ray construction, pose transform, Newton projection, and `grid_sample`. Approximate guidance for `640x512`, `batch_size=1`, `n_views=3`, and `n_depths 8 32 48`:

* 6 GB: use `--freeze_refractive_params --refractive_depth_chunk 1 --cost_reg_checkpoint`.
* 8 GB: try learnable refractive parameters with `--refractive_depth_chunk 1` and checkpointing, or keep parameters frozen and increase chunk size.
* 10-12 GB: more suitable for full learnable refractive training.
* 16 GB+: comfortable for larger chunks, more views, or larger depth counts.

If CUDA OOM occurs, try these in order:

1. Set `--refractive_depth_chunk 1`.
2. Add `--cost_reg_checkpoint`.
3. Add `--freeze_refractive_params`.
4. Reduce `--n_depths`, for example `4 16 32`.
5. Reduce `--n_views`.

## DTU dataset

### Data download

Download the preprocessed [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view) and [Depth_raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip) (replace the `Depths` folder in `mvs_training/dtu` with the `Depths/scanXX` in `Depth raw`). from original [MVSNet repo](https://github.com/YoYo000/MVSNet) and unzip. For the description of how the data is created, please refer to the original paper.

### Training model

Run (example)
```
python train.py \
   --dataset_name dtu \
   --root_dir $DTU_DIR \
   --num_epochs 16 --batch_size 2 \
   --depth_interval 2.65 --n_depths 8 32 48 --interval_ratios 1.0 2.0 4.0 \
   --optimizer adam --lr 1e-3 --lr_scheduler cosine \
   --exp_name exp
```

Note that the model consumes huge GPU memory, so the batch size is generally small.

See [opt.py](opt.py) for all configurations.

### Example training log
![log1](assets/log1.png)
![log2](assets/log2.png)

### Metrics
The metrics are collected on the DTU val set.

|           | resolution | n_views | abs_err | acc_1mm  | acc_2mm   | acc_4mm    | GPU mem in GB <br> (train*/val) |
| :---:     |  :---:     | :---:   | :---:   |  :---:   | :---:     | :---:      | :---:   |
| Paper     |  1152x864  | 5       | N/A     | N/A      | 82.6%     | 88.8%      | 10.0 / 5.3 |
| This repo <br> (same as paper) |  640x512   | 3       | 4.524mm | 72.33%  | 84.35%     | 90.52%     | 8.5 / 2.1 |
| This repo <br> (gwc**) |  640x512  | 3       | **4.242mm**| **73.99%** | **85.85%** | **91.57%**    | **6.5 / 2.1** |

*Training memory is measured on `batch size=2` and `resolution=640x512`.

**Gwc with `num_groups=8` with parameters `--depth_interval 2.0 --interval_ratios 1.0 2.5 5.5 --num_epochs 50`, see [update](#update) 1. This implementation aims at maintaining the concept of cascade cost volume, and build new operations to further increase the accuracy or to decrease inference time/GPU memory.

### Pretrained model and log
Download the pretrained model and training log in [release](https://github.com/kwea123/CasMVSNet_pl/releases).
The above metrics of `This repo (same as paper)` correspond to this training but the model is saved on the 10th epoch (least `val_loss` but not the best in other metrics).

------------------------------------------------------------------------------------------------------------------------

## BlendedMVS

Run
```
python train.py \
   --dataset_name blendedmvs \
   --root_dir $BLENDEDMVS_LOW_RES_DIR \
   --num_epochs 16 --batch_size 2 \
   --depth_interval 192.0 --n_depths 8 32 48 --interval_ratios 1.0 2.0 4.0 \
   --optimizer adam --lr 1e-3 --lr_scheduler cosine \
   --exp_name exp
```
The `--depth_interval 192.0` is the product of the coarsest `n_depth` and the coarsest `--interval_ratio`: `192.0=48x4.0`.

### Some modifications w.r.t original paper

Since BlendedMVS contains outdoor and indoor scenes with a large variety of depth ranges (some from 0.1 to 2 and some from 10 to 200, notice that these numbers are not absolute distance in mm, they're in some unknown units), it is difficult to evaluate the absolute accuracy (e.g. an error of 2 might be good for scenes with depth range 10 to 200, but terrible for scenes with depth range 0.1 to 2). Therefore, I decide to scale the depth ranges roughly to the same scale (about 100 to 1000). It is done [here](https://github.com/kwea123/CasMVSNet_pl/blob/cc483ce7e421329e163f965af809f62e5b0d5a35/datasets/blendedmvs.py#L98-L103). In that way, the depth ranges of **all scenes** in BlendedMVS are scaled to approximately the same as DTU (425 to 935), so we can continue to use the same metrics (acc_1mm, etc) to evaluate predicted depth maps.

Another advantage of the above scaling trick is that when applying model pretrained on DTU to BlendedMVS, we can get better results since their depth range is now roughly the same; if we do without scaling, the model will yield very bad result if the original depth range is for example 0.1 to 2.

### Pretrained model and log
Download the pretrained model and training log in [release](https://github.com/kwea123/CasMVSNet_pl/releases).

------------------------------------------------------------------------------------------------------------------------

## Some code tricks

Since MVS models consumes a lot of GPU memory, it is indispensable to do some code tricks to reduce GPU memory consumption. I tried the followings:
*  Replace `BatchNorm+Relu` with [Inplace-ABN](https://github.com/mapillary/inplace_abn): Reduce the memory by ~15%!
*  `del` the tensor when it is never accessed later: Only helps a little.
*  Use `a = a+b` in training and `a += b` in testing: Reduce about 300MB (don't know the reason..)

# Testing

For depth prediction example, see [test.ipynb](test.ipynb).

For point cloud fusion from depth prediction, please go to [evaluations](evaluations/) to see the general depth fusion method description, then go to dataset subdirectories for detailed results (qualitative and quantitative).

A video showing the point cloud for scan9 in DTU in different angles and [me](https://github.com/kwea123/VTuber_Unity) (click to link to YouTube):
[![teaser](assets/demo.gif)](https://youtu.be/wCjMoBR9Nh0)

## Point cloud to mesh

You can follow [this great post](https://towardsdatascience.com/5-step-guide-to-generate-3d-meshes-from-point-clouds-with-python-36bad397d8ba) to convert the point cloud into mesh file. **Poisson’ reconstruction** turns out to be a good choice. Here's what I get after tuning some parameters (the parameters should be scene-dependent, so you need to experiment by yourself): ![a](https://user-images.githubusercontent.com/11364490/80682209-0ac0fc00-8afd-11ea-86c7-30ee81fc3ad1.png)
