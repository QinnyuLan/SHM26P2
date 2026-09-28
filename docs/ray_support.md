# 训练 SfM 稀疏深度支持

`src/bridge_rgs/ray_support.py` 提供经典稀疏深度约束的独立辅助模块，不是新的学术贡献。它不修改 train/model、不启动训练、不读 RGB 或语义 GT；只读取初始 train-only SfM 几何、轨迹和相机，接收调用者提供的有效像素 mask。

## 调用接口

```python
from bridge_rgs.ray_support import SparseDepthSupport

# manifest 可以是路径，也可以是已经读取的 dict；初始化一次。
support = SparseDepthSupport.from_manifest(
    manifest, config=config.get("sparse_depth", {}))

# 保留 support.provenance 到运行记录；由调用者设置哪些 scene 参数可训练。
loss, audit = support.loss(
    rendered["depth"], rendered["alpha"], data["valid"],
    data["K"], data["w2c"], view["image_id"])
if loss is not None:
    total = total + config["sparse_depth_weight"] * loss
stats.update(audit)
```

权重为 0 时调用者应完全跳过本分支。没有有效目标时返回 `None`，避免用人为零损失构造梯度触发旧优化器动量。`loss` 支持 H×W 或 H×W×1 的相机 z 期望深度及 alpha；不是沿单位射线的欧氏距离，也不是逆深度。gsplat `RGB+ED` 的当前输出符合相机 z 约定。

所有目标、像素位置、置信度、相机参数和 valid 均 detach。Alpha 只作为 detach 后的门槛/权重，防止直接通过降低权重减少损失；深度本身的渲染计算图仍可把梯度传到 opacity。调用者若只允许 opacity 更新，必须冻结位置、尺度、旋转、SH、语义、相机及其他参数；本辅助模块不控制优化器。

## 可审计筛选与默认配置

```yaml
sparse_depth:
  min_observations: 3
  max_reprojection_error_px: 1.0
  max_current_reprojection_error_px: 2.0
  max_relative_depth_std: 0.10
  relative_depth_std_scale: 0.02
  observation_saturation: 8
  min_confidence: 0.05
  border_px: 2
  min_alpha: 0.5
  max_targets_per_view: 4096
  uv_mode: observed
  loss_kind: log_huber
  huber_delta: 0.1
  near_error_fraction: 0.1
```

- 初始化缓存每个 TRAIN view **实际保存的观测成员**，不将全部点投到全部相机充当证据。任何 NPZ 观测 ID 不属于 TRAIN、重复观测计数或非训练 view 的调用均报错。
- `reprojection_error` 是准备阶段每点训练观察的 RMS 像素误差，阈值按源原图像素定义；本数据这些 RMS 的中位数为 0.32674 px。当前观察额外检验的是原始 UV 与当前投影之差，单位为 manifest 原生网格像素，随训练下采样保持同一门槛。
- 默认从 `dataset_root/camera_parameters/images.txt` 恢复原始 TRAIN keypoint，经源相机 K/畸变去畸变得到归一化射线，再乘当前 K。读取验证观察行后立即丢弃，不解析其数值。数据中重复 image/track keypoint 按 `geometry.triangulate_tracks` 一样取最后一项：此次共 8,465 个重复项，记录在 provenance。
- 若源轨迹缺失，默认报错。只有显式 `uv_mode: projected` 才改用当前相机投影的点位置作为采样 UV，provenance 明确标为近似；它与原始测量 UV 并不相同。没有修改 prepared NPZ 文件。
- 使用正深度、有限值、位置协方差半正定检查、原生边缘与 eroded valid 支持。后者要求双线性采样的邻域都有效。目标过多时按固定索引等间距取最多 4,096 个，没有随机改变训练视图序列。
- 使用固定相机条件下的位置深度方差 `var_z = R_z @ Sigma_SfM_position @ R_z.T`，相对标准差 `s = sqrt(var_z)/z`。**Gaussian 椭球形状不参与此不确定度**，也没有把该近似协方差称为校准后验；忽略相机不确定度是这里的明确条件。

通过门槛的点置信度为观测数饱和项 `min(n/8,1)`，乘以三个 `1/(1+x²)` 因子：全轨迹重投影误差/1px、相对深度标准差/0.02、当前观察重投影误差/2px。它是工程可靠性权重，不是正确率。损失为置信度及 detach alpha 加权均值的 Huber(`log(rendered_z/target_z)`)，可显式选 `relative_huber`。场景长度和位置协方差分别按 a、a² 同比缩放时，目标筛选、权重与损失不变。

## 实际输入审计与测试

仅 CPU 读取原始训练轨迹，60,000 点中 52,547 点通过观测数/全轨迹误差/协方差初筛。350 个 TRAIN view 都有可用几何目标；经过正 z、当前 UV、相对方差和有效边缘筛选及 cap 后，每视角数量 min/median/max 为 120/832/4,096，共 382,880 个 view-point 目标。邻近 235 的训练相机 234/236/237 分别有 572/485/489 个目标；235 是验证相机，模块拒绝调用。以上尚未计入 rendered alpha 门槛，真实有效数量由运行日志报告。完整审计见 `artifacts/prepared/ray_support_audit.json`。

`uv run pytest tests/test_ray_support.py -q`：10 个 CPU 测试通过，涵盖 TRAIN 隔离、重复观测、原始 UV 与双线性采样、质量/边缘/正 z/协方差门槛、尺度不变性、梯度隔离与无支持分支。每步日志含候选/目标/有效数量、置信度质量、relative error 中位数、alpha 均值和 `rendered_z < 0.9*target_z` 的过近比例。

**限制：期望深度可以被前后层抵消。** 例如深度 1 与 3 的等权混合满足目标 2，却依然存在前方层；单测明确覆盖此反例。现模块没有测量前方累计不透明度，不是完整自由空间约束、第二矩损失或终止深度分布监督，不预言能移除所有浮层。首轮应采用相同 1k 预算的 opacity-only RGB 对照与 RGB+depth 对照；深度权重与门槛预先固定，不从 235 GT 或其他验证视角选择。支持目标覆盖率和训练 RGB 损失应一起报告，不能只看深度损失降低。

## 一手先例与表述边界

[DS-NeRF（CVPR 2022，作者项目页）](https://www.cs.cmu.edu/~dsnerf/) 已使用 SfM 稀疏深度及不确定度监督射线终止分布。本模块只实现期望深度的鲁棒误差，没有复现其完整分布损失。[LoopSparseGS（作者官方仓库，IEEE TIP 2025）](https://github.com/pcl3dv/LoopSparseGS) 将可靠 SfM 稀疏深度与单目深度用于 Gaussian 几何监督，直接说明“向 GS 加 SfM 深度”本身不能作为创新点。[原始 3DGS 官方实现的 depth regularization](https://github.com/graphdeco-inria/gaussian-splatting#depth-regularization) 也已提供深度正则，并说明跨场景收益可能为正、很小或负。这里的作用是一个可审计、严格 TRAIN-only 的经典控制项，任何最终收益必须由同预算配对实验证明。

## 首次真实 CUDA 预检

`runs/opacity_depth_preflight/preflight_report.json` 保存了从完整语义 8k checkpoint 开始的两步测试。唯一 trainable 与唯一变化的 state key 都为 `splats.opacity_logits`；其他所有模型 tensor、语义头和训练相机逐位不变。完整 SH degree=3；两步有效目标数 1,062/120，单独深度项的 opacity 梯度范数为 5.07e-4/1.55e-3，均有限且非零。随后正式配对使用 `configs/opacity_rgb_control.yaml` 与 `configs/opacity_sfm_depth.yaml`，仅 output 与 sparse_depth_weight=0/.05 不同，第二组复用第一组的 source_snapshot。

训练配置允许 `sparse_depth.colmap_images_path`：train setup 把它分离为 `from_manifest(..., colmap_images_path=...)`，不混入严格数值配置 schema。开启深度的运行会保存 `sparse_depth_provenance.json`，包含 init NPZ 和原始轨迹文件 SHA；checkpoint/manifest/source SHA 由 `run_experiment.py` 保存。首次真实预检也覆盖了这条显式路径配置。
