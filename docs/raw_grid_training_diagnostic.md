# TRAIN 原图往返插值与畸变训练接口诊断

2026-09-26 的固定 16 TRAIN 视图 CPU 诊断显示：**原图经 v2 prepared 去畸变、再映射回原图，会产生可测的平滑**。实际 prepared PNG 加最终 uint8 量化，在共同完整窗口上的平均 PSNR 为 **39.773805 dB**、7×7 box SSIM **0.97514149**、11×11 Gaussian SSIM **0.97387455**；亮度 Sobel 平方能量保留 **74.0572%**。这不是模型评价、性能上限或新机制，不能从 held-out 模型分数中直接减去该误差。

## 先锁定的来源与比较定义

运行前生成 [固定计划](../artifacts/raw_grid_training_diagnostic_plan.json)，SHA `84fca9e2a1da491a2b800f9c874d5900eefb184e1096f3a071834c2d8e31f0a8`。随后才解码任何原图/prepared RGB。固定样本完全复用已有 teacher signal audit 的名单：将 259 个 labeled TRAIN 按名字排序，取索引 `i*258//15, i=0…15`：

`002, 021, 041, 059, 079, 100, 118, 137, 156, 176, 200, 220, 241, 259, 278, 300`。

只读取这 16 个 TRAIN 的 raw RGB、v2 prepared RGB 与 valid 图片；是否带标签只取 manifest 中的路径存在信息，**不读取 mask 或 annotation 内容，也不读取 VAL 图像像素**。相机与 split 元数据允许读取，无 pose 优化、模型推理、LPIPS 或 GPU 调用。

计划 SHA 绑定 raw/prepared/valid、v2 manifest、原样本计划、uv.lock、runner 与实际调用的三个模块。映射调用来自两阶段训练的冻结 source `runs/corner_v2_preparation/source_snapshot`，而非后续 main：

- `data.undistortion_maps`：显式 `colmap_corner_v2`，原生 1320×989、原 K、不缩放；OpenCV 双线性、常数外边界。
- `evaluate.distortion_render_grid`：按 v2 corner 共轭得到原图像素到 pinhole 的反向采样位置；先去掉 render canvas 的整数平移，再从实际 native prepared 采样。本相机最终 canvas 仍为 1320×989。
- 全部 16 张实际 prepared PNG 均由原图和该冻结去畸变映射**逐像素精确重现**。运行前后输入 SHA 均保持不变。

预先固定三种变体，不选择最好的一个：

1. **全浮点往返**：raw uint8/255 → float32 去畸变 → float32 回原图，避免中间/最终 uint8 量化。
2. **实际 prepared，浮点输出**：实际 prepared PNG/255 → float32 回原图，包含 prepared 保存时的舍入。
3. **实际 prepared，uint8 输出**：上一结果以 `np.rint(*255)` 量化后再除以 255，对应最终输出量化。

所有映射固定为本机 `cv2.INTER_LINEAR`；没有更换插值核或调整畸变系数。**随后实现可导对应算子时发现并纠正了一项预设**：安装的 OpenCV 5.0.0 float-map remap 使用连续双线性权重，不能假定它经过 1/32 离散系数表。原计划中的表近似描述作为历史记录保留，不重写；取数脚本一直调用实际 cv2，故上报数值不变。[后续 CPU 合同](../artifacts/raw_warp_cpu_contract.json)记录该勘误以及显式 fixed-map 与 float-map 的区别。OpenCV 官方迁移页也说明新版 warping 对表近似的修改，但本环境合同以实际探针为准。[官方迁移说明](https://github.com/opencv/opencv/wiki/OpenCV-4-to-5-migration)

## 支持区域与指标

原图的一个像素只有在其反向坐标位于 prepared 中，且所有 floor/ceil 双线性邻居均为 prepared-valid 时，才进入共同支持。该规则对 OpenCV 权重表舍入为零的邻居仍保守检查。映射外部可以在运算中产生零，但**这些像素及其受影响窗口不参与评分；不复制 GT、补边或将结果称作全图评分**。

每个视图共有 1,305,480 像素：

| 区域 | 像素数/视图 | 原图覆盖率 | 用途 |
|---|---:|---:|---|
| 共同双线性有效支持 | 1,300,776 | 99.639673% | 另报像素 PSNR |
| 共同支持内完整 7×7 窗口中心 | 1,286,886 | 98.5757% | 只记录覆盖，不作为主比较支持 |
| 共同支持内完整 11×11 窗口中心 | 1,277,666 | 97.869443% | 三个主指标及边缘能量共用 |

主 PSNR、7-box SSIM 和 11-Gaussian SSIM **使用完全相同的 11×11 完整窗口中心集合**。因此两种 SSIM 的差别不混入支持区域变化。SSIM 均使用 population covariance、range=1、常数 .01²/.03²、RGB 通道平均；Gaussian 为 11×11、sigma=1.5。计算采用 float64。7-box 与 `losses.ssim_map` 的训练损失窗口/公式一致，但这里使用完整窗口支持。实际 native 评价（包括 107b/de8f 冻结快照）本来就是 skimage 11-Gaussian 加 valid 的 11×11 侵蚀；不能将 7-box 称为 native 评价定义。

边缘统计在相同集合上用 encoded RGB 的固定 Rec.709 亮度、3×3 Sobel/8，记录平方梯度均值与梯度模长均值。能量比例是按像素汇总后的比值，不是挑选某种边缘阈值。没有 VAL 阈值搜索。

## 固定结果

下表全部在共同完整 11×11 支持上；PSNR/SSIM 为 16 个视图的算术均值，边缘为 pooled 能量比：

| 变体 | PSNR dB ↑ | SSIM 7-box ↑ | SSIM 11-Gaussian ↑ | 平方边缘能量保留 | 梯度模长保留 |
|---|---:|---:|---:|---:|---:|
| 全浮点往返 | 39.816415 | 0.97564138 | 0.97438442 | 73.8849% | 83.5169% |
| 实际 prepared，浮点输出 | 39.800783 | 0.97545468 | 0.97419954 | 73.9603% | 83.7898% |
| 实际 prepared，uint8 输出 | 39.773805 | 0.97514149 | 0.97387455 | 74.0572% | 84.1781% |

在侵蚀前的共同有效像素上，三者平均 PSNR 分别为 **39.823768 / 39.808125 / 39.781176 dB**。主支持 pooled-MSE PSNR 分别为 **39.527850 / 39.514126 / 39.490244 dB**，与平均逐视图 dB 的口径不同。最后一种逐视图主 PSNR 范围为 **37.849501–44.368993 dB**。

在本样本/支持下，中间与最终量化合计只让平均主 PSNR 从 39.816415 降至 39.773805，差约 **0.042611 dB**；双线性平滑仍存在于全浮点变体。量化使高频能量略增加，不表示恢复了真实细节。最后一种在同支持上，7-box 与 11-Gaussian SSIM 相差约 **0.001267**；不能将这个 TRAIN 变换实验的差值套用于模型的 VAL 预测。

这里比较的是输入图像的往返插值。真实模型可能欠拟合、拟合插值后的图像，或产生不同频谱误差；误差之间会相关。**39.77 dB 不是模型可达到分数的上限，也不是损失中可独立扣除的常数。**原 native 与共同 official 评价改变原图/去畸变网格、完整边界支持、uint8 与畸变回映射；两者 SSIM 核均为 11-Gaussian，不能归因于 SSIM 窗口改变。已有两份模型的指标变化不能归因于模型回退，本诊断也没有逐项分解其 VAL 指标差。

## 本地 gsplat 1.5.3 梯度合同

本节是源码审查，不是 CUDA 数值验证。安装包 metadata 为 1.5.3；[机器可读合同](../artifacts/raw_grid_training_gradient_contract.json)记录实际本地三个文件的路径/SHA。官方 tag 链接用于核对设计，不声称本地文件与 tag 逐字节相同。

| 路径 | 已确认的梯度范围 | 不能直接沿用的部分 |
|---|---|---|
| `with_ut=True, with_eval3d=False` | 2D 合成仍可更新颜色/opacity；UT 投影输出对输入不可导 | 无投影几何梯度；SH view-direction 可能产生间接 means 梯度，因此只检查 `means.grad != None` 不充分 |
| `with_ut=True, with_eval3d=True` | 从世界空间 ray–Gaussian response 的独立 backward 返回 means、quats、scales、colors、opacities，背景也有梯度 | UT 投影/离散 tile 调度仍不可导；相机、intrinsics、畸变、AbsGrad 及部分深度路径另有限制 |

`fully_fused_projection_with_ut` 明确警告对所有输入不可导；但 `_RasterizeToPixelsEval3D.backward` 调用独立 CUDA 反向算子，并返回上述几何导数。不能仅凭 UT 的警告推论整个 eval3d 分支完全不能训练几何。[官方 wrapper](https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/gsplat/cuda/_wrapper.py)

CUDA backward 直接对 ray–Gaussian 响应计算 means/quaternion/scale VJP，再累加相应梯度；这条路径不依赖 UT 投影的反向。pinhole 畸变读入的是 6 个 radial 系数，tangential 为 2 个；本 SIMPLE_RADIAL 应显式映射为 `[k1,0,0,0,0,0]` 和 `[0,0]`，不能把 OpenCV 的混合 4/5 元数组直接当 radial 传入。像素中心仍为 `(j+.5,i+.5)`。[官方 CUDA backward](https://raw.githubusercontent.com/nerfstudio-project/gsplat/v1.5.3/gsplat/cuda/csrc/RasterizeToPixelsFromWorld3DGSBwd.cu)

本地接口还存在下列限制：

- 畸变要求 `with_ut=True`；UT/eval3d 要求显式 quats/scales、`packed=False`、`sparse_grad=False`。
- eval3d 的 `viewmats.requires_grad` 会抛 `NotImplementedError`；Ks、畸变和相关相机参数不返回梯度，必须固定相机。
- eval3d 调用未传 `absgrad`，`means2d` 又来自不可导 UT，现有依赖 `means2d.absgrad` 的增密流程不能直接复用。
- antialias 的 UT opacity compensation 对几何不可导；ED 使用的 UT 中心深度值也不可导。合成权重仍可能对几何有导数，但这不等价于完整中心深度监督梯度。[官方 rendering 分支](https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/gsplat/rendering.py)

因此当前不能将 `with_ut` 一行开关当作等价的原图训练替代。若后续获授权，最小前置验证应是固定相机、classic 模式、少量 Gaussian、关闭 SH 方向混杂，用 RGB/alpha 的有限差分分别验证 means/quats/scales/opacity/colors，并验证畸变系数布局、零畸变前向差异及 metadata。通过后仍需独立处理增密和深度合同；不得预言原图训练一定改善成绩。本轮**未实现或运行**这些 GPU 实验，也未修改模型/训练源代码。

## 复核入口

- [诊断脚本](../scripts/diagnose_raw_grid_roundtrip.py)：`plan` 先固定来源与定义，`run` 验证 SHA 后执行；已有计划/结果时拒绝覆盖。
- [结果 JSON](../artifacts/raw_grid_training_diagnostic.json)：含 16 视图逐项结果、pooled 与 mean-view 统计、运行前后输入校验。SHA `44037c9617d7761ad7b5336b39ff678bb2b913ca2af87639abc24e90026e3c16`。
- [梯度合同 JSON](../artifacts/raw_grid_training_gradient_contract.json)：只读代码证据、文件 SHA、明确未进行数值检验。
- `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 uv run pytest tests/test_raw_grid_roundtrip.py -q`：**5 passed**，覆盖边界/无效邻居排除、完整窗口、常量/identity、Gaussian SSIM 与 skimage 一致、棋盘插值及无效区域隔离；Ruff 通过。

结果只增加小型 JSON 与文档，不保存派生图像，不删除原产物。
