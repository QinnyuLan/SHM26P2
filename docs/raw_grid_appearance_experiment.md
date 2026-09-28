# 原始畸变网格颜色精修：固定双臂工程实验

2026-09-26，先登记方案，再执行。动机来自[16 TRAIN 的往返插值诊断](raw_grid_training_diagnostic.md)：去畸变与映射回原图会平滑细节；这不证明任何模型训练策略有效。已有共同原图评价中，新 v2 未整体超过旧 support，因此本轮也保留未经继续训练的 v2 和旧 support 结果。

**这是一项训练目标/网格的工程对照，不是学术创新，也不单独识别插值的因果效应。** 相机、几何、opacity、三维语义、精修器都冻结，但颜色改变可能影响精修器的输入与最终 mask，必须重新评价同场的 RGB 与语义。

## 固定变量

- 两臂从 v2 semantic 8k 的同一个完整终点出发，SHA `a77d304f32de4356c1a4608ed5192258b6a9ced4c67a4297809f439481408d89`，绑定 v2 manifest `91b41aeecedc352e4882eb80a1bb635d479ae432251a4c96c85a9dc110cbb327`。
- 全部 350 TRAIN，包含无标签图；相同排序、独立 `random.Random(42)` 无放回循环、相同 3,000 步。不用标签、teacher、额外图像、增密或相机更新。
- 只训练 `splats.sh0`、`splats.sh_rest`、`background_logits`，Adam 的学习率固定为 `2.5e-4 / 1.25e-5 / 1e-4`，betas .9/.999，epsilon 1e-15，无调度、权重衰减。没有中间 VAL 或中间选点。
- 两臂都用现有 antialiased pinhole renderer 渲染同样的覆盖画布，完整 SH degree 3，先显式 clamp RGB 到 [0,1]。
- 00 native：裁取原生 pinhole 区域，对 prepared RGB/valid 监督。01 original：固定可导畸变映射，对原始 RGB 全图监督。两臂完整像素支持/角度采样、目标平滑程度、边缘覆盖因此不同，不能把差值只解释为去除平滑。
- 共同损失为 .8 masked RGB L1 + .2 masked (1−7×7 box SSIM)。L1 使用各自有效像素；SSIM 仅使用各自 valid 内的完整 7×7 窗口，不将边界填成 GT。它是训练损失，最终评价仍用固定的 11×11 Gaussian SSIM。

## 前向合同与磁盘

CPU 检查发现，本机 OpenCV 5.0.0 的浮点坐标 remap 使用连续双线性权重，不能用 `convertMaps` 的 1/32 权重表代替。`raw_grid.py` 先测定实际分支；本次计划锁定 `continuous_float32`，固定四邻索引/权重，只对图像颜色求梯度。官方导出代码及历史评价不变。纯合成图、实际 002 相机的[CPU合同](../artifacts/raw_warp_cpu_contract.json)里，四邻 gather 相对官方 float remap 最大误差约 1.79e-7；常规 normalized `grid_sample` 有更大的坐标舍入误差，未采用。

正式训练前，在排序第一张 TRAIN 上进行一次无优化检查：可导 warp 相对原 cv2 的 float RGB 最大差 ≤2e-6；覆盖画布裁图相对原生直接渲染最大差 ≤2e-5；两种损失只有三组颜色参数有有限且非零梯度。若失败，停止并保留回执，不能悄悄放宽容差。这里不要求不同后端浮点输出逐位相同。

仅保存最终[appearance delta](appearance_delta_checkpoints.md)，精确依赖不可变 base SHA；每臂约 91.22 MiB，不复制冻结的完整场，不保存 optimizer/RNG，不能普通 resume。两臂 checkpoint、原图评价 PNG、原子保存临时文件与至少 256 MiB 余量合计在 1,152 MiB 初始空闲门槛内。不删除既有文件。每臂最多 1,800 秒训练；超时停止，不能暗改步数。

源码、350 张 TRAIN 的两种 RGB/valid、原始 base 评价、checkpoint/manifest/uv.lock 在准备时绑定 SHA。训练后逐张量核对冻结状态，完整重构 delta 后再次核对所有模型张量与相机。官方评价仍为单场单次精修，不加 TTA/teacher；先生成所有 50 张预测，再读取原图/标注评分。

## 终点与解释

两臂固定 final 都执行共同原图 50 RGB / 41 semantic 评价。报告 raw−native、两臂各自−未经续训 v2 的全部 RGB/语义指标与 5,000 次配对视图 bootstrap；旧 support 同协议成绩也保留用于判断是否存在新的整体改进。

本轮的工程继续投入门槛预先定为：original 相对 native 和原始 v2 的 PSNR 都至少 +0.15 dB，且对应配对区间下界大于零；SSIM 点估计不降低、LPIPS 不增加，五类/拉索 IoU 相对原始 v2 不下降超过 .20 个百分点。这些是本项目的投入规则，不是通用显著性定义。未满足时完整报告并停止，不扫描学习率、训练步数或挑选中间 checkpoint。即使通过，也不能据此宣布同协议胜过同学或形成学术贡献。

运行入口为 `scripts/run_raw_grid_appearance.py --prepare`，执行时使用生成的源快照与 `--run <plan.json>`。结果目录为 `runs/raw_grid_appearance_v1`；本文登记时尚无结果。
