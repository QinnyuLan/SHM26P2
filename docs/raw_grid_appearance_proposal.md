# 原图网格 appearance-only 工程对照：接口与预检方案

推荐先使用**现有 antialiased pinhole splat → 官方 overscan canvas → 固定相机可导畸变 warp**。只更新 SH 与背景颜色，保留当前 plain 官方推理路径。`with_ut+with_eval3d` 虽有独立几何 backward，但前向高斯响应也改变；它应另立 renderer 协议，不能混入本次 native/raw 目标网格对照。

此文件记录实现边界与 CPU 证据，不表示已完成 GPU 预检或训练。主 runner 由根任务管理，新增 helper 本身不会启动训练。

## 固定双臂

共同 base 为 `runs/corner_v2_semantic_coupled/last.pt`，SHA `a77d304f32de4356c1a4608ed5192258b6a9ced4c67a4297809f439481408d89`。两臂各 3,000 步、相同 350 TRAIN 名单与独立 shuffle seed42，相同 overscan renderer、full SH3，从 fresh Adam 开始。允许更新的张量严格只有 `splats.sh0`、`splats.sh_rest`、`background_logits`；相机、位置、形状、opacity、语义特征、decoder/refiner、人工 prior 全冻结。

| 项目 | `00_native` | `01_original` |
|---|---|---|
| 共同预测 | 同一 overscan RGB，先 clamp(0,1) | 同左 |
| 取出监督网格 | 按整数 offset 裁出原 prepared canvas | 连续 float-map 四邻可导 warp 到完整 raw grid |
| 目标 | v2 prepared RGB | 原始 RGB |
| RGB 支持 | prepared valid | 原始全图；要求所有非零插值 tap 被 renderer canvas 覆盖 |
| SSIM 支持 | valid 内完整 7×7 窗口 | 图内完整 7×7 窗口 |

两臂统一 loss 为 `.8*masked_L1 + .2*mean(1−SSIM7box)`，SSIM 使用完整有效窗口；不读取语义标签，不作 region weighting，不填补边界，不在训练输出上作 uint8 量化。固定学习率为 SH0 `2.5e-4`、SH-rest `1.25e-5`、背景 `1e-4`；Adam betas=(.9,.999)、eps=1e-15、无 schedule/weight decay。

这是目标网格、目标预处理及有效支持共同变化的工程对照，**不是单一插值因素的因果实验**。原始全图监督覆盖了 native valid 未监督的区域；最终应报告两臂实际 RGB 像素数和 SSIM 窗口数。要隔离目标插值本身，未来需要两臂均在同一 raw lattice/common support、只改变 raw 与 roundtripped-prepared target 的另一项实验；本次不扩充该臂。

冻结 semantic 参数不保证最终语义预测不变：refiner 读取渲染 RGB，因此颜色更新仍可影响其输出。终点必须用同一 plain 官方协议重测完整 50 RGB / 41 semantic，不只评价 RGB，也不宣称最终 mask 不变。

## 可导 warp 的实际合同

`src/bridge_rgs/raw_grid.py` 提供：

```python
layout = build_raw_grid(K_numpy, opencv_distortion, width, height,
                        protocol="colmap_corner_v2")
layout.warp.to(device)
rendered = scene.render(K=..., w2c=..., width=layout.render_width,
                        height=layout.render_height, semantics=False,
                        absgrad=False)["rgb"].clamp(0, 1)
native_rgb = layout.crop_native(rendered)
original_rgb = layout.warp(rendered)
loss, stats = appearance_rgb_loss(prediction, target, valid)
```

`layout.render_K`、render_width/height 为原 official `distortion_render_grid` 的相同 covering canvas。`native_left/top/width/height` 给出整数平移裁剪；本最小接口要求 prepared 与 source 的 native K/size 一致，不支持任意 resized/different-K canvas。`layout.warp.receipt()` 记录坐标/权重 SHA、实际 OpenCV 分支、完整覆盖数量。warp 无训练参数；图像值可导，坐标/K/畸变固定。

最初的 1/32 表假设被 CPU 测试否定：当前 OpenCV 5.0.0 的 float-map remap 使用连续权重；`convertMaps` 后显式 fixed-map remap 才符合量化表。helper 用确定性小探针识别本机实际分支，未知或歧义分支直接报错。本轮 runner 还必须固定 `warp.weight_policy == "continuous_float32"`，不允许更换环境后悄悄改协议。官方 warp 源代码不变。[OpenCV 官方迁移说明](https://github.com/opencv/opencv/wiki/OpenCV-4-to-5-migration)

因此当前采用原 float32 map 的 floor/ceil 位置和连续分数，torch gather 四个邻居后作固定加权；不用 `grid_sample` 的归一化坐标往返。所有非零 tap 必须落在渲染 canvas 内；没有用 GT 补边。训练先 clamp 后 warp，与现 plain 官方导出计算顺序一致；clamp 的饱和区梯度为零是这项共同协议的一部分。

### CPU 证据

`artifacts/raw_warp_cpu_contract.json` 使用固定 002 TRAIN **相机元数据**与合成 ramp/checker/random 图，不读任何真实 RGB/GT。全 1,305,480 个 raw 输出像素均有完整插值支持。

| 合成输入 | 四tap vs 官方 float-map 最大误差 | float32 grid_sample 最大误差 | 显式 1/32 表 vs 官方 float-map 最大误差 |
|---|---:|---:|---:|
| 线性渐变 | 1.788e-7 | 2.384e-7 | 1.591e-5 |
| 棋盘 | 5.960e-8 | 1.806e-4 | 3.035e-2 |
| seed42 随机 | 1.788e-7 | 1.638e-4 | 2.676e-2 |

应用前向容差固定 `2e-6`，未放宽来隐藏错误。CPU unit test 对小随机输入使用更紧的 `2e-7`；double gradcheck 通过。12 个相关测试覆盖：显式 float/fixed-map 区别、固定坐标图像梯度、越界拒绝与零权重边界、identity/native crop/负畸变 overscan、完整 SSIM 窗口、RGB 图边界保留、target/valid detach、无效像素隔离，以及前轮 roundtrip 支持/指标。Ruff 通过。

helper SHA：`989476c5940228c0bbd4cda97450e6e29540833ac2b7bfde6d73684c530c2e6d`。这只是 CPU warp 合同；尚需根任务在同一个 002 TRAIN 上进行真实 CUDA forward/grad smoke：四tap 对官方 float-map `≤2e-6`，overscan native crop 对直接 native render `≤2e-5`，只三个允许参数有梯度，全部冻结参数/相机不变。通过并锁定 source/input 后才能进入固定双臂。

## 紧凑 delta 与磁盘

当前 base 有 498,136 高斯。三个允许参数的实际 FP32 张量合计 **95,642,124 bytes = 91.2114 MiB**，其中 SH0 5,977,632 bytes、SH-rest 89,664,480 bytes、背景 12 bytes。两套最终 delta 约 182.423 MiB，无需重复保存完整 geometry/semantic scene。

最小 delta loader 必须将一个完整 base 路径/SHA 与 manifest SHA/profile 绑定；只接受上述三个 key，严格检查 shape/dtype/finite、SH degree/高斯数量。只在内存中叠加，拒绝 delta chain 或未知 key；完整 base 必须保留，不可把它当可清理临时文件。最终派生权重明确标为 inference/warmstart，不携带旧 stage 的 optimizer、RNG、density 状态，不可作为普通 resume checkpoint。新实验配置/步数单独 namespaced，保留原场景与 held-out lineage。

若连同两份 Adam 动量保存，每臂约 273.634 MiB，两个可恢复 checkpoint 加 atomic 峰会占约 820.903 MiB，另加完整评测 PNG 后过于紧张。本轮 root runner 选择**只写最终 inference delta、不支持中断恢复**；如任务中断，应如实报告失败，不能从缺失 optimizer 的 delta 冒充严格 resume。

最终 delta 文件按新文件原子发布，控制臂+实验臂再计一份写入峰约 273.634 MiB；两次 official PNG 保守约 521 MiB，再加 256 MiB 余量和小型日志/快照，启动门槛固定为至少 1,152 MiB free，运行保留 256 MiB floor。实际启动前仍需读磁盘并检查其他任务预留；本诊断没有删除文件以满足预算。

## 为什么 eval3d 暂不混入

`with_ut=True, with_eval3d=True` 可以直接计算畸变相机的 world-space ray–Gaussian response，颜色与 opacity、几何有其独立 backward；appearance-only 本身不需要解冻几何。但响应模型、UT compensation/调度和 SH 射线语义与当前 2D splat+warp 不等价。只在训练换 eval3d、推理仍用旧 renderer，会产生 train/inference mismatch；连推理一起换则需要独立 official renderer 协议与零训练基线。当前 AbsGrad 和 ED 也不能直接沿用其原合同。详见 [梯度审计](raw_grid_training_diagnostic.md)。

本轮先验证低成本、保持现有推理 renderer 的工程对照，不预言 RGB 或语义一定改善。
