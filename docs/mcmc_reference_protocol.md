# MCMC RGB 工程参考：固定执行协议

本轮采用 **gsplat 1.5.3 MCMC 完整工程配方的本地适配**，不是新采样算法、不是 relocation 单因素消融，也不是作者全部 benchmark 配置的精确复现。目的只有一个：在同一桥梁划分与 500k 上限下，建立比继续叠加局部正则更有解释力的 RGB 参考。没有预先假定 RGB 或共享几何语义会改善；语义投入必须先通过本文固定 RGB 门。方法归属见 [MCMC 原论文](https://arxiv.org/html/2404.09591v3)、[官方 gsplat v1.5.3 策略](https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/gsplat/strategy/mcmc.py)与[同版本训练示例](https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/examples/simple_trainer.py)。文献决策另见 `docs/mcmc_reference_decision.md`。

## 数据、初始化与训练

- 固定 `artifacts/prepared_corner_v2/manifest.json`：350 TRAIN / 50 VAL，`colmap_corner_v2`，原固定相机。初始化仍为同一 60,000 个 SfM 点及颜色，加同一确定性的 2,000 点远景 shell；不读取 mask 像素，不采用 NPZ 内的语义计数。初始化位置、颜色和 shell 规则不因本参考改变。
- seed=42。视图用独立 `numpy.random.default_rng(42)` 打乱，每轮遍历全部 350 TRAIN；记录并保存顺序、游标及 RNG。MCMC 抽样与噪声使用 PyTorch RNG，checkpoint 保存其状态。只声明可恢复状态，不声明 CUDA 逐位重放。
- 初始 sigmoid opacity=.5；现有三近邻平均距离所得的 Gaussian scale 乘 .1；其余 RGB 初始化沿用本项目。`scene_scale=s` 是包含 shell 的初始点到坐标中位数距离的 90% 分位，固定不随训练重算。该尺度不是 SfM 点半径或作者相机范围的同义词。
- 30,000 步，每步一幅原生 1320×989 TRAIN RGB 与 valid；`train_scale=1`、`progressive_resolution=false`。只优化 RGB 几何、SH 与全局背景颜色；semantic/refiner 不参与前向监督，无教师、mask、相机优化、归因、PCA、回收配额、深度监督或 Mip 3D filter。保留现有 `antialiased` 2D AA。
- 光度目标仍为本项目 `.8*valid-L1 + .2*valid-(1-SSIM)`，SSIM 为已修复 contiguous NCHW 的 7×7 box 实现。加 opacity 均值正则 `.01` 与下文 canonical scale 均值正则 `.01`；替代旧 scale 系数 `1e-4`，不叠加旧项。不是官方示例的 fused-SSIM 全图支持合同。
- 每个 splat 参数单独 Adam：means 的物理 LR 为 `s*1.6e-4*(.01**(step/30000))`，quaternion `.001`、log-scale `.005`、opacity-logit `.05`、SH0 `.0025`、其余 SH `.000125`，splat Adam epsilon 保持 `1e-15`。背景颜色仍为本项目 head optimizer 中的 LR `.001` / 默认 epsilon。SH 阶数按 `min(3, step//1000)`；没有额外 SH。语义随机参数可随 topology 复制，但无语义梯度、prior 始终为零，不报告其语义性能。

## 单位映射与非等价边界

令 canonical 坐标为 `μ′=μ/s`，Gaussian 协方差为 `Σ′=Σ/s²`；概念上的相机平移同时为 `t′=t/s`，旋转与 K 不变。实现不改实际坐标。令 canonical means LR 为 `η`，则物理 LR 为 `η_phys=sη`。installed gsplat 噪声乘的是 **协方差 Σ，而非其平方根**：

`Δμ = lr_strategy * 5e5 * sigmoid(100*(.005-opacity)) * Σ * ξ`，`ξ~N(0,I)`。

因此从 canonical 更新乘回 s，得到 `lr_strategy=η_phys/s²`。canonical scale 正则也恰为 `.01*mean(exp(log_scales))/s`；opacity 无量纲不变。在相同随机数下，这两项是代数一致的单位转换。

**完整 Adam 轨迹并非精确缩放等价。** 若 canonical 梯度/矩为 `g′,m′,v′`，物理量为 `g′/s,m′/s,v′/s²`，固定物理 epsilon=e 的更新是 `−sη*m′/(sqrt(v′)+s*e)`；对应 canonical epsilon 是 `s*e`，不是 e。要严格匹配同一 canonical epsilon 才应使用物理 `e/s`，本参考不改既定 `1e-15`。近零矩、FP32 舍入及离散重定位也不能被“epsilon 很小”排除；不声称精确坐标重参数化或作者轨迹复现。

渲染 near/far 继续物理 `.01/1e6`，对应 canonical `.01/s,1e6/s`，不是另行保持 canonical 裁剪数值。MCMC 分支跳过旧分支的 post-Adam quaternion 原地归一化、log-scale 上下界及 opacity-logit clamp；render/noise 算子按其原有合同解释 quaternion。仅保留 installed relocation 的 opacity 修正与其内部 ratio clamp，避免把旧约束无意带入新配方。

## 策略、容量及状态

直接调用 installed `MCMCStrategy`，不修改 site-packages。参数：`noise_lr=5e5, min_opacity=.005, refine_start_iter=500, refine_stop_iter=25000, refine_every=100, cap_max=500000`。按本项目一基步骤，在 Adam 后调用；重定位与增长发生于 **600,700,…,24900**，共 244 次。先重定位低 opacity 槽，再令 `N_next=min(500000,int(1.05*N))`；没有世界尺度剪枝、opacity reset 或旧 split/clone。噪声每步施加，包括 25k 后及最终 30k，最终 checkpoint 是最后一次噪声后的固定端点。

保留 v1.5.3 的真实 Adam 迁移语义：relocation 清零被采中 donor 的非 step 矩，替换的 dead 槽旧矩保留；sample_add 保留 donor 旧矩，仅新增尾部矩置零。不得简写为“父子全重置”。ratio 的内部上限为 51；500k 未触发 `2**24` 以上 NumPy sampling fallback。全 dead、非有限、形状或状态不一致均停止，不能临场修改阈值或重跑。上限相同不等于点数历程、像素预算或时间相同；报告 render-before-topology Gaussian-step 总和、post-topology noise Gaussian-step 总和、原生 pixel 总和、峰值 N、墙时及显存。

checkpoint 持久保存策略 policy、版本/单位 convention、binomial 表、事件/点数/噪声累计、优化器与 RNG。旧 density 窗口只保留合法尺寸，MCMC 不消费它们。active resume 严格核 policy/尺度/形状/步骤；普通 load/render 保留 metadata，不执行策略。任何后续冻结语义 stage 是另行决策，本轮不运行。

## 相比现有 RGB 参考的明确差异

| 项目 | 既有 RGB140-inspired mixed | 本轮 MCMC |
|---|---|---|
| 数据、初始位置/颜色、相机、上限 | v2、350/50、62k、固定、500k | 相同 |
| 初始 opacity / scale | .1 / 现有近邻尺度 | .5 / 现有尺度×.1 |
| 分辨率 | .5→.75→1 日程 | 全程 native |
| topology | signed clone + abs split / prune / reset | opacity 重定位 + 固定 5% 增长 |
| opacity LR | .025 | .05 |
| 正则 | 旧 scale `1e-4` | opacity `.01` + canonical scale `.01` |
| means 噪声 | 无 | 每步 covariance 噪声 |
| 视图 RNG | 原配置采样路径 | 独立 NumPy shuffle |
| post-Adam 通用约束 | normalize/clamp | 跳过，仅策略自身修正 |

官方 MCMC 示例常用归一化世界坐标、不同初始化尺度细节、AA 关闭、不同 SSIM 与支持；本项目保留 bridge 数据坐标、近邻平均初始化、shell、有效像素和 AA，故只能称 **本地 MCMC 工程参考**。相对强 SSIM-fixed hybrid 的比较同样是完整配方比较，不能把差异归因于 relocation 或任何单一机制。

## 一次执行与固定投入门

先做 610 步、180 秒硬限时、无 VAL smoke；610 是自身 LR horizon，不是 30k 轨迹前缀。通过实际 finite、600 步增长、Parameter/Adam 迁移与 CPU endpoint 审计后，冻结相同 package source 执行一次 30k，硬限时 3600 秒，训练中不评价 VAL、不挑中间模型。完整训练只评价固定 last：native 50 RGB+LPIPS，以及独立共同官方原始网格的 plain 50 RGB。官方协议交付 uint8 PNG 后评分，不能把 native float 数值与官方数值相减。

固定对照是 `/mnt/data/SHM2026/runs/ssim_fixed_corner_v2_rgb_full/evaluation_official/official_metrics.json`，原始网格指纹 `21a2f19c5d4d703403a0107402d5dd5e006dcd98e5f8e023820c388c3e69a85d`。其 checkpoint SHA 为 `bd5097e3aa5d0a8fcc6c326c7f653447c4683979727a9277fdde73c4eaf16239`，metrics SHA 为 `12bd34d0536da05f2a70a10f3b736312c679989c2a7fab3d42f84cae18d08ea8`。比较 candidate−reference，按同 50 相机配对 bootstrap 5000 次、seed=20260926、95% 双侧 percentile 区间。继续投入语义必须同时满足：

1. 平均官方 PSNR 提高至少 **0.15 dB**；
2. PSNR 差的配对 95% CI 下界 **严格大于 0**；
3. 平均官方 SSIM 点估计不下降；
4. 平均官方 LPIPS 点估计不增加。

任何一项失败不扫描 noise、初始化、正则或后处理，也不选择某些 VAL 帧解释为通过。该投入门是本项目资源决策，不是通用统计最优性或所有 MCMC 变体的否定。run 内保存本文不可变副本、配置、源码及输入 SHA；正文不是结果报告，smoke/30k 的实际完成与门结果须由各自 receipt 和 CPU 审计证明。
