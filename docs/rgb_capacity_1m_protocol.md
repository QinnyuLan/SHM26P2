# 同配方 1M 上限 RGB 容量工程对照

2026-09-27。只检验当前强 RGB 配方的容量上限是否限制性能，不提出新增密方法或学术创新，也不是同学 RGB148/RGB150 的精确复现。已披露 peer RGB148 约 991,322 点，而此前本项目强 RGB 实验最多设置 500,000 上限；这是计算容量的实际差异，但不是 peer 分数差距的已证原因。

## 唯一数值变化与冻结来源

固定基线是已完成的 `/mnt/data/SHM2026/runs/ssim_fixed_corner_v2_rgb_full`。复制其 **`replay_config.yaml` 与实际 `source_snapshot/bridge_rgs` 全部 30 个 Python 文件**；不从当前工作区主源重新组合，不采用 MCMC、Mip 或 FP64 appearance 新分支。

| 来源 | 固定 SHA256 |
|---|---|
| 原 `plan.json` | `81b378c4d58b586d22d4eb3718a9f678f8a6650d5d0003b4f5e01c28b19e6b6e` |
| 原 `replay_config.yaml` | `b897e47518b4e8b8120b3b930c3c4c83df6c8004dd2e5a0a46423ac1c8c4be1c` |
| 原实际 `train.py` | `99c27c929253b17e71753a6a2afb080d1cdc0e464ce8f2550d78d8177c3e04d0` |
| 原实际 `densification.py` | `0b1f4073086afce3358d0c7fb4fcc6cb1d87a18cb0981ac411c4aabadeb5cfea` |
| 原实际 `losses.py` | `1e114069c2fdb181e243d7cb8a986659b2e229d2ec49c34821cb45013d9cf2b2` |
| 原 `launch_receipt.json` | `8a9dc28a60850fbf7b7a6fd4e4233e0656dd7879a0a9861f85c956748e4250c5` |
| 原 `experiment_receipt.json` | `c92914f7f8e323829937bfb1d5448f9130dcad9b57a8a3917dfb5803bf6afdc0` |
| 原 `stage_audit.json` | `aea06dafce6d3ff21bdedcc4abbd6d88b23d2c19afdcac545d97b92bb3311f14` |

旧外层已记录 completed/observed_exit_code=0，experiment completed、stage audit passed。本次 CPU 检查再次确认 30 个实际 package 文件 SHA 全部匹配旧 plan。新 plan 必须继续逐文件及输入字节核验，不以这段文字代替执行证据。

配置只允许两键不同：**`max_gaussians: 500000 → 1000000` 和新的 `output` 路径**。其余 fresh seed42、30,000 步、350/50 划分、corner-v2 数据、原相机、RGB/valid、初始化位置/颜色及 2,000 shell 点、Adam/LR、SH 升阶、原 `.contiguous()` FP32 SSIM、AA、渐进分辨率、原 NumPy shuffle、结构分裂/剪枝/opacity reset 全部相同。不能从 500k 终点续训来代替 fresh 同配方对照。

数据仍为 `artifacts/prepared_corner_v2/manifest.json`，SHA `91b41aeecedc352e4882eb80a1bb635d479ae432251a4c96c85a9dc110cbb327`；`init_points.npz` SHA `544917a1697f9363168ad1632f8f9370f7ff1efa0ef240091501b60e45b158c2`。起始 60,000 点加同 2,000 shell，合计 **62,000**。旧配方虽然不训练语义头，仍含 `region_rgb_weight=.15` 的 TRAIN mask 前景 RGB 强调、初始化语义 prior 与 `.65` 前景分配比例；为了只改容量，这些不能静默关闭，不能把新实验称为完全不使用 TRAIN 标签的 RGB 基线。语义 loss/教师/融合/相机优化仍关闭。

## 原增长窗口的可达性与限制

冻结实现的线性预算是

`B(step)=round(62000+(step−500)/(24000−500)*(max_gaussians−62000))`，

仅在 `500 ≤ step < 24000`、每 100 步且不处于 reset 暂停时调用增密。6k/12k/18k opacity reset 后各暂停 400 步；可调用次数共 **223**。最后一次是 **23,900**，所以两配置的最终调度上限分别是 **498,136 / 996,009**，并非精确 500k / 1M。保持这个行为，不为凑整新增第 24,000 步增长。`recycle_count=500` 只在当前点数达到硬 cap 时启用；从该 fresh 线性路径不会达到硬 cap，因此两组都不会因改 cap 新启用回收。

每次最多净新增 `min(12000, round(.06*N))`，split 或 clone 都是净加一个；实际还受多视角阈值、候选数和剪枝影响。在“不剪枝且候选无限”的纯算术上界重放中，1M 方案从第 900 步追上调度并可在 23,900 步到 **996,009**；这是窗口/每次新增额度不存在硬阻塞的证明，**不是预测真实增长**。

旧日志可见 223 次调用中 211 次结束点数等于当时预算；最后一次有 72,154 个记录的 eligible candidates，新增 2,219、剪枝 355，结束 498,136。增大 cap 后相邻常规预算增量约从 1,864 变为 3,991，低于后期 12,000 额度。这支持工程可行性，但新场的梯度、遮挡、opacity 和 pruning 会改变，不把旧候选计数当作新轨迹。必须记录实际峰值/终点点数；若未填满预算，照实报告，不调阈值/窗口补足。

## 执行、成本与终点审计

只准备一次新 run；相同已完成 CUDA 配方无需另开新 seed 或 smoke。正式执行仍须 root GPU 交接及空闲检查。冻结的新外层 worker 按相同 CLI 参数依次执行 train/native，明确 workspace root，不把搬离原目录后的旧 `run_experiment.py.__file__` 错当项目根；30 个生产 package 文件逐字节相同，外层 worker 自身另绑定 SHA。外层硬限时固定 **2400 秒**，含 30k 训练和既有 native 终点评价；超时/失败保留记录，不自动重跑、缩分辨率或改上限。旧训练日志耗时 1115.77 秒、`torch.cuda.max_memory_allocated` 峰值 **1.463 GiB**，它不是设备总显存占用。较大场的内存、tile 交叠与排序成本不必线性增长，不能据此保证 1M 可运行或满足限时。

旧终点完整训练 checkpoint 为 467,917,643 字节；按点数比例估算新文件约 **892 MiB**，仅为磁盘规划，另需 atomic-save 临时文件、native/official PNG 和来源副本。所有大文件写 `/mnt/data`。新 run 记录训练与外层时间、峰值 allocated/可得的设备观测、实际 Gaussian 数和文件大小。保留旧 snapshot/checkpoint/结果，不清理它们为新实验腾空间。

每 100 步原日志含点数与增密/剪枝信息，供独立 CPU 重建增长轨迹；跳过 reset 暂停时旧 stats 会沿用，不能误计为新事件。本次仅允许外层对 `GaussianScene.render` 加成本观察包装，读取 `len()`/整数尺寸并计数，不改变 tensor、RNG、调用参数或返回对象；分别记录 train/native 的 calls、sum-N、pixels 与峰值 N。新计数是实际调用观察，旧成本仍只能由 source+logs 重建，不能把两者都称为逐步运行 trace。原 hybrid 仍每 20 步进行低分辨率原始残差/可见性渲染，需计入额外成本，不冒称只有 30k scene renders。

仅在自然完成后打开终点 checkpoint，检查 step30000、完全相同来源/输入、配置仅两键差异、62000 初始化、相机逐位、corner-v2 manifest/profile、模型/optimizer/density 有限及形状一致、原采样终态、actual N≤996009、无中途 VAL 或另选 checkpoint。按原 save_every5000 更新 last，不保留“最佳 VAL”模型。未训练语义头的结果不作语义排名。

## 固定 RGB 投入门及语义后续

唯一 RGB 对照是上述原 500k 强参考的 `evaluation_official/official_metrics.json`，SHA `12bd34d0536da05f2a70a10f3b736312c679989c2a7fab3d42f84cae18d08ea8`；原 checkpoint SHA `bd5097e3aa5d0a8fcc6c326c7f653447c4683979727a9277fdde73c4eaf16239`。新终点首先执行同 native 50 图，再按共同原始像素网格 plain 50 RGB + LPIPS 评分；只有 official 结果用于此门，native 不作为替代选择。

candidate−reference 的四项固定门全部满足才投入语义：**平均 PSNR ≥+.15 dB，50 同名相机配对 bootstrap 95% 下界 >0，平均 SSIM 不降，平均 LPIPS 不增**。bootstrap 固定 5000 次、seed20260926；共同 official fingerprint 为 `21a2f19c5d4d703403a0107402d5dd5e006dcd98e5f8e023820c388c3e69a85d`。同一次 30k 的单 seed、单场开发集比较不等于盲测或跨场景统计证明。

失败保持当前 selected，不扫描容量、训练步数或损失参数。若四项全部通过，最短后续仅复制已完成 `ssim_fixed_corner_v2_semantic_coupled` 的同 **259 TRAIN / 8k** 配方，只改 warmstart/output，从新 RGB 冻结几何训练，并核新语义场的 RGB 与新 RGB 终点逐张完全一致。此处不接 legacy H+，不把 RGB 投入门通过当作联合系统采用；任何语义后续需独立来源与完整同协议评价。

本文锁定的是工程方案，不是完成结果或性能预报。新 run 必须保存本文不可变副本并绑定 SHA；root 最终决定执行时间，本文不自行启动 GPU。
