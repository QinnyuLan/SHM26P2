# 两臂匹配 RGB 域 H+ 适配：结果记录

**两臂训练、共同原图终评及独立 CPU 核验均已完成；系统门 4/7、域匹配证据门 0/2，均未通过，保留原 selected。** 新域适配对同预算对照恢复了部分拉索精度，但全类 mIoU 的独立增益未成立；不追加步数、选 best 或扫描配比。

首个终评尝试在 GPU 空闲检查处退出 1：RustDesk 桌面服务被原严格 compute-process 检查拒绝。该次 6.264 s，0 预测、0 GT 读取、峰值 CUDA 分配 0 B，输入及源码未变；原 `evaluation` 目录和 failed 回执保留。恢复版仅允许显式 G 进程及 `/proc/<pid>/exe` 确认为 `/usr/share/rustdesk/rustdesk` 的 C 进程，其余计算进程仍拒绝；未终止用户应用，推理、数据及数值门均未变。[恢复等价性审核](/mnt/data/SHM2026/runs/matched_rgb_teacher_adaptation_v1/evaluation_v2/environment_recovery_review.json) 已通过；`evaluation_v2` 自然退出 0。

固定方案见 [协议](matched_rgb_teacher_adaptation.md)。两臂均由同一历史 H+ EMA 出发，固定 2,000 步、50% 真实 RGB；另一半分别使用原 selected 渲染域或新 RGB 等权组合域。终评两臂均只输入同一批新组合 RGB，以相同原图适配器和 TTA 独立生成新 mask。

## 固定端点与共同原图评分

| 系统 | 训练 / 适配域 | 终评 RGB | 全类 mIoU (%) | 索 IoU (%) |
|---|---|---|---:|---:|
| 同预算对照 | 真实 + 原 selected，2k | 新组合 | 94.477276 | 93.235118 |
| 新域候选 | 真实 + 新组合，2k | 新组合 | 94.493989 | 93.623609 |
| 冻结 H+ 参考 | 原历史 H+，无本轮适配 | 新组合 | 94.386572 | 93.123743 |
| 当前完整 selected | 历史 H3 + H+ 固定组合 | 历史 selected | 95.109016 | 94.786655 |

共同原始网格评分固定为 50 个 RGB 视图及其中 41 个有语义标注的视图。两臂共 100 张 mask 生成并绑定后才解码语义 GT；两臂 RGB 均逐 PNG 字节一致，继承新组合 **PSNR 30.035661 dB / SSIM 0.874420736 / LPIPS 0.264741845**，终评未重新渲染或读取真实 RGB GT，未拼接任何旧 mask。完整 selected 是另一套历史系统参考，不能冒充同输入、同预算对照。

## 配对比较与预定门

差值均为候选减参考；语义差值及区间单位为百分点。固定按视图配对 bootstrap 5,000 次，seed 20260926。

| 参考 | 全类 mIoU 差值 [95% CI] | 索 IoU 差值 [95% CI] |
|---|---|---|
| 同预算对照 | +0.016712 [−0.093626, +0.112428] | +0.388491 [+0.135774, +0.660500] |
| 冻结新 RGB H+ | +0.107417 [+0.007796, +0.200077] | +0.499865 [+0.065601, +0.897796] |
| 当前完整 selected | −0.615027 [−1.427925, +0.295798] | −1.163046 [−2.802411, −0.185226] |

- 系统采纳门：沿用已绑定的四项 RGB 门；相对完整 selected，全类 mIoU 至少 +0.20 pp、配对区间下界严格大于 0，索 IoU 不低于 −0.10 pp。**四项 RGB 通过，三项语义均失败：4/7。**
- 域匹配证据门：相对同预算对照，全类 mIoU 至少 +0.20 pp，且配对区间下界严格大于 0。**两项均失败：0/2。**

类别权衡不能省略：相对同预算对照，候选 foundation IoU **−0.341717 pp [−0.778089, −0.059976]**，背景 +0.033628 pp [+0.009029, +0.066269]；deck −0.004619 pp、tower +0.007779 pp，后两者区间跨零。索的恢复和 foundation 的下降抵消了大部分全类均值增量。这是输出类别分布的实测权衡，未识别对比度、模糊或几何错误等具体因果机制。

因此不能概括为“域适配无用”，也不能用对冻结教师的 +0.107417 pp 替代同预算域匹配证据。上述类别区间为同一固定开发集的描述性结果，不作多重类别检验后的机制证明。

## 成本、来源与解释边界

| 阶段 | 已记录墙时 (s) | 峰值 CUDA allocated (B) | 范围 |
|---|---:|---:|---|
| 训练 RGB 缓存 | 319.759 | 1,869,171,712 | 三场各 350 TRAIN + 三次固定复现，共 1,053 次 scene 调用 |
| 同预算对照训练 | 195.661 | 3,327,421,440 | 固定 2k；内部训练计时 188.128 s |
| 新域候选训练 | 200.437 | 3,327,421,440 | 固定 2k；内部训练计时 192.805 s |
| 完成的共同终评 | 103.597 | 2,181,768,192 | 既有 RGB 上两端点推理、100 mask、评分及 I/O；0 scene render / 0 optimizer step |

缓存 scene 调用会同时执行原场语义分支，不能将 1,053 写作纯 RGB raster 次数。每臂监督更新覆盖原 259 有标注 TRAIN，未用缓存中另外 91 张无标签图；1000 real / 1000 rendered 的顺序、视图、裁剪、增强及末态 RNG 均实际逐步核验一致。训练外层计时包含端点核验，不能与内部训练计时相加。表中终评不含已有两场训练、缓存及本轮适配；桌面 RustDesk C 进程及其他 G 进程被记录保留，因此不是独占 GPU 延迟基准。

本轮是匹配渲染域的工程控制，未提出或验证新的共享几何机制。纯 H+ mask 不等于原生三维语义场；开发集已反复用于选型，配对区间不覆盖该选择偏差、跨场景、跨 seed 或同学未知划分。当前完整 selected 保持不变，本轮结果不授权扩步、扫参或重选端点。

来源闭合：[训练自然完成回执](/mnt/data/SHM2026/runs/matched_rgb_teacher_adaptation_v1/training_execution_receipt.json)、[逐步匹配核验](/mnt/data/SHM2026/runs/matched_rgb_teacher_adaptation_v1/matched_trace_review.json) 和[独立训练审计](/mnt/data/SHM2026/runs/matched_rgb_teacher_adaptation_v1/independent_training_review.json) 均通过。终评 [plan](/mnt/data/SHM2026/runs/matched_rgb_teacher_adaptation_v1/evaluation_v2/plan.json) SHA `527ca121a5bd5cfbfd92cfb216280cbbc05e6f062ca84bc324c9133bee773d71`；[completed 回执](/mnt/data/SHM2026/runs/matched_rgb_teacher_adaptation_v1/evaluation_v2/execution_receipt.json) SHA `24610509c595282ebb918d9873b671e54cebea1e42ee1031a12090c551dd11e6`；[实际两类门](/mnt/data/SHM2026/runs/matched_rgb_teacher_adaptation_v1/evaluation_v2/system_gate.json) 与三份配对报告位于同目录。

[独立 CPU 终评审计](/mnt/data/SHM2026/runs/matched_rgb_teacher_adaptation_v1/evaluation_v2/independent_cpu_review.json) SHA `9ed5b83124984872bb2405f6465175e3b6f9563967a88e2dcc837ed626b15031` 已通过：100 RGB 字节、100 mask 哈希/尺寸/类别、41 个原始标注独立栅格化、82 个 CM、三组 5,000 次配对 bootstrap 及所有门均一致。它未再次调用教师、GPU 或 RGB 评分；RGB 分数在字节一致后复用。预测早于 GT 的时序依据完成回执、预测回执及已审代码，并非独立运行期 I/O 跟踪。
