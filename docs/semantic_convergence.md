# 语义学习率收敛对照

原生裁剪对照未显示稳定增益，固定 3000 步的全图 / 混合裁剪全类 mIoU 分别为 94.6071% / 94.6423%。源码与 optimizer checkpoint 显示既有 sem_features、classifier、refiner 学习率分别恒定在 .01、.001、.0003。本项只检验降低后期步长是否改善收敛，属于工程控制，不作为新机制或原始创新。

计划在正式训练前写入 `configs/generated_semantic_lr/lr_plan.json`。两组共同 warmstart 为 `runs/strong_semantic_averaged/last.pt`，SHA256 `c6060e559b9d006a97c4fb7ae198c3fb629ec976de95a05e0825d377835dd9fd`，498,136 个高斯，完整保留平均后的所有模型参数，各自建立全新的 Adam。geometry、RGB、background 和训练相机冻结；仅用 259 张 TRAIN GT，独立 seed42 shuffle，native 1320×989 全图，固定 3000 次更新。teacher、fusion、crop、geometry/depth/opacity 正则均关闭，raw / final 语义损失及类别权重不变。

00 保持公共倍率 m(t)=1。01 对三个语义优化器组统一使用 m(t)=0.1+0.9×[1+cos(π(t−1)/2999)]/2，t=1…3000。因此第一步和原学习率相同，最后一步为原来的 10%；不存在额外 warmup、权重搜索或 checkpoint 选择。两组只有 output 和 `semantic_lr_schedule.type` 不同，`final_multiplier=.1` 是预先共同记录的 cosine 终值，在 constant 模式不生效。主终点固定为最后 3000 步，1500 步验证只作诊断。

`semantic_schedule.py` 在没有显式配置时不改任何 optimizer group，也不增加日志字段。显式启用时仅修改 sem_features 的唯一 group 和 heads 的前两组，背景第三组、SH、几何、Adam moments、参数值不受 helper 修改。base learning rates 在恢复 optimizer 前捕获，加载后重新绑定 group；checkpoint 中记录 base LR，严格恢复时核验已保存步数对应的 LR。更改 type、终值、总步数或启用状态必须新阶段 warmstart，避免悄悄拉长已经执行的曲线。旧版没有 schedule 的 strict resume 行为保持原样。

CPU 契约测试涵盖默认逐位优化结果不变、端点与单调性、背景/SH 的独立更新不受影响、严格恢复 Adam 和参数逐位等同连续运行、非法输入与恢复配置变化拒绝。连同 crop 回归共 35 项测试通过，Ruff 通过。两个真实 2-step CUDA 预检分别执行 constant / cosine，所有模型值有限、三个语义模块均更新、全部冻结 tensor 与相机逐位不变、view sampler 状态一致。cosine 在此短预检中用自身 2-step 曲线，倍率为 [1,.1]；正式曲线仍是预先固定的 3000 步。两次峰值约 5.212 GiB，同时验证 `save_every=0` 只产生 final checkpoint。预检结果为 `runs/semantic_lr_preflight/preflight_audit.json`。

正式两臂复用与预检完全相同的 immutable source snapshot；train.py SHA256 为 `b139ecd2430f108e86be452c6ea24882a1738478bf3443fc7564ef012639b69b`。输出为 `runs/semantic_lr_pair/00_constant` 与 `01_cosine`。两臂已完成 3000 步及全部 50 张 RGB / 41 张有标注验证图的正式评价。`scripts/report_semantic_lr_pair.py` 已通过 source/input SHA、冻结 tensor、原生 RGB 逐像素、view sampler、逐日志倍率和各 optimizer 最终 LR 的全部审计。259 张 TRAIN 图每张访问 11–12 次。50 张 RGB PNG 在两臂及 warmstart 之间完全一致。

| 固定最后 3000 步 | 全 5 类 mIoU (%) | 前景 mIoU (%) | Cable IoU (%) | Cable 边界 F1 (%) | Raw 3D mIoU (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Constant | 93.9586 | 92.6449 | 93.4428 | 88.6729 | 78.4960 |
| Cosine 1→0.1 | 93.8998 | 92.5713 | 93.5735 | 88.3440 | 78.6698 |

Cosine 减 constant 的最终全类 mIoU 差值为 **-0.0588 个百分点**，5000 次视图配对 bootstrap 的 95% 区间为 **[-0.1626, +0.0464]**；Cable IoU 差值为 **+0.1307 个百分点**，区间为 **[-0.1749, +0.4699]**。这两项均没有稳定的正收益。Raw 3D mIoU 有小幅提升 **+0.1738 个百分点**，区间为 **[+0.0903, +0.2756]**，但未转化为最终分割收益，不能据此声称完成精度目标。

作为部署精度参考，共同 warmstart 平均模型的全类 mIoU 为 **93.9892%**。额外训练的 constant / cosine 分别低 0.0305 / 0.0894 个百分点，均未超过它，因此不替换已选模型，也不继续搜索该学习率终值。两臂及 warmstart 的 RGB 均为 **PSNR 30.27351 / SSIM 0.9028182 / LPIPS 0.2244944**。累计到最后训练步时间为 413.99 / 479.28 秒，包含中途验证且并发条件不同，不能用作独占速度比较。

完整审计和指标为 `runs/semantic_lr_pair/lr_report.json`，配对结果为同目录 `paired_comparison.json`。单一 seed 与反复使用的开发验证划分仍限制泛化结论；视图 bootstrap 不覆盖训练 seed 方差。本项保留可选接口及负结果，结论只适用于当前底座、损失和预先固定的曲线，不能排除其他优化策略，但没有证据支持继续把本项当作主要精度改进方向。
