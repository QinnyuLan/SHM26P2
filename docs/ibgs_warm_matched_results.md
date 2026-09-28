# warm-IBGS 固定双臂结果

2026-09-27。两臂各6000步、四个固定RGB末态和独立CPU终评核验均已完成。预先指定的主候选仍是 **full/fused**。它在50个开发验证视角上明显优于相同训练步数的 `no_source/fused`，但未满足相对原AA单场的全部RGB条款，更未满足相对当前工程组合E的条款，**不替换E**。本轮不更新或重新评价语义分支，不产生新的语义或完整联合系统性能结论。

这是已有 [IBGS](https://arxiv.org/html/2511.14357v1) 的暖启工程参考。图像来源融合、法向／平面优化和残差网络不是本项目提出的新方法；本次端口修订与6000步暖启配方也不是作者默认配方的精确复现。训练和评价分别遵守[训练协议](/home/sky/workspace/SHM2026/docs/ibgs_warm_matched_protocol.md)与[固定终评协议](/home/sky/workspace/SHM2026/docs/ibgs_warm_evaluation_protocol.md)，未按结果挑选迭代、分支、源阈值或系数。

## 50视角正式RGB结果

四组预测均按同一50个相机，输出1320×989原始畸变网格的uint8 PNG。先完成200个native浮点预测，再完成200个交付PNG，最后读取50张RGB目标评分。复用已冻结的官方扭曲映射、量化、PSNR、SSIM和LPIPS实现与本地权重；没有读取语义标注、运行DINO或教师。

![固定四输出与原AA、E的RGB均值比较](/home/sky/workspace/SHM2026/docs/figures/ibgs_warm_matched/rgb_metrics.png)

图中为逐视角均值，不是多种子区间；[SVG图](/home/sky/workspace/SHM2026/docs/figures/ibgs_warm_matched/rgb_metrics.svg)及[数据来源](/home/sky/workspace/SHM2026/docs/figures/ibgs_warm_matched/sources.json)一并保存。

| 固定输出／参考 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| full/raw | 29.304543 | 0.868000 | 0.267374 |
| **full/fused（主候选）** | **29.846757** | **0.872022** | **0.250069** |
| no_source/raw | 29.335564 | 0.868194 | 0.267560 |
| no_source/fused | 29.335884 | 0.868196 | 0.267545 |
| 原AA 1M单场，996009点 | 29.496148 | 0.872983 | 0.263820 |
| 当前E交付RGB | 30.035661 | 0.874421 | 0.264742 |

下表均为候选减参考，括号为固定5000次、seed20260926、按相机配对重采样的95%区间。条款固定为：PSNR均值增加至少0.15 dB、PSNR区间下界大于0、SSIM均值不降、LPIPS均值不升；它们只是后续投入参考，不是自动系统采用规则。

| 固定比较 | ΔPSNR，dB [95% CI] | ΔSSIM [95% CI] | ΔLPIPS [95% CI] | 条款 |
|---|---|---|---|---:|
| full/fused − no_source/fused | +0.510873 [0.433969, 0.589825] | +0.003827 [0.002999, 0.004696] | −0.017476 [−0.019942, −0.015147] | 4/4 |
| full/fused − 原AA 1M | +0.350608 [0.168893, 0.567677] | −0.000961 [−0.002024, 0.000099] | −0.013751 [−0.017248, −0.010562] | 3/4 |
| full/fused − E | −0.188904 [−0.369591, 0.009896] | −0.002398 [−0.004706, −0.000127] | −0.014673 [−0.017807, −0.011847] | 1/4 |
| full/raw − no_source/raw | −0.031020 [−0.041140, −0.020596] | −0.000194 [−0.000277, −0.000108] | −0.000186 [−0.000332, −0.000033] | 1/4 |

其余六项预定对参考比较均为0/4：full/raw、no_source/raw、no_source/fused分别对原AA 1M和E。全部四组逐图结果和十组比较保存在[评分回执](/mnt/data/SHM2026/runs/ibgs_warm_evaluation_v1/score_execution_receipt.json)，没有省略不利分支或改选主候选。

来源输入的作用主要体现在最终融合输出：full的原始场RGB相对no_source的PSNR反而低0.031020 dB，区间全负，SSIM也下降；不能称为基础场RGB全面改善。full/fused相对原AA单场获得明确PSNR和LPIPS改善，但SSIM点估计下降，原定四条未全部通过。相对E则只有LPIPS改善，PSNR点估计较低、SSIM区间为负。因此不能用感知指标优势宣称全面超过E，也不能把既有E语义成绩与本次RGB拼接成已实测的新系统。

## 16 TRAIN视角的移植与恢复

另一个预定诊断使用相同16张已参与训练的视角、相同native有效像素支持。表内是clip到[0,1]后逐相机PSNR的等权均值；这不是50视角原始畸变完整图评分，二者不能直接混为一个评价总体。

| 阶段／输出 | 16 TRAIN PSNR，dB |
|---|---:|
| 原AA 1M场 | 32.300961 |
| 移植到IBGS的初始场 | 28.661841 |
| full/raw末态 | 31.638356 |
| full/fused末态 | 32.154370 |
| no_source/raw末态 | 31.703568 |
| no_source/fused末态 | 31.701134 |

full/fused相对端口初始场恢复 **3.492529 dB**，但仍比原AA起点低 **0.146591 dB**。这说明相当一部分暖启优化在恢复移植后的差距；不能把全部恢复幅度算作相对原模型的新收益。不同光栅器同时涉及AA与其他数值／合成差异，现有检查没有将起点差全部归因于AA。TRAIN表仍低于原AA与50视角PSNR超过原AA并不矛盾：视角、支持和交付网格不同。

初始结果见[移植诊断](/home/sky/workspace/SHM2026/docs/ibgs_start_transfer_results.md)，末态见[恢复分析](/mnt/data/SHM2026/runs/ibgs_warm_train_recovery_v1/analysis.json)。末态两臂各自刷新350源深度，32次目标调用保存64个native输出后才载入16份缓存目标；源库为全部350 TRAIN，每个目标排除自身，其他诊断目标仍可作为来源。它是已拟合TRAIN上的恢复描述，无采用门，也不用于选端点。未clip结果同时保留在分析文件。

## 对照范围与训练不确定性

两臂均从同一996009点、SH3的已训1M场开始，不增删高斯、不更新相机。相同seed42、初始场／网络／背景记录和6000步相机顺序；各执行1000步几何暖启、1000步融合梯度阻断暖启、4000步联合拟合。full使用真实源残差及相机特征；no_source仅将网络入口的这七个源通道置零，保留相同网络、自己的渲染RGB、射线、源支持，以及真实source-photo几何监督与渲染成本。

因此差异衡量的是**共同来源照片监督下，源输入通路及其联合训练后果的总作用**。no_source不是完全不使用多视图的模型；联合阶段两个场本身会分化，该比较也不是固定同一几何后的纯读出消融。不能由这次结果识别唯一的遮挡、薄索、法向或语义机制。

[训练合同复核](/mnt/data/SHM2026/runs/ibgs_warm_matched_v1/training_contract_review.json)确认相同的已记录初始化身份和6000相机轨迹，但在来源消融尚未启用的前1000步，五个保存的日志位置已存在数值差异：总损失最大绝对差0.0002529174，normal项最大差0.0017000437。它们不能归因于源输入消融，日志也不足以确定非确定性的具体来源。没有保存完整独立step-zero张量快照供该复核重新生成初始化。**本轮只有一个seed；相机配对区间不包含训练重复运行方差。** 相同步数与相机序列不等于逐位训练轨迹、相同墙时或已测相同FLOPs。

## 实际成本与资产

| 阶段 | 实际内部时间 | 峰值allocated显存 | 实际工作 |
|---|---:|---:|---|
| full训练 | 749.970 s | 7109035520 B | 6000场更新、5000网络更新、6350光栅调用 |
| no_source训练 | 701.958 s | 7089284096 B | 同上述调用与更新数 |
| 两臂正式渲染 | 20.060 s | 3230416384 B | 700源深度、100目标调用、200 native输出 |
| 正式交付与评分 | 69.740 s | 348000256 B | 200 PNG、50 RGB目标、四组指标 |
| 16 TRAIN末态恢复 | 19.912 s | 3230564352 B | 700源深度、32目标调用、64 native输出 |
| 独立CPU终评核验 | 16.046 s | 无GPU | 200交付重建、200 PSNR、十组配对比较 |

两臂训练总worker时间1453.004 s，外层1455.21 s；正式渲染和评分外层分别21.53 s与71.12 s；恢复外层20.35 s。以上包含各阶段实际加载、源库深度刷新或I/O，不是单帧FPS。训练是增量成本，未计入原1M场30k训练、端口编译／修复和之前预检。原16 TRAIN移植检查另有16次原始渲染、内部5.971 s。

模型还依赖350张TRAIN来源图像与相机资产；测试调用只需要目标相机，来源图像是冻结的模型资产。每臂正式评价使用自己的最终源深度，不复用另一臂或旧训练深度缓存。端口包含像素中心／相机接线、逐交点源有效支持、纹理导数与平面梯度累积以及带符号相机特征支持修正；这些明确修订保存在冻结来源中，不能称为未修改官方实现。

## 独立核验与可追溯性

[独立CPU终评报告](/mnt/data/SHM2026/runs/ibgs_warm_evaluation_v1/independent_cpu_review.json)与[自然退出回执](/mnt/data/SHM2026/runs/ibgs_warm_evaluation_v1/independent_audit_launch_receipt.json)均通过。使用冻结审计脚本、`CUDA_VISIBLE_DEVICES=''`及主项目`uv run --no-sync`，唯一执行自然exit0，外层16.123 s。200个native到PNG重建逐字节一致；独立PSNR最大差为1.07e−13；十组配对重采样／门的最大差为3.33e−16；核验64份源文件、375份输入及四组共同相机／来源身份。

CPU审计未重新计算SSIM或LPIPS像素函数，而是核验固定评分源码、权重及产物身份，并独立汇总已有逐图分数和配对区间。预测先于GT的证据来自绑定的程序流程与时间记录，不冒称另行进行了运行时I/O追踪。没有GPU、语义目标、教师或优化更新。

16 TRAIN恢复的[独立CPU复核v2](/mnt/data/SHM2026/runs/ibgs_warm_train_recovery_v1/independent_cpu_review_v2.json)也已[自然exit0](/mnt/data/SHM2026/runs/ibgs_warm_train_recovery_v1/independent_audit_v2_launch_receipt.json)，外层9.380 s：核验64个预测数组、16份目标缓存及544个浮点标量，最大差7.11e−15。首次审计在`full/fused/259.png`未clip PSNR的独立FP64 BLAS归约差为1.01608e−12，略超原1e−12容差，失败报告及回执保留。v2仅将审计者平方求和改为独立longdouble归约，容差、生产输出和分数均未修改；这是审计数值归约修复，不是训练或生产评价失败。该复核不重渲、不反序列化模型、不重新解码原图；起点参考数值绑定旧分析，不冒称重新运行了起点。

| 关键证据 | SHA256 |
|---|---|
| [训练plan](/mnt/data/SHM2026/runs/ibgs_warm_matched_v1/plan.json) | `c1af18ac17a1ce934208fa91f2b987f7a46c9920b7e35eb3ba9c95dafc056691` |
| [训练execution](/mnt/data/SHM2026/runs/ibgs_warm_matched_v1/execution_receipt.json) | `aea010d45783547953f6ca16ade0612503f16a53d5fc91640a9f7fa0126ae965` |
| [终评plan](/mnt/data/SHM2026/runs/ibgs_warm_evaluation_v1/plan.json) | `84425a934197cd32e08b4b7262dad9b371449731c996d9c53159c66d05f43e7b` |
| [评分execution](/mnt/data/SHM2026/runs/ibgs_warm_evaluation_v1/score_execution_receipt.json) | `b15334babce388a35b824916b17f2b51f167e23a4f71e8f8d7c4bbd25973cdff` |
| [独立终评报告](/mnt/data/SHM2026/runs/ibgs_warm_evaluation_v1/independent_cpu_review.json) | `22533b163fc9aee1293eba2028ed0b50d23bfe73945109f190ca95782436bf86` |
| [16 TRAIN恢复execution](/mnt/data/SHM2026/runs/ibgs_warm_train_recovery_v1/execution_receipt.json) | `6c8264fba03914b63f4c15b87453fb48f3e646e44a99dcf00fd62f122459f681` |
| [16 TRAIN恢复analysis](/mnt/data/SHM2026/runs/ibgs_warm_train_recovery_v1/analysis.json) | `ca18a367b1dc06590ec0400541ae38e7bb1c88eda889bfbaf362e1bf6706881b` |
| [16 TRAIN独立复核v2](/mnt/data/SHM2026/runs/ibgs_warm_train_recovery_v1/independent_cpu_review_v2.json) | `f35f15983c74a25f344526eb8d41653e7e464ad60a2f4542ff146a39168286a6` |

当前结论限于这一次预定工程对照：来源输入在融合RGB上有可测价值，原场RGB并未随之全面改善，E保持不变；不据此宣称已验证桥梁结构新机制、语义收益或训练跨seed稳定性。
