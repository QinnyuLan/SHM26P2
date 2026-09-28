# 单桥面轴四臂实测：未形成精度增益

2026-09-27。四臂各完成2,000步，共8,000次反向与更新，随后在固定原图50相机/41标注协议上统一评价。**真实投影方向未超过原条带、每相机最佳常方向或错误方向；本固定实现停止，工程E保持不变。** 不能将桥面轴估计稳定、几何场存在图内变化，解释为语义性能提升或学术创新已经成立。

## 固定终点

新三臂仅替换冻结H3场的二维精修头横向上下文聚合；三种采样共用逐查询、逐偏移有效支持。四臂都从同一H3起点更新完整精修头，使用相同2,000步TRAIN顺序、fresh Adam、损失与固定学习率。完整设计见[冻结合同](projective_deck_pooling_protocol.md)。

下表联合语义均采用固定H3/DINOv3 H+概率各半、一次soft畸变回投后argmax。新输出的RGB输入逐图精确匹配既有教师缓存；本轮复用教师soft和E RGB，不把旧RGB收益或缓存计时写成新方法收益。

| 固定终点 | 联合五类mIoU | 联合前景mIoU | 联合索IoU | 精修头单独五类mIoU |
|---|---:|---:|---:|---:|
| 当前工程E | **95.109016%** | **94.036190%** | **94.786655%** | — |
| original：原条带续训 | 95.020907% | 93.934176% | 94.350774% | 94.551104% |
| per_camera：每相机常方向 | 95.026651% | 93.940473% | 94.390450% | 94.543799% |
| **projective：真实桥面投影** | **95.018896%** | **93.931386%** | **94.358407%** | **94.525830%** |
| wrong：错误相机方向 | 95.050653% | 93.968497% | 94.467118% | 94.577923% |

所有四臂原始三维语义mIoU均为 **79.312262%**。其不变符合冻结设计，不构成三维语义改善。错误方向虽在四个新终点中点估计最高，仍低于既有E；不把它改选为科学主候选。

## 配对比较与采用判定

单位均为mIoU百分点，固定5,000次视图配对bootstrap，seed20260926。

| 真实投影减参照 | 差值 | 配对95%区间 | 原定最低增益 |
|---|---:|---:|---:|
| original | −0.002011 | [−0.027065, +0.026421] | +0.15 |
| per_camera | −0.007755 | [−0.028975, +0.010761] | +0.10 |
| wrong | −0.031757 | [−0.085022, +0.006867] | +0.10 |
| 当前E | −0.090120 | [−0.392349, +0.129395] | +0.20 |

四项增益点估计门和四项正区间门全部失败。索IoU相对wrong为−0.108711pp、相对E为−0.428248pp，也越过−0.10pp容差。共28条中18条通过，仅来自其余类别的退化容差；**整体失败**。本轮不扫描轴、跨度、插值日程、步数、学习率或融合权重来挽救该固定实现。

![固定终点和配对区间](/mnt/data/SHM2026/runs/projective_deck_pooling_v1/figures/projective_context_endpoints.png)

[矢量图](/mnt/data/SHM2026/runs/projective_deck_pooling_v1/figures/projective_context_endpoints.svg)及生成脚本、收据已保存。左图使用明确标注的94%–96%局部纵轴，右图展示差值区间和预先固定的最低增益。

这是已重复使用开发集上的固定终点对照，仍不是跨桥、多seed或官方盲测。与Bundle Pooling等既有方向池化技术存在明显重合；目前也没有本实现的正向消融证据，不作新颖性结论。

## 执行、缓存与审计

- 活动源码和冻结副本各78项CPU测试通过，Ruff通过。43份冻结文件、731项输入，uv环境锁定。
- 350 TRAIN相机的独立NumPy预检最大数值差6.11×10⁻¹⁶：共同支持72.444153%，平均12.315506/17样本；约5.092832%查询仅中心样本，1,801,100个查询中4个为空。它只验证实现，不评价语义。
- 四臂首步预测SHA完全相同，完整2,000步顺序相同，初始head相同，非refiner张量哈希相同且训练前后未变。仅head获得实际更新。
- original/per_camera/projective/wrong训练计时分别229.72/232.31/233.23/233.54秒；CUDA峰值约4.771/4.779/4.779/4.779 GiB。GPU与桌面RustDesk共享，这不是独占硬件比较。
- 自然退出0，worker总计1046.005秒，其中终点推理、文件写出和评分113.261秒、CUDA峰值1,874,964,480字节。终评实际200次scene调用、0次teacher调用、0次新E RGB渲染。
- 全部200份soft、200份联合mask、200份scene-final mask及200份raw mask先保存，再读取41份annotation；没有新的真实RGB评分。E RGB只按已验证PNG及原分数复用。不能以本轮耗时代表完整三场部署速度。

存在一个**仅报告计数的勘误**：冻结runner的顶层`new_val_annotation_payload_reads`停留在初始化0，没有在评价返回后累加；嵌套评价和独立评价回执均记录真实41次。原始源码、回执与SHA均保留，[单独勘误](/mnt/data/SHM2026/runs/projective_deck_pooling_v1/reporting_errata.json)说明该问题；活动runner已补入计数复制。它不改变预测屏障、训练、混淆矩阵或判定。

独立CPU终审通过，耗时36.55秒：400份soft重新生成scene/joint mask逐字节一致，200份raw mask在四臂一致；400份新H3原图/画布RGB与源精确一致；492个新混淆矩阵、41个旧E混淆矩阵、12组指标、4组各5,000次配对bootstrap和28条门全部一致，数值差为0。4份delta、各122个head张量及Adam终态、2000步顺序/alpha/首步/RNG均核验。未用GPU、未重评分真实RGB；非head不变的证据来自运行时哈希、基场摘要及新RGB/raw输出，并非保存了完整终点场后再次逐张量复现。

[独立审计报告](/mnt/data/SHM2026/runs/projective_deck_pooling_v1/independent_result_review.json) SHA `c8a5c78813bb4a4e9ae1fa4285dab2220d69ae95f5e2416a6268ba4d2d6b9bd5`。

- [冻结计划](/mnt/data/SHM2026/runs/projective_deck_pooling_v1/plan.json)：SHA `24be93a805cf348d2ce006caabdb431fe9bb75e2895340b521a18d480d82dd19`。
- [执行回执](/mnt/data/SHM2026/runs/projective_deck_pooling_v1/execution_receipt.json)：SHA `f183936782e3d224eab47f6cc0d0970ae7a6a33f068780779b20837bc64646db`。
- [完整门判定](/mnt/data/SHM2026/runs/projective_deck_pooling_v1/evaluation/system_gate.json)。
- [几何说明图](/mnt/data/SHM2026/runs/projective_deck_pooling_v1/figures/projective_context_geometry.png)，只展示实际固定TRAIN相机的采样差异，不展示标签或性能。

可用工程E仍为PSNR **30.035661 dB**、SSIM **.874420736**、LPIPS **.264741845**、五类mIoU **95.109016%**。相对本地同协议SEM386-style参考的工程优势已在[此前独立审计](multifield_h3_teacher_results.md)建立，但未复现同学缺失清单的精确Dev30，也不能替代尚未建立的创新机制增益。
