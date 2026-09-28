# H2 的监督强度：CPU 末态与日志审计

2026-09-26，只读既有 checkpoints、immutable source snapshots、日志及 TRAIN confidence calibration。没有新训练、GPU 推理、重新产生伪标签、修改源 checkpoint 或读取 VAL 像素。当前证据**不支持把 H2 概括为“监督几乎关闭”**：3D KD 确实稀疏、部分批次没有索 argmax 目标，但损失有有效权重归一；2D final-output KD 的阈值保留了绝大多数无标签训练像素。两者对最终精度的作用仍不能从标量损失单独推断。

证据分别保存在 [3D CPU 审计](../artifacts/diagnostics/h2_fusion_cpu_v2.json)、[2D CPU 审计](../artifacts/diagnostics/h2_pseudo_cpu.json) 和 [待执行的 16 TRAIN 视角计划](../artifacts/diagnostics/h2_contribution_plan.json)。`h2_fusion_cpu.json` 是未补软类别概率项的第一版，保留但以下使用 v2。脚本为 `scripts/audit_h2_fusion_cpu.py`、`scripts/audit_h2_pseudo_cpu.py`；两者 Ruff 通过。执行时显式 `CUDA_VISIBLE_DEVICES=''`。保存的源码 SHA、输入 SHA 及实际导入的旧 losses 函数可追溯数值；2D 脚本执行副本另保留为 `artifacts/diagnostics/h2_pseudo_cpu_execution_source.py`，其后主脚本只整理一处 import 空行。

## 1. 先分清 checkpoint 中的两个时间点

旧训练快照在每步 optimizer 更新**之后**，每 200 步刷新 `fused_target`，再写日志和 checkpoint。因此：

- `last.pt` 内 target 是 **3000 步新生成、未用于任何更新**的一批。其 `fused_reliable_points` 也是新批的统计。
- 同一份 `last.pt.stats` 的 `fusion_weighted_kd`、`fusion_accepted_*` 仍来自当步使用的 **2800 批**。不能把它们和新存的 target 拼成同一轮。
- `step_001500.pt` 中 target 是 **1400 批**，已经用于 1401–1500 共 100 步，随后原训练还用于 1501–1600；CPU 重算用的是更新后的 1500 状态，不是当步 backward 前的实际梯度。
- 真正进入训练的是生成于 200、400、…、2800 的 14 批，各使用 200 步。历史比较只取对应 `generation+100` 的一条日志，不能把重复日志当成独立证据。

四个读取的 checkpoints（两臂 × 1500/3000）的 geometry、SH、background、prior buffer、training cameras 逐位相同；两臂同一保存步的 4096 个候选索引完全相同且批内无重复。两臂全部 snapshot 文件 SHA 与 receipt 一致，末态 checkpoint SHA 也一致。读取中未恢复优化器、未修改任何参数。

## 2. 3D 融合 KD：稀疏不等于除以 4096

实际配置是 `multiview_weight=.05`、`local_teacher_kd=true`、GT priority 开启、`semantic_geometry_weight=0`。设过滤后权重为 r，target 的 argmax 类为 c，`m_c=sum_{i:c_i=c}r_i`，只对有正质量的类求平均质量 m。旧 losses 函数实际计算：

```
b_c = min(m / max(m_c, 1e-8), 3)
a_i = 0.05 * r_i * b[c_i] / max(sum_j r_j*b[c_j], 1)
L_3D = sum_i a_i * KL(target_i || softmax(W_detached*f_i+b_detached))
```

它既不除以 4096，也不除以全场 69,726。两份保存状态的分母均远大于 1。全体 reliability 同比例缩小通常会被这个归一抵消，不能把 projection 的权重总量减少直接解释为梯度同比减少。类别间质量差异、缺失类别和 target 内容仍会改变梯度。

### 14 批实际使用记录

| 指标 | Fixed | Projection |
|---|---:|---:|
| 平均接受 unique 目标数/批（批内不重复） | 244.21 | 226.71 |
| 平均索 argmax 目标数/批 | 6.64 | 5.86 |
| 含索 argmax 目标的批数 / 14 | 8 | 7 |
| 索组归一后损失权重份额，含无索批的均值 | 8.698% | 7.365% |
| 有索批中的索组份额中位数 | 15.38% | 13.51% |
| 索组总系数均值（再乘 .05） | 0.004349 | 0.003683 |
| 已乘 .05 的 local KD，14 条日志均值 | 0.00091264 | 0.00102286 |
| 同一日志 local KD 中位数 | 0.00010889 | 0.00011235 |

类别顺序 bg/deck/cable/tower/foundation 的平均归一份额分别是 fixed **29.49/29.57/8.70/22.24/10.01%**，projection **29.75/29.10/7.37/23.61/10.18%**。所以原始背景目标数较多，没有使背景独占归一后的 KD。Projection 甚至有略高的日志 KD 均值；不能只按 accepted weight sum 判它更接近关闭。

这些是每批平均，不是整个训练曾接受的独立高斯并集。日志没有完整历史索引或每步梯度；无法恢复 14 批的跨批去重率、各点累计更新量、渲染贡献或与 GT 梯度的方向关系。也没有理由把 200 次重复训练一批解释为 200 次新的多视角观测。

### 真实保存的目标集合

| 保存状态 / 生成批 | 臂 | 教师正权重数 | GT priority 后 unique | 全场覆盖 | 过滤后总权重 | 接受数 bg/deck/cable/tower/foundation |
|---|---|---:|---:|---:|---:|---|
| 1500 / 1400，确实用过 | Fixed | 90 | 77 | 0.1104% | 58.9851 | 18 / 46 / 0 / 3 / 10 |
| 1500 / 1400，确实用过 | Projection | 70 | 61 | 0.0875% | 40.3190 | 15 / 37 / 0 / 3 / 6 |
| 3000 / 3000，从未用过 | Fixed | 249 | 226 | 0.3241% | 194.8223 | 31 / 81 / 0 / 114 / 0 |
| 3000 / 3000，从未用过 | Projection | 235 | 217 | 0.3112% | 172.6815 | 31 / 76 / 0 / 110 / 0 |

两臂接受集合 Jaccard 为 1400 批 0.7037、3000 批 0.8771。它们都是少量共同候选的不同筛选，不是完全不同的监督实验。这两个保存批没有 cable **argmax** 目标；这不等于 cable logit 无梯度。KD 使用软 target，归一后的 cable 概率质量分别仍有 0.3428% / 0.1431%（1400）、0.3400% / 0.4719%（3000），softmax 的类别耦合也会对所有 logits 产生梯度。

使用 snapshot 的实际 `balanced_local_distillation`，在 CPU 上重算保存状态的局部 loss 与对选中特征的梯度；独立解析梯度及重建公式数值一致，classifier 梯度为空：

| 保存状态 | Fixed loss / feature gradient L2 | Projection loss / feature gradient L2 |
|---|---:|---:|
| 1500，更新后 proxy | 0.00274744 / 0.00038409 | 0.00148903 / 0.00027037 |
| 3000，新未用目标 proxy | 0.06056592 / 0.00463568 | 0.05848481 / 0.00471243 |

这些数值排除了当前保存批“loss 被掩成零”或梯度数值消失的解释。不能把新 3000 批的较大 loss 说成训练已施加的大监督，也不能以梯度 L2 判定与实际 GT 梯度相比足够强。Adam 的特征 LR=.01、eps=1e-15，更新使用混合损失与历史一阶/二阶矩；没有历史合成梯度就无法从上述 proxy 还原 KD 的实际参数更新占比。3D local KD 直接更新 sem_features；共享 classifier、refiner 和冻结 geometry 都没有来自这条 loss 的直接梯度。

## 3. 2D final-output KD：.8 阈值没有把像素覆盖压到接近零

这一支与上面的局部 3D KD 必须分开。实际配置为 `pseudo_refiner_weight=.1`、`pseudo_weight=0`、`pseudo_threshold=.8`、类别质量重平衡上限 3。只在无官方 GT 的 TRAIN view 上使用。总 3000 步中，2,215 步有 GT，785 步无 GT；无 GT 视角来自 91 张 301–400 图像。

confidence 不是单纯 max probability，而是已存的 `max_probability*(1-normalized_entropy)*(1-disagreement)`。阈值形成 binary reliability mask，之后乘 pseudo valid 和图像 valid，再对各 teacher argmax 类做 capped mass balance、按有效质量归一。日志 `pseudo_refiner_loss` 是**未乘 .1**的 KL。

只取确定性 sampler 显示当前 view 确实无 GT 的日志步 **1/300/800/1100/1500/1600/1700/2100**。其余有 GT 行残留的 pseudo 字段是旧值，全部排除；这里仅有 8 条实际观测，不能冒称 785 步的精确时间均值。

| 实际日志统计 | Teacher only | Fixed fusion | Projection fusion |
|---|---:|---:|---:|
| 乘 .1 后 final KD 均值 | 0.00116364 | 0.00119505 | 0.00120812 |
| 乘 .1 后最小值 | 0.00049207 | 0.00049362 | 0.00048974 |
| 乘 .1 后最大值 | 0.00253395 | 0.00252345 | 0.00252433 |
| 接受像素数范围（3 臂相同） | 1,236,507–1,280,150 | 同左 | 同左 |
| 占完整 1320×989 网格范围 | 94.72–98.06% | 同左 | 同左 |

这不是“threshold 后只剩几个易像素”的覆盖模式。低 KL 可能表示概率接近，或者类别平均掩盖少量困难区域；并不能仅凭这个标量判断学习信号有用或没用。这里所谓 final-only 是 **loss 作用于最终输出**，不是 refiner-only parameter scope：`refiner_field_grad=true` 仍让它更新 sem_features；共享 classifier 按设计 detach。不能把 H2 当成已经隔离过的“只训练二维头的压缩实验”。

### 已有 TRAIN calibration，阈值 .8

| 类别 | labeled259 真类覆盖 | labeled259 预测类覆盖 | 接受预测的错误率 | unlabeled91 预测类覆盖 |
|---|---:|---:|---:|---:|
| background | 98.99% | 98.97% | 0.0127% | 98.83% |
| deck | 94.29% | 94.20% | 0.1023% | 90.22% |
| cable | 90.02% | 90.37% | 0.1846% | 89.26% |
| tower | 86.26% | 86.03% | 0.1966% | 85.28% |
| foundation | 82.38% | 82.65% | 0.6763% | 82.49% |

前三列直接来自已有 calibration JSON；最后一列仅把其 `all_train` 计数减去 `labeled_train` 计数，分母是该 teacher 在无标签区域预测为对应类的像素数，并非未知 GT。同样算得 unlabeled91 总覆盖 **97.4014%**，没有新增推理或从图像推断覆盖。Calibration SHA 与 pseudo provenance 声明、teacher SHA、manifest SHA 一致。

阈值确实更常去掉小类预测，但当前没有证据把留下的 82–99% 全部判为“已会的易样本”，或断言被丢弃部分具有正确且未利用的教师信号。错误率列使用 teacher 曾训练过的 labeled259，不能外推到 unlabeled91 或 VAL。现有记录没有逐类 student KL、teacher/student 正误转移、被拒像素上的教师增量及实际梯度；不能据此立刻提高权重或去掉阈值。最新有收益的 renderer-adapted ensemble 使用不同教师、输入域与推理协议，也不能反推本 H2 已经有相同的可蒸馏信号。

## 4. 仅准备、不执行的 16 TRAIN 可见贡献计划

当前 CPU 证据已经排除了简单的“数值 off”，但留下空间送达是否有效的未知。若继续审计，主集合是**确实用于训练的 1400 批**；3000 新批仅作明确未训练的次要描述。不能用这两批无 cable argmax 目标的情况替代全部 8/7 个有索批。

固定清单已经写入带 SHA 的 plan：`013/038/063/090/113/137/161/190/216/242/266/291/316/340/363/386.png`，全部来自 TRAIN，按排序后 `floor((k+.5)*350/16)` 选出，不依据 GT 或错误。后四张无官方 mask，仍参与无标签覆盖汇总；类别分层仅对前 12 张有 GT 视角进行，明确分母不同。

拟议执行约束如下：

1. 使用 H2 的同一 immutable snapshot 与 legacy manifest/相机/原生尺度，校验 CPU 报告里的输入 SHA。geometry、opacity、RGB、prior 和相机已证四状态相同，只加载一套共同场。绝不换成当前 498k 场、v2 坐标或新的 source。
2. 完整场保留所有点、原 opacity、原深度排序。额外颜色通道只作为只读合成探针：每组保存目标的 accepted 类别指示 5 通道，四组共 20；teacher-positive 指示 4；实际局部 KD 系数 4；同生成步共同 candidate 指示 2，总 30 通道，背景为零。绝不把不选中的高斯删除后重算 transmittance。
3. 返回的 alpha 与共同 RGB render 的 alpha 做数值一致检查，所有类别贡献之和应等于 accepted 总贡献，且 binary 组贡献应在 [0,alpha] 容差内。记录实际源加载路径和 frozen tensors 前后 hash；不创建 optimizer、不给任何参数梯度。
4. 主读数为 `kappa=sum_selected(T*alpha)/sum_all(T*alpha)`。分别报告 valid 区域、GT 类别和每视角的均值/中位数/P95、低 alpha 的数量、unselected 及背景剩余质量。KD 系数通道只称“系数加权的贡献”，不把其数值当成概率或可见性百分比。逐像素贡献求和只是空间送达代理，不是 final refiner 的有效梯度上界。
5. GT/valid 只在探针输出后用于统计；不据这些数值改变选点、模型或训练。最多 16 个不同 TRAIN 相机，零 VAL、零 teacher 推理/伪标签重算，只保存小 JSON 与 hash，不保存大的概率/特征文件。

即使得到低 kappa，也只支持这些已存目标的局部覆盖不足；不能恢复历史有索批的覆盖，更不能证明改变采样会涨分。高 kappa 也不能证明跨视图 target 正确。如果问题必须具体回答“有索批在索区域贡献多少”，现有保存 target 不足，应先报告这一缺口，不能新抽一批后伪称历史重放。

按真实合成贡献分配语义是已有方向，[VALA](https://arxiv.org/abs/2509.05515) 已直接做贡献门控与稳健跨视图聚合。本计划是对既有负结果的工程解释控制，不作为创新。标准无阈值 soft KD、调大权重或增加 teacher 监督同样不构成新的学术机制。本次不启动这些训练。
