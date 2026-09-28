# H3 类别不可见特征投影：两张 TRAIN 的实际读取诊断

2026-09-27。固定四次渲染已自然完成，数值门和独立 CPU 复核均通过。**原生 raw 分类不变，冻结精修器明显依赖被移除的特征及其 cross-moments；但这两图的主要变化是背景被错误扩张为索，原有索区域纠正仍保留 99.13%。** 结果不支持把该分量简单解释成“索区域补全所必需的信息”。

这是一次固定头的输入依赖诊断，不是新方法、性能改进、三维语义真值或泛化验证。当前 selected 模型和生产检查点没有改变，未启动后续实验。

## 固定合同与执行核验

对象是已完成 H3 cross 场 `runs/h3_moments/02_cross/last.pt`，SHA `22bc8a2ddb260f93cb01b17857c97b2bb0873038efdb9318545cb2bdbb045226`。沿用其完整原26文件 package，只新增诊断入口和设计副本。全部350 TRAIN按名称排序的索引0/175固定为 **002.png、205.png**，采用 legacy 原生1320×989、原相机、SH3。没有选择错误最多的视图。

唯一变换沿用前次 CPU 报告中的 `D=W[1:]-W[0]` 行空间正交投影：从每Gaussian的16维 `sem_features` 源头临时置换为 `FP32(FP64(f)P)`，所有渲染特征和深度—特征矩通过原路径重新生成，没有复用旧 feature/moment 缓存。geometry、opacity、RGB参数、decoder、refiner不变。原场和投影场分别渲染两相机，共 **4 scene / 8 raster / 0 backward / 0 optimizer**；没有教师、新检查点或真实RGB输入。

四次预测全部结束后才解码两张TRAIN mask和对应valid（同一valid文件读取两次，共4次允许读取）。CE锁定五类权重均为1，复用原H3 `semantic_loss(lovasz_weight=0)`：概率下限1e−7，分母为有效且非255像素数，汇总为两view等权平均。这是**描述性无权CE，不是原H3类别加权训练目标**。IoU汇总使用两图混淆矩阵相加，不能与完整41 VAL或官方原图成绩混合。

| 全图数值门 | 002 | 205 |
|---|---:|---:|
| RGB / depth / alpha | 三者逐位相同 | 三者逐位相同 |
| p3d最大绝对变化 | 2.38419e−7 | 1.78814e−7 |
| p3d预定门 | ≤1e−6，通过 | ≤1e−6，通过 |
| raw argmax变化 | 0 | 0 |

执行内部8.483s，外层9.383s、自然exit0；完整state、相机、requires-grad标志及模块train/eval模式在finally恢复。两状态之间所有非feature参数未变，source/input SHA前后一致。四份压缩NPZ合计189,904,841字节，仅保存p3d和final概率。

独立 CPU 审计重新读取四NPZ，核SHA、dtype/shape、全图p3d变化；用NumPy重算全部CM、IoU、索纠正计数与CE，CM/计数完全相同，CE最大差3.87e−8（FP32原计算与独立FP64求和的舍入差）。也将实际原checkpoint的全部tensor SHA、350原相机与runtime前后SHA核对一致。RGB/depth/alpha数组没有额外持久化，因此本次独立审计核的是已保存的对应tensor SHA一致性，没有冒称再次逐像素渲染核验；运行调用计数及flags/modes恢复同样是持久化runtime证据。

## 结果：依赖明显，主要表现为背景抑制受损

| 两图 pooled / 等view CE | 原场 | 投影场 |
|---|---:|---:|
| raw 五类mIoU | 74.9635% | 74.9635% |
| raw 索IoU | 8.4756% | 8.4756% |
| raw CE | 0.242636 | 0.242636 |
| final 五类mIoU | 99.0165% | 84.3228% |
| final 索IoU | 99.1779% | 53.6102% |
| final CE | 0.005782 | 0.330417 |
| final 索precision | 99.4409% | 53.9411% |
| final 索recall | 99.7340% | 98.8690% |

这两图raw索分数特别低，**不代表完整验证集的约30% raw索IoU**；它们是事先按名称固定的TRAIN样本，没有外推该数值。

| 每view final | 002原场→投影 | 205原场→投影 |
|---|---:|---:|
| 五类mIoU | 98.2865% → 73.9310% | 98.9150% → 92.4187% |
| 索IoU | 95.4119% → 7.3964% | 99.4072% → 83.8833% |
| CE | 0.005271 → 0.553059 | 0.006294 → 0.107776 |

原来由head把“GT索、raw判背景”纠正成索的150,992像素中，投影后仍正确149,682，丢失1,310，保留率 **99.1324%**；同时有278个新的GT索纠正。相反，新增索假阳性148,973，其中背景→索CM计数净增148,105，索真阳性只净减1,535。两图所有类别共新增164,855个错误、反向纠正936个。单看索IoU的大幅下降会掩盖这个precision/recall不对称。

因此，这次干预显示当前head的输出依赖原生softmax看不见的feature分量，且在这些样本上其移除主要伴随**索区域外的背景抑制失效**。它没有显示绝大多数原有索补全依赖该分量。后续解释应区分“补全能力”和“抑制错误扩张”，不能直接据此推出新的loss或宣称raw低分由decoder维数不足导致。

## 解释边界

投影是分布外输入，改变feature幅度、相关性与GroupNorm之前的激活，还移除了null全局均值中可能的alpha/可见性编码。当前结果不能排除这些数值分布变化造成的错误扩张，也不能证明被移除分量包含必要物理语义。普通feature通路及其cross-moment通路共同改变，不能把效果单独归因矩通路；没有重训补偿head，因此不构成训练后可达性能的上界。

两图均为TRAIN，没有VAL、教师或生产模型变更。正交零空间能量依赖feature坐标度量，不是内在信息量。标准线性投影和读取消融不作为创新。当前结论到此为止，不追加投影、阈值、视图或训练。

## 可复核产物

- [锁定计划](/mnt/data/SHM2026/runs/h3_semantic_readout_diagnostic_v1/plan.json)，SHA `cd0e6336a3df4e7d01890926f6590f10f493993a435a6640f5efc7e54055e81f`。
- [实际执行回执](/mnt/data/SHM2026/runs/h3_semantic_readout_diagnostic_v1/execution_receipt.json)，SHA `fadabf80c0f1ddd501f9774ea50b01697ae67b4a9208aef67bd1cca81c3be88c`。
- [独立CPU审计](/mnt/data/SHM2026/runs/h3_semantic_readout_diagnostic_v1/independent_cpu_review.json)，SHA `98f6cb08e4da9cbf33e97bab9e2e0d7f0c4b360b36ef14b21b8653425e5eec73`；[审计入口](/home/sky/workspace/SHM2026/scripts/check_h3_semantic_readout.py)。无新增GPU调用。
- [原CPU代数审计与设计](/home/sky/workspace/SHM2026/docs/h3_semantic_readout_diagnostic.md)保留原字节，本页独立记录结果。
