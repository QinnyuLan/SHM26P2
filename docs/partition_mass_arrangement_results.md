# 完成端点的 TRAIN 类别总量与内部排列检查

2026-09-27。固定四臂实验未通过采用门后，只检查其预定 integrated 终点，不挑最高分控制组。**已学到的空间排列在这 16 张 TRAIN 上有小幅作用，但精修后的差异极小，不能改写验证集负结果或支持创新增益。** 没有新训练或模型采用。

## 固定干预与结果

同一完成端点、同一精修头、16 个此前固定的 TRAIN 相机（002/021/041/059/079/100/118/137/156/176/200/220/241/259/278/300），每相机三次 camera-only 渲染：

- learned：原学习到的条件积分分区。
- own_qbar：把每 Gaussian 换成其自身解析无条件类别平均 `qbar=a*qin+(1−a)*qout`，去掉内部排列。
- perpendicular：保持类别端点及带中心/宽度不变，把局部法向确定性转到垂直方向；无条件占比 a 与 qbar 不变。

额外保存同一 context 的原 H3 p3d，没有另做 renderer 调用。继承真实训练的冻结 Cholesky projection，未使用后来新增的 OT helper。三次读取的 RGB/深度/alpha/原 p3d 哈希完全相同。固定的是**每个完整、连续、归一化 Gaussian 的无条件类别平均**，不是遮挡后的像素类别面积。

48 次预测及 112 个 FP32 NPY 全部完成、核验后才读取 16 份 prepared TRAIN mask 与 valid（共32次解码）；没有真实 RGB、原图 annotation、VAL、teacher、反向或优化步骤。下面是 legacy TRAIN 网格上的拟合诊断，**不能作为官方原图或留出泛化分数**。

| 读出 | weighted CE | Brier | pooled 五类 IoU % | 拉索 IoU % |
|---|---:|---:|---:|---:|
| 原 context p3d | .103967140 | .086570573 | 78.767115 | 27.506932 |
| learned raw | .103015168 | .085152626 | 78.943214 | 27.849781 |
| own_qbar raw | .103880766 | .085954978 | 78.863910 | 27.508845 |
| perpendicular raw | .104163849 | .086203902 | 78.868882 | 27.526336 |
| learned + 同一精修头 | .002255081 | .001461928 | 99.235873 | 99.106533 |
| own_qbar + 同一精修头 | .002268113 | .001465158 | 99.233727 | 99.102614 |
| perpendicular + 同一精修头 | .002265488 | .001464425 | 99.231997 | 99.103239 |

CE/Brier 先对每图已知有效像素取均值再等相机平均；IoU 来自 pooled CM。2000 次固定相机配对 bootstrap、seed20260927，仅描述，没有新增模型采用门。

learned raw 相对自身 qbar 常量的五类增量为 **+.079303 pp**，95% 区间 **[+.021983,+.122298] pp**；拉索 **+.340937 pp**，区间 **[+.063158,+.532196] pp**。相对垂直法向，五类 **+.074332 pp**，区间 **[+.012762,+.118851] pp**；拉索 **+.323445 pp**，区间 **[+.031219,+.525909] pp**。所以在这批拟合 TRAIN、这一完成端点上，内部排列的影响不完全由无条件类别总量变化解释。

同一精修头后，learned 相对 qbar 的五类增量只有 **+.002146 pp**，相对垂直法向只有 **+.003876 pp**。固定头读取改变后的先验可能出现分布偏移；这些干预不是各自重新训练的同预算比较，也不能判定剩余错误是否主要由排列造成。原 base 已训练过这些相机，bootstrap 不将其变成独立验证。

## 优化状态提供的另一个假设

[只读端点状态检查](/mnt/data/SHM2026/runs/semantic_partition_matched_v2/optimizer_endpoint_diagnostic.json)对三臂完成 optimizer state 做 CPU 计算。Integrated 的非零二阶矩分量中，`sqrt(vhat)/(sqrt(vhat)+eps)` 中位数：内端点 **.00093284**、外端点 **.00078882**、方向 **5.11e−7**；其中比值小于 .1 的分量分别 **92.38%/93.00%/98.22%**。端点概率 TV 的按 Gaussian 数量中位数为 **2.11e−6**，p99 约 **.09758**，说明也确有一部分较大变化。

这量化了同一保存 Adam moments 下 epsilon 对更新幅度的影响，**不是另一条训练轨迹、不按实际可见像素加权、不证明失败原因或充分收敛**。历史底座的 per-Gaussian feature optimizer 使用 `eps=1e−15`，本轮新场为 `1e−8`。因此后续可以单独检验 field epsilon，而不同时改 LR、宽度、投影或其他模块；普通优化配置修订本身不是学术机制。旧失败保留，不追加步数到旧端点来重选结果。

端点状态脚本 [inspect_partition_optimizer.py](/home/sky/workspace/SHM2026/scripts/inspect_partition_optimizer.py)运行 1.10 秒，未使用 GPU、模型渲染或标签。独立 NumPy/SciPy 重算三个端点的 618 个数值项，最大差 `2.22e−16`。按同一保存 moments 算出的 FP64 最后一步更新幅度是描述量，不冒称逐位复现 PyTorch FP32 Adam。

## 成本与核验

本次渲染诊断 worker **24.327 秒**、含启动 **24.701 秒**，预测屏障时刻为 **14.471 秒**，峰值 CUDA allocated **1,943,388,160 B**。共享桌面 GPU，时间包含模型加载、NPY I/O/hash、CPU统计，不是部署 FPS。

九项冻结 CPU 测试通过。场、分区参数、相机、flags/modes 前后完全恢复；输入/源码/预测哈希未变。[独立 CPU 复核](/mnt/data/SHM2026/runs/partition_mass_arrangement_v1/independent_cpu_review.json)已一次自然退出 0、9.759 秒通过：55 份源码、27 份输入、4 个已安装来源与112个NPY前后 SHA 一致；112 个视图 CM 和 7 个 pooled CM 精确一致；616 个浮点项、6 组 2000 次 bootstrap 最大差 `1.11e−16`。checker 未导入生产 score/summarize，不加载 Torch 或使用 GPU。

- [完整产物](/mnt/data/SHM2026/runs/partition_mass_arrangement_v1)
- plan SHA：`24f780c57979603ee9af35fc76684432f78744e1eabefe1d1cc56cbc0c54bd5e`
- execution receipt SHA：`b3348398cad0a9523aee403d8cce262175c54bdd8a993bc4de0a85cd7f39ae2b`
- independent CPU review SHA：`eedd8b8bcd571942022e1cba755e3b251be7c4b8705b93383d3a4531caa38f1c`
- optimizer 状态诊断 SHA：`5c16c634a21297c37d2b0157703aacf923f4c4940e7cfe50c620c024e32c585a`

原四臂验证集结论仍见[实际结果](semantic_partition_matched_results.md)。当前工程 E 不变。
