# H3：深度与语义特征交叉矩的信息消融

[These Magic Moments（2025）](https://arxiv.org/html/2503.14665v2) 的式 (9) 已用同一 alpha 合成过程计算任意特征的高阶矩，覆盖深度和语义 embedding。H3 沿用这种已有计算框架，检验交叉项作为语义细化输入是否有用；不把矩渲染、方差公式或协方差本身作为原创。固定终点实验未支持交叉项相对方差的独有增益。

共同底座是 `runs/support_split_semantic_coupled/last.pt`，SHA256 `a329fc70d582972fc338715765304b99da208c821ee34288809db500d5acd6b9`，498,136 个高斯；这是 support 几何上的语义 8000 步最终模型，全类 mIoU 94.5118%。不使用之前学习率实验的旧 strong average，也不混用它的 RGB 分数。全部输入仍由相机和同一场生成。

三组都给既有 head 增加同样的 17 通道零初始化投影，保留已训练卷积的形状和值：`zero` 提供全零；`variance` 仅提供相对深度离散度，另 16 维为零；`cross` 再提供有符号、经 asinh 压缩的深度—特征标准化协方差。三组 head 均为 564,757 参数，比原 head 多 7,344 个。主要比较为 cross 减 variance；只看 cross 减 zero 无法区分深度离散度与交叉关系的贡献。

预定仅训练 refiner：geometry、opacity、RGB、background、语义 features、共享 classifier 和相机全部冻结；259 张 TRAIN GT，独立 seed42 shuffle，原生 1320×989，全图 3000 步，固定 head LR .0003。没有学习率 schedule、teacher、pseudo、fusion、crop、depth、front 或 opacity 损失。新投影在原始 head 上以零初始化插入，禁止 reset head。save/eval_every 均为 0，只评价固定最后 3000 步。v2 预检和独立复核通过后，三组正式实验已按相同快照完成于 `runs/h3_moments/{00_zero,01_variance,02_cross}`，全部原生 50 张 RGB / 41 张带标注视图的评价已完成。

矩采用全体高斯同一完整遮挡权重，矩通道背景为零；z 先除以固定 scene scale，再用 alpha 归一化。实体远背景 shell 仍是场的一部分。该统计描述当前混合，不是校准的几何不确定性，也不保证单层近相机浮层可被识别。

第一版短预检保留于 `runs/h3_moments_preflight`。三组各 2 步中仅 refiner 变化，所有原生 raw 语义、RGB、geometry 和相机保持逐位不变；增加零支路后的初始概率和残差在固定 TRAIN `002.png` 上实际最大差均为 0。独立 gsplat 调用以全 498,136 点渲染黑背景 `[z,z²,z*f,f]`，alpha 与完整 RGB render 完全相等，特征和独立公式 context 与模型输出最大差为 0；归一化深度均值换回场景单位后，相对 RGB+ED 的最大绝对误差 4.9591e-5、最大相对误差 1.5993e-6。这是浮点误差量化，未预设 CUDA 必须逐位相同。

此第一版已标为 **numerical_revision_pending**：独立审查发现远距离单层的 float32 方差消减误差可能超过固定 epsilon，形成伪小方差。第一版不作为正式实验的数值门槛，全部记录保留。

修订版采用 `floor = 8*finfo(dtype).eps*max(abs(E[z²])+E[z]²,1)`，支持条件为 `alpha>=1e-4` 且 `variance>max(floor,1e-6)`；其余公式与固定场景尺度不变。这是数值退化保护，不是统计校准。root 的合成回归覆盖 z=10/20/1000 与 alpha=.3/.99 的单层零输出，以及 z=990/1010 的可分辨混合保留。修订不影响 depth_moments=off 的既有模型或学习率实验。

**v2 预检已完成并通过**，目录为 `runs/h3_moments_preflight_v2`，计划/正式配置为 `configs/generated_h3_moments_v2`。depth_moments.py SHA256 为 `b59d81036f34d24bacf2f3085dc01a6f097958dad48b54d6e49d59d618020ed8`。再次执行三组各 2 步和相同 TRAIN `002.png` 的独立矩渲染，全部冻结、head 初始前向、参数量、采样和矩公式合同仍通过。此相机上 floor 为 9.5367e-7…1.7746e-5，没有额外拒绝原先大于 1e-6 的像素，支持比例为 98.9804%；远深度单层风险由合成回归补充，不能由这一相机单独排除。

v2 的主审计进程及真实训练 worker 均断言实际导入的 model/train/refinement/depth_moments 来自指定快照，并记录路径与 SHA。worker 仅在真实训练器建立 optimizer 时添加只读检查，直接确认首步前所有旧模型 tensor 与 warmstart 逐位相同、新投影为零、仅 refiner 开梯度、Adam 状态全新；这与手工构造 head 的初始 CUDA 前向误差检查分开记录。三组两步后，非 refiner tensor 及相机逐位不变，raw 语义、features、RGB、alpha、depth 也逐位相同。zero 新投影全零；variance 仅第一个输入通道更新；cross 的深度离散度与交叉通道均有更新。

独立渲染指相同 gsplat 的另一次完整调用、独立通道排布和手工矩公式，用于检验数据连接、背景和 ED 一致性；它不是独立物理射线积分器。所有原生单相机初始概率/残差最大差实际为 0，矩 context 与手工公式最大差为 0，ED 误差仍为上述量级。预检峰值约 5.368–5.450 GiB，无验证图输入、无 50 图预检评测。

| 固定最后 3000 步 | 全 5 类 mIoU (%) | 前景 mIoU (%) | Cable IoU (%) | Cable 边界 F1 (%) |
| --- | ---: | ---: | ---: | ---: |
| Zero | 93.0724 | 91.6595 | 87.8950 | 74.4987 |
| Variance | 94.5747 | 93.3887 | 93.7742 | 88.7626 |
| Cross | 94.5363 | 93.3440 | 93.8271 | 88.7286 |

预先指定的主要比较 **cross − variance**：全类 mIoU **-0.0384 个百分点**，5000 次配对视图 bootstrap 的 95% 区间 **[-0.2049, +0.1796]**；Cable IoU **+0.0529 个百分点**，区间 **[-0.6782, +1.0451]**。未支持交叉关联比单独深度离散度更有效这一假设，因此不能据本结果声称交叉矩带来了有效的学术创新。

次要比较 variance − zero 的全类 mIoU 为 **+1.5023 个百分点 [0.8537, 2.2757]**，cross − zero 为 **+1.4638 个百分点 [0.8917, 2.1865]**。但共同 warmstart 是 94.5118%，zero 在额外训练后退化到 93.0724%；variance 和 cross 相对 warmstart 只高 **0.0629 / 0.0245** 个百分点。不能把大于 zero 的差异全部称为新增信息提升部署精度，更不能将其解释为交叉项独有作用。相同存储参数量也不等于相同活跃输入维数：zero 的新增投影始终收零输入，variance 只有一维有效输入，cross 有 17 维。

最终审计全部通过：三组 source/input SHA 一致，只 output/mode 不同；所有非 refiner tensor 和训练相机逐位保留，冻结 optimizer state 为空；50 张 RGB PNG 以及整体、逐视图 raw 3D confusion matrix 均与 warmstart 完全一致。原生 3D mIoU 均为 **79.1987%**。Torch、CUDA、NumPy RNG 的组合 SHA 三组同为 `1d8b2422433b0f010999a1a05f15befc56821810e2ad1060d530f1ae0cf43b6a`，view sampler order/cursor/RNG 一致，259 图每张访问 11–12 次。实际新投影更新范围符合三种输入设定；这只是路径检查，不是特征重要性证据。

全量记录为 `runs/h3_moments/h3_report.json` 和三份 `paired_*.json`，报告脚本为 `scripts/report_h3_moments.py`。三臂约 785–786 秒训练时间来自并发运行，不用于独占延迟比较。没有中途验证、选择 checkpoint、改权重或额外种子搜索。结论仅限单场景开发划分与单训练 seed，视图配对区间不覆盖跨 seed 或跨场景不确定性。

另存三份 `paired_{arm}_minus_source.json`，补充相对原始 support 8000 步源模型的配对区间。Zero 的全类变化为 **-1.4393 个百分点 [-2.2474, -0.7418]**；Variance 为 **+0.0629 [-0.0555, +0.1972]**；Cross 为 **+0.0245 [-0.2154, +0.3543]**。两个有效矩输入相对原始模型的整体区间均跨零。该次要比较同时包含额外训练和输入支路的变化，不能单独识别信息作用；主要比较仍为 cross − variance。

正式训练结束后另行独占 GPU 测量 50 个原生相机、每个 3 次、先 warmup 5 次的 RGB 与最终语义 argmax。同步 wall-clock 计时，不含模型加载、JIT、照片/标签读取、PNG 写入、回传 CPU 或畸变重映射。四份 `benchmark_{base,zero,variance,cross}.json` 均记录测量前后无其他 GPU 客户端，不能与并发训练耗时混用。

| 模型 | 平均延迟 (ms) | FPS | 峰值分配显存 (GiB) |
| --- | ---: | ---: | ---: |
| 原始 support 8000 步 | 36.628 | 27.30 | 1.589 |
| Zero | 42.308 | 23.64 | 1.823 |
| Variance | 41.537 | 24.08 | 1.823 |
| Cross | 42.098 | 23.75 | 1.740 |

Cross 相对原始模型增加约 14.94% 推理延迟，在主要精度比较未见稳定收益。此测量只覆盖当前硬件和实现，不代表其他部署环境。
