# 结构轴驱动的 projective line pooling：有限审查

2026-09-26。只读当前 `refinement.py` 并检索一手论文；未实现模块、计算真实轴或运行 GPU。**建议至多先做 TRAIN-only 轴/方向诊断，暂不进入训练，也不把它列为已成立创新。** H3 的失败不构成本候选的正反证据；它与正在准备的配对外观诊断应保持独立。

## 直接重合与可陈述差异

| 一手工作 | 已有内容及本候选边界 |
|---|---|
| [Strip Pooling, CVPR2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Hou_Strip_Pooling_Rethinking_Spatial_Pooling_for_Scene_Parsing_CVPR_2020_paper.html) | 长条窗口聚合远程上下文；当前横/竖均值分支属于这一普通设计族。换方向本身不足以构成新机制。 |
| [Bundle Pooling, CVPR2020，§5.1、§6.1](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zeng_Bundle_Pooling_for_Polygonal_Architecture_Segmentation_Problem_CVPR_2020_paper.pdf) | **最直接重合**：沿 vanishing-line 束提取特征并 mean/max pooling，用于建筑多边形边界与 VP 分配；还比较了整流后 axis pooling。原文操作对象是候选框内多条线，不是当前逐像素 dense refiner，但不能宣称首次 VP/投影线 pooling。 |
| [VPSeg, CVPR2024](https://openaccess.thecvf.com/content/CVPR2024/html/Guo_Vanishing-Point-Guided_Video_Semantic_Segmentation_of_Driving_Scenes_CVPR_2024_paper.html) | VP 已用于语义分割中的跨帧运动对应与远处特征增强。我们的单个渲染视图、TRAIN 3D 轴来源不同；“VP 改善分割”这一动机已有。 |
| [Z-ACN, 2022，§III-B](https://arxiv.org/html/2206.03939v1) | 已有无需学习参数的相机/深度几何采样偏移与 adapted average pooling。其局部平面假设不应移植成桥梁虚拟支撑面，但“纯几何偏移、同参数量”不新。 |
| [Light Touch Multi-view Geometry, CVPR2023](https://arxiv.org/abs/2211.15107) | 已用极线约束 cross-attention。这里是同一视图的结构平行线投影，不是两个相机之间的极线；不应把两者混称。 |

可能保留的区别仅是**用 TRAIN 真实三角化点估计稳定结构方向，再将相机投影得到的线场作用于冻结 Gaussian 证据的 dense refiner**，无需检测测试图 VP、无需跨视图 RGB 输入、无需新增几何或改变原生类别。目前这是特定任务上的组合和输入来源差异，尚无证据证明方法学贡献；本次有限检索也不能证明没有更近工作。

## 几何与代码约束

当前 `PanoramaContext` 位于 1/16 feature map：4 个 pyramid 分支，2 个横/竖条带分支（均值后 Conv/GroupNorm/SiLU），再 merge；输入已有渲染 RGB、深度、alpha、16维特征、p3d、熵及梯度。因此模型可能已从边缘和长程上下文恢复足够方向；显式轴是否提供剩余信息必须验证，不能由现有错误反推“缺相机几何”。

设世界方向为 d，world-to-camera 旋转为 R，齐次消失点 a=KRd。对图像坐标 p=(u,v)，线方向可写为

`g(p)=(a_x−u a_z, a_y−v a_z)`。

这是 `Jπ(z K⁻¹[p,1]) R d` 的方向，其尺度含 1/z，**归一化后深度消去，平移也不决定该全局轴线场**。深度只在限定三维步长、局部变轴或遮挡判断时增加作用；第一项控制宜只测方向，避免同时加入不可靠 rendered depth 的尺度/可见性机制。

- `a_z≈0` 是 VP 在无穷远，应使用齐次公式，不能除以 `a_z`。真正退化是 `||g||≈0`（轴接近该像素视线、强缩短），以及两个投影轴近共线。对称线段 pooling 应对 d↔−d 完全不变。
- 同投影线不表示同类或可见连通：线会跨背景、塔/桥面交叉与遮挡。相机校准正确不能证明平均传播正确，也不会修复近相机浮层。若后来加深度门控，必须独立变量测试，不能顺手绑定进方向模块。
- 多座塔点整体 PCA 可能得到塔间水平方向；桥面点云也受视角密度、栏杆/索污染、弯曲及接近重复特征值影响。应先做真实点的局部/空间分块方向共识，报告失败，不强加 Manhattan 正交或把标签区域当平面。
- 1/16 特征中心须从实际 stride/padding 和模型像素 profile 推导；当前连续四次 stride2、kernel3、padding1 对 corner 图像的中心位置是 `16j+0.5`。不能按输出尺寸比例盲缩 K，尤其是 1320/989 非整除时。模型输出仍沿自己协议处理畸变/导出。

## 仅建议的下一道门：TRAIN 诊断

固定 TRAIN tracks 和已有重投影/观测数质量规则；按稳定 deck/tower 点分别估轴，来源及每类保留数记录下来。使用空间 block bootstrap 与移除相机组的重复估计，报告无向角置信区间、特征值间隔、空间覆盖及跨组稳定性；不能仅随机抽相邻点得出过窄区间。

在预先固定的 TRAIN 相机和均匀 feature 网格，报告：无定义/轴共线比例；方向场与固定横竖、TRAIN 汇总的最佳常数角之间的无向角残差；within-view 与 between-view 变化；定长采样的有效比例和跨图边界比例。可用 TRAIN RGB 梯度张量的高置信切向作独立对照，查看已有外观是否已提供同样方向，不能用它选 VAL 上表现最佳的轴。需要语义诊断时，另用 TRAIN 标注统计线采样跨类率，并明确全图、类内分别统计；GT 永远不参与推理网格。

若轴区间宽、方向大多退化，或 projective 场几乎等同一个常数旋转/容易从 RGB 得到，应停止；不能为得到复杂场而人工旋转轴或挑相机。阈值与固定样本集在查看诊断前登记，只可用于 TRAIN；本备忘没有已测稳定性数值。

## 仅在诊断成立后考虑的最小控制

保留原模型零训练结果和同预算原始 strip 续训作为必要基线。四个可比较处理使用相同双分支通道、权重、样本数 S、距离偏移、bilinear kernel、边界归一化和优化步数：固定横竖；TRAIN 方向分布拟合的全局常数旋转（两个方向不强制投影后正交）；真实 projective；固定 seed 的错误 TRAIN 旋转分配/打乱 pose。后者保持原 K，并报告有效采样率差异，不能把更频繁的越界造成的失败解释为相机方向有效。原始整行均值本身与有限 S 采样不等价，必须保留其独立续训基线，不能只和弱化的 sampled-strip 比。

**相同参数数目不等于相同计算与初始函数。** 当前均值可跨位置共享，新逐像素 grid sampling 会更贵；记录实际样本和时间。直接替换旧 pooling 会改变 GroupNorm 输入及旧权重语义，不能声称 warmstart 初始输出完全相同。若需原函数起点，应为四组共同预定无参数插值日程，从旧聚合逐步切换到各自聚合，统一核算计算；不要只给 projective 更温和初始化。

几何、RGB、Gaussian 语义 features/decoder 全冻结，仅训练同一 refiner，验证 raw p3d 逐位不变。不得加入 camera ID、测试照片、H3 moments、teacher 或配对外观干预。只有 projective 同时超过**原 warmstart、原 strip 等步续训和强固定旋转/相同采样控制**，且错误 pose 控制不能解释收益，才值得扩大验证；否则收敛为负/普通工程结果。即使一桥单种子获得正结果，也不因此证明首创或跨桥泛化。
