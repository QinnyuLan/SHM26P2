# 曲线约束细构件：范围很窄的先例记录

2026-09-27。三维滤波30k没有改善本项目指标之后，只读检索了“用参数化细索／中心线耦合高斯”的可能方向。未实现该方法、未新增实验、未将其纳入当前诊断。现有直接先例足以排除把曲线锚定本身称为创新。

[Curve-Aware Gaussian Splatting，ICCV 2025原文](https://www.openaccess.thecvf.com/content/ICCV2025/papers/Gao_Curve-Aware_Gaussian_Splatting_for_3D_Parametric_Curve_Reconstruction_ICCV_2025_paper.pdf)已经把三次Bézier曲线／线段与高斯耦合，由曲线采样规定位置、切向规定主轴、曲线厚度规定横向尺度，直接以多视图边缘监督优化，并包含线性化、合并、分裂、剪枝。该工作重建的是曲线，不能直接据其结论推断本桥完整RGB＋区域语义会改善；但“把细构件的多个高斯绑到同一条曲线”存在明确重叠。[作者代码](https://github.com/zhirui-gao/Curve-Gaussian)。

[FrameTwin作者预印本](https://arxiv.org/abs/2605.09362)的摘要进一步描述了曲线锚定高斯、稀疏视图中的细杆变形及神经变形场，全局一致性用于线框打印的反馈。这里只依据作者摘要确认设计重合，未审查完整数学与代码；不能把全局曲线一致性本身视为未有工作覆盖。

本项目的官方stay_cable mask是区域标注，不能当作单根索中心线真值。若以后研究真实细构件约束，仍需从TRAIN RGB／可信三维观测建立对应证据，检验遮挡、误关联、粗初始化与欠支撑情况下的行为，并将完整RGB与语义收益分开验证。当前没有这项证据，所以不因此扩大实验范围，也不替代正在准备的有限opacity前向诊断。
