# 第四角的50视角评价与2×2交互

只在median4_normalized固定6000步完整训练且root观测自然exit0后准备。绑定新训练plan8455dc63…af7d19、已完成三臂评价plan5c744495…55929、原三个端点/分数、相同50相机和GT字节指纹。新训练的初始化已在首次更新前与原三臂逐参数哈希核对；评价再次检查6000步、有限FP32参数、原场/source缓存/后端及共同渲染模块完全相同。

复用原已执行评价源码1ccdfed8…def43c8c中的渲染、被动CNN特征观察、float clip、官方畸变warp、uint8量化、SSIM/LPIPS与配对统计路径。源码适配只涉及新端点协议、单臂数量/预测屏障与绑定旧分数的位置，不改变逐像素算法。50次原raster＋50次selector，0source-depth更新，50个原生FP32数组；独立评分进程先保存/校验全部50交付PNG，然后解码50个原GT RGB，各一次LPIPS。0语义、teacher、优化或新场更新。

固定主机制比较为**top4_normalized − median4_normalized**。另外报告新median4_normalized减median4_mass、top4_normalized、旧full/fused及E；其中新减top4_normalized是主机制比较的反向表示，不是额外独立证据。E四门仍为PSNR≥+.15dB、配对95%下界>0、SSIM不降、LPIPS不升，无自动采用或新语义结论。

2×2交互按同一视角计算`(top4_normalized−median4_normalized)−(top4_mass−median4_mass)`，再求均值及5000次整视角配对bootstrap，seed20260926。三个RGB指标分别报告，LPIPS的负向差改善；必须保持四臂相同GT哈希/网格/人口，不能分别独立重采样四臂。交互显著不自动表示top4更好，需同时看条件下的选择效应和绝对画质。

此实验是在原三臂结果之后提出的补充对照，原旧分数直接绑定，不能追认为原预注册三臂的一部分。开发集已重复使用、相邻视角相关且只有一个训练seed，区间没有实验选择校正，也不是跨桥梁或盲测证据。没有通过新的机制和完整系统检验时，当前E保持。

渲染内限300秒/外限360秒，使用原IBGS隔离uv环境；评分内限200秒/外限240秒，使用主uv环境。root顺序执行两个阶段并记录真实退出；使用新fresh目录`/mnt/data/SHM2026/runs/ibgs_factorial_evaluation_v1`。

状态更新：完整训练、渲染和评分均已自然exit0。归一化条件下top4比median4的PSNR提高.127846dB、三个指标的配对区间均改善；新median4仍未通过相对E的采用门。完整数字、交互和证据见[四组结果](ibgs_factorial_results.md)。冻结快照保存执行前的原协议文本。
