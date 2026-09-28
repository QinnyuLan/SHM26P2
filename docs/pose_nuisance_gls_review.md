# Pose-nuisance GLS：数学与可行性短评

**结论：值得考虑一次预先固定的梯度诊断，尚不值得直接安排完整训练；不构成新的 Schur、位姿不确定性或边缘化方法。** 本页是实施前的数学与文献审查。后续已完成[诊断代码和固定计划](pose_profile_gradient_diagnostic.md)的 CPU 检查，GPU 诊断仍待当前教师训练与终点评价结束；没有启动对应场训练。

令有效 RGB 残差为 \(r=R_\theta(T)-I\in\mathbb R^n\)，相机左扰动雅可比 \(J=\partial r/\partial\delta\) 有六列，顺序为平移、旋转；\(K=\partial r/\partial\theta\)。在线性模型 \(r+J\delta\)、\(\delta\sim\mathcal N(0,\Sigma)\)、先验 \(\Sigma\succ0\)、像素噪声方差 \(\sigma^2>0\) 下，固定 \(J,\Sigma,\sigma\) 时：

\[
C=\sigma^2I+J\Sigma J^\top,\qquad
C^{-1}=\sigma^{-2}I-\sigma^{-4}J(\Sigma^{-1}+\sigma^{-2}J^\top J)^{-1}J^\top,
\]
\[
E=\frac{r^\top C^{-1}r}{2n}
=\min_\delta\frac{\|r+J\delta\|^2/\sigma^2+\delta^\top\Sigma^{-1}\delta}{2n},
\qquad \nabla_\theta E=\frac{K^\top C^{-1}r}{n}.
\]

这就是带高斯先验的线性 nuisance 消元；联合位姿/场 Gauss–Newton 的相机块 Schur 消元产生相同局部二次形式。完整线性高斯边缘似然还含 \(\log\det C/(2n)\)：stop-gradient 的 \(C\) 使它当步不贡献场梯度，但随着场更新反复重算 \(J\)，得到的是移动的冻结二次代理，不能称完整非线性边缘似然优化。联合 BA 会累积、重线性化位姿均值；本候选丢弃局部 \(\delta\)，保持名义相机与测试相机不变，区别主要是这个操作约束。

[Robust Gaussian Splatting §4.1](https://arxiv.org/html/2404.04211) 已将相机位姿分布传播为高斯均值、形状协方差及 opacity 修正，联合处理位姿与模糊；本候选在图像残差的协方差中积分线性位姿噪声，保持均值渲染器不变，区别不等于已证实的新颖性。[LongSplat（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/papers/Lin_LongSplat_Robust_Unposed_3D_Gaussian_Splatting_for_Casual_Long_Videos_ICCV_2025_paper.pdf) 已进行增量式相机/高斯联合优化，结合学习到的三维先验和八叉树锚点；不能把联合几何与位姿或鲁棒相机本身作为贡献。

**与 H1 的证据关系。** [主循环](../src/bridge_rgs/train.py) 的 `total = loss_rgb` 仍来自原始 L1/SSIM；`camera_quality` 乘的是用于增密的 `means2d` 梯度统计，周期诊断在优化器更新后运行，并不投影主损失梯度。[H1 四组空结果](pose_stress_results.md) 因此没有直接测试上述 \(K^\top C^{-1}r\)。已有[恢复诊断](camera_attribution_mechanism_review.md) 的 self 8/8、真实 RGB 增量 7/8 必要门通过，只说明局部求解可恢复已知增量，不能证明全部可解释残差来自相机。其有限位移线性预测相对 RMSE 为 0.227–0.460，也不支持把局部近似当成精确物理分解。

**主要反证风险。** 设 \(J\Sigma^{1/2}/\sigma\) 的奇异值为 \(s_i\)，相对于 raw L2，相应图像方向被乘以 \(1/(1+s_i^2)\)，其余方向保留。任何真实几何变化 \(Ku\) 若落入这些方向，也会被压低；近似平面、整体刚体规范以及结构边缘位移都可能混淆。有限先验能限制损失，不能创造可辨识性；若实际训练，需明确场/相机规范和锚点。\(\Sigma\) 与 \(\sigma\) 之比决定抑制强度，现有经验先验并非校准的 BA 后验。有效像素、RGB 标量数、sum/mean 归一化和世界长度/rad 单位必须一致；生产 solver 的额外对角 damping、步长截断及非线性接受规则也不能偷偷等同于这个无约束 GLS。实现上可用 \(\Sigma=LL^\top\) 和六维 Cholesky 避免稠密 \(n\times n\) 逆；这些都属数值合同，而非创新。

**最小值得讨论的检验，仅设计、未执行。** 固定已有冻结场与八个 TRAIN 相机，不读 VAL/标签，沿用既有半幅相机扰动；另预注册一个非全局刚体、纯几何选择的局部场位移，不按残差挑位置或调幅度。用实际非线性渲染产生两种已知来源的残差，比较 raw L2、固定生产等价二次正则的 GLS，以及一次固定行置换 \(J\) 的控制（保留 \(J^\top J\)、秩与收缩谱）。零优化，以同单位的共享 means 梯度报告相机扰动污染是否下降，以及已知局部修复方向的符号/幅度是否保留（不混加位置、SH、opacity 的梯度范数）；保留原始能量和数值底噪，不把必然变小的 GLS loss 当收益。行置换控制只能排除部分任意低秩抑制解释；几何信号不可测则为 inconclusive。若真实 \(J\) 没有可分辨的特异作用，或同样消除了可测几何修复信号，就不推进这个固定先验候选，不扩大噪声/协方差追结果。若通过，也只支持再审议同预算训练；必须以 raw L2 对照 GLS，不能把 L1/SSIM → L2 的损失族变化混成相机收益，更不能据一次诊断宣称准确率或学术创新。

**2026-09-27 Energy-GS 正式全文复核。** 已从 [CVF 正式页面](https://openaccess.thecvf.com/content/CVPR2026/html/Gao_Energy-GS_Image_Energy-guided_Pose_Alignment_Gaussian_Splatting_with_redesigned_pose_CVPR_2026_paper.html) 的链接取得并阅读主文方法/实验及补充材料，不再停留于摘要。**其公开公式没有对场景 means 主梯度实施带 pose 先验的 Schur/profile 消元。** 主文 §3.3（p.7313，Eq.2–3）在姿态对齐阶段固定中心及初期点数，调整 tile 投影支撑以稳定参与反传的点集；§3.4（p.7314，Eq.5）按图像能级延后增密，中心不更新时仍计算其梯度供 clone/split。§3.5–3.6（pp.7314–7316，Eq.6–9）逐步开放目标图像的 SVD 能量分量，是监督图像课程，并非按 pose 协方差投影残差。[正式主文 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Gao_Energy-GS_Image_Energy-guided_Pose_Alignment_Gaussian_Splatting_with_redesigned_pose_CVPR_2026_paper.pdf)

补充 §1.3（pp.2–3，Eq.32、38）推导 alpha 与累积透射率到 SE(3) 的链式姿态梯度，并明确不使用 SH 颜色到姿态的第三条路径；其中形状协方差不是 pose-prior covariance。§1.1 固定第一相机作为规范。§1.4 承认初始点数、能级调度与姿态学习率敏感；这些不能用“已稳定梯度”消除。其学习和累积更新相机，未定义我们候选的六维正则最小化问题。[官方补充 PDF](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Gao_Energy-GS_Image_Energy-guided_CVPR_2026_supplemental.pdf)

实验检验的是联合重建/姿态恢复：主文 §4 使用 synthetic/Mip-NeRF360，分别训练 50k/100k；Table 4（p.7317）的 ship 逐项消融，PSNR 8.08→12.38→24.12、ATE .179→.150→.011，支持其完整策略，不是固定相机下 means 梯度的来源区分测试。这里未复现其实现，也不能把该训练预算或分数与我们的零更新探针比较。[正式实验](https://openaccess.thecvf.com/content/CVPR2026/papers/Gao_Energy-GS_Image_Energy-guided_Pose_Alignment_Gaussian_Splatting_with_redesigned_pose_CVPR_2026_paper.pdf#page=7)

与已锁定 `pose_profile_gradient_v1` 的**必要区别**是：我们保持名义相机，按每个条件固定真实有限差分 J 和生产等价正则，消去局部增量后检查共享 means 的梯度；stop-J/stop-δ 给出的是**冻结线性化二次代理的 envelope/GLS 梯度**，不是精确非线性联合 BA 消元。诊断用相同收缩谱的行置换控制，另检查已知局部几何修复方向是否保留；尚未执行，必要门通过也不等于端到端收益。Energy-GS 未覆盖这个具体操作不代表全领域新颖；标准变量消元、GLS、相机相关梯度稳定均不能独立作为首创。此次只收紧重叠边界，不改变冻结代码、先验、门槛或待执行状态，不扩展检索。

2026-09-26 的另一项摘要范围核验保持不变：[UGS-Loc（CVPR 2026）](https://openaccess.thecvf.com/content/CVPR2026/html/Kong_Rethinking_Pose_Refinement_in_3D_Gaussian_Splatting_under_Pose_Prior_CVPR_2026_paper.html) 在定位时结合位姿采样与 Fisher 信息几何不确定性；任务与本候选的固定名义相机下场训练不同，但也排除了“首次联合考虑位姿和几何不确定性”的表述。
