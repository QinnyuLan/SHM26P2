# 直接畸变渲染与共享相机不确定性：有限研究审查

2026-09-27；仅本地源码与六项一手来源审查，未运行 GPU、未改训练或选择模型。**3DGUT 值得作为直接原图训练的标准工程参考候选；目前没有已验证的新颖性，也没有足够证据启动“相机标定不确定性边缘化”的新训练。**它与当前教师 RGB 迁移相互独立。

**本机可训练性有明确边界。** gsplat 1.5.3 的 `fully_fused_projection_with_ut` 不对输入求导，但 `with_eval3d=True` 的独立世界空间 backward 返回 means、quats、scales、colors、opacity 导数；不能由前一警告断言几何不可训练。畸变要求 UT，且只支持 `packed=False/sparse_grad=False`；相机、K、畸变不支持梯度。当前 `means2d.absgrad` 增密不能直接接入；UT 的 AA compensation 和追加的中心深度没有相应几何导数，ED 仍有权重导数，但不是完整深度梯度。SIMPLE_RADIAL 须映射到六维 radial、二维 tangential，像素中心是 `j+.5`。这些是源码合同，尚无本机 eval3d 数值验导。[已存本地路径/SHA与具体限制](raw_grid_training_diagnostic.md#本地-gsplat-153-梯度合同)

| 最接近的一手来源 | 已覆盖内容与本候选边界 |
|---|---|
| [3DGUT §4.1–4.4，Eq.6–11](https://arxiv.org/html/2412.12507v2) | 七个 sigma points 来自 **Gaussian 形状协方差**；UT 用于投影支撑，3D ray response 独立反传。不是相机后验积分。原文另有每射线排序、3D 位置梯度增密及 L2/SSIM 配方，不能把本地开关称为完整复现。 |
| [gsplat v1.5.3 rendering / eval3d](https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/gsplat/rendering.py) | 本地仍由投影深度构建 tile 排序；有世界响应梯度不代表实现论文全部排序策略。任意颜色通道允许共享 RGB/语义合成，但不构成新的共享可见性方法。 |
| [Robust Gaussian Splatting §4.1 / Algorithm 1](https://arxiv.org/html/2404.04211v1) | 已把位姿分布一阶传播为每个 Gaussian 的均值、协方差与 opacity 补偿，处理位姿误差/运动模糊。不能声称首次 pose uncertainty splatting；逐 Gaussian 传播也不等于精确保留共同位姿导致的遮挡相关性。 |
| [UGS-Loc §3.2–3.3，Eq.4–7](https://arxiv.org/html/2603.16538v1) | 已结合位姿粒子、先验与渲染梯度 Fisher 几何不确定性进行定位。任务不是共享场的 RGB—语义联合边缘似然，但“pose prior + uncertainty”本身已被覆盖。 |
| [Continuous Semantic Splatting §III-B/C，Eq.12–16](https://arxiv.org/html/2411.02547v1) | 已用遮挡合成权重更新 Dirichlet 语义，并渲染均值/方差；独立 Gaussian 语义变量假设不包含一个共同相机潜变量。不能把概率语义或不确定性合成本身称新。 |
| [SGS-SLAM §3.1–3.2，Eq.3/5/6](https://arxiv.org/html/2402.03246v2) | RGB、深度、语义共用 alpha/transmittance，语义参与位姿跟踪。共享几何、共享 visibility、语义辅助相机优化均已有直接先例。 |

**剩余差异必须先成为可识别问题。** 已知镜头畸变、Gaussian 形状、曝光期间运动与未知但固定的标定误差是不同随机对象。共同相机扰动 \(\delta\) 使整条射线的 \(W_i(\delta)=\alpha_i(\delta)\prod_{j<i}(1-\alpha_j(\delta))\) 相关变化；其期望一般不能由各 alpha 期望的乘积替代。然而，若目标仍是 \(\mathbb E L_{RGB}+\lambda\mathbb E L_{sem}\)，两模态共用或独立抽取同分布的位姿，**期望目标完全相同**，区别最多是 Monte Carlo 方差；分别输出均值图也不能识别该关联。真正不同的候选例如 \(-\log\mathbb E_\delta[p(I\mid\delta)p(Y\mid\delta)]\)，而非两项独立边缘似然，但这需要明确观测模型、尺度及可信的相机分布。这是标准潜变量建模的区别，不能称新数学或据上述六项有限检索认定全球新颖。

本项目 [camera_uncertainty](../src/bridge_rgs/geometry.py) 是固定三维点、经验 pose prior 的条件 GN 近似，明确忽略点—相机相关、共享内参及系统误差，**不是校准后验**。当前证据也未定位出由共同相机误差导致的 RGB/语义联合遮挡失败：[H1 6k](pose_stress_results.md) 没有端到端鲁棒性增益；[H2](h2_multiscale.md) 的投影采样增量与 [H3](h3_depth_moments.md) 的 cross−variance 主比较均未支持收益；[pose-profile](pose_profile_numerical_calibration.md) 后续方向校准仅 3/5 通过，原门仍失败。这些不是对所有相机模型的否定，但更换 renderer 不能自动修复旧数值门或提供新机制的正证据。

**可证伪的缺口仅保留为条件问题，不安排新实验：**在独立已知的 TRAIN 小位姿扰动下，共同位姿导致的 RGB/语义可见性关联，是否比匹配两侧边缘分布的平滑更能识别正确修复？若未来具备可信观测模型，可用固定等权位姿样本，比较原配对与仅置换语义样本索引的联合分数；两边均值/边缘分布、场、预算保持相同。若配对关联没有可分辨信息，或收益只来自分别模糊两张图，就停止这个具体联合关联假设；不能通过增大扰动或重调协方差补救。此时还不能推断训练收益，尤其错误语义赋值可能反向污染 RGB 相机证据。

当前建议止于**标准直接畸变渲染的可行性核验**：保持固定原始相机/原图像素约定，先验证独立 3D 响应导数，再明确增密、AA、深度与排序的配方差异；训练和推理必须使用同一新后端，结果走共同 official 输出评分且单独记录 inference protocol。已有 TRAIN 往返插值损失说明工程问题存在，却不是模型性能上限或标定误差证据。不建议现在把它与不确定性、语义机制合并成新方法，也不增加语义支撑面。
