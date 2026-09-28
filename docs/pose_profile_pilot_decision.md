# 必要梯度门之后：两臂 pilot 决策备忘

**2026-09-27 状态更新：root 已授权此两臂 1,050 步方案的隔离 CPU 实现，尚未运行。** 原 v1 结果仍为 `inconclusive_measurability`；新的独立数值校准已完成，结果为 `direction_calibration_not_met`，所以已准备入口保持 **prepared_not_started**、gate 拒绝启动。标准线性 profile/Schur 消元不是新方法。原不可变计划 `0b5872b2…8006048` 与门槛不改变；实现合同见 [两臂入口说明](pose_profile_pilot_implementation.md)。

**可以复用的证据与输入。** H1 的 350 TRAIN/50 VAL 划分、原 RGB/valid、clean 与固定 mild 相机文件、未扰动的 VAL 相机、scene_scale/scene_radius 定义及旧渲染器快照全部保留。共同起点使用已核验的 H1 clean compensated 6k/177,378 点 checkpoint，故两臂都从相同成熟场开始，SH3 与语义张量相同。历史四臂的预测和指标只能作背景及基底参考：它们训练的是 L1/SSIM+前景加权，同时增密并改变归因排序，**不能充当新 raw-L2 基线**。[历史合同](pose_stress_results.md)

**最小两臂建议。** 两臂均在已有 mild TRAIN 相机和真实 RGB 上做 1,050 步，即 350 个视图的三轮独立固定 shuffle（seed 42），相同顺序、宽 320、旧 valid 规则；不重新抽噪声，不加大扰动。只优化全场共享 means，冻结 scale/quaternion/opacity/SH/背景/语义和相机；关闭增密、剪枝、opacity reset、原 H1 排名/质量加权、前景标签权重、L1/SSIM、教师及额外正则。这样测的是“成熟场在真实 RGB 错配训练中，梯度度量能否减少有害位置适应”，不是从头重建、细索 native 优化或一般容量收益。

| 臂 | 唯一训练差异 |
|---|---|
| raw-L2 | 原始加权 RGB 标量平方和除以 `2 sumW` |
| profile | 同分母、同残差，使用当前场/当前 TRAIN 相机的冻结 J 与精确生产 Λ 的六维正则 profile 梯度 |

两臂同用新置零的 means Adam，复用起点 checkpoint 的固定末态 LR/betas/eps，不承接一臂的旧动量。profile 每步重算 J，使用诊断中固定的差分 eps、先验、damping、有效行与单位；无 δ 截断、非线性接受、相机累计更新或自适应调强度。为压低成本，raw 不计算无用的 J；profile 每步 13 次 FD+1 次训练 render，raw 每步 1 次，合计两臂 **15,750 render、2,100 backward**，另有固定末点评价。两臂只匹配优化步数、所见真实图与活跃参数，明确不匹配 render/墙时，不能据此声称计算效率或等算力优势。先依据已授权梯度诊断的真实耗时评估资源是否值得，不承诺墙时，也不按结果改步数。若这一预算仍不合算，直接不做，不把更短无功效试验当否定证据。

**怎样判读。** 预先锁定只评起点与固定末点，不用中途 VAL 选参数或停止点；在同一 H1 原生评分指纹下完整报告 50 视图 PSNR（主量）、SSIM、LPIPS，以及相对共同起点的变化和 profile−raw 配对差。只使用两臂都成立的共同指纹，不能跨 native/official 网格混报。若 profile 比 raw 少退化，却仍低于起点，只能称这一短续训下的保护作用，不能称超越原模型。同步记录真实 means 更新/相对起点漂移、raw RGB loss 和 profile loss：Adam 可能抵消单纯梯度幅值缩小，因此 loss 下降或梯度范数缩小不足以说明最终更新更好；不事后换 SGD。配对视图区间仅描述本场，不能替代独立场景或训练种子。

**与 joint BA 的边界事先声明。** 固定 J 的 profile 梯度就是局部带先验相机块消元；joint BA 会保留并重新线性化位姿均值，本试验每步丢弃 δ、保持名义/测试相机不动。没有 BA 臂就不能声称比 BA 更优、更准确识别相机误差，或首次利用位姿不确定性；[Robust Gaussian Splatting](https://arxiv.org/html/2404.04211) 和 [LongSplat](https://arxiv.org/html/2508.14041) 已覆盖相关位姿分布/联合优化方向。只有 mild 两臂也不能证明噪声特异收益——历史 clean/raw-L1 不能补成交互对照。若两臂没有可辨认差异或共同明显退化，就停止这个已固定的 pilot，不扩大噪声、追加损失或筛指标追结果；正结果也仅支持另审 clean 配对/联合 BA 等必要控制，**不自动授权多组训练或学术创新宣称**。
