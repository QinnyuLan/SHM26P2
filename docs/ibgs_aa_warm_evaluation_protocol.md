# FP64-factor AA IBGS 固定末态评价 v2

评价协议为 `ibgs_aa_stable_evaluation_v2`，只在 `ibgs_aa_stable_matched_v2` 两组各6000步自然完成后准备并执行。原 AA v1 在第62步的行列式数值失败保留，不作为可评价端点。训练数据、视角顺序、优化器、损失和步数保持此前对照；本轮共同使用 FP64 因子 AA 与 near=.01。固定主候选仍为full/fused，同时报告full/raw、no_source/raw、no_source/fused，不按结果换主候选。

新端点必须同时匹配新protocol、`renderer_profile.id=ibgs_centered_corner_v2_aa_factored64_near001_v2`、AA模块SHA、隔离二进制及完整训练specification；顶层 `aa_provider=bridge_rgs.ibgs_antialias_stable` 和 `precision_policy` 必须与训练plan及冻结模块 `PRECISION_POLICY` 完全一致。prepare只解析该源码字面量，渲染时再次核实际导入模块SHA和policy，不从live模块替换。原无AA和FP32-Gram AA端点均拒绝。继承旧helper时，仅临时赋予已锁新spec：协议／profile／provider／policy，以及训练每臂3600秒、外层7500秒上限；训练仍6000步，不能借加载身份改优化器或损失。

渲染时显式加载该隔离后端，并在同一稳定AA context内刷新两组各350个TRAIN来源深度、渲染两组各50个VAL相机，共800次AA调用。`capture_last=False`，所有源深度与目标使用相同补偿。读取来源TRAIN RGB；不读VAL目标或语义。所有200份native FP32预测完成后，评分阶段按既有官方畸变映射、clamp、uint8舍入保存200张PNG，再读取50张原始VAL RGB。

评分函数与此前一致：交付uint8 PSNR、Gaussian11 SSIM、AlexNet LPIPS；固定50相机、逐视图配对5000次bootstrap、seed20260926。报告主候选对相同训练步数的no_source/fused、原AA单场、当前E；四种输出均公开。固定10组比较和门不变：四输出各对两个旧参考，加两种读出的full减no_source。通过条件保持PSNR均值至少+.15 dB、其配对区间下界>0、SSIM均值不下降、LPIPS均值不增加。评分没有自动更新E。

这些视图已被反复用于开发，配对区间不包含训练重复方差或选择偏差，不当盲测结论。source输入对照包含联合训练路径的总效应；AA兼容性修复本身不构成学术创新。新FP64因子补偿不改CUDA内部协方差、剔除或已有近似VJP，也不保证逐位匹配原gsplat。该轮只评价RGB，语义沿用当前系统且不声称重新测得提升。

渲染内部600／外部660秒，评分内部300／外部360秒，记录实际墙时和显存。使用独立uv环境渲染、主uv环境评分。旧训练、旧评价、旧二进制保持可复现。
