# AA 训练第 62 步：有限参数下的 FP32 行列式消减

2026-09-27。`ibgs_aa_warm_matched_v1` 的 full 臂只完成 61 次场更新、350 次初始深度和 61 次目标渲染，head 更新为 0；第 62 步 `004.png` 在外部 AA 补偿计算中失败。训练没有完成，两臂比较及性能结论均不存在。原失败目录、checkpoint 和阈值保留。

根进程的一次投影捕获复现了原异常，在 996009 个槽中仅找到 index 740284。该诊断没有调用 raster、backward、optimizer 或解码图像。本次独立检查只用 NumPy 读取保存的 `bad_slots.npz` 与相机元数据，不加载 checkpoint，不调用 Torch/CUDA 或原诊断计算函数。

## 独立数值核对

从实际 activated FP32 的 means、scale、wxyz quaternion 和 w2c 直接提升到 FP64；**不重新 exp、不重新归一化 quaternion**。四元数平方范数为 0.9999999795487098。用独立叉积矩阵表达 `R = I + 2w[v]× + 2[v]×²`，按原 clamp 与 FP32 tan(FOV) 设置重建投影因子 `B = J Rcamera Rquaternion diag(scale)`，再以两行叉积平方范数求行列式。

| 量 | 保存的 FP32 路径 | 独立 FP64 因子路径 |
|---|---:|---:|
| det(C) | −536870912 | 717437286.107254 |
| det(C + 0.3 I) | −469762048 | 795449059.950191 |
| rho | 原 guard 拒绝 | 0.949698572930051 |

独立因子与保存因子的最大绝对差为 1.82e−12，rho 差约 3.85e−14。不同 FP64 矩阵乘法排列仍使 det 差约 5.94e−4，但相对差仅 8.27e−13；不要求条件数较差的直接 `ad−bc` 逐位等于因子计算。

原 C32 为 `[[257871168, 23652720], [23652730, 2169497.5]]`，两项非对角元素已经相差 10。独立 C64 的特征值约为 2.75895775 与 260039243.084，条件数约 9.43e7；C32 相对 Frobenius 偏差只有 5.48e−6，却足以改变极小特征方向与行列式符号。这是有限参数下的矩阵乘法／行列式舍入问题，不需要以参数 NaN 解释。

实际 camera mean 为 `[1.16283044, −0.79599424, 0.0207088553]`，高于固定 near=.01；scale 为 `[.298689187, .0000161575517, .228656799]`，最大／最小比约 18486。未 clamp 的 x/z、y/z 为 56.15、−38.44，超过 ±0.9269、±0.6944 的限幅。所有保存输入有限，scale 为正，opacity 为 .0630427。训练源码在每次已完成更新后检查场参数及 Adam 状态有限；回溯也表明本次通过输入有限性检查后，在 determinant guard 处、进入第 62 次底层 raster 前终止。这不证明该槽实际覆盖任何像素，也不证明优化方向稳定。

## 可修复范围

因子表达 `D = ||B₁ × B₂||²`、`H = D + 0.3 ||B||² + 0.09` 在实数中与原 AA 相同，不需要加 floor 或调整训练阈值。正 D 的自动微分等价；rank-one 处采用安全零分支仍是零次梯度约定。FP64 可避免本例主消减，但不是对任意极端尺度、乘积溢出或近共线 minors 的普遍保证。

**IBGS CUDA 内部 covariance/conic 的计算没有因此改变。** 外部稳定 rho 不能证明内部足迹、剔除、离散支持与排序正确，也不修复既有 alpha cap、clamped Jacobian 或 median 加权反传近似。因此后续独立稳定模块必须使用新 renderer profile，重新绑定模块 SHA；旧失败不追认为通过。本检查不启动 v2、不修改模型或 helper，不构成训练收益／新方法证据。

来源：[原失败训练回执](/mnt/data/SHM2026/runs/ibgs_aa_warm_matched_v1/full/training_receipt.json)、[回溯](/mnt/data/SHM2026/runs/ibgs_aa_warm_matched_v1/stdout.log)、[一次投影诊断](/mnt/data/SHM2026/runs/ibgs_aa_determinant_diagnostic_v1/execution_receipt.json)、[保存坏槽](/mnt/data/SHM2026/runs/ibgs_aa_determinant_diagnostic_v1/bad_slots.npz)。诊断 execution SHA `75b0b739a646b9097e09b64a076b9a1def64ab9f058328a112185f0ddc492bea`，NPZ SHA `7929ccb8b6f910efe709dfcb1f7e6c3a61b1590a13c0266244963f941bb0ab37`。
