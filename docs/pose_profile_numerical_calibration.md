# 原五视角局部修复方向：独立数值校准

状态：root 已按锁定版本完成一次 GPU 校准，结果为 `direction_calibration_not_met`（5/5 可测、3/5 方向可靠、3/5 通过组合保留门）；以下先保留原协议，末节记录实测。未授权 pilot 训练。这是新诊断；原 `pose_profile_gradient_v1` 的 `inconclusive_measurability` 和不可变 plan/audit/receipt 不改写。

原报告中，058/170 的全图 RMS 分别为 1.01978e−5/5.10580e−6，低于预定 3.81470e−5 下界；但 raw 修复内积分别为 9.95620e−11/2.33103e−11，明显超过各自原方向下界 1.30465e−14/4.07393e−15。决定旧门的是绝对 FP32 强度尺度常数，不是实测重复抖动。若局部变化仅覆盖有效像素的比例 f，全图 RMS 会有 √f 稀释；22 个 Gaussian 本身不能确定 f，也不能确定遮挡。旧报告没有保存逐像素预测与完整梯度，不能据其标量补算独立有限差分，更不能把稳定重复等同于正确导数。

**固定内容与唯一目的。** 复用旧 H1 177,378 点场、旧源快照、SH3、宽 320、旧 valid/权重/J 差分/先验/置换。仅看原 003/058/115/170/344 五个 TRAIN 视角、原 22 IDs、原 θg=θ₀+d；不读真实 RGB、语义或 VAL。自目标仍为 I₀=R(θ₀,T)，修复 u=−实际 FP32 d。只校准这些方向上的 means 反向、真实渲染变化及 profile 修复信号，不声称全部 means Jacobian 正确或真实照片中的相机误差可分。

三个端点固定为 θ(h)=float32(θg+h u)，h=1/8、1/16、1/32；都在原 θ₀ 至 θg 的修复区间内。没有中心差分向外增加扰动，也不按信号重选步长。CPU 计划逐点保存实际 Δθ(h)=double(θ(h))−double(θg)、量化偏差；GPU 必须逐位复核。非选中的全部点、其余所有模型/相机张量不改。

**同一条链的三项检查。** 每视图只在 θg,T 构建一次原 13-render 相机 J 和 Λ；三臂和全部端点共用，不重新估计或调正则。raw、真实 J profile、固定等谱置换三臂都使用原 FP32 renderer/残差相减，再 FP64 聚合损失与内积。每个端点都重新求最优 δ；不把旧 δ 固定后当成 profile value。

令 r̄=√W r，S=ΣW，L 为该臂固定二次损失。对两次独立基础前向和两次独立端点前向，分别报告：

- A=g_means(θg)·实际 Δθ；使用真实 CUDA backward 与实际 FP32 坐标变化。
- B=g_r̄(r̄g)·Δr̄；图像侧梯度独立由六维解的闭式表达 `(r̄g+J̄δ*)/S` 计算，raw 则为 r̄g/S。
- D=L(r̄(h))−L(r̄g)，以及独立正项余量 Q=L(Δr̄)。固定二次式应满足 **D=B+Q**。

A 与 B 的差包含 renderer 有限割线/反向差别；B 与 D 的差由明确的二次余量解释。A 与 D 还直接检查实际小步损失变化是否接近反向预测。因为 gsplat 为 FP32 且步长有限，不能把这些割线宣称为精确连续 Jacobian；FP64 聚合不等于 FP64 renderer。此链不调用 SSIM，因此不借另一个 SSIM 修复来假定 means backward 已正确。

**预先固定的独立数值门。** 1/8 全量报告但不参与最终门；两个较细步 1/16、1/32 必须分别成立，不能事后选较好步。每臂每步的 floor 取以下最大值：基础/端点/差值的实测重复 loss 差、原场两次 null raw loss、重复 A/B 差、32ε₆₄×对应 loss/图像内积绝对项尺度、32ε₃₂×Σ|g_means,i Δθᵢ|。它是诊断容差，不是已证明的 CUDA 误差上界。必须同时满足：

1. |A|、|B|、|D| 都严格超过 10×floor，不使用任意全图绝对 RMS 门；
2. `|A−B|/max(|A|,|B|)` 与 `|A−D|/max(|A|,|D|)` 都≤10%，并且 D=B+Q 相对组成项误差≤1e−9；
3. 原五视图三臂都达到该方向可靠性，且 raw 的 A、D 均为负。至少 4/5 视图的 profile A、D 保持负号，并分别保留同一步 raw 下降量的≥50%。

10% 是本次预声明的有限步方向一致性容差，不由 058/170 结果拟合。三步均保存全部重复值、A/B/D/Q、各 floor、相对误差。误差不过门只能报告“此固定方向的校准未成立”；可能来自非线性、量化、排序不光滑或错误 backward，不能直接归咎某个 CUDA kernel。变化不超过 floor 则为新诊断 `inconclusive_numerical_signal`。旧结果始终仍为 `inconclusive_measurability`。

另只作描述地记录真实基础重复 RMS、null RMS 和残差能量逆参与率 `(Σr̄²)²/Σr̄⁴` / 有效标量数，区分全图平均与能量集中；这不定义可见支持、遮挡标签或新筛选规则，不用它调门。

**预算与来源。** 每视图 1 target+2 null+13 J+2 base+3×2 endpoint=24 render，2 null+2×3 means backward=8；总 **120 render、40 backward、0 optimizer**。原 264/112 worker 包含加载/核验共 5.447 秒，因此保留 180 秒子进程硬上限充足但不承诺实际耗时；超时不重试。外层 CPU 准备和首次哈希不计 worker 上限。执行必须等 root 明确 GPU 交接，compute client 非空则拒绝。旧模型/source/input 字节与原 SHA 核对后复制只读快照；正常/异常均恢复参数、buffer、梯度权限与原 grad，末端核验所有张量/相机及磁盘来源。硬超时无法保证进程内 finally，回执明确记录；不写候选 checkpoint。

实现：`scripts/calibrate_pose_profile_direction.py`；合同：`tests/test_pose_profile_calibration.py`。CPU 覆盖独立密集矩阵的图像梯度/二次恒等式、真实 FP32 区间量化、重复稳定但错误 backward 的拒绝、低强度正确方向与实测噪声区分、不可用粗步替换细步、五视角完整性、异常恢复和固定预算。旧冻结 helper 的数学/载入/权重/置换/恢复代码保持原字节。

即使新校准支持方向，也只允许另行评审主训练梯度消融：历史 H1 的 L1/SSIM+增密排序训练不能充当 matched raw-L2。需要新的 raw-L2/profile 同源匹配；若要声称 clean/stress 鲁棒性交互，两条件都必须有这两臂。没有联合 BA 对照不能声称优于 BA；这是桥梁场上冻结线性化 profile 主梯度的应用问题，不是新 GLS 数学。本次不实现或授权任何 8k 训练。

CPU 已锁定：`/mnt/data/SHM2026/runs/pose_profile_numerical_calibration_v1/plan.json`，SHA256 `d14423c9772793648d2e401af946418ce9c12cde00f4d54f514157faa7892f69`；冻结 runner SHA256 `89c89efee6aa2602f228c68fb7d9a4ff82f4fa7bbe0c973110501302946a9323`。新 13 项与旧 25 项 CPU 合同合计 38 passed，Ruff 通过；准备记录 CUDA 未初始化、0 像素解码。未执行 GPU，仍等 root 交接。


## 2026-09-27 一次执行：可测五帧，方向校准未过

root 执行冻结计划后自然 exit0，receipt 为 completed，耗时 **3.197 秒**，120 render / 40 backward / 0 optimizer，0 真实 RGB/语义解码。全部模型张量/保存相机逐位恢复，输入/source SHA 未变。这里的 completed 指协议成功执行；科学结果仍为 **`direction_calibration_not_met`**，不能写成校准已通过。CPU 独立复核匹配 plan/audit/receipt SHA、从 A/B/D/floor 重算门及逐视图/总体 summary；未重跑 GPU。

下表列两个预定细步的 raw 链误差，格式为 A/B、A/D 相对误差；profile 和置换三臂数值非常接近，组合门要求全部臂成立。

| TRAIN | h=1/16（%） | h=1/32（%） | 五帧预定数值门 |
|---|---:|---:|---|
| 003 | 0.0509 / 3.3368 | 0.6219 / 1.1182 | 通过 |
| 058 | 0.2942 / 3.3354 | 0.0620 / 1.5890 | 通过 |
| 115 | 7.4315 / **10.5285** | **10.1436 / 11.6124** | 未通过 |
| 170 | 0.2071 / 3.9549 | 0.1736 / 2.2455 | 通过 |
| 344 | 7.7417 / 4.6522 | **13.9646 / 12.2602** | 未通过 |

**058/170 的实际方向信号可以分辨。** 其 h=1/32 的 raw A/B/D 分别为：058 的 −3.11542e−12 / −3.11349e−12 / −3.06592e−12，combined floor=4.07797e−17；170 的 −7.29877e−13 / −7.28610e−13 / −7.13487e−13，floor=1.27346e−17。两帧的两细步均满足本次独立验导容差。原绝对全图 RMS 门在这两个局部干预上较保守；这不回写原 v1 的不可测判定。它们的残差能量逆参与率/有效标量数分别约 0.002592% 和 0.004035%，表明能量高度集中；该统计不等于实际可见面积、也不是遮挡判定。

**相对 profile 修复量尚可描述，但不补足精确链校准。** 五视图两个细步的 raw、profile 实测 D 都为负，profile/raw 实际下降量保留率为 **99.4027%–99.9711%**；058 为 99.7879%/99.7818%，170 为 99.4040%/99.4027%。115 和 344 的 profile/raw 也接近 1，但 `retained_views=3` 是预先规定的“数值方向可靠 AND 保留量”组合门，不是说这两帧没有观察到下降。不能用相对量相近代替 A/B/D 的绝对一致性，也不能称“5/5 修复门已经通过”。

全部 45 个 D=B+Q 恒等式通过，组成项归一化误差最大 **4.432e−15**；全部变化相对 combined floor 的最小比率约 **52,463**，远超预定 10 倍。因而本次问题不在所记录重复噪声遮住信号，也没有观察到固定二次代理恒等式失效。115/344 的失败位于参数线性预测 A 与有限图像割线 B/真实有限损失变化 D 的匹配；三臂同样失败，不支持把问题特指为 profile 解/消元。有限位移非线性、FP32 图像/参数量化、排序或覆盖的不光滑性、renderer backward 偏差尚未被此设计区分，不能直接归因 CUDA 或某个 kernel。重复一致和恒等式一致都不是整个 means 渲染 Jacobian 正确的证明；也不能把这个局部门失败推成整个候选机制无效。

冻结 pilot 的 `calibration_gate()` 已在 CPU 实际复核：对该 completed 校准明确抛出 `Independent numerical calibration did not support this pilot`。已准备的 1,050 步两臂保持 **prepared_not_started**，无执行 receipt、无训练/新 checkpoint；不改 gate、不补样、不增幅、不重试 GPU。原 v1 仍为 `inconclusive_measurability`。

来源：[audit.json](/mnt/data/SHM2026/runs/pose_profile_numerical_calibration_v1/audit.json)，SHA256 `27533f04dee52f0a169c307e3c6af7fd7d74f4a58fc5cace68b6f19b9ca4a4c2`；[execution_receipt.json](/mnt/data/SHM2026/runs/pose_profile_numerical_calibration_v1/execution_receipt.json)，SHA256 `ba6885185e4dd3dfcbb420c096860a5ff66fc335c883cc452cc263ff399441ba`。原锁定 plan/runner 字节保持不变。
