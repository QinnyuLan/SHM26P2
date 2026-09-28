# Semantic partition 数值 transport 的坐标依赖

当前 `actual_footprint_whitened_transport_v2` 使用 Cholesky transport，在理想 footprint 与实际 footprint 不完全相同时，**其条件均值存在可复现的屏幕坐标轴依赖**。这不是 quaternion 正负号错误，也不改变合成权重 W。固定高条件数合成例子显示该依赖不能普遍视为微小量；仅重放此前保存的 002 几何时，幅度明显较小。下面均为 CPU 代数或旧几何重放，未读取本轮训练 checkpoint、soft 输出、图像或 GT，未运行 renderer/GPU；不能据此声称当前实验发生了准确率损失。

独立新模块 [partition_transport.py](/home/sky/workspace/SHM2026/src/bridge_rgs/partition_transport.py) 实现标准 2×2 对称 Gaussian optimal transport（OT）供未来数值修订审查。它未接入 `partition_projection.py` 或当前训练。OT 选择是最小二次位移的协方差映射，不是对真实相机误差的估计，也不是新的数学算子。本轮匹配训练的源码、采用门和结果解释均保持原合同。

## 当前 Cholesky 依赖的精确范围

令 `C0=P0 P0ᵀ+0.3I`，实际 rasterizer 的 footprint 为 `Cr`，且两者 SPD。当前实现取 `L0=chol(C0)`、`Lr=chol(Cr)`、`B=L0⁻¹P0`，条件均值系数 `A=BᵀLr⁻¹`，条件协方差 `V=I−BᵀB`（实际用 Joseph 形式计算）。

对正交屏幕变换 O，若 `L0*=O L0 Q0ᵀ`、`Lr*=O Lr Qrᵀ`，则

`A* O = Bᵀ Q0ᵀ Qr Lr⁻¹`。

只有相应白化旋转相容时，才得到 `A*O=A`；例如 `Cr=C0` 时成立。`V` 在精确算术下保持不变。小的普通矩阵范数扰动，在条件数 κ 较大时仍可能产生约 `θ√κ` 量级的标准化条件均值差。因此，约 `1e−4` 的 footprint 相对差本身不是类别概率误差上界。

当前和候选修订都是与 `Cr` 一致的局部条件代理。它们不等同于原 `P0` 加固定 `0.3I` 噪声的物理模型；transport 后噪声一般为各向异性的 `0.3 T Tᵀ`。此分析也不声称完整透视投影、像素栅格、裁剪、排序和 early termination 对任意图像旋转精确等变。

## 已固定合成样本与加权幅度

[固定规格](/mnt/data/SHM2026/runs/semantic_partition_coordinate_synthetic_v1/specification.json) 的 40 个矩阵组合使用 κ∈{1,10,100,10⁴,10⁶}，屏幕主轴角 0°/30°，footprint 微扰旋转 0/`1e−4` rad，坐标旋转 30°/90°。相同查询集合为标准化半径 0/1/2/3、16 方位；固定三个 normal、b=0/.5、w=.75、τ=.3，三种模式均报告。没有按结果追加方向或挑选样本。

以下是微扰 `1e−4` rad 时 integrated gate 的最大差。加权列使用**合成单 primitive、opacity=.9、无前序遮挡、两端为相反 simplex 顶点**，权重 `W=.9 exp(−||z||²/2)`，是该查询集合上最大的 `W|Δg|`。

| κ | Cholesky gate 最大差 | 合成 W 加权类别概率最大差 |
|---:|---:|---:|
| 1 | 3.33e−16 | 1.21e−16 |
| 10 | 8.43e−4 | 1.52e−4 |
| 100 | 3.39e−3 | 5.94e−4 |
| 10⁴ | .0352043 | .00613326 |
| 10⁶ | .361886 | .0651397 |

这说明最坏固定合成条件下不能笼统称可忽略；**不是场景全局上界或真实误差发生率**。实际输出差还乘以 `q_in−q_out`，并受 opacity、前序遮挡和多 Gaussian 合成影响。等端点初始化时类别输出不随 gate 改变，但两端参数的梯度责任 g 与 1−g 可以不同。这里没有测当前训练端点对比度或其优化轨迹。`Cr=C0` 的原 FP64 合成对照最大 gate 差约 `5.26e−11`，marginal 模式坐标差为 0。

原始 [report.json](/mnt/data/SHM2026/runs/semantic_partition_coordinate_synthetic_v1/report.json) 与 [weighted_report.json](/mnt/data/SHM2026/runs/semantic_partition_coordinate_synthetic_v1/weighted_report.json) 保存每个固定样本。

## 对称 OT：数学与有限精度分别核验

标准 SPD Gaussian OT 为

`T=C0⁻¹ᐟ² (C0¹ᐟ² Cr C0¹ᐟ²)¹ᐟ² C0⁻¹ᐟ²`。

二维 SPD 矩阵平方根恒等式给出等价形式：

`T=(Cr+s·solve(C0,I))/sqrt(trace(C0 Cr)+2s)`，`s=sqrt(det(C0)det(Cr))`。

新 helper 先用同一标量归一化两协方差，s 从 Cholesky 对角线积计算，`trace(C0 Cr)=||LrᵀL0||²_F`，避免形成条件数可能约 κ² 的中间 eigensystem；不裁剪特征值，非 SPD 显式拒绝。输出 FP64。令 `P'=T P0`、`E'=.3 T Tᵀ`、`A=A0 T⁻¹`，保留理想条件模型的 Joseph `V0`。需分别验证 `P'P'ᵀ+E'=Cr` 与 `A Cr Aᵀ+V0=I`。

该 OT 在精确算术下对**正交**屏幕坐标变换等变，并非任意仿射变换等变。共同屏幕单位改变时，应同时按单位平方改变噪声方差。局部 Gaussian 正交基变换在同步变换语义 normal 后亦保持相同模型。

同 40 个样本的独立 NumPy 比较见 [ot_report.json](/mnt/data/SHM2026/runs/semantic_partition_coordinate_synthetic_v1/ot_report.json)：

| FP64 方法 | integrated 旋转 gate 最大差 | 合成 W 加权最大差 | 最大白化 footprint 残差 |
|---|---:|---:|---:|
| 当前 Cholesky | .361886 | .0651397 | 8.34e−11 |
| 直接 eig 根式 OT | 1.32e−5 | 4.23e−6 | 3.05e−5 |
| 2×2 稳定恒等式 OT | 4.01e−11 | 1.28e−11 | 1.01e−10 |

直接 eig OT 的普通相对 footprint 残差仅 `3.05e−11`，但白化残差达 `3.05e−5`，说明只看最大特征方向的归一化残差会掩盖短轴误差。稳定二维形式的 T 和有效噪声在全部案例中 SPD，没有 clipping。旧比较 JSON 中 Cholesky 的 `T_eigenvalue_min` 对 `sym(T)` 求值；它可能为负，**不能据此说非对称 Cholesky transport 非法**，其噪声和 joint covariance 另行核验。

实际新 helper 的同样 40 案例又通过当前 `prepare_shader_coefficients` 路径生成系数，δ 同步旋转；[helper_precision_report.json](/mnt/data/SHM2026/runs/semantic_partition_coordinate_synthetic_v1/helper_precision_report.json) 明确区分算术精度：

| 新 OT 的系数与查询精度 | integrated gate 最大差 | integrated 合成加权最大差 | point gate 最大差 |
|---|---:|---:|---:|
| FP64 | 4.07e−11 | 1.30e−11 | 8.12e−11 |
| FP32 | 9.36e−5 | 2.53e−5 | 1.82e−4 |

FP32 条件系数与大各向异性查询中的消减仍然破坏有限精度等变性；不能把 FP64 代数性质升级为 shader 逐位等变声明。这里 CDF 为 SciPy 的稳定尾部差，**没有执行 Triton**，也没有为这些数值建立或放宽真实预检门。16 项 CPU tests 包括三模式 FP32 系数/查询旋转的固定普通条件合同；它们不是上述极端案例的新接受阈值。

## 旧 002 保存几何：全部活跃点重放

原独立诊断的 [active_geometry.npz](/mnt/data/SHM2026/runs/partition_projection_diagnostic_v1/active_geometry.npz) 仅含几何、实际 FP32 scales、K/pose、conics/means/radii。此次读取全部 33,323 个 active 行，使用原已审 `diagnose.calculate` 的 FP64 clamped Jacobian 构造 P0，Cr 从保存 conics 的 SPD solve 得到。没有模型加载、新 render、标签、真实 RGB 或当前训练产物读取。运行 23.75 s，CUDA 未初始化，输入/源 SHA 前后相同。

[重放规格](/mnt/data/SHM2026/runs/semantic_partition_coordinate_synthetic_v1/old002_replay_specification.json) 与 [结果](/mnt/data/SHM2026/runs/semantic_partition_coordinate_synthetic_v1/old002_replay_report.json) 记录：

- C0 条件数中位数 22.89，p99 1,633.72，最大 80,312.82。
- 新 OT 的白化 footprint 最大残差 `4.65339e−13`，普通行相对最大残差 `1.35018e−15`，总协方差残差 `1.36557e−13`。
- C0、Cr、T、有效噪声、V 的最小特征值依次为 `.3000082`、`.3000083`、`.9997565`、`.2998539`、`3.37255e−8`；全部正，无行剔除或特征值修补。
- `max|T−I|=2.01396e−4`；`max|E'−.3I|=1.20823e−4`。与 Cholesky 的 V 差 `1.11e−16`；两者条件均值改变方式不同。

在同样固定 slab 参数/查询下，以下均不是实际已训练的 q、实际 alpha 或遮挡权重：

| 比较（002 全 active） | FP64 integrated gate 最大差 | FP64 合成加权最大差 | FP32 integrated gate 最大差 |
|---|---:|---:|---:|
| Cholesky，90° 坐标变换 | 6.60e−5 | 1.31e−5 | 6.60e−5 |
| OT，90° 坐标变换 | 2.66e−13 | 6.67e−14 | 0 |
| 同坐标 OT 对 Cholesky | 6.55e−5 | 1.30e−5 | 6.56e−5 |

OT 的 FP32 0 仅发生在本次 90° 符号交换、固定查询与 CPU 算术下；前面的 30°/高 κ 案例已经展示非零舍入误差。旧 002 的 Cholesky 加权差 p99 约 `7.03e−7`，并不支持将高 κ 合成 worst case 当作实际普遍问题。同样，也没有由此证明误差对任意学得的窄 τ/端点对比度可忽略。

## Quaternion 表示与后续边界

q 与 −q 表示同一旋转。固定合成检查中 q、−q、2q、1.7q 经归一化后 A/V 相同。更一般地，若同一世界协方差因尺度/轴置换写成 `M'=M U`，同一 slab 应同步令 `n'=Uᵀn`；其 gate 差为 `5.55e−16`。若改了 Gaussian 局部基却重新固定 n=e1，描述的就是不同世界方向，固定例子的 gate 差为 `.538777`。这属于语义参数/初始化的坐标约定，不能指认为 quaternion 双覆盖 bug。signed permutation 下可对应变换参数和优化器状态；一般正交重参数化下，逐坐标 Adam 并不自动保持相同训练轨迹。

本次可交付的是一个未接入生产的可审查数值替代。若未来采用，需要单独冻结 revision，验证同一真实场的等端点恒等、实际 FP32 shader/梯度、footprint 与资源成本，再按预先匹配的训练计划比较。不得把当前试验结果事后归因于 Cholesky，或因新 helper 改写当前门、重选 arm。OT 既不解决表征可识别性，也不提供模型质量/新颖性证据。

## 可复核来源

| 文件 | SHA-256 |
|---|---|
| 当前 projection（未改） | `5605a78fb3422c6929f14173abaaaaba62465ec7759f062d3a42b760062ba24c` |
| 新 partition_transport.py | `d1860babf4a6e1e5f52af97facc1d30dff9937e6fc55271cf8b625f0333b512d` |
| 新 16 项 tests | `df935cfd825a73f84fa9650fa13b9268fc499b6e4908eeab223c3ab0eec44a69` |
| 原 002 active_geometry.npz | `ac51934988370b3b8e74bafffc0289e9c6be7e24c886a3e25c98d090a9acc797` |
| 原合成 report | `1719a3f06b9e49864294c6207400b6cfa3c44eab55cdb4d0e45ccbf576ca754c` |
| 原合成 weighted report | `d30166cc230897c180096c0483f5238175dec0aa933a933992865a37a38ae6aa` |
| 原同 case OT 比较 | `8744078378d62a0220407541c9cde1ba489abc8d03877f55b66697dd51c7953b` |
| 新 helper FP64/FP32 比较 | `2d24fac3ee22821aa0837cc71f95ff2721c94a26da905a3adbcf91c533948213` |
| 旧 002 新 helper replay | `3cc6bb4c38336619dccb7580d976571c0dc95c5b5a30f685eb776941aa3009ca` |

CPU 执行使用项目现有 uv 环境、`CUDA_VISIBLE_DEVICES=''`、禁 bytecode；16 tests 与 Ruff 通过。新 helper 已独立只读数学复核。结果脚本都保存在上述独立合成目录，原诊断和当前训练源没有修改。
