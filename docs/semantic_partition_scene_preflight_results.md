# Gaussian 内语义带：真实场景预检结果

截至本页记录，**新 footprint transport 修订通过了两个固定 TRAIN 相机的接线预检；没有训练收益或泛化收益结果**。全部尝试保留，v2 原条件模型的 conic 门失败没有被改写为通过。本页只汇总已完成的数值与工程证据，不改变任何冻结计划、源码或采用门。

固定输入为 H3 cross 场 `runs/h3_moments/02_cross/last.pt`，SHA `22bc8a2ddb260f93cb01b17857c97b2bb0873038efdb9318545cb2bdbb045226`，498,136 个 Gaussian；相机为 `002.png`、`118.png`，使用原生 `legacy_mixed_v1` 网格和原始 pose。真实预检没有打开 RGB/语义标签，没有 VAL、优化器更新或新模型 checkpoint。

## 四段证据链

| 阶段 | 结果与范围 | 回执 |
|---|---|---|
| v1 准备 | CPU 父回执路径遗漏 `execution` 子目录，准备失败；没有冻结 plan，也没有 GPU 执行。 | [preparation_failure.json](/mnt/data/SHM2026/runs/partition_scene_preflight_v1/preparation_failure.json) |
| v2 原条件模型 | 002 第一次 partition forward 在 conic 一致性门失败。原门为相对矩阵最大差 `256 × FP32 eps = 3.0517578125e-5`；没有得到六项预检通过结论。worker 1.5991 s，外层 2.5440 s，退出 1。 | [execution](/mnt/data/SHM2026/runs/partition_scene_preflight_v2/execution_receipt.json)、[launch](/mnt/data/SHM2026/runs/partition_scene_preflight_v2/launch_receipt.json) |
| 独立 002 数值定位 | 固定一次 `scene.render(refine=False)`，只保存几何与实际投影 metadata，0 图像/标签读取、0 backward、0 optimizer。1.7132 s，执行自然退出 0。 | [plan](/mnt/data/SHM2026/runs/partition_projection_diagnostic_v1/plan.json)、[analysis](/mnt/data/SHM2026/runs/partition_projection_diagnostic_v1/analysis.json)、[execution](/mnt/data/SHM2026/runs/partition_projection_diagnostic_v1/execution_receipt.json) |
| v3 新条件模型 | 新 transport 修订的两个相机 × 三模式全部通过；原失败目录保持原样。worker 4.7371 s，外层 5.7016 s，自然退出 0。 | [plan](/mnt/data/SHM2026/runs/partition_scene_preflight_v3/plan.json)、[analysis](/mnt/data/SHM2026/runs/partition_scene_preflight_v3/analysis.json)、[execution](/mnt/data/SHM2026/runs/partition_scene_preflight_v3/execution_receipt.json)、[launch](/mnt/data/SHM2026/runs/partition_scene_preflight_v3/launch_receipt.json) |

002 数值定位的 CPU 准备阶段也曾因 `np.float32` JSON 标量序列化失败，未使用 GPU；不完整 plan 留在诊断目录的 `prepare_incomplete_plan.json`。修复序列化后冻结独立 plan，再执行上述唯一一次 render。

## 原 conic 差异与修订理由

诊断保留了 002 的全部 33,323 个活跃 Gaussian，没有按误差剔除点。原 adapter 先算 `P = J Rcam Rquat S` 再算 `P Pᵀ`；安装的 gsplat CUDA 先算世界协方差，再变换到相机和图像平面，最后做 FP32 二阶矩阵求逆。两者代数等价，但有限精度的乘法、消减与求逆路径不同。

- 诊断的 CUDA FP32 `P Pᵀ + Cholesky` 重算有 153 行超出原 conic 门，最大相对差 `2.42454e-4`。这些行的协方差条件数中位数为 1,289.61，全活跃集合中位数为 22.89，最大为 80,312.82。
- TF32 原本关闭；显式关闭后的结果相同。仅将 Torch 运算改为 CUDA 源码的协方差乘法顺序也未解决：225 行仍超门。
- CPU 与 CUDA 的 FP64 重算一致，但相对实际 FP32 conic 仍有 99 行超门，最大相对差 `1.80639e-4`。因此不能把“改为 FP64”说成原实际 footprint 已被精确复现，也没有证据将此归因于某个单独 kernel bug。
- 直接使用原 `P` 和实际 `C_r` 构造 `I−PᵀC_r⁻¹P`，最小特征值为 `−4.24887e-4`，3 行低于原 `−128eps` 容差。放宽 conic 门后硬接这两个不一致对象，会得到非法条件协方差。

这些事实支持明确改写条件模型，而非调大旧门限。各变体、条件数、失败行及原始几何均保存在诊断目录；没有从这些结果挑点、改相机或改可见性。

## 新模型：实际 footprint 下的合法条件 Gaussian

源码 revision 为 `actual_footprint_whitened_transport_v2`，与运行目录 `partition_scene_preflight_v3` 的编号含义不同。先以输入 dtype 计算 `exp(log_scales)`，对应 renderer 实际递交的尺度，再用 FP64 构造

\[
C_0=P_0P_0^T+0.3I,\quad L_0=\operatorname{chol}(C_0),\quad B=L_0^{-1}P_0.
\]

实际同一次 render 的 SPD conic 定义 `C_r`，令 `L_r=chol(C_r)`、`T=L_r L_0⁻¹`：

\[
P'=L_rB,\quad E'=0.3TT^T,\quad P'P'^T+E'=C_r,
\]
\[
A=B^TL_r^{-1},\qquad V=I-B^TB.
\]

`V` 使用等价 Joseph 形式 `D Dᵀ + 0.3 A₀A₀ᵀ` 计算，其中 `D=I−BᵀB`、`A₀=BᵀL₀⁻¹`，避免高各向异性下用消减产生负方差。全部求解使用结构化 solve，不显式求逆。

**有效噪声 `E'` 通常不再是 `0.3I`。** 这是与实际 footprint 一致的新条件代理模型，不是原 `J` 和固定 `0.3I` 模型通过了旧门。Cholesky transport 依赖固定屏幕坐标顺序，不宣称是唯一、任意旋转等变的修正。它不改变 RGB、opacity、排序、裁剪、early termination 或共享合成权重 `W`。

实际中心和 conic 直接使用同一次 render 的 metadata；同场/同相机身份由 caller 的来源合同保证。旧相对误差仍输出诊断，近零投影中心不再以 `8eps × max(1,|mean|)` 强行判断身份。不会静默删掉活跃点或夹出大幅度合法协方差。

对已保存 002 几何的 [CPU 重放](/mnt/data/SHM2026/runs/partition_projection_diagnostic_v1/cpu_revision_replay.json) 没有新增 render，保留全部活跃点且 actual means 逐位一致：footprint 重建相对误差最大 `1.08921e-15`，Joseph 对原式差 `1.21014e-13`，`A C_r Aᵀ+V−I` 最大绝对残差 `1.16351e-13`；`max|T−I|=2.78519e-4`，有效噪声相对 `0.3I` 的最大绝对元素变化为 `1.46058e-4`。报告仍列出旧 conic 99 行、旧 mean 1,901 行超门。

## v3 实际通过的范围

冻结的 49 项 CPU 测试通过，包括 28 项语义公式、16 项 projection 与 5 项 field 合同。实际预检运行 `marginal`、`point`、`integrated` 三模式，共六个初始化比较。两端类别 logits 从原 classifier 等值初始化，六项 RGB、raw `p3d`、最终概率、alpha 与各自原场 baseline 的最大绝对差均为 **0**；simplex 最大误差为 `1.7881393432617188e-7`。

point/integrated 另用固定、相反符号的端点扰动触发非零 slab 梯度。两相机的 raw 和完整 final 标量探针，对 inside/outside logits、direction、offset、width 五组参数的梯度均有限且整体范数非零；head 每个参数张量都有有限梯度，head 总梯度平方范数非零。marginal 本轮仅核前向恒等，没有把方向梯度为零当作失败。这里的非零梯度接线检查不等于在真实场上做了完整有限差分校准。

| 相机 | 模式 | 初始化 forward / s | 扰动 forward + raw VJP + final backward / s |
|---|---|---:|---:|
| 002 | marginal | 0.97924 | 未测 |
| 002 | point | 0.06113 | 0.59231 |
| 002 | integrated | 0.05969 | 0.17687 |
| 118 | marginal | 0.34853 | 未测 |
| 118 | point | 0.06466 | 0.53021 |
| 118 | integrated | 0.06641 | 0.18942 |

峰值 Torch allocated 为 **5,753,475,584 bytes，约 5.358 GiB**。前向计时范围 0.05969–0.97924 s，包含首用/分辨形状相关开销；point/integrated 的四次前向范围 0.05969–0.06641 s，复合前反向为 0.17687–0.59231 s。这不是完整训练峰值、独立 FPS 基准或最终系统速度。运行前存在原 RustDesk 进程占用 537 MiB，未干预，因此也不是整卡完全独占的性能测量。

## 来源与结论边界

v3 plan 绑定 55 个源码/测试文件及 6 个输入，worker 结束核验来源不变。回执保存 15 个实际加载的 `bridge_rgs` 模块路径/SHA；本页编写时只读复核其均属于冻结 snapshot 并匹配 plan，也复核了实际 gsplat 二进制 SHA。环境为 Torch `2.8.0+cu128`、Triton `3.4.0`、gsplat `1.5.3`、NumPy `2.4.6`。

| 关键产物 | SHA-256 |
|---|---|
| v2 失败 execution receipt | `960d34858b8e8f32b254ae6698f80ecbf153b1161d3863ca92ac6d862f8a9ca0` |
| 单相机诊断 plan | `d9c5b83960009cb52d5e99b4fac74a5d73d4f2738f6b327d391988f0998db754` |
| 单相机诊断 analysis | `df77821c108429561a907bc295125f92081591f2d6901ad60bd32b043db02a08` |
| v3 plan | `6e19cf78d7e54092328dc6c4ef65f4dcadeb51e3cb920252d970aa22cba3d49d` |
| v3 execution receipt | `729e70f250f8608d2595f912969e13aff9e00b6dafc9fe820697e0a8c0bdd14c` |
| v3 analysis | `2d55dc0e34db1873aa55e61460023138f72ca7ad981e4ecd36b16cb38bf0b3e3` |
| projection 新源码 | `5605a78fb3422c6929f14173abaaaaba62465ec7759f062d3a42b760062ba24c` |
| field 源码 | `b8142ca1cdac5e275be115ba1d4ccaace6fd45b4f913dc3fd2704d807d0cb575` |
| shader 源码 | `27207fbcf89da463390909380dbb51e38df9c2dc3cb7f6e2a957db32fdd01196` |
| 实际加载 gsplat CUDA binary | `b08674757c8904d8cce210c3f5f9801dedf39f62fffd067b1600c4cb12046106` |

v3 证明新修订能在这两个真实相机上保持等端点恒等并传回有限、非零的 raw/final/head 梯度，允许随后按独立冻结合同进行匹配训练。**本页没有任何训练质量、held-out 增益或采用结论，现有 selected 系统不变。** Gaussian clipping/条件 CDF、primitive 内空间属性和上述数值传输均不能直接当作新颖性证据；对应先例与数学边界见 [semantic_partition_prior_art.md](/home/sky/workspace/SHM2026/docs/semantic_partition_prior_art.md)。
