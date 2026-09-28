# Semantic partition：固定合成数值验证结果

CPU 解析积分审计和 GPU 固定可见性 shader 审计均按各自冻结门通过，自然退出 0。**这只确认固定合成数值合同；尚无真实桥梁训练、语义提升或新颖性证据。** 3D-HGS/XClipGS 已有条件 Gaussian 半空间积分，NTS 已有 primitive 内空间变化纹理；归因与数学边界见 [先例和协议](semantic_partition_prior_art.md)。

## CPU FP64：条件积分与归一质量

根任务授权后只执行一次，48 个固定条件 × integrated/point/marginal 三模式，共 144 项记录。执行前冻结包内 44 项 CPU tests 全过（1.51 秒，禁 pytest cache）。10 个固定条件 × 三模式 × 16 个局部参数坐标进行独立五点差分。

| 检查 | 实际最大误差 | 固定门 |
|---|---:|---:|
| Torch 闭式值 vs 独立 SciPy 标量积分 | 6.9009e−13 | 绝对 1e−8 |
| 二维 Gauss–Hermite 归一质量 | 5.3291e−15 | 绝对 1e−8 |
| 非零梯度分量 vs 独立 NumPy 五点差分 | 相对 8.1797e−7 | 相对 1e−4，信号 >1e−7 |
| 全部梯度分量绝对差（描述） | 1.5415e−12 | 近零分量门 1e−8 |
| 图像/世界单位重表达 | 2.7756e−17 | 绝对 1e−10 |
| 类别 simplex | 1.1102e−16 | 绝对 1e−12 |
| 固定 W 合成概率 / alpha | 1.3878e−17 / 0 | 1e−8 / 1e−12 |

CPU 内部 2.273742 秒，外层 2.667921 秒；CUDA 未初始化，真实数据读取、渲染和优化步均为 0。SciPy quad 最大误差估计 7.5975e−12。归一质量门对 integrated/marginal 检查 prior 质量；point 只对其自己的解析屏幕期望核验，不因偏离 prior 而判失败。

[CPU audit](</mnt/data/SHM2026/runs/semantic_partition_numerical_v1/execution/audit.json>) SHA `0d1d4cc577f1ed337aca86e52bce9e0c919d2b0b21895c6cdd7d9ef2adb11e64`；[plan](</mnt/data/SHM2026/runs/semantic_partition_numerical_v1/plan.json>) SHA `2195e208966370acc48a6b5f51e4d2ae415200ca8befb63113e74120bd192f87`；[自然退出回执](</mnt/data/SHM2026/runs/semantic_partition_numerical_v1/launch_receipt.json>) SHA `c404798a8cb6cacff3d315a9c1ca3b1bb71c54810a413c05b17b57591bfc2d78`。

## GPU FP32：固定 W 栅格合成

根任务执行的七种合成情况包括常规、部分 tile、空交集、提前终止、阈值邻居、极小中心区间及尾部。与独立 CPU FP64 合成参考比较；`q_in=q_out` 时另外与安装的 gsplat `rasterize_to_pixels` 比较。

| 检查 | 七例实际最大误差 | 固定门 |
|---|---:|---:|
| 独立参考概率 / alpha | 1.4619e−7 / 1.2410e−7 | 各 2e−6 |
| gsplat 常量端点概率 / alpha | 0 / 0 | 各 2e−6 |
| 类别 simplex | 1.1921e−7 | 2e−6 |
| q_in/q_out/系数梯度绝对差 | 3.8130e−9 | 2e−5 |
| 同梯度相对 L2 差 | 3.4578e−6 | 非小信号时 5e−4 |
| 两个固定方向 FD 绝对差 | 4.1951e−7 / 4.3477e−8 | 2e−4 + 2e−3 × \|analytic\| |

两个 FD 的相对差分别约 4.762% 和 0.794%，通过的是预设**绝对加相对混合门**，不能误报为二者均达到 0.2% 相对准确度。独立 FP64 VJP 对照提供了另一个较强的梯度检查。尾部微小量也只在该绝对容差范围内得到验证。

GPU 内部 3.015512 秒，外层 3.817422 秒，PyTorch peak allocated 164,864 B；RustDesk 常驻约 537 MiB 已记录，因此不是独占 GPU 吞吐测试。无真实图像、训练或优化。

[GPU analysis](</mnt/data/SHM2026/runs/partition_rasterizer_synthetic_v1/analysis.json>) SHA `414d20dcb6a0fd5f1af8c632fe137b8160321580ef297ecec753182121838629`；[plan](</mnt/data/SHM2026/runs/partition_rasterizer_synthetic_v1/plan.json>) SHA `f9ff1e6dae231ebac5e8bf6866fc19b7bf2851fb86bb4f50808117aced0a841e`；[执行回执](</mnt/data/SHM2026/runs/partition_rasterizer_synthetic_v1/execution_receipt.json>) SHA `8e6c14050ef23348c0757299bf962bccf8402842c4c241c6684d040f6ac47689`。

## 来源与结论边界

原 GPU worker 检查的是复制的 gsplat 源文件 SHA，实际常量对照调用安装目录中的 wrapper；它未在执行当时逐文件绑定实际安装源码。执行后的 [supplemental dependency audit](</mnt/data/SHM2026/runs/partition_rasterizer_synthetic_v1/supplemental_dependency_audit.json>)（SHA `ae637413cc3631e76cb75cb4264c023875b85792443835b86dcc90ca277a8df3`）确认当前安装的 `_wrapper.py`、前向 CUDA 源码、`Utils.cuh` 与冻结副本及计划 SHA 全相同，版本也相同。此补充没有修改旧计划/回执，**不能追溯证明执行时的文件字节或未绑定的编译扩展二进制**。

CPU 测的是连续、完整、归一 Gaussian 的仿射 EWA 概率恒等式，不是有限画布或遮挡后绝对类质量守恒。世界单位测试仅核指定 `P=JS` 的代数重表达，未替代真实相机 Jacobian 接线。GPU 七例的 gsplat 输出误差 0 不推广到所有场景、投影适配器或训练轨迹；投影→slab→真实场景 shader 的联合梯度和语义收益仍需后续独立检验。现阶段不改变选中模型，也不作学术创新声明。
