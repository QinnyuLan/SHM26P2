# IBGS 数值端口 revision 1

2026-09-27。这是官方 commit `977e96c6574f20b456c760942e615031154b4cd5` 的明确工程修订，不是原实现逐位复现。原构建、identity-source 烟测失败和原 median-depth 差分失败全部保留；它们没有被新端口结果覆盖。[原构建报告](ibgs_build_validation.md) 只证明构建／导入成功。

本轮修订限定如下，未调烟测容差或深度一致性阈值：

1. 按 [相机合同](ibgs_camera_contract.md)，Python Camera 与 CUDA 前后向全部采用数组主点 `(W−1)/2,(H−1)/2`；反向独立 ray 也同步。保留 NDC、整数数组坐标、texture 采样 `+.5`。适用域仍为共同居中 K、共同 native 尺寸；不冒称任意异源内参支持。
2. 前后向使用同一逐 buffer 投影与支持函数：源点须 finite、正 z、投影在图内，采样处 source depth 大于零。上层用 radius-1 erosion 的 RGB-valid 乘 source depth 来屏蔽无效 RGB；这里仅检验非零支持，没有添加新的逐交点深度一致性阈值。最终 median-source 接受还须有正的已累积颜色权重，避免 median 有效、实际颜色支持全空时输出假黑色源。
3. 修复 plane 梯度写入位置。每个 buffer 交点先累加所有合法源的颜色→深度项，再将深度梯度一次传给 plane。原代码在每个源内重复写累计值，导致 M 个源重复直接深度项，并使无源时直接深度梯度丢失。分母同步前向的 `S+1e−8`，投影导数也包含前向的 z epsilon。
4. 纹理梯度采用连续双线性近似：对数组 uv 取 floor，四邻点从 texture 中心 `(i+.5,j+.5)` 取值。原代码对 `uv+.5` 取 floor、再从整数 texture 坐标取平均值，错位半像素。这个修补**不**对硬件纹理插值分数的量化求导，也不认证完整 rasterizer 的梯度。

设固定 buffer 权重为 `w_i`、交点深度为 `z_i`，本次修正的标量为

`D_i = g_depth w_i/(S_depth+eps) + Σ_m [w_i/(S_m+eps)] g_color,m · ∇I_m · p′_m(z_i)`。

交点 `z_i=−d_i/t_i` 的 plane pullback 只执行一次：`∂d_i=−D_i/t_i`，`∂n_i=D_i d_i ray/t_i²`。这项局部推导不涵盖离散 buffer/遮挡切换、硬件 texture 量化或整个 alpha/transmittance 梯度链。根代理现已完成固定 0/1/4 源的实际 GPU 差分，18/18 通过，最大绝对差 `4.63724e−5`；合成后端和无效 source RGB 污染检查也自然通过。具体数值、原失败与范围见 [验证结果](ibgs_validation_results.md)，不能据此宣称真实场景质量或完整梯度已经验证。

随后 996009 点 SH3 的真实 `002/041` TRAIN 接线也自然通过：8 source-depth + 2 target forward、2 backward、0 optimizer，field 参数不变、field/融合网络梯度有限，内部 3.431237 秒；详见同一 [结果报告](ibgs_validation_results.md)。这是资源与接线验证，不是训练质量或新 VAL 成绩。

五个文件的前后副本、相对原构建的独立 patch、相对官方的完整 patch 和原二进制均保留。此次重建自然 exit 0，用时 **21.4988 s**；纯导入自然 exit 0，用时 **2.7674 s**，CUDA 未初始化。现有 8 项 CPU 相机合同通过（pytest `0.58 s`），两名代理完成有限源码交叉审查。未由本代理运行 GPU 前向、训练或真实图像读取。主环境 `uv.lock` / `.venv/pyvenv.cfg` 未变。

- [端口回执](/mnt/data/SHM2026/third_party/ibgs_port_revision1/port_receipt.json)：`bccd8bc3a409c57d6118cfe717b56e5414e91ae73f6d38b021e4507bbf15874e`
- [独立数值 patch](/mnt/data/SHM2026/third_party/ibgs_port_revision1/port_revision1.patch)：`6003020b60ce0b58143fd31326b93a2814d208bfdd758f6eafff9fc7de0f219a`
- 新 `diff_plane_rasterization._C`：`436b2b55df2b37cddda59e606ec0855ad125eb211058b502197674bd2b93adaf`，已安装到原独立 IBGS venv 并另存端口目录，`sm_120`；simple-knn 未变。
