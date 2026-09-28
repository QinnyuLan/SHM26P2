# 连续共享几何：新接线预检与匹配训练草案

2026-09-27。仅新 CPU 实现；尚未 prepare 或 GPU 执行。旧有限结构交换、pose-profile 及各已完成结果保持原状。本页不把连续位置修正、PCGrad 或约束优化称为创新。实际强语义阶段冻结几何；本轮检验的是固定 Gaussian 数量下，语义监督能否通过同一组 means 改善真实共享渲染，而非要求先证明几何容量不足。

## 本次可执行部分：两 TRAIN 接线校准

继承已完成、独立审计通过的 `raw_simplex_em_fw_v1/source_snapshot`，原 H3 `22bc8a…` 几何/SH/opacity 与该 run 的 `final_q_delta.npz`。q 取保存的 FP32 renderer 值，不归一、不加 floor、不映回 features。相机和所有非 means 参数冻结；means 使用独立 FP32 leaf，因此不修改原场。仅固定 `002.png`、`041.png` 的 legacy prepared RGB/mask/valid；不读 VAL、不运行 head/teacher/优化器。

每次调用原标准 gsplat 两遍：原 SH 的 `RGB+ED`（取 RGB）与显式固定五类 q，后者使用 class0 residual background。两者用相同 means、原 covariance/opacity、相机、AA、near/far；要求 alpha 逐位相同。**两个任务的 means 梯度均来自完整标准 projection、AA 与 compositor，不使用 direct-q 的 detached-info/q-only VJP。** RGB 目标为 valid 像素三个通道的 clamp-to-[0,1] MSE；语义目标为旧固定 class weights、`δ=5e−7` 的未归一 raw affine-noise CE；loss 聚合 FP64，场和 raster FP32。关闭 TF32/AMP，绑定原运行环境和实际 gsplat binary。

对每个任务自己的基线梯度 g，用原协方差 Σ 构造 `d_i=Σ_i g_i / max_j sqrt(g_jᵀΣ_jg_j)`。理想 d 的最大 Mahalanobis 长度为1；固定 h=`1/64,1/128,1/256`，与结果无关。每个 ± 端点均从同一原 FP32 means 升 double、加减后独立 cast，比较中心 FD 与 `g·(means_plus−means_minus)/(2h)` 的 FP64 内积。另存单边实际位移点积、实际 Mahalanobis 幅度和 cast 差；不拿理想 d 冒充实际移动。

每图一次 baseline＋同态重复＋2方向×3幅度×2端点＝14次双任务前向；合计 **56 standard rasters、4 means VJP、0参数更新**。两 loss 始终同步测量。全部12项 own-direction 主校准须相对误差≤5%，且实际 analytic 大于10倍数值 floor：同 loss 的重复差/h、FP64标量 spacing/(2h)、FP64点积归约 gamma_n 界的最大值。零梯度、端点量化重合或主项不可测不放行；交叉导数仅描述，不因其近零而失败。floor 是本次局部数值分辨率控制，不是全 renderer 的严格误差界。内部120s/外部180s，不重试、不改变 h 或阈值；完整恢复 flags/modes/gradients/数值设置并核所有 state SHA。

数值完成状态与 `numerical_status` 分开；通过仅表明这些采样方向的接线一致，不能推断全 Jacobian 正确、几何需要修正或性能收益。失败也不能直接归因某个 CUDA kernel。冻结 tests 必须 `-p no:cacheprovider`，通过 `uv run --no-sync` 启动，执行前创建独占 started 标记。

## 后续最简三臂（规格草案，不由此脚本执行）

共同 H3＋上述 q；只 means 更新，q/SH/opacity/scale/quaternion/camera/features/decoder/head 全冻，无 densify。259有标注 TRAIN，每轮固定独立 seed42 shuffle，batch7，37 batch/轮，4轮 **148 attempts/臂**；同初始 means、同序、同全batch均值目标与真实前后双任务 render 预算。fresh Adam `lr=1.6e−6*scene_scale, betas=(.9,.999), eps=1e−15`，对应本项目30k末态 means LR。每 Gaussian 候选实际 FP32位移按原协方差 cap `1/64`；超界且无法实现的行回原值。拒绝须回滚 means、Adam moments和step，不能只撤参数。

| 臂 | 交给同一个 Adam 的梯度 | 候选规则 |
|---|---|---|
| ordinary joint | `g_RGB + .03*g_sem` | 有限合法候选提交；仍测真实双目标 |
| PCGrad | 两个原始 task 梯度 `g_RGB`、`.03*g_sem` 做标准 symmetric PCGrad后相加 | 同上；不声称 Adam 实际步天然 RGB 安全 |
| RGB trust | `g_sem` | 仅实际同 batch RGB guard 与 CE下降均通过才提交 |

`.03` 来自已有但未完成强匹配验证的 semantic_geometry_weight 配置，不按新结果调参。batch接受不保证全TRAIN或VAL不退；所有臂必须报告 actual位移/损失、接受数与最终全TRAIN/统一末态评价，不能把梯度夹角当性能结论。具体 RGB guard 与正式运行限时须在训练 plan 前由 root 锁定；不得把本页草案当自动执行授权。等预算仅这些匹配 attempts，既有 EM/FW 成本另列。标准算法先例包括 [PCGrad](https://arxiv.org/abs/2001.06782)；形状尺度位移约束与多view gradient surgery亦已有先例，不据命名作新颖性主张。
