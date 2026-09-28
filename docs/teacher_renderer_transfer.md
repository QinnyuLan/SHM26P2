# 固定 DINOv3 教师向新 support RGB 场迁移

这项评价检验较大语义教师与学生的互补性，是成本较高的工程对照，不是新机制或性能上界。没有新增训练、teacher 更新、权重网格搜索或验证集图像输入。固定的 0.5 概率平均在四个预定学生上均有收益；最高组合为 95.0231% 全类 mIoU，仅略过 95%，不足以表述为明显超过目标。H3 的交叉项独有收益仍未得到支持。

计划在推理前保存于 `configs/generated_renderer_transfer/support_plan.json`，最终 SHA256 为 `f08bb1bbdb29499d1282d1ec2c2661909c7ba1f35c7a9bc2cb208545f390e433`。实际推理严格从 `runs/h3_moments_preflight_v2/source_snapshot` 导入 model、train、refinement、depth_moments、teacher，逐模块校验实际路径与 SHA；未混入之后主树的坐标或类别图缩放修订。执行代码、计划和完整快照另存于 `runs/teacher_renderer_transfer_support`，完成回执记录输入散列和指标散列。

教师固定为 `runs/teacher_render_adapt_v1/best.pt`，SHA256 `00f5b84ac9a56c39512c5b8e43f70110397923feea1a2b4c78bdffd1f5524bff`：ModelScope DINOv3 H+/16 冻结骨干加 DPT 头，之前用 TRAIN 真实 RGB 与旧 strong 场 TRAIN 渲染 RGB 各 50% 做 2000 步域适配。训练渲染场为 `runs/strong_rgb/last.pt`，SHA256 `3f57584a785110a70a2b4b52a4b73105b23448b223ca398b1ca160a0c91e1edb`。本次推理渲染场则为 `runs/support_split_semantic_coupled/last.pt`，SHA256 `a329fc70d582972fc338715765304b99da208c821ee34288809db500d5acd6b9`。两域被明确区分，旧教师在旧场上的 94.3866% 不被当成本次新场结果复用。

输入链为原生相机 → 同一 support 高斯场 RGB → 教师。四个学生为 support 8000 步原模型与 H3 的 zero / variance / cross 最后 3000 步模型，均使用这一相同 RGB 场。教师固定 tile 768、stride 512、水平 flip、0.25 全图 context、context 短边 768；每个相机只执行一次完整的 tiled/flip/context 预测协议，所得概率复用于四个固定 0.5 平均。这里“一次协议”包含多次 backbone 调用，不是单次神经网络前向。

审计通过：四学生所有非 refiner tensor、训练相机、scene scale、feature dimension、SH degree 完全一致；新渲染的 50 张 RGB 与四组各自既有 native PNG 逐像素相同。各学生在 41 张标注图上的逐视图和整体 confusion matrix 与原评测完全复现。每张相机的四个学生、一教师、四融合共九份预测全部完成后才打开 GT 和 valid mask，真实 VAL 照片不被读取为预测输入。预测仅保存 uint8 类别 PNG 和 JSON，未保存大体积概率文件。

| 同一 support RGB 场上的输出 | 全 5 类 mIoU (%) | 前景 mIoU (%) | Cable IoU (%) | Cable 边界 F1 (%) |
| --- | ---: | ---: | ---: | ---: |
| 固定迁移教师 | 94.1303 | 92.8323 | 94.0136 | 88.6827 |
| 原始 support 学生 | 94.5118 | 93.3197 | 93.5493 | 88.5672 |
| H3 Zero 学生 | 93.0724 | 91.6595 | 87.8950 | 74.4987 |
| H3 Variance 学生 | 94.5747 | 93.3887 | 93.7742 | 88.7626 |
| H3 Cross 学生 | 94.5363 | 93.3440 | 93.8271 | 88.7286 |
| 原始 support + 教师，固定 0.5 | 94.9530 | 93.8513 | 94.3341 | 90.7307 |
| H3 Zero + 教师，固定 0.5 | 94.6991 | 93.5450 | 93.5068 | 89.1694 |
| H3 Variance + 教师，固定 0.5 | 95.0003 | 93.9040 | 94.5068 | 90.7167 |
| H3 Cross + 教师，固定 0.5 | 95.0231 | 93.9316 | 94.7097 | 91.1027 |

所有输出共同 RGB 为 **PSNR 30.1024 / SSIM 0.902981 / LPIPS 0.224007**，使用同一场的原生 50 相机结果；没有与旧 strong RGB 指标拼接。各模型的语义分数均来自固定的 41 张标注视图。

5000 次配对视图 bootstrap 中，四个融合减去各自单学生的全类 mIoU 差异依次为 **+0.4412 [0.2351, 0.7946]、+1.6267 [1.0321, 2.3853]、+0.4256 [0.2088, 0.7913]、+0.4868 [0.2534, 0.8806]** 个百分点。Zero 的大差值主要涉及退化学生，不能用来声称矩输入贡献。Cross 融合比 variance 融合只高 **0.0228 [-0.1281, 0.2205]** 个百分点，比原始 support 融合高 **0.0701 [-0.1460, 0.3722]**；均不支持新增矩支路带来稳定收益。所有十个预定配对结果均保留，没有只呈现最有利的比较。区间描述单场景开发视图重采样，不覆盖训练随机种子、跨场景或隐藏测试集的不确定性。

教师共有 846,968,966 参数（骨干 840,592,640、decoder 6,376,326），单学生加教师共约 884.9M 参数；学生自身约 37.9M 参数。一次加载四个 head 共用同一场的审计 bundle 为 886,580,938 参数。50 相机的平均教师完整协议耗时 767.69 ms；四次学生渲染加一次教师协议及 RGB 审计 I/O 的 bundle 平均 1112.51 ms，峰值 PyTorch 分配显存 3.422 GiB。执行时没有其他 GPU 作业，但该 bundle 包含重复渲染和审计 I/O，不能当作单一融合部署方案的独占 FPS。轻量学生的正式独占基准另见 `docs/h3_depth_moments.md`，约 36.6–42.3 ms。

运行脚本为 `scripts/evaluate_renderer_transfer.py`，CPU 配对报告为 `scripts/report_renderer_transfer.py`；结果与九组预测位于 `runs/teacher_renderer_transfer_support`，其中 `execution_receipt.json`、`metrics.json`、`report.json` 和全部 `paired_*.json` 可追溯整个实验。旧域适配教师、原 teacher pseudo350、旧场 ensemble 和 H3 三组权重均保持不变。

固定组合已接入 `bridge-rgs render --teacher-checkpoint ...`，可与上述四个学生任一检查点配合。`src/bridge_rgs/render_ensemble.py` 固定完整预测协议和0.5权重，校验像素协议、类别顺序、骨干配置/权重散列及已知manifest散列；原生或overscan的渲染RGB经uint8量化后输入教师，融合概率再执行官方畸变映射。只接受相机和模型，外部RGB/GT不是推理输入。

新接口的001原生相机RGB及cross组合mask与本报告存档逐像素相同，记录在 `runs/render_ensemble_smoke/reproduction.json`。这是一相机实现合同，不是重新计算完整95.0231%成绩；官方overscan可能改变教师全图上下文，官方输出也不冒用原生验证分数。当前整仓373项测试（含CUDA）和Ruff通过。
