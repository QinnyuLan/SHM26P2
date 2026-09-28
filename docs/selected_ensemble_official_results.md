# 已选 DINOv3 组合的原图复核

2026-09-26。**既定组合在共同原图网格达到五类 mIoU 95.1090%、前景 94.0362%、拉索 94.7867%。比同场语义头提高 0.5310 个百分点，配对 95% 区间 [+0.2794,+0.9352]；50 张 RGB PNG 的 SHA 全部相同。** 这是既有组合的语义工程收益，不是新的学术机制结论。

沿用[迁移实验](teacher_renderer_transfer.md)已选定的 `runs/h3_moments/02_cross/last.pt` 与 `runs/teacher_render_adapt_v1/best.pt`，概率固定各 0.5。没有新训练、挑选系数或阈值。教师只输入本场渲染并量化的 RGB，固定 tile768/stride512/flip/context0.25/short-side768。在覆盖视野的 pinhole canvas 上融合，软概率 warp 到原图后才 argmax。

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 五类 mIoU % ↑ | 前景 mIoU % ↑ | 拉索 IoU % ↑ |
|---|---:|---:|---:|---:|---:|---:|
| support 联合模型 | 29.400950 | .870650 | .271237 | 94.552931 | 93.369764 | 93.565369 |
| 同场 H3 cross 语义头 | 29.400950 | .870650 | .271237 | 94.578062 | 93.395017 | 93.833060 |
| 上一行 + 固定 DINOv3 组合 | 29.400950 | .870650 | .271237 | **95.109016** | **94.036190** | **94.786655** |

使用 `official_original_grid_v1`，fingerprint 为 `21a2f19c5d4d703403a0107402d5dd5e006dcd98e5f8e023820c388c3e69a85d`。固定 50 RGB / 41 有标注视图、原始相机/畸变网格、交付 uint8 PNG、全图 PSNR/LPIPS、11×11 Gaussian SSIM 完整窗口、合并混淆矩阵。全部预测完成后才读取原始 RGB/标注评分。相机输入白名单排除了照片与标签。

组合五类 IoU：背景 99.400320%、桥面 97.387959%、拉索 94.786655%、塔柱 94.364776%、基础 89.605369%。基础仍是最低项。

5000 次相机配对 bootstrap、seed20260926，以下单位为百分点：

| 组合 − 对照 | 五类差 [95% CI] | 前景差 [95% CI] | 拉索差 [95% CI] |
|---|---:|---:|---:|
| 同场 H3 cross | +0.5310 [+0.2794,+0.9352] | +0.6412 [+0.3347,+1.1388] | +0.9536 [+0.5247,+1.5484] |
| support | +0.5561 [+0.2288,+0.9797] | +0.6664 [+0.2752,+1.1736] | +1.2213 [+0.1671,+2.6598] |

三种模型的 50 张 RGB PNG 逐字节相同，RGB 差及区间严格为零。

![同場固定教师组合的原图增量](../artifacts/optimization/selected_ensemble_official.png)

新产物放数据盘，旧产物保留：

- [锁定计划](/mnt/data/SHM2026/runs/official_selected_ensemble_v1/plan.json)，SHA `edcdda7e61026e49f452e6e662fe3cfa28722702f8d9b01a8b39701a2cce5c66`；[完成回执](/mnt/data/SHM2026/runs/official_selected_ensemble_v1/execution_receipt.json)。输入与 34 个源码文件前后 SHA 一致。
- [同场头指标](/mnt/data/SHM2026/runs/official_selected_ensemble_v1/cross_plain/official_metrics.json)、[组合指标](/mnt/data/SHM2026/runs/official_selected_ensemble_v1/cross_teacher/official_metrics.json)。各目录保存 PNG、逐图分数与独立回执。
- [组合减同场头](/mnt/data/SHM2026/runs/official_selected_ensemble_v1/paired_teacher_minus_cross.json)、[组合减 support](/mnt/data/SHM2026/runs/official_selected_ensemble_v1/paired_teacher_minus_support.json)，图的 [PDF](../artifacts/optimization/selected_ensemble_official.pdf)。

评测入口新增可选 `--teacher-checkpoint`，默认单模型数值路径保留；59 项相关 CPU 合同测试与 Ruff 通过，包括预测/GT 顺序、覆盖画布融合、软概率 warp、无效输出/索引/SHA 变化拒绝。两组自然退出 0，耗时 24.30/67.66 秒包含加载、哈希、写图和评分，不能作部署延迟。

```bash
uv run --no-sync python scripts/evaluate_official.py \
  runs/h3_moments/02_cross/last.pt \
  --teacher-checkpoint runs/teacher_render_adapt_v1/best.pt \
  --output /mnt/data/SHM2026/runs/new_official_ensemble_evaluation
```

教师约 8.47 亿参数，运行完整滑窗/翻转/全景推理，并非等容量或等时延对照。原 native 95.0231% 与本原图 95.1090% 不直接求增益。缺少同学 Dev30 清单与 RGB150 留出指标，不能宣布公平胜出。开发视图已反复用于研究和选择；配对区间不涵盖研究选择、重训种子或新桥梁的不确定性。H3 消融仍无可靠交叉项增量，不能用组合收益代替创新证据。
