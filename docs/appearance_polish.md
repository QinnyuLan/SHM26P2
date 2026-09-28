# 固定 15k 几何后的 RGB appearance polish

这是一个经典的颜色拟合工程控制：固定较早的几何、透明度和语义参数，只继续优化 RGB 的 SH 系数与背景颜色。选择 repaired-support 的 15k checkpoint，是观察到后期浮层后的**适应性开发决策**；不是预注册盲测，也不把该参数范围称为学术创新。它可以改变已有浮层的颜色和可见外观，不能移动或删除浮层，也没有预设一定改善验证指标。

输入为 `runs/support_split_rgb_full_resumed/step_015000.pt`，332,255 个高斯，SHA256 `4c13642cf041b9236cfbe333d7ee38b57f61a028dd7083fb61f6c6c72d7b5f6d`。原生 50-view RGB 基线 PSNR 28.824443 / SSIM .880276，来自现有 `metrics_015000.json`；该基线没有计算 LPIPS。验证 fingerprint 为 `750b9f53046bc104093715c6c26c090837746c467445364584570feb21f10b7b`。本实验不覆盖原 15k/30k 模型和运行目录。

`parameter_scope: appearance_only` 的模型参数白名单只有：

- `splats.sh0`，Adam lr .0025；
- `splats.sh_rest`，Adam lr .000125；
- `background_logits`，Adam lr .001。

位置、旋转、尺度、opacity、语义特征、decoder、refiner、人工 prior_counts 和相机全部保留。使用 warmstart 新优化器，不沿用源模型的旧 Adam/density 状态。不重置 refiner，拒绝在此 scope 改变其架构。`freeze_geometry: true`、`freeze_rgb: false` 正确描述实际权限；原先语义阶段 `freeze_geometry` 自动冻结 RGB 的行为仅对此显式 scope 例外。SH 从第 1 步使用完整 SH3，不重启 progressive degree。scope 拒绝低阶 SH、densification、相机更新、teacher/fusion、语义训练、稀疏深度/opacity entropy 正则。

配置 `configs/support15k_appearance_polish.yaml` 固定 3000 步、native 1320×989、不递进分辨率、全部 350 TRAIN、独立 RNG shuffle seed42；`region_rgb_weight=.15` 与 full RGB 一致，semantics/refine_start=1e6。区域 RGB 权重会读取有标签 TRAIN 的前景 mask，作用是 RGB 误差加权，不更新语义参数。无标签 TRAIN 步仍有完整 RGB 监督。

## 预检

```bash
uv run python scripts/preflight_appearance_scope.py
```

预检调用真实 `train()` 执行 2 步，先保存源代码快照，再安装仅记录梯度的 hook。训练后对照原 checkpoint 的全部 `model` 张量与相机，检查允许改变的键精确等于上述三项，冻结参数没有 Adam state，架构/scene metadata 不变，保存的 freeze 标志与实际权限一致。

已通过：`runs/appearance_scope_preflight/preflight_report.json`。SH0 两步梯度范数 .003407/.001739，SH_rest .013194/.006737，background .001601/.000091；均有限且非零。全部其他模型张量及相机逐位相等，SH3 从 step1 生效。两步训练约 .79 秒，峰值 PyTorch allocated .688 GiB；这里只验证权限，不作为性能 benchmark。scope/crop/entropy 回归共 42 项测试通过。

## 正式运行与待记录结果

等其他实验释放 GPU 槽后运行；复用已通过真实预检的同一源快照：

```bash
uv run python scripts/run_experiment.py configs/support15k_appearance_polish.yaml \
  --source-snapshot runs/appearance_scope_preflight/source_snapshot
```

`run_experiment` 记录 source/config/input SHA，完成 3000 步后原生 50 RGB/41 GT + LPIPS 评估。该模型的语义头未在本阶段训练，不能把它当作联合语义完成模型。结束后还需再次核对全部非 appearance 张量/相机的逐位不变性，并报告全部验证 RGB 指标和 235 等局部可视化；不得只凭平均 PSNR 声称浮层解决。

15k 实验已经完成并通过最终不变性检查，见 `runs/support15k_appearance_polish/appearance_invariance_audit.json`。只有3个允许的颜色张量变化，所有其余参数/人工prior/相机/架构元数据逐位保留，raw3D confusion matrix 也与源模型相同。正式 source hash 与2步预检完全一致。

| 指标 | 源15k | 15k+颜色3k |
|---|---:|---:|
| 50-view PSNR | 28.82444 | 29.33766 |
| SSIM | .880276 | .884864 |
| LPIPS | 未计算 | .261710 |
| 235 PSNR | 20.99018 | 20.93599 |
| 235 SSIM | .805965 | .806477 |

平均RGB改善，235并未随之改善PSNR；不足以说浮层已经消除，也仍未达到已有强RGB模型的整体指标。不能填造源15k LPIPS或报告不存在的LPIPS差值。

观察此结果后，下一次适应性工程迭代选择 `runs/support_split_rgb_full_resumed2/step_020000.pt`，425,447点，源PSNR29.68201/SSIM.894647。`configs/support20k_appearance_polish.yaml` 相对15k配置只改变 warmstart/output，训练预算仍3000步，source仍复用同一预检snapshot。选择20k是为了检验较好初始颜色/容量与较少后期浮层的折中，不是独立盲测或证明特定训练阶段最优；不直接启动其语义训练。

20k 实验也已完成：PSNR **29.94362**（+.26161），SSIM **.896792**（+.002144），LPIPS **.238845**（源未测）；235 PSNR20.75788→20.76670、SSIM.806903→.807859。`runs/support20k_appearance_polish/appearance_invariance_audit.json` 通过全部非颜色张量/相机逐位不变、raw3D confusion相同、source相同及仅warmstart/output配置不同检查。结果仍是RGB工程折中，不是新的联合语义模型。

**可复现性限制**：原20k输入随后因团队保留依赖同步冲突被清理，未找到真实备份。严格不变性审计于2026-09-26 17:49:25 UTC在原文件存在时完成；清理审计时间17:51:40 UTC，记录相同源SHA `f730749483f199d8ea004ecbec71e024e470fc19c9c99ecd9f56cc672212c21c`。派生 `support20k_appearance_polish/last.pt`（SHA `88f1c0577bc2bd02a23a5255151b4f0d45545c2f813fccef4b673a2465757c73`）、评测和不变性审计均保留有效，但目前不能逐字节重放这一次原20k warmstart；派生last不能冒充已删除的原输入。15k原输入仍保留。
