# 固定 E＋stable-AA RGB 平均：不替换 E

唯一全局 0.5 候选已自然完成。它改善 E 的 SSIM 和 LPIPS，但 PSNR 仅增加 **0.043369 dB**，未达预定 .15 dB，且配对区间跨 0；主四门仅 **2/4** 通过，因此不替换当前 E，不扫描比例。输入和算法见[冻结协议](ibgs_e_fixed_blend_protocol.md)。

| 固定 RGB 输出 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| E | 30.035661247 | .874420736 | .264741845 |
| stable-AA full/fused | 29.825302158 | .875525202 | .252678474 |
| 固定 0.5 平均 F | 30.079030156 | .876132687 | .257035658 |

F−E 的 5000 次同视角配对 bootstrap（seed 20260926）：PSNR **+.043368908 dB**，95% CI **[−.092071634, +.179028570]**；SSIM **+.001711951**，CI **[+.000484683, +.003061943]**；LPIPS **−.007706187**，CI **[−.009710909, −.005820236]**。F 相对 stable 成分的 PSNR 增加 .253727998 dB，但 LPIPS 恶化 **+.004357184**，CI **[+.002137422, +.006340475]**，不能称支配两个成分。这些是重复使用的 50 个开发视角，不是盲测。

本轮只组合 100 张已有成员 PNG，保存 50 张新 PNG 后读取 50 张 GT RGB并评分；0 渲染、优化、teacher 或 mask 输出。worker **14.540236 秒**，root 外层 **15.43 秒**，仅为缓存组合和评分成本。完整交付还需要 E 的两个 RGB 场、新增 IBGS 场/融合网络及 TRAIN 源照片库，并保留独立 H3 语义路线；本轮未重新运行语义或完整多场系统，不产生新语义成绩，也不构成方法创新。

来源：[plan](../runs/ibgs_e_fixed_blend_v1/plan.json)、[执行回执](../runs/ibgs_e_fixed_blend_v1/execution_receipt.json)、[自然退出回执](../runs/ibgs_e_fixed_blend_v1/launch_receipt.json)、[四门](../runs/ibgs_e_fixed_blend_v1/rgb_gate.json)。plan SHA `051650888b7f092d33ddc345f1ba0e07d872cc9cfc816c8b29404824312a1509`；execution SHA `efc22ad103a53e874345d8b77418253ec5757e228dbf06cf3d279d049ceb0481`。旧 E 的 completed execution、预测屏障与独立审计仍原样保留；其历史目录没有独立 launch，不补造记录。

[独立 CPU 统计复核](../runs/ibgs_e_fixed_blend_v1/independent_rgb_statistics_review.json)通过，耗时 **.197183 秒**，核验 37 个源文件、120 个绑定输入、100 张成员与 50 张新 PNG 的字节哈希、来源/预测屏障，并独立重算每图分数均值、两组配对统计和四门。最大标量差 **3.553×10⁻¹⁵**（比较容差 10⁻¹²）。未解码图像或 GT、未重算像素级 PSNR/SSIM/LPIPS、未使用 GPU；平均运算依据已绑定源码/CPU测试，未在本次复核中逐像素重建。报告 SHA `6b9f3f0ac31a0a3b9135ecd7c6e4dc41b2a89d41f1b3f6bedf9c6a556a9c3968`。
