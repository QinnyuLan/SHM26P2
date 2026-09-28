# F RGB 加权成员候选结果

## 候选定义

保持 F 的两个已验证 RGB 成员和全部渲染 PNG 不变，仅将等权融合改为

\[
I_{\alpha}=\alpha I_{\text{top4\_normalized}}+(1-\alpha)I_{\text{MCMC}},\qquad \alpha=0.6.
\]

候选脚本只读取 50 张官方组件 PNG，先写出并校验全部 50 张预测，再读取 50 张目标 RGB；没有重新训练、teacher 调用、几何更新或语义读取。分数与 F 使用同一官方 PSNR/SSIM/LPIPS 实现，并做 5000 次 paired bootstrap。

## 结果

| 版本 | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|
| F（α=0.5） | 30.236593434 | .875024123 | .259300123 |
| 候选（α=0.6） | 30.252776551 | .875538013 | .257947664 |
| 候选 − F | +.016183118 | +.000513890 | −.001352459 |

配对 95% 区间分别为 PSNR **[−.021356,+.055054] dB**、SSIM **[+.000117,+.000962]**、LPIPS **[−.002254,−.000560]**。因此候选的 SSIM 与 LPIPS 区间改善，但预先登记的工程门要求 PSNR 至少 +0.15 dB 且区间下界为正，候选只通过 SSIM/LPIPS 两项，`candidate_gate.json` 为 `passed=false`。

还有一个选择边界：α=.6 是在同一开发集 50 视角上探索得到的，因此本次结果是可复核的候选证据，不是盲测或跨桥泛化证明。独立 CPU 复核读取 50 张 PNG、验证 hash 与屏障回执，并以 float32 语义精确重算 PSNR/SSIM；`no_gpu=true`、`psnr_ssim_exact=true`。LPIPS 保留生产者回执，未把 CPU 近似重算冒充官方分数。

运行目录为 `/mnt/data/SHM2026/runs/f_rgb_weight_candidate_v1`。F 的正式部署仍使用 α=.5；若要采用 α=.6，应先锁定权重并在新的盲/留出相机协议上确认四项门。
