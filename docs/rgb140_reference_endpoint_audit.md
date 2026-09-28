# RGB reference 终点与成本审计

审计脚本为 `scripts/audit_rgb140_reference_endpoint.py`，CPU-only。root确认训练/native自然完成后已执行一次，全部合同通过；未修改脚本或训练产物，未使用GPU。入口必须先确认实验回执 completed（训练和原生评价都完成），再确认 root launch 回执 completed/已有退出码为0；随后才允许读取 checkpoint。任何未完成状态均拒绝且不输出成功报告。

固定比较 `/mnt/data/SHM2026/runs/rgb140_inspired_mixed_500k` 与 `ssim_fixed_corner_v2_rgb_full`。核验两者 plan/config/source/input SHA、30k 最终步、model/optimizer/density 张量有限性、350 TRAIN 名单、原始相机逐位相等和 corner-v2 manifest；新参考另核禁用 mask loading、类别初始化及类别权重的固定源码分支、全零语义先验，以及直接累计成本计数和最终 checkpoint/log 一致。**no-mask 证据来自固定源程序与配置，不冒充所有30000步逐次像素打开的观测 trace。**

原生评价必须完整50视角、1320×989、同 native fingerprint；从冻结 evaluator 和实际 manifest 重建 fingerprint。只复用已有 paired helper 的 `rgb_only=True`，输出 PSNR/SSIM/LPIPS及视图配对区间。未训练的语义字段不参与比较。root 另行进行共同原图 RGB 评价，不将两种网格的数字混算，也不把此多项配方差异说成独立增密消融。

旧30k没有累计成本仪表，采用 **来源+日志重建**：旧 `train.py` SHA `99c27c929253b17e71753a6a2afb080d1cdc0e464ce8f2550d78d8177c3e04d0` 唯一改变 Gaussian 行数的 `apply_densification` 位于每100步、主渲染/optimizer之后；每个该边界的 post-update N 都被日志捕获（共step1和300个整100步）。因此第t步渲染使用上一边界N，绝不能把t步增密后的N计入该步。recycle也在同一结构选择内，不增加漏记的行数变化事件。resize源码 SHA `08fafa9f20ee07cddd867db4d4c02a33e706435ec40a13a15e22db4a7600a65d` 使用 Python round；所有350 TRAIN原生尺寸相同，故不需要假设其逐步相机顺序。

重建的渐进尺寸为：4499次660×494，13500次990×742，12001次1320×989；不是近似4500/13500/12000。旧主要训练 Gaussian-step 合计 **9,581,235,400**，平均每次render N **319,374.51**，主渲染像素 **27,050,749,440**。其最终和记录峰值N均为 **498,136**。旧hybrid另在500…23980的整20步执行raw residual与visibility两次320×240无梯度诊断，共2350次、额外 Gaussian-render 合计 **655,269,668**、额外像素 **180,480,000**。新mixed无这些诊断render，日志有直接累计主训练Gaussian-step、renderpixels、field peakN；审计会再用其post-N日志独立交叉检查直接计数。

最终同时报告真实最终N、记录的field峰值N、累计主训练/额外诊断成本、末条日志训练时间、含原生评价的实验时间与显存峰值。Gaussian求和与像素求和是成本代理，**不是精确FLOPs**；主训练含反传，额外诊断不含反传。峰值N是渲染前/增密剪枝后场大小，不代表瞬时构造中间张量的峰值；显存采用实际已记录的allocated峰值。旧累计项明确标注重建，不能写成历史直接观测。

5项 CPU 合同通过：活动checkpoint早拒、等待自然退出、增密前后计数边界、渐进尺寸精确取整、缺日志/不同尺寸拒绝。运行必须等 root 确认终点完成：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/mnt/data/SHM2026/runs/rgb140_inspired_mixed_500k/source_snapshot uv run --no-sync python scripts/audit_rgb140_reference_endpoint.py
```

默认新输出为该 run 的 `endpoint_audit.json`，不覆盖已有报告、不启动 GPU、不修改任何输入模型或训练状态。

## 已完成结果

[审计报告](/mnt/data/SHM2026/runs/rgb140_inspired_mixed_500k/endpoint_audit.json) SHA `10c4c8b9afc689c7438730a1f4e9d29c6b2806bbed5bcb2516e26c78c26e24ef`。来源、配置、30k完整步、有限性、350原始TRAIN相机及native fingerprint全部通过。

| Native RGB | 修复后旧配方 | RGB140-inspired | 新−旧，95%视图配对区间 |
|---|---:|---:|---|
| PSNR dB | 30.172306 | 29.898078 | −.274228 [−.615262,+.093614] |
| SSIM | .9031500 | .9029932 | −.0001567 [−.0016515,+.0016299] |
| LPIPS | .2234777 | .2236167 | +.0001390 [−.0039635,+.0037008] |

三项区间均跨零，没有新参考性能胜出的证据；这里只解释RGB，语义未训练。

新/旧最终N为499,821/498,136；累计主Gaussian-step为13,052,285,500/9,581,235,400（1.36228倍），主像素数相同27,050,749,440。计入旧额外诊断，Gaussian-render求和比为1.27507倍，仍不是FLOPs比较。日志训练时长954.63/1115.77秒（新为旧.85558倍），含原生终评实验时长972.87/1136.57秒；allocated峰值1.316/1.463GiB。新计数为直接记录并经日志交叉核验，旧为上述源+日志重建。实际N相近并不意味着训练容量轨迹相同；时间差不能仅归因于某一个配方分量。
