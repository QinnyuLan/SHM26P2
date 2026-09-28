# 参数级 opacity 熵：完整负结果

这是标准 Bernoulli 熵正则的工程对照，不是新研究机制。两组从同一 `strong_semantic_coupled/last.pt` 开始，固定498,136点拓扑及除opacity外的所有参数；相同350 TRAIN视图独立shuffle、1,000步、原生分辨率、RGB损失、完整SH3。没有语义/教师/深度损失或相机更新。唯一处理变量为平均opacity熵权重0与0.05，不做验证集权重搜索。

`H(p) = -p log(p) - (1-p) log(1-p)` 的稳定实现处理极大logit。最小化使参数趋向0/1，但单个图元的opacity不是射线占用概率；二值化本身并不证明几何正确。

| 全50 RGB / 41语义原生视图 | RGB控制 | 熵权重0.05 | 差异 / 配对95%区间 |
|---|---:|---:|---|
| PSNR | 30.34757 | 30.20432 | −0.14325 dB / [−0.23689,−0.03697] |
| SSIM | .903026 | .900622 | −.002404 / [−.003291,−.001733] |
| LPIPS | .224156 | .227085 | +.002928 / [+.001973,+.004084] |
| 最终五类mIoU | 93.87982% | 93.89998% | +0.02016个百分点 / [−0.11586,+0.21851] |
| 最终拉索IoU | 93.49739% | 93.24294% | −0.25445个百分点 |
| raw3D五类mIoU | 78.69901% | 78.37061% | −0.32840个百分点 |
| raw3D拉索IoU | 30.30408% | 29.02316% | −1.28091个百分点 |

结论：RGB三项均退化，最终语义差异区间跨0；未采用该正则继续训练。已固定的235诊断视图PSNR20.1864→20.2469，不改变全量负结论，也不构成浮层已解决的证据。参数平均熵确实从控制组0.21956降到0.02310，opacity<0.01的比例从10.81%增至41.19%、>0.99从28.69%增至51.66%；这验证了处理生效，但没有验证目标获益。

`scripts/audit_opacity_pair.py` 逐项核验仅 `splats.opacity_logits` 改变，相机与其他模型张量逐位不变；两组源码/输入SHA相同，唯一配置差异为output和熵权重，最终torch/CUDA/numpy及视图sampler RNG相同。回执中的最终检查点和评价SHA重新实算通过。完整数据与5000次配对视图bootstrap位于 `runs/opacity_entropy_pair_audit.json`，仅反映该开发视图集合，不是跨场景显著性。

```bash
uv run python scripts/audit_opacity_pair.py \
  runs/strong_semantic_coupled/last.pt \
  runs/opacity_entropy_control_v2 runs/opacity_entropy_005_v2 \
  --treatment-key opacity_entropy_weight --output runs/opacity_entropy_pair_audit.json
```

首次控制配置将save_every设为0，旧源码在第1步发生除零异常。失败目录 `runs/opacity_entropy_control` 保留，不参与比较；两组有效v2均使用其相同源码快照，save_every=1000，全部正常完成。主开发源码随后修复了save_every=0仅保存最终模型的契约，不追改已完成实验快照。
