# E RGB 与 stable-AA IBGS 的固定半权平均

本协议只定义一个工程候选，尚未执行。状态以独立输出目录中的 plan、execution 与 root launch 回执为准；不自动替换当前 E，不宣称学术创新。

输入固定为 `rgb_fixed_ensemble_v1/rgb_metrics.json` 对应的 50 张 E 原图 PNG，以及 `ibgs_aa_stable_evaluation_v2/predictions/full/fused` 的 50 张 PNG。后者必须来自固定末态 full、稳定 FP64 因子 AA/.01 near 的自然完成 render/score 两阶段。E 的历史目录只有 completed execution、预测屏障和 passed 独立审计，没有独立 launch 回执；新计划如实绑定现存证据，不补写历史。两路每张 camera（含 K、pose、畸变、原图尺寸）、GT 路径/SHA 元数据、评分协议和公共 fingerprint 必须一致；每图旧 metric 的 RGB SHA 必须对应实际输入 PNG。

唯一预测为：

```python
F = np.rint((E.astype(np.float32) + stable.astype(np.float32)) * .5).astype(np.uint8)
```

这是已交付、已量化的编码 RGB 平均，采用 round-to-even；不再 warp、不做线性光变换。不扫描比例，不比较或选择旧 non-AA 版本，不用目标图设权重。全部 50 张新 PNG 必须写完并保存带 SHA 的预测屏障，之后才读取 50 张原始 GT RGB。CPU prepare 只读取来源记录和输入 PNG 字节用于哈希，不读取 GT 像素或 GT 字节。

评分与统计直接复用 E 冻结的 `evaluate_fixed_rgb_ensemble.py` 中 `score_rgb_only`、`paired_rgb`、`gate_clauses`，及同 SHA 的官方评分包。原图全像素 PSNR、既定 SSIM、AlexNet LPIPS 均重新评价新 PNG；50 次 LPIPS，数值设置为 cuDNN TF32=True、matmul TF32=False、benchmark=False、precision=highest，结束恢复设置。旧 scorer 的 mask 形参仅使用内存零占位，`target_mask=None`，返回字段严格限 RGB；不保存、推理或计分任何语义 mask。

固定报告 F−E 与 F−stable full/fused 两组比较。每组都是同 50 个已反复使用的开发视角，按 name 排序，5000 次 paired-view bootstrap，seed=20260926；不能解释为盲测或选择校正后的证据。主四门只针对 E：PSNR 增益≥.15 dB、其 paired 95% 下界>0、SSIM 点估计不下降、LPIPS 点估计不上升。无其他新门；无自动采用。两组成分原指标一并列出，不能仅凭某一指标称支配全部成分。

本轮 0 scene render、0 optimizer、0 teacher、0 annotation decode、0 mask 输出。E 的语义路线拟保持，但本轮未实际重新运行或测量语义，也不是完整多场系统的新端到端执行。部署此 RGB 组合会增加第三个 RGB 场、IBGS 融合网络和 TRAIN 源照片/深度库；若保留 E 的 H3 语义场，完整系统还包含该独立语义场。组合与评分耗时不代表系统渲染 FPS，亦不省略这些资产成本。

新入口 `scripts/evaluate_ibgs_e_fixed_blend.py`。prepare 只允许 fresh `/mnt/data` 路径，复制 E 冻结评分包与 helper 并绑定本入口、测试、协议、uv.lock、已有 LPIPS 权重、全部来源记录及 100 输入 PNG。执行拒覆盖并绑定显式 plan SHA；内限 300 秒、root 外限 360 秒，无重试。root 独占 GPU，仅 LPIPS 评分使用 GPU。prepare 与实际评分均须另获 root 启动，本次代码交付不执行它们。
