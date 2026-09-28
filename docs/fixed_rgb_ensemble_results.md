# 固定1M＋MCMC等权RGB组合结果

2026-09-27。固定两个已完成成员各0.5，在共同原始像素网格的50张交付RGB PNG上计算 `rint((A.float32+B.float32)*.5).uint8`（round-to-even）。全部50张预测保存并hash后才读取GT。本轮仅执行这一个预设组合，成员是依据已有开发结果选出的两场，本轮未搜索成员或权重。相对固定SSIM修复版500k参考，预定RGB门 **4/4通过**。

| 指标 | 500k参考 | 固定组合 | 差值（组合−参考） | 配对95%区间 |
|---|---:|---:|---:|---:|
| PSNR ↑ | 29.467484 | **30.035661** | **+0.568177 dB** | [+0.355616, +0.823303] |
| SSIM ↑ | 0.870854827 | **0.874420736** | **+0.003565909** | [+0.002326347, +0.005053416] |
| LPIPS ↓ | 0.270773566 | **0.264741845** | **−0.006031721** | [−0.008991534, −0.003477983] |

四门为PSNR增幅≥0.15 dB、配对区间下界>0、SSIM点估计不降、LPIPS点估计不增。区间来自固定50相机的5000次配对bootstrap，seed20260926；[配对结果](/mnt/data/SHM2026/runs/rgb_fixed_ensemble_v1/paired_minus_reference_500k.json)和[门结果](/mnt/data/SHM2026/runs/rgb_fixed_ensemble_v1/rgb_gate.json)均保留。

三个解释限定：

1. **工程RGB收益，不是单场创新或精确peer胜出。** 保留两个场共1,496,009个Gaussian，新视角需要两次场渲染。本次15.815秒、峰值CUDA allocated 348,000,256字节仅计缓存PNG组合与评分，0 scene render／0 optimizer，不能当双场端到端耗时或显存。平均的是量化后的编码RGB，不是线性光或未量化辐射。这里只评共同原始网格，不混报native指标；50个开发视角已被反复使用，区间未作选择校正。
2. **没有在全部指标上优于每个成员。** 相对1M成员，PSNR +0.539513 dB（区间[+0.325107,+0.769362]），但LPIPS点估计 **+0.000921933**，区间[−0.003480014,+0.004860027]跨0；SSIM差值区间也跨0。因此不能称组合全面优于所有成员，详[成员配对](/mnt/data/SHM2026/runs/rgb_fixed_ensemble_v1/paired_minus_capacity_1m.json)。
3. **未评价或采用RGB—语义联合系统。** 本轮只读取RGB GT，无标注读取、mask输出或语义成绩，没有拼接旧mask，也未自动启动之前停止的1M 8k语义链。若未来评估联合系统，须从实际组合RGB重新产生语义并独立验证；当前联合selected不变。

[锁定plan](/mnt/data/SHM2026/runs/rgb_fixed_ensemble_v1/plan.json) SHA `77091ff41a50d1931f092e8ee2505662bc98120c74658a49f4f8fbce1879697f`；[自然完成回执](/mnt/data/SHM2026/runs/rgb_fixed_ensemble_v1/execution_receipt.json) SHA `7acb6259f976873214cb4b09dd06ec6ea5f8804bb12627cd4288ca4136f74ca5`，root观察session39266 exit0。源与绑定输入未变。继承共同评价identity `21a2f19c…e69a85d`，本轮仅重新验证RGB字节，不新增语义协议证据。[独立CPU复核](/mnt/data/SHM2026/runs/rgb_fixed_ensemble_v1/independent_cpu_review.json)已通过，SHA `dc3b35b88dbea8311865ff5b869564c2f93f649393edef4b5071d20758a26d56`：50张组合逐像素符合整数round-to-even，100张成员／50张GT来源hash核验，PSNR重算最大差0，三组配对区间及4/4门一致；SSIM／LPIPS沿用绑定的冻结评分值，只独立重算汇总与bootstrap，未重新像素评分。

**后续状态：纯H+显式迁移已实际重评。** 新RGB对应五类94.3866%、拉索93.1237%，未通过联合采用门；原联合selected保留。上述“语义未评”描述的是本RGB实验的范围，后续独立结果见[迁移评价](rgb_teacher_transfer_results.md)，不能将新RGB与旧95.1090%语义成绩拼接。
