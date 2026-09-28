# Factored FP64 AA 的固定验证结果

2026-09-27。四个新验证均自然退出 0，`status=completed`、`numerical_status=passed`。新 provider 是 `bridge_rgs.ibgs_antialias_stable`，profile 为 `ibgs_centered_corner_v2_aa_factored64_near001_v2`；旧 FP32 AA 模块、完成的预检和第 62 步失败记录均保留。**本页没有新的完整训练或终评结果，不替换 E。**

## 修正范围

旧训练在第一臂完成 61 步后，于 004.png 的一个高斯触发 `det_blur <= 0`。保存的实际激活输入对应 FP32 determinant 为 `−536870912`、blurred determinant 为 `−469762048`；同一输入经 FP64 因子计算分别为约 `7.174372861e8`、`7.954490600e8`。这是已捕获实例中的行列式消减，不是通过剔除该点或放宽 guard 解决。

新模块用投影因子三个 2×2 minor 的平方和求 determinant，在 FP64 中计算补偿，最后把有效 opacity 转回原 dtype；不重新激活参数、不重新归一化 quaternion、不加 floor。实际 renderer 的 FP32 covariance/conic、alpha cap 及近似 VJP 均未改变。名义 FP64 `.3` 也不声称与 CUDA `.3f` 逐位相同。详见[固定验证协议](ibgs_aa_stable_revision_protocol.md)和[原失败诊断](ibgs_aa_determinant_diagnostic.md)。

## 四项固定验证

| 检查 | 实际预算 | worker / 外层秒 | 结论 |
|---|---|---:|---|
| 原失败态 004，step=61 | 1 raw forward / 1 backward，0 GT | 4.036801 / 4.33 | 六类 raw 参数梯度记录均存在、有限；field 未更新 |
| 原始 field 的固定16 TRAIN兼容 | 16 forward，缓存目标评分 | 7.875680 / 8.17 | 原有限性、计数、保存屏障与状态门通过 |
| 固定合成局部 FD | 33 forward / 3 backward | 3.536390 / 3.82 | 原四个主 FD 门及 zero control 通过；cap 反例仍不通过 |
| 原始 field 的002/041接线 | 8 source depth + 2 target / 2 backward | 4.812547 / 5.15 | 两图八类 field 参数和网络梯度记录均有限，参数未更新 |

总计 60 次光栅、6 次 backward、0 次 optimizer。004 回归不建 source bank、不读取目标图像；normal/offset 不参与 raw RGB 路径，因此没有对它们强加梯度要求。两 TRAIN 接线仍使用原先的 10 张 TRAIN RGB／1 份 valid，未使用 VAL 或语义标签；峰值已分配显存为 `7,217,923,584 bytes`。全部 provider、hook 与数值标志恢复。

16 图的独立缓存复算为：

| 输出 | 等图平均 MSE | 等图平均 PSNR / dB |
|---|---:|---:|
| 原 gsplat AA | 0.0008853589035895 | 32.3009613255 |
| stable IBGS，原始 RGB | 0.0008848921504177 | 32.2999801985 |
| stable IBGS，clipped RGB | 0.0008848918620983 | 32.2999821043 |

clipped 平均逐图 PSNR 相对原 AA 为 **−0.0009792212 dB**；有效像素平均绝对 RGB 差为 `9.8978048e−5`，原始 RGB 最大像素差为 `0.04290533`。这保持了修订前的前向兼容水平，但不是逐像素相同，也不是训练收益；16 图都已参与原 field 训练。

合成主步长 `h=.002` 保持原门 `|FD−g·d_actual| ≤ 1e−5 + .01|g·d_actual|` 和原信号门，不改阈值：

| 方向 | 保存梯度 × 实现位移 | 保存 RGB 重算 FD | 相对差 |
|---|---:|---:|---:|
| means | −0.0826673090 | −0.0826444472 | 0.027655% |
| log scales | 0.0075693461 | 0.0075677409 | 0.021206% |
| raw quaternion | 0.0257819562 | 0.0257849692 | 0.011686% |
| opacity logits | 0.0978952171 | 0.0978970895 | 0.001913% |

两组既定敏感性步长也按保存数组复算；zero-rho 输出与背景逐位一致。**cap-active 反例保留**：主步长的保存 analytic dot 为 `0.0017585843`，图像 FD 为 `0`。这说明本次数值稳定修订没有使后端完整反向变成真实导数，局部未饱和通过不能外推为全 renderer 认证。

## 薄 CPU 核验与来源

CPU 核验全部四套各 40 个冻结源、实际导入和共 1977 个去重绑定输入／后端文件，SHA 一致。独立读取 16 个新 RGB、16 个缓存目标／原 AA，以及合成的 33 个保存前向数组，重算指标和 FD；185 项标量对照最大差 `1.0842021725e−19`，绝对对照容差 `1e−12`，耗时 `3.41540601 s`。未调用 producer 评分或 FD 函数。

实际 dot 仅由保存的梯度与实际方向记录重新求和，**不是独立 VJP**。004 与002/041只核保存的有限梯度、计数和恢复证据，没有重新执行真实场光栅或有限差分。没有 GPU、检查点反序列化、原图/valid 解码或活动训练 checkpoint 读取；绑定字节只用于 SHA。

- [合并 CPU review](/mnt/data/SHM2026/runs/ibgs_aa_stable_compatibility_v2/revision_independent_cpu_review.json)：`1e80dcab0ea71dc3444df47bc12d23d5185924d8d339eada86dd9aa7502c0885`。
- [004 回归](/mnt/data/SHM2026/runs/ibgs_aa_stable_failure_v2/execution_receipt.json)：`70d9550e84dd45dab1e0be022a70d632c0dc45f9363d4bb549592dd9e73a99d3`。
- [16 图兼容](/mnt/data/SHM2026/runs/ibgs_aa_stable_compatibility_v2/execution_receipt.json)：`01e43a3bc95c54e9ea5bdc0e8d84306f1268170ef1a8040b77628581671e94b4`。
- [合成 FD](/mnt/data/SHM2026/runs/ibgs_aa_stable_gradient_v2/execution_receipt.json)：`f4a4a4076188338b757fcd83649b5b05ae200574a637137de3674655eea0cddd`。
- [两 TRAIN 接线](/mnt/data/SHM2026/runs/ibgs_aa_stable_preflight_v2/execution_receipt.json)：`bd1f5ab8e2e5fcaa7cbda6876feb9cad89a04f0211d661a36367c532ab921ca6`。

新模块 SHA 为 `ea151cc69520261d50337101b219e9561a15b4f94c1cce9197a9415b096bdcd2`；独立 near=.01 二进制仍为 `c504257ae703800e90fc79c8f0541f0cd7757cd3e293e078760a5535c2bae35d`。各目录均有独立 plan 与自然结束 launch receipt。

Root 已另行启动两臂各 6000 步的重训，均从原始 `35b489…` checkpoint 重新开始，未从失败态续训。该训练正在进行，结果另记；本页的局部验证不预告其完成、效果或采用结论。
