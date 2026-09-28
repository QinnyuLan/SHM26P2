# IBGS AA 与近裁剪兼容性结果

2026-09-27。固定原 996009 点、SH3 检查点，同一组 16 个已参与训练的 TRAIN 相机；没有优化、源图融合或模型选择。新独立后端将近裁剪改为 `< .01`，Torch hook 在原 opacity 上乘既有二维 AA 补偿，其余 IBGS 光栅规则不变。原后端和历史结果保留。

| 输出 | 等图平均 MSE | 等图平均 PSNR / dB |
|---|---:|---:|
| 原 gsplat AA 缓存 | 0.0008853589035895 | 32.3009613255 |
| IBGS AA + near=.01，原始 RGB | 0.0008848922075059 | 32.2999794764 |
| IBGS AA + near=.01，clipped RGB | 0.0008848919191869 | 32.2999813822 |

clipped 输出相对原 AA 的平均逐图 PSNR 为 **−0.0009799433 dB**。有效像素 RGB 平均绝对差为 `9.8984844e−5`，全图原始 RGB 最大绝对差为 `0.04290527`。均值接近不等于逐像素等价；平均 MSE 与平均逐图 PSNR也不是同一统计量。相对[此前起点分解](ibgs_renderer_decomposition_results.md)的约 3.64 dB 迁移损失，这次工程兼容已恢复绝大部分差距，但不是训练收益，也不是新方法。

![固定002与041对照](/mnt/data/SHM2026/runs/ibgs_aa_compatibility_v1/compatibility_002_041.png)

图依次为缓存目标、原 AA、未兼容 IBGS、AA 与近裁剪兼容 IBGS。图由既有数组绘制，没有追加推理。

唯一 GPU 执行自然退出 0：16 次前向、16 次 AA hook，0 backward、0 optimizer、无原图或语义/VAL 解码；所有预测保存后才加载缓存目标，field 逐项保持不变且 hook 已恢复。worker 用时 `6.56411561 s`，root 外层记录 `6.94 s`。该前向检查不认证反向或融合训练：Torch 补偿的正值支路使用真实自动微分，gsplat 手写补偿反向另有稳定化项；IBGS 的 alpha cap、足迹边界和投影运算仍不同。

独立 CPU 复算全部 16 个新数组、16 个缓存目标与 16 个原 AA 数组，未调用 producer 评分函数；FP64 平方及归约、等图汇总与原报告最大差 `7.1054273576e−15`，原定绝对容差 `1e−12`，用时 `2.72002350 s`。同时核验绑定源码、输入、隔离后端和实际导入记录。运行次数、field 未变及 hook 恢复属于已保存的运行证据，并非 CPU 重新执行光栅验证。没有 GPU、检查点反序列化或原图解码。

来源与回执：

- [冻结计划](/mnt/data/SHM2026/runs/ibgs_aa_compatibility_v1/plan.json)：`15b149b1a4e477d69a69a8fc530c611f87df794dccdfb3f9d5102b9ac340a263`。
- [执行回执](/mnt/data/SHM2026/runs/ibgs_aa_compatibility_v1/execution_receipt.json)：`5e00f82d0439cb6352df8addf4f055337a1fbe24c7c207e7fc9d514d54dd28a1`；[自然结束回执](/mnt/data/SHM2026/runs/ibgs_aa_compatibility_v1/launch_receipt.json)。
- [独立 CPU 复算](/mnt/data/SHM2026/runs/ibgs_aa_compatibility_v1/independent_cpu_review.json)：`d4e0a387a2f6e6ab3b8945bc832b25da30470d2a85655e622c479fb2b7e81eaf`。
- 隔离后端：`c504257ae703800e90fc79c8f0541f0cd7757cd3e293e078760a5535c2bae35d`。原后端未覆盖。

当前结论只支持继续固定范围的合成梯度与两 TRAIN 视图真实反向预检；没有新增训练结果或采用结论，当前系统 E 不变。

## 后续两 TRAIN 视图反向接线预检

同日，固定 002.png、041.png 的真实预检已自然退出 0，`status=completed`、`numerical_status=passed`。沿[固定协议](ibgs_aa_port_preflight_protocol.md)，8 次 source depth、2 次目标 render、2 次 backward、10 次 AA hook、0 次 optimizer 全部闭合。350 相机的 TRAIN-only bank 实际解码 10 张 TRAIN RGB 和 1 份共享 valid；无 VAL 或语义标签解码。

| 目标 | 完整损失 | 融合支持比例 | 8 类 field 梯度／网络梯度 |
|---|---:|---:|---|
| 002.png | 0.1138709337 | 71.76525% | 均存在且有限 |
| 041.png | 0.1046755910 | 46.21074% | 均存在且有限 |

保存的 8 类 field 梯度范数在两图均非零；这只是观察值，没有新增非零阈值门。原 field 和新鲜融合网络的参数均未更新，AA hook 和 TF32/cuDNN 数值标志恢复。worker 用时 `4.09974196 s`，外层 `4.44 s`，峰值已分配显存 `7,059,280,384 bytes`。这里采用已完成 warm-training 的无符号支持判断 `abs().sum(0)>0`，并未改写早期预检记录。

独立薄 CPU 核验用时 `0.66304219 s`：35 个源码、363 个输入、1550 个后端文件 SHA 全同，19 个实际导入均来自冻结包；隔离二进制、相机邻居名单、计数、有限梯度标量和参数/状态恢复记录一致。仅从保存分量重新组合损失，最大差 `6.4075e−9`，符合该 FP32 运算顺序与 FP64 标量组合的差异；这不是新的导数校准。

- [预检计划](/mnt/data/SHM2026/runs/ibgs_aa_port_preflight_v1/plan.json)：`e17b5f1b8827f03da05ac6707573ccba40feb1a27ceccc4fc409528d2d58427e`。
- [预检执行](/mnt/data/SHM2026/runs/ibgs_aa_port_preflight_v1/execution_receipt.json)：`c44fc23436634568fb5c56a1ba9c489ac5a7f5b7b28b5571e04fc309dcfe3e25`；[自然结束回执](/mnt/data/SHM2026/runs/ibgs_aa_port_preflight_v1/launch_receipt.json)：`c7897a4ac8920b39afda87e8f15e00cbbb711623e16d04695395ce1f45607735`。
- [独立 CPU 元数据核验](/mnt/data/SHM2026/runs/ibgs_aa_port_preflight_v1/independent_cpu_review.json)：`d22e68da4b753264d3ac7281fa4abc704300e0d4bf5a726040c16c58c2448699`。

CPU 核验没有重新执行 VJP、光栅或全场有限差分，也没有反序列化检查点、解码图像或读取正在训练的末态。两张已参与训练的视图和新鲜网络只能支持有限反向接线成立，不能认证完整梯度、泛化收益或采用资格。后续训练结果另行记录，本页不预告其效果。
