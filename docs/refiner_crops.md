# 学生细化头的原生裁剪对照

本项吸收既有 768 裁剪与全图训练经验，作为工程数据增强对照。`configs/generated_refiner_crop/crop_plan.json` 在正式训练前锁定：共同 warmstart 为 `runs/refiner_multiscale_v1/last.pt`，两组均不重置已有 head，各训练 3000 步；仅用 259 张有标注 train 视图，以独立 seed42 shuffle 采样。geometry、RGB、训练相机冻结，teacher、fusion、深度与 opacity 正则均关闭。已有 refiner 学习率 3e-4、raw-GT 权重与其余损失不变。

00 始终全图训练。01 以 0.5 概率选择原生 768×768 裁剪，以 0.5 概率使用全图。坐标由 stateless(seed, step) RNG 产生，不改变视图采样流；固定 3000 步序列实际为 **1536 次裁剪、1464 次全图**。共同的高斯渲染仍在原始 K 与完整 1320×989 网格上完成。裁剪分支先 `render(refine=False)`，再对同坐标的 rendered features/RGB/depth/alpha/p3d 切片运行 refiner。只有最终分割损失及 residual 正则使用切片，**raw p3d 的 GT 监督每一步仍覆盖完整原生网格**。GT/valid 只做精确索引切片，无重采样、填充或类别值变化；推理和验证始终全图。

`refiner_crops.py` 对配置作严格校验：只允许冻结场景、labeled-only、独立 view RNG 的纯 GT 模式，拒绝 pseudo、fusion、semantic geometry、sparse depth、opacity entropy、额外 region RGB 等混用，避免空间尺寸不匹配被静默接受。`model.render` 仅增加返回 rendered features 和对应 gradient route 的 prior，不修改模型参数或默认前向数值。默认 crop probability=0。

CPU 验证覆盖 stateless 坐标、全局 NumPy/Torch RNG 不变、完整类别 ID/255 与 valid 的精确对齐、裁剪最终损失的梯度范围、完整 raw 监督、RGB/depth/alpha 断梯度及不支持配置拒绝。与既有 model/routing CUDA 回归一起 **34 项通过**。两个真实旧 69k 场景的 2-step CUDA 预检分别强制 probability0/1，以确保两个分支都执行；这与正式 treatment=.5 区分。两次预检均确认所有模型参数有限，sem_features/classifier/refiner 更新，geometry/RGB/camera 逐 tensor 不变，view sampler 状态一致；峰值已分配显存分别约 4.771/2.360 GiB。记录见 `runs/refiner_crop_preflight/preflight_audit.json`。

正式配置和输入 SHA 完全一致，仅 `output` 与 `refiner_crop_probability` 两键不同。00 建立源码快照，01 复制同一快照。主要终点固定为最后 3000 步；1500 仅用于诊断，不用于改预算或选择替代终点。两组更新次数相同，最终分割的像素曝光量由于裁剪处理而不同；这不是同像素预算实验。裁剪同时改变 head 的全景上下文以及内部 depth 最大值归一化的统计范围，本对照不能分离这些因素。

执行目录为 `runs/refiner_crop_pair`；两组已完整训练并完成 50 张 RGB / 41 张有标注视图的最终评测。`scripts/report_refiner_crop_pair.py` 核验了相同 source/input SHA、geometry/RGB/camera 逐 tensor 不变、50 张 RGB PNG 完全一致、sampler 顺序/cursor/RNG 一致、日志中的 crop 坐标和视图名称可重放。259 张训练图每张访问 11–12 次。审计及结果见 `runs/refiner_crop_pair/crop_report.json`，5000 次视图配对 bootstrap 见同目录 `paired_comparison.json`。

| 固定最后 3000 步 | 全 5 类 mIoU (%) | 前景 mIoU (%) | Cable IoU (%) | Cable 边界 F1 (%) | Raw 3D mIoU (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 全图 | 94.6071 | 93.4609 | 92.3866 | 83.1178 | 75.1766 |
| 50% 原生裁剪 / 50% 全图 | 94.6423 | 93.5086 | 92.1951 | 82.6741 | 75.1527 |

裁剪减全图的全类 mIoU 差值为 **+0.0352 个百分点**，视图配对 95% 区间为 **[-0.2064, +0.2707]**；Cable IoU 差值为 **-0.1915 个百分点**，区间为 **[-0.8058, +0.4108]**。该对照没有显示稳定收益，不替换已选模型，也不作为新机制的有效性证据。单一训练 seed 和反复使用的开发验证划分限制了结论，视图 bootstrap 不衡量训练 seed 方差。

两组共享旧 69,726 高斯几何，RGB 指标均为 **PSNR 27.69713 / SSIM 0.8632775 / LPIPS 0.3143150**；这些语义结果不能与其他新几何的 RGB 分数拼接。训练至最后一步累计时间分别 731.36 / 628.35 秒，包含中途验证且存在并行 GPU 工作，不可视为独占训练吞吐。

1500 步全图为 94.7503%，最后 3000 步回落到 94.6071%；裁剪组为 94.3426% → 94.6423%。仅作为收敛诊断，未据此改变固定终点。源码及最终 optimizer state 证实 sem_features / classifier / refiner 学习率始终为 .01 / .001 / .0003，且 Adam state 实际更新。后续可以单独检验公共语义学习率的 cosine 衰减，当前数据尚不能将非单调验证表现归因于常数学习率。
