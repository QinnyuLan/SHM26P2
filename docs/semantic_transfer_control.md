# 旧语义到新 RGB 几何：固定 1NN 工程控制

本项将 `runs/refiner_multiscale_v1/last.pt` 的 Gaussian semantic features、完整 classifier 和 multiscale refiner 迁移到 `runs/strong_rgb/last.pt`。它是可验证的工程 warmstart 控制，不是新算法，也不能由几何覆盖率推断语义正确率。没有启动训练。

## 兼容性与迁移风险

两次原始 experiment receipt 中 manifest SHA 完全一致；350 TRAIN 相机 tensor 逐位相同，world 坐标系与 `scene_scale=16.261470794677734` 相同；feature_dim 均为16。目标 strongRGB 的 geometry/RGB/prior_counts 又与 strong_semantic_coupled 8k 逐位一致。源有69,726点，目标498,136点。强RGB checkpoint原 head 是legacy，必须用源 multiscale64 / bound6 / pyramid_strip 的 architecture metadata 完整替换；只复制部分权重无法加载正确的网络。

预先固定 Euclidean 1NN，无 k 搜索、融合平均或验证阈值。精确等距时选源索引最小者；实现检查第二近距离仅用于定位 tie，不构成第二种邻居数实验。最近邻距离单位是 COLMAP 任意 world scale，不是米。

| 目标点分组 | 点数 | 最近源中心距离 p50 / p90 / p99 | 位于该最近源3σ形状椭球内 |
|---|---:|---|---:|
| 全部 | 498,136 | 0.03705 / 0.45465 / 3.97307 | 72.04% |
| 稳定继承TRAIN前景票 | 283,624 | 0.02711 / 0.13385 / 0.71265 | 67.10% |
| 稳定继承TRAIN背景票 | 113,732 | 0.08152 / 0.68633 / 4.49211 | 79.95% |
| 其他 | 100,780 | 0.06484 / 2.23892 / 6.69350 | 77.04% |

这里“稳定票”仅指继承的 `semantic_prior_counts` 总票≥2、最大类占比≥0.8。增密后同一初始证据被多个后代复制，**不代表独立点级3D真值**。3σ椭球来自源 Gaussian 形状，仅描述几何 footprint，不是位置协方差、语义可信区间或接受门槛；此次所有点均按1NN迁移。

最近源点的继承票主类与目标稳定票主类一致率，bg/deck/cable/tower/foundation 为81.93/76.84/47.92/50.78/25.09%；最近源点的学习分类器同类率为69.96/71.32/78.73/59.41/21.18%。这些是相互一致性代理，不能称迁移准确率，但说明 foundation/tower 等空间接近的类别边界和已位移后代存在明显风险。

源 feature 范数中位数8.85，目标未训练语义 feature为5.67；完整复制源 decoder 和 head 能保持来源特征的内部坐标约定，却不能保持**渲染后**特征分布：新 Gaussian 的数目、alpha混合、遮挡、深度和 RGB 都不同。源 refiner 依赖这些输入及空间上下文，尤其新235的浮层仍保留；因此不能承诺复制旧94.713%的能力到新几何。源的高final而低raw成绩也说明它依赖视图条件修正，不能简单理解为可靠点标签的近邻传播。

## 实现与审计

- 实现：`src/bridge_rgs/semantic_transfer.py`；命令：`uv run python scripts/transfer_semantic_1nn.py`。
- 派生 checkpoint：`runs/semantic_transfer_1nn/last.pt`，source/target/manifest/source-snapshot SHA 与覆盖风险写入 `last.provenance.json`。输入 manifest 还必须与各自原始 experiment receipt 的 hash一致，避免只检查当前同名文件。
- 仅修改 `splats.sem_features`、完整 `semantic_decoder.*`、完整 `refiner.*`，更新 head architecture。目标所有 geometry、RGB、background、`semantic_prior_counts`、训练相机逐位保留；输入checkpoint不修改。
- 丢弃 optimizer、density、RNG、旧stats；标记 `semantic_transfer_inference_or_warmstart`、step=0。train 的 derived-checkpoint guard 禁止严格 resume；以后如训练必须启动新 warmstart stage。
- 固定本次 source_snapshot 后进行一次完整50 RGB /41有GT、原生1320×989、original-camera、含LPIPS零训练评价；无真实图像输入模型、无测试图片输入。

8项迁移CPU测试、6项averaging测试及8项scope测试共22项通过，检查非语义参数逐位不变、输入不被修改、1NN/tie规则、缺失SHA、manifest/frame/camera/feature/schema/dtype不一致、NaN与不合法head拒错。Ruff通过。较详细CPU可行性统计在 `runs/semantic_transfer_feasibility.json`。

## 后续最小受控训练建议（尚未执行）

先报告本次零训练 final/raw/各类/尾部视角与RGB不变性。若它有继续训练价值，最小两臂可用**同一迁移feature+decoder**，仅比较保留源head与重置head；冻结新几何/RGB，使用相同259 labeled TRAIN独立shuffle、相同步数/权重/梯度路由。这样检验的是head迁移的价值。若改用“全新语义初始化”作对照，则同时改变feature、decoder和head，结论应明确是整套warmstart的工程比较。不要在同一轮同时搜索 k、距离阈值和loss权重，更不要按235单视图定制迁移规则。

## 完整零训练结果：明显负结果

| 指标 | 旧源模型 | 新几何已完成语义8k（参考） | 本次固定1NN零训练 |
|---|---:|---:|---:|
| PSNR | 27.69713 | 30.27351 | 30.27351 |
| SSIM | 0.863278 | 0.902818 | 0.902818 |
| LPIPS | 0.314315 | 0.224494 | 0.224494 |
| final all5 mIoU (%) | 94.71320 | 93.90821 | 78.03719 |
| final cable IoU (%) | 92.50966 | 93.64356 | 43.15018 |
| final tower IoU (%) | 94.31433 | 91.31644 | 89.82803 |
| final foundation IoU (%) | 90.29992 | 89.06267 | 77.92524 |
| raw all5 mIoU (%) | 75.46445 | 78.64341 | 73.10326 |
| raw cable IoU (%) | 15.69540 | 29.78754 | 20.69864 |

三者同一 evaluation fingerprint，完整50/41视角保留；本次迁移的全部50张已存RGB PNG与新几何参考逐字节一致，不仅是均值指标一致。`evaluation_summary.json` 保存checkpoint/metrics SHA和不变性检查。

迁移未能保存旧94.7能力，不能作为当前最终模型。结果符合NN跨类别错配与渲染特征/上下文分布变化的风险，但**本次不能分离两者的因果贡献**。是否经过新几何上的有标签微调可以恢复仍未验证；没有启动后续训练，也没有根据该验证结果搜索邻居数或阈值。旧源6k、参考新语义8k与本次零训练预算不同，此表是工程可用性检查，不是同预算训练算法比较。
