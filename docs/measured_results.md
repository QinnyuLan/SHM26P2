# 原生分辨率实测汇总

由 `uv run python scripts/write_optimization_report.py` 从已完成的原始评价生成。
固定50个RGB/41个语义开发验证视图，1320×989去畸变网格，原始验证相机。
每行是一个实际检查点或明确标出的固定模型组合；不拼接不同场的最优指标。
DINOv3组合额外需要约8.47亿参数教师；raw3D栏属于组合中的同一学生场，未被教师更新。

| 实验 | 高斯数 | PSNR↑ | SSIM↑ | LPIPS↓ | 五类mIoU↑ | 前景mIoU↑ | 拉索IoU↑ | raw3D五类/拉索↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [原联合模型，3,000步](../runs/refined/evaluation_native/metrics.json) | 69,726 | 27.697 | 0.86328 | 0.31432 | 89.952% | 88.151% | 74.003% | 76.104% / 19.217% |
| [旧架构延长至9,000步](../runs/refined_long_v2/evaluation_native/metrics.json) | 69,726 | 28.666 | 0.87769 | 0.28312 | 91.266% | 89.645% | 78.222% | 77.023% / 21.867% |
| [局部精修±6，相同6,000步](../runs/refiner_bound6_control/evaluation_native/metrics.json) | 69,726 | 27.697 | 0.86328 | 0.31432 | 92.515% | 91.073% | 83.563% | 75.383% / 15.621% |
| [多尺度精修，相同6,000步](../runs/refiner_multiscale_v1/evaluation_native/metrics.json) | 69,726 | 27.697 | 0.86328 | 0.31432 | 94.713% | 93.593% | 92.510% | 75.464% / 15.695% |
| [强RGB，6,000步](../runs/strong_rgb_sanity/evaluation_native/metrics.json) | 177,378 | 28.971 | 0.88565 | 0.26411 | — | — | — | — |
| [强RGB，30,000步](../runs/strong_rgb/evaluation_native/metrics.json) | 498,136 | 30.274 | 0.90282 | 0.22449 | — | — | — | — |
| [强几何＋语义，允许精修回传](../runs/strong_semantic_coupled/evaluation_native/metrics.json) | 498,136 | 30.274 | 0.90282 | 0.22449 | 93.908% | 92.582% | 93.644% | 78.643% / 29.788% |
| [强几何＋语义，隔离精修梯度](../runs/strong_semantic_detached/evaluation_native/metrics.json) | 498,136 | 30.274 | 0.90282 | 0.22449 | 93.330% | 91.866% | 92.325% | 78.699% / 30.444% |
| [强几何，后期语义参数平均](../runs/strong_semantic_averaged/evaluation_native/metrics.json) | 498,136 | 30.274 | 0.90282 | 0.22449 | 93.989% | 92.680% | 93.776% | 78.670% / 29.741% |
| [旧多尺度，后期语义参数平均](../runs/multiscale_semantic_averaged/evaluation_native/metrics.json) | 69,726 | 27.697 | 0.86328 | 0.31432 | 94.816% | 93.714% | 92.803% | 75.335% / 14.959% |
| [仅opacity续训RGB对照](../runs/opacity_rgb_control/evaluation_native/metrics.json) | 498,136 | 30.348 | 0.90303 | 0.22416 | 93.880% | 92.549% | 93.499% | 78.697% / 30.296% |
| [仅opacity＋SfM深度](../runs/opacity_sfm_depth/evaluation_native/metrics.json) | 498,136 | 30.348 | 0.90298 | 0.22406 | 93.906% | 92.582% | 93.463% | 78.818% / 30.563% |
| [仅opacity熵实验，匹配RGB对照](../runs/opacity_entropy_control_v2/evaluation_native/metrics.json) | 498,136 | 30.348 | 0.90303 | 0.22416 | 93.880% | 92.549% | 93.497% | 78.699% / 30.304% |
| [仅opacity，熵正则0.05](../runs/opacity_entropy_005_v2/evaluation_native/metrics.json) | 498,136 | 30.204 | 0.90062 | 0.22708 | 93.900% | 92.577% | 93.243% | 78.371% / 29.023% |
| [稀疏射线前方质量实验，匹配RGB＋ED对照](../runs/sparse_front_pair/00_rgb_ed/evaluation_native/metrics.json) | 498,136 | 30.429 | 0.90305 | 0.22399 | 93.814% | 92.466% | 93.519% | 78.934% / 31.282% |
| [仅opacity，稀疏射线前方质量约束](../runs/sparse_front_pair/01_rgb_ed_front/evaluation_native/metrics.json) | 498,136 | 29.882 | 0.90105 | 0.22633 | 94.034% | 92.745% | 93.109% | 78.851% / 29.382% |
| [H3，相同容量零矩输入](../runs/h3_moments/00_zero/evaluation_native/metrics.json) | 498,136 | 30.102 | 0.90298 | 0.22401 | 93.072% | 91.659% | 87.895% | 79.199% / 30.808% |
| [H3，仅深度方差输入](../runs/h3_moments/01_variance/evaluation_native/metrics.json) | 498,136 | 30.102 | 0.90298 | 0.22401 | 94.575% | 93.389% | 93.774% | 79.199% / 30.808% |
| [H3，深度与语义特征交叉矩](../runs/h3_moments/02_cross/evaluation_native/metrics.json) | 498,136 | 30.102 | 0.90298 | 0.22401 | 94.536% | 93.344% | 93.827% | 79.199% / 30.808% |
| [仅精修器续训3k，翻转匹配对照](../runs/refiner_flip/00_control/evaluation_native/metrics.json) | 498,136 | 30.102 | 0.90298 | 0.22401 | 93.184% | 91.788% | 88.318% | 79.199% / 30.808% |
| [仅精修器续训3k，水平翻转0.5](../runs/refiner_flip/01_flip/evaluation_native/metrics.json) | 498,136 | 30.102 | 0.90298 | 0.22401 | 94.551% | 93.362% | 93.997% | 79.199% / 30.808% |
| [修正结构分裂，6,000步RGB](../runs/support_split_rgb_sanity/evaluation_native/metrics.json) | 177,378 | 29.075 | 0.88652 | 0.26196 | — | — | — | — |
| [修正结构分裂，再次恢复完成30,000步RGB](../runs/support_split_rgb_full_resumed2/evaluation_native/metrics.json) | 498,136 | 30.102 | 0.90298 | 0.22401 | — | — | — | — |
| [修正几何＋相同8,000步语义](../runs/support_split_semantic_coupled/evaluation_native/metrics.json) | 498,136 | 30.102 | 0.90298 | 0.22401 | 94.512% | 93.320% | 93.549% | 79.199% / 30.808% |
| [修正15k几何，3k仅颜色优化](../runs/support15k_appearance_polish/evaluation_native/metrics.json) | 332,255 | 29.338 | 0.88486 | 0.26171 | — | — | — | — |
| [修正20k几何，3k仅颜色优化](../runs/support20k_appearance_polish/evaluation_native/metrics.json) | 425,447 | 29.944 | 0.89679 | 0.23884 | — | — | — | — |
| [修正20k几何，邻近阶段1NN语义迁移](../runs/support20k_semantic_transfer/evaluation_native/metrics.json) | 425,447 | 29.944 | 0.89679 | 0.23884 | 93.857% | 92.538% | 92.163% | 78.890% / 30.371% |
| [旧多尺度，3k全图续训对照](../runs/refiner_crop_pair/00_fullframe/evaluation_native/metrics.json) | 69,726 | 27.697 | 0.86328 | 0.31432 | 94.607% | 93.461% | 92.387% | 75.177% / 14.441% |
| [旧多尺度，3k裁剪/全图混合](../runs/refiner_crop_pair/01_mixed_crop/evaluation_native/metrics.json) | 69,726 | 27.697 | 0.86328 | 0.31432 | 94.642% | 93.509% | 92.195% | 75.153% / 14.472% |
| [强语义平均，3k常数LR续训](../runs/semantic_lr_pair/00_constant/evaluation_native/metrics.json) | 498,136 | 30.274 | 0.90282 | 0.22449 | 93.959% | 92.645% | 93.443% | 78.496% / 29.307% |
| [强语义平均，3k余弦LR续训](../runs/semantic_lr_pair/01_cosine/evaluation_native/metrics.json) | 498,136 | 30.274 | 0.90282 | 0.22449 | 93.900% | 92.571% | 93.573% | 78.670% / 29.681% |
| [旧语义1NN迁移，零训练控制](../runs/semantic_transfer_1nn/evaluation_native/metrics.json) | 498,136 | 30.274 | 0.90282 | 0.22449 | 78.037% | 75.296% | 43.150% | 73.103% / 20.699% |
| [H2：仅GT续训](../runs/h2_multiscale/00_gt_continuation/evaluation_native/metrics.json) | 69,726 | 27.697 | 0.86328 | 0.31432 | 94.769% | 93.652% | 92.908% | 75.311% / 15.072% |
| [H2：教师，无融合](../runs/h2_multiscale/01_teacher_no_fusion/evaluation_native/metrics.json) | 69,726 | 27.697 | 0.86328 | 0.31432 | 94.767% | 93.654% | 92.698% | 75.316% / 15.122% |
| [H2：教师，固定范围融合](../runs/h2_multiscale/02_teacher_fixed_sampling/evaluation_native/metrics.json) | 69,726 | 27.697 | 0.86328 | 0.31432 | 94.793% | 93.685% | 92.747% | 75.312% / 15.037% |
| [H2：教师，投影范围融合](../runs/h2_multiscale/03_teacher_projection_sampling/evaluation_native/metrics.json) | 69,726 | 27.697 | 0.86328 | 0.31432 | 94.789% | 93.676% | 92.904% | 75.315% / 15.062% |
| [原support＋固定翻转TTA](../runs/refiner_flip_tta/base/evaluation_native/metrics.json) | 498,136 | 30.102 | 0.90298 | 0.22401 | 94.576% | 93.396% | 93.744% | 79.199% / 30.808% |
| [续训对照＋固定翻转TTA](../runs/refiner_flip_tta/00_control/evaluation_native/metrics.json) | 498,136 | 30.102 | 0.90298 | 0.22401 | 93.196% | 91.806% | 88.158% | 79.199% / 30.808% |
| [翻转训练＋固定翻转TTA](../runs/refiner_flip_tta/01_flip/evaluation_native/metrics.json) | 498,136 | 30.102 | 0.90298 | 0.22401 | 94.621% | 93.447% | 94.136% | 79.199% / 30.808% |
| [原support＋DINOv3固定0.5](../runs/teacher_renderer_transfer_support/metrics.json) | 498,136 | 30.102 | 0.90298 | 0.22401 | 94.953% | 93.851% | 94.334% | 79.199% / 30.808% |
| [H3零输入＋DINOv3固定0.5](../runs/teacher_renderer_transfer_support/metrics.json) | 498,136 | 30.102 | 0.90298 | 0.22401 | 94.699% | 93.545% | 93.507% | 79.199% / 30.808% |
| [H3方差＋DINOv3固定0.5](../runs/teacher_renderer_transfer_support/metrics.json) | 498,136 | 30.102 | 0.90298 | 0.22401 | 95.000% | 93.904% | 94.507% | 79.199% / 30.808% |
| [H3交叉项＋DINOv3固定0.5](../runs/teacher_renderer_transfer_support/metrics.json) | 498,136 | 30.102 | 0.90298 | 0.22401 | 95.023% | 93.932% | 94.710% | 79.199% / 30.808% |

同学SEM386的Dev30数字为五类93.7418%、前景92.3300%、拉索91.4347%。
本地缺少其具体Dev30视图清单及RGB150留出指标，不能将不同划分的数值差称为同协议提升。
研究机制配对实验与局限见[研究定位](research_positioning.md)、[优化记录](optimization_log.md)。
机器可读来源与SHA见[汇总JSON](../runs/optimization_results.json)。
