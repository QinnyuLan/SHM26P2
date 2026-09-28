# 坐标修正重训：共同原始网格实测

2026-09-26，两个固定模型均完成全部50 RGB／41标注视角的原始网格评分和5000次配对视图bootstrap。**没有证据支持新v2模型整体优于旧support基线，因此不替换现有最佳模型，也不把正确性修复写成性能创新。**

| 指标 | 旧support模型 | 新corner-v2模型 | 新−旧及95%配对区间 |
|---|---:|---:|---:|
| PSNR↑ | 29.40095 | 29.49663 | +0.09568 dB [−0.00602,+0.20314] |
| SSIM↑ | 0.870650 | 0.870772 | +0.000122 [−0.000072,+0.000311] |
| LPIPS↓ | 0.271237 | 0.271328 | +0.000091 [−0.000770,+0.001019] |
| 五类mIoU↑ | 94.55293% | 94.26299% | −0.28995 pp [−0.97927,+0.15428] |
| 前景mIoU↑ | 93.36976% | 93.02348% | −0.34629 pp [−1.18389,+0.18755] |
| Cable IoU↑ | 93.56537% | 93.21935% | −0.34601 pp [−1.13255,+0.55660] |
| Deck IoU↑ | 97.03543% | 96.35668% | −0.67875 pp [−1.93409,−0.04756] |
| Tower IoU↑ | 94.00920% | 93.38779% | −0.62141 pp [−2.43891,+0.30891] |
| Foundation IoU↑ | 88.86906% | 89.13008% | +0.26103 pp [−0.34559,+0.92309] |

六个主要RGB/语义终点的区间均跨零；Deck这一单类区间全为负，其余类别详见完整JSON。这里没有多种子或跨场景重复，类别区间也是描述性的逐类区间。不能据此把任何小的点估计变化称为稳定机制收益。

![共同原始网格的配对差异](../artifacts/optimization/corner_v2_official_comparison.png)

两组均只输入原始相机，单次精修，无DINO教师推理或TTA。最终语义与RGB来自各自同一个场和检查点。旧模型依其历史`legacy_mixed_v1`约定导出，新模型依`colmap_corner_v2`导出；二者在同一原始畸变图和同一LabelMe栅格真值上评分。完整规则见[协议](official_original_grid_evaluation.md)，包括对交付uint8 PNG评分、全图PSNR/LPIPS、11×11 Gaussian SSIM完整窗口、忽略未知标签及pooled confusion。

公共fingerprint为`21a2f19c5d4d703403a0107402d5dd5e006dcd98e5f8e023820c388c3e69a85d`。这不是两组各自的native fingerprint；后者仍不同，不能将旧native与新native直接差分。冻结的native评价代码也使用11×11 Gaussian SSIM，但在去畸变网格的侵蚀有效支持上评分；7×7 box SSIM用于训练loss，并非这里的native评价。原始网格评价还改变了RGB支持、畸变插值和PNG量化，不能将两种评分的绝对差解释成模型性能退化，或只归因于某一个插值环节。

## 执行与来源

- 共同源码树：`de8f1299f2e96c3eaa304317aa4ec1e1f8ee87e8f828f3852da741a5f4ab1c4a`。
- 旧support checkpoint：`a329fc70d582972fc338715765304b99da208c821ee34288809db500d5acd6b9`。
- 新v2 checkpoint：`a77d304f32de4356c1a4608ed5192258b6a9ced4c67a4297809f439481408d89`。
- [旧模型指标](../runs/official_common_grid_v1/legacy_support/official_metrics.json)、[新模型指标](../runs/official_common_grid_v1/corner_v2/official_metrics.json)、[完整配对](../runs/official_common_grid_v1/comparison.json)。
- [外层执行回执](../runs/official_common_grid_v1/pair_execution_receipt.json)记录三个子命令自然退出0，两个scorer分别约23秒；这包括PNG和评分，不是部署时延。

两组各100张交付PNG、实际导入8个核心模块、模型和训练来源、相同LPIPS权重、指标SHA及公共GT指纹均通过检查；末端再次核对所有绑定输入字节。每组都先完成全部预测，再解码该组评分GT。第一次GPU预检错误地阻止graphics-only桌面进程，尚未加载模型即退出；失败回执和原runner保留，外层检查修复后通过18项回归再执行，详情见[执行计划](official_pair_execution_plan.md)。没有修改冻结scorer来修复这一问题。

该对照比较两条完整训练链的实际交付效果。训练源码、旧模型恢复历史及初始化图像颜色等也有差异，所以不是像素约定的严格单变量因果实验。两组仍用本项目多次使用的开发划分，不能替代同学未公开的Dev30或主办方盲测；共同协议也是本项目定义，不冒充主办方评分代码。DINO固定组合的95.0231%属于旧native协议，不能搬进本表。

下一步只继续有明确诊断依据的方向。配对外观敏感性使用TRAIN图作干预，不能把其真实图输入结果列为部署成绩；新的原始网格训练可行性检查也只用于工程诊断。已有H1/H2/H3负结果完整保留。
