# 固定水平翻转推理对照

本实验对三个事先指定 checkpoint 各执行一次固定的水平翻转 TTA：原始 support 8000 步、无翻转继续训练 3000 步、50% 翻转继续训练 3000 步。只复用一次相机渲染的三维证据，运行原始和水平翻转两个 refiner 分支，将翻转分支还原后按 0.5 概率平均。没有新增训练、teacher 或权重搜索，这是标准工程对照，不作为创新机制。

计划为 `configs/generated_refiner_flip_tta/plan.json`，SHA256 `9fb22e1dc182ea6485f20bd079c1b1588c6d9262897b379f0aa40a45499724ba`。源快照固定于 `runs/refiner_flip_tta/source_snapshot`，按排序文件散列计算的树 SHA256 为 `8f26a9af7f84865dc6eb526d97cabbc5bdce75172740d3a36c3990ca1d130dc3`。runner 为 `scripts/evaluate_refiner_flip_tta.py`，预定 SHA256 `938f9a6cb4b4c202af4e44c56bfe3b73bc283be7960c005553c8918176a7b4d4`。运行时实际导入的关键模块路径必须在此快照中，并在执行前后校验源文件与所有输入字节。后续主树变化不参与本次运行。

输入均从已完成的 experiment receipt 验证 checkpoint/evaluation SHA，另逐一锁定每个模型原评测中的 50 张 RGB 与 50 张 mask。三个 checkpoint 分别为 `runs/support_split_semantic_coupled/last.pt`、`runs/refiner_flip/00_control/last.pt`、`runs/refiner_flip/01_flip/last.pt`；三者均保持 legacy 协议，不以当前 manifest 推断升级。新代码计算的原生评价 fingerprint 必须与各自旧评测相同。

每组使用原生 1320×989 全 50 个 VAL 相机、41 张有效语义标注和 LPIPS。runner 在 `scene.render` 返回、TTA 尚未执行时检查 K、原始相机姿态和网格，保存此时的 argmax，逐像素核验它与原评测 50 张 mask 完全一致，同时核验新 RGB 与原 PNG。最终还检查整体和逐视图 raw 3D confusion matrix、全部模型 tensor、训练相机、50 RGB PNG 及 RGB 指标不变。若任何一项不符，停止并保留失败回执，不用新的预处理结果代替旧基准。

评价采用标准 `evaluate_scene`：scorer 会读取真实 RGB、GT 和 valid 计算指标；传给渲染器的只有相机参数，传给 TTA 的只有渲染出的 features/RGB/depth/alpha/prior。没有把评价输入交给预测器，但也不声称这些评分文件没有被打开。runner 的 `render_seconds` 包含审计 readback、参考 PNG 读取和保存，不能作为纯部署推理延迟；必要的独占基准须另行测量。

对三个 checkpoint 分别报告 TTA 减去同一 checkpoint 原始预测的全部配对结果，使用 5000 次视图 bootstrap。该区间只反映固定开发划分的视图变化，不覆盖跨随机种子、跨桥梁或隐藏测试集不确定性。三组正式推理与完整审计均已成功完成，完整指标、plain 预测、冻结审计与配对结果保留于 `runs/refiner_flip_tta`。

| 固定 checkpoint | 原始 mIoU (%) | TTA mIoU (%) | TTA 前景 mIoU (%) | TTA Cable IoU (%) | TTA Cable 边界 F1 (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原始 support 8000 步 | 94.5118 | 94.5763 | 93.3957 | 93.7440 | 88.9715 |
| 无翻转继续训练 3000 步 | 93.1845 | 93.1965 | 91.8064 | 88.1578 | 75.7203 |
| 50% 翻转继续训练 3000 步 | 94.5506 | 94.6210 | 93.4468 | 94.1363 | 90.2461 |

相对同一 checkpoint 原始预测，TTA 的全类 mIoU 增益依次为 **+0.0645 个百分点 [-0.0991, +0.2100]**、**+0.0120 [-0.2302, +0.2069]**、**+0.0704 [+0.0136, +0.1278]**。对应 Cable IoU 变化为 **+0.1947 [-0.4345, +0.7910]**、**-0.1604 [-1.4236, +0.7627]**、**+0.1389 [-0.0846, +0.3560]**。只有经过翻转训练的 checkpoint，其全类小增益在该配对视图区间中保持正值；Cable 区间仍跨零。最高分 94.6210% 尚未到 95%，不构成明显超过目标或学术创新证据。

另保存三组TTA相对原始support底座的事后描述性配对，未更改锁定计划、预测或主报告。翻转训练加TTA的完整组合相对原底座为全类 **+0.1093个百分点 [-0.0098,+0.2571]**，Cable **+0.5870 [-0.0601,+1.3497]**，两者区间均跨零。因此“相对自身plain有小增益”不能写成“完整优化相对起始模型已可靠提升”。文件为`paired_{base,00_control,01_flip}_tta_minus_original_source.json`。

三个模型各 50 张未经 TTA 的 argmax 均与旧 native mask 逐像素一致，证明本次坐标协议兼容修改没有改变这些 legacy checkpoint 的原始语义输出。150 张 RGB PNG、全部整体与逐视图 raw 3D confusion matrix、模型 tensor、相机、原生评价 fingerprint 和 RGB 数值均通过精确不变检查。共同 RGB 为 **30.1024 / 0.902981 / 0.224007**；TTA 不改变三维场、RGB 或参数量。各组含标准 scorer/LPIPS 的峰值分配显存为 1.955–1.957 GiB，不是单纯部署 forward 的显存测量。

执行回执 `execution_receipt.json` 为 completed，三组固定顺序完成后再次核对全部输入与源码 SHA。每组的 `audit.json`、`paired_tta_minus_plain.json` 和 `evaluation_native/metrics.json` 提供逐视图证据，汇总为 `report.json`。未新增种子、改翻转权重、筛除验证视角或重复选择 checkpoint。

之后另做独占 GPU 部署路径基准：50 相机各 3 次，warmup 5 次，batch=1，无其他 GPU 客户端，不含上述审计 I/O。原始 support 的 plain/TTA 为 **35.822 / 66.536 ms**（27.92 / 15.03 FPS），翻转训练模型为 **35.934 / 66.628 ms**（27.83 / 15.01 FPS）。峰值分配显存均由 plain 的 **1.589 GiB** 增至 TTA 的 **1.651 GiB**。四份 `benchmark_{base,flip}_{plain,tta}.json` 独立保存，没有改动已锁定的指标汇总。约 85% 的延迟增量应与约 0.07 个百分点的小增益一并评估；这不是免费改进。
