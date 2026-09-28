# 原生语义质量与可见质量：固定 8 TRAIN 诊断

这项固定诊断已执行完成，结果附于末节，不改变选中模型。目标是区分当前原生概率场的决策偏置与一个很宽的共享可见性限制，而不是预设哪一项解释了拉索低分。参考场为 SSIM 修复后的联合终点 `391a0577…`；其 native 原生五类 mIoU 79.094334%、拉索 31.841606%，最终 refiner 成绩不作为这里的目标。

## 数学条件与不能推出的结论

在理想的前向透明度混合中，令 `w_i=T_i α_i`，总可见质量 `A=Σw_i`，每高斯五类概率为 `q_i`。当前实现给未覆盖部分背景 one-hot，因此 `P_c=Σw_i q_ic+(1−A)1[c=background]`。代码随后执行 `clamp_min(1e-7)` 和归一化，实际 FP32 输出不是这条恒等式的逐位实现。

如果一个**已知正确的拉索高斯子集**贡献总质量 M，所有其它贡献及 residual 都是背景，即使该子集完美 one-hot，仍有 `P_cable=M`、`P_bg=1−M`。此时 M≤0.5 不能让拉索获胜，等值的 argmax 也优先背景。若其它前景类别分摊剩余质量，精确条件变为拉索质量超过每一个竞争类别；0.5 不再是必要条件。若背景至少占 `1−M−ε`，只能在明确这个 ε 条件后给出 `M≤(1−ε)/2` 的对应限制。

当前 `p_cable` **不能**充当正确拉索子集质量的严格上界：它同时包含分类概率、错误标签及可见权重。无需三维类别真值的更宽限制是总 semantic alpha：即使把所有有贡献的高斯都赋给拉索，最多也得到 `P_cable=A`，而 residual 背景至少 `1−A`；因此理想混合下 A≤0.5 阻止任意单一前景类获胜。这是逐像素、不要求跨视图类别一致的宽松 oracle，**A>0.5 不证明正确拉索质量足够**。实际统计同时输出直接 A≤0.5、A<0.5−2e−6 和 |A−0.5|≤2e−6 边界带，避免把 FP32、clamp/归一化和理论条件混同。

## 已有工作检查

有限 `rg` 检查覆盖现有语义误差、refinement、方法/研究定位、联合阶段建议及 scripts/docs 中 raw3D 与 bias/calibration/temperature/logit-render 组合词。没有找到已执行的 raw3D 类别偏置校准或 logit-versus-probability 渲染对照。`docs/semantic_error_audit.md` 中有 refiner base floor/temperature 的未执行备选；教师置信度校准针对 DINO pseudo，和本项不同。此结论仅限所查记录，不声称穷尽所有历史。

正的单一温度不会改变 argmax，因此这里只拟合 4 个类偏置，背景偏置固定为 0。`softmax(log(P)+b)` 保留共享几何/可见性，只校准最终类别决策；即使改善，也不是恢复了正确三维覆盖。渲染 logits 后再 softmax 是另一种聚合模型，需要另立实验，不纳入本次，也不称创新。

## 已锁定的最小执行范围

从 sorted 259 labeled TRAIN 中均匀固定 8 个索引 `[0,36,73,110,147,184,221,258]`，名单在读取标签计数前确定：

| 用途 | 固定 TRAIN 名单 |
|---|---|
| 4 fit | 002.png、084.png、167.png、257.png |
| 4 TRAIN-check | 043.png、125.png、215.png、300.png |

TRAIN-check 只对 bias 拟合留出；这些视图仍参与过原模型训练，不能视作泛化评价。每视图仅一次 native 1320×989 场景 render，SH3、`semantics=True/refine=False`；原 checkpoint sorted350 相机逐张量核验。通过临时包装现有 `gsplat.rasterization` 返回值，捕获同一次 semantic pass 的 alpha，不增加渲染、不改任何颜色/几何，并记录与 RGB pass alpha 的最大差。包装在 finally 恢复。保存 FP32 `p3d` 和 semantic alpha，全部 8 次预测结束后才打开其 TRAIN mask/valid 做诊断；不打开真实 RGB 或 VAL 像素。

原训练 class weights 从全部 259 TRAIN mask 的类别计数按原公式重建：`min((max(freq,.002)^−.25)/mean,3)`，不加 valid 权重，保留原训练口径。计数为 `[285818023,17050754,23347376,6091203,2011398]`；FP32 权重为 `[.45604560,.92277277,.85304344,1.19358921,1.57454896]`。这一步只为复现既有权重，不把其它 251 个视图加入诊断拟合。

四个 fit 视图各均匀、不放回抽取最多 50,000 个有效像素，固定 NumPy seed 20260927。一次 CPU FP64 L-BFGS-B，4 bias 范围均为 [−6,6]，最多 100 次迭代，gtol=1e−7、ftol=1e−12；目标是原类别权重的 CE 像素均值，不优化 mIoU、不搜索权重或阈值。保留实际收敛状态、样本索引哈希和前后 CE。

输出三组 pooled 混淆矩阵/IoU（fit、TRAIN-check、全部8图）、各视图逐类真实类别概率分位数、`p_bg>p_true`、`p_true≤.5`、总 alpha 分位数及上界/边界计数。拉索作为 class 2 单独可读。总 alpha 上界和低 p_c 的经验统计分开。所有模型/相机张量与源码、输入在运行前后核验不变；只保存诊断数组/JSON，不写新场景 checkpoint。

解释预先限定：bias 在 TRAIN-check 上改善，支持“类别决策校准是问题的一部分”，不能排除几何质量不足；GT 拉索中的低总 alpha 比例高，支持宽松 oracle 下仍有覆盖障碍；若 alpha 高但 p_c 低，仅凭这些输出仍不能区分分类器错误与正确拉索子集可见质量不足。无论结果如何不据此宣称新方法、跨桥泛化或采用收益。

## 最初锁定与检查记录

- Runner：`scripts/audit_raw_semantic_mass.py`，SHA `0ce9d9af3d65eed7fd53dd0ed0c0939e3520b5be6294bb2ad9e7207768111a0f`。
- 已锁 plan：`/mnt/data/SHM2026/runs/raw_semantic_mass_diagnostic_v1/plan.json`，SHA `0b05743bbf061467682fb7265ce21d0dab2abc01b8bd0eb8b0aa8000353c5b6a`。
- 来源：直接复制该 completed semantic run 的既有 30 文件 package，加诊断入口，共 31 个源文件；263 个输入哈希绑定 checkpoint、manifest、uv.lock、259 TRAIN mask 与本组 valid。未使用正在编辑的 main package。
- 6 个诊断 CPU 测试通过：固定 TRAIN 列表、加权 CE 解析梯度/有限差分、bias 可复现/有界、alpha 与概率统计分离、捕获 finally 恢复、温度与两类质量边界。与缓存合同合跑共 14 测试通过，Ruff 通过。

以下保留最初 v1 命令作为来源记录；v1 在入口 CPU 检查失败，实际完成的是末节记录的 v2，不应重新运行 v1：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/mnt/data/SHM2026/runs/raw_semantic_mass_diagnostic_v1/source_snapshot OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 uv run --no-sync python /mnt/data/SHM2026/runs/raw_semantic_mass_diagnostic_v1/source_snapshot/audit_raw_semantic_mass.py --execute /mnt/data/SHM2026/runs/raw_semantic_mass_diagnostic_v1/plan.json
```

预计只需 8 次短推理，之后释放场景进入 CPU 拟合；FP32 数组约 240 MiB，写数据盘。运行不读取 VAL 选偏置，也不自动升级为训练实验。


## 固定结果：宽上界只能解释极少部分，CE 偏置未改善拉索

2026-09-27，v1 在入口 `verify()` 因导入历史包中不存在的 `bridge_rgs.checkpoints` 而自然 exit 1，发生在 scene/CUDA/诊断标签解码之前，0 次 render。原 plan/source/log 均保留。root 授权的 v2 **仅移除这个不使用的依赖检查**，原 30 个 package 文件逐字节不变，样本、权重、拟合和统计协议不变；在冻结旧包下实际 CPU imports、全部源码和 263 输入哈希通过后运行一次。

v2 目录 `/mnt/data/SHM2026/runs/raw_semantic_mass_diagnostic_v2`；plan SHA `084cdc53b9c115ba698afae618564730ef13a048382e0ce0dfa24edb3f9cfc37`，runner SHA `a6e47fee59652b85eb05bd2f27c6bbd6d30527b8b16c5fcfbf796eac18428916`。session 27379 在 180 s 外部上限内自然 exit 0，receipt completed，内部耗时 **3.565 s**；正好 8 次 scene render，所有模型和相机张量前后相同，8 图 semantic/RGB alpha 的最大差均为 0。GPU 已明确交还 root，没有追加渲染、样本或拟合。技术失败与修复链保存在 `launch_failure_audit.json`（v1）、`entrypoint_repair`（v2 plan）及 `launch_audit.json`（v2）。

L-BFGS-B **12 次迭代正常收敛**（projected gradient 门槛），拟合样本共 200,000 个；加权 CE 从 0.08988547 降至 0.08667142。五类偏置为 `[0, +0.640900, −0.249755, +1.089813, +0.694000]`。固定 CE 目标给拉索的是负偏置，不能把 CE 下降等同于拉索 IoU 提升。

| 固定集合 | raw 五类 % | calibrated 五类 % | raw 拉索 % | calibrated 拉索 % |
|---|---:|---:|---:|---:|
| 4 fit TRAIN | 79.330518 | 77.650557 | 27.560216 | 23.538091 |
| 4 TRAIN-check | 78.325586 | 78.714415 | 28.872666 | 24.935129 |
| 全部 8 TRAIN | 78.918449 | 78.246228 | 28.426861 | 24.484369 |

全部 8 图中共有 **764,951 个有效 GT 拉索像素**。其中总 semantic alpha≤0.5 只有 **4,450 个（0.581737%）**，全部来自 125.png；这 4,450 个也都低于 `0.5−2e−6`，拉索中没有落入 ±2e−6 边界带。fit 组比例为 0，TRAIN-check 为 0.841894%。各图拉索 alpha 中位数为 0.96396–0.99987，除 125 外其余均至少 0.99969。因此，**总 alpha 不足导致 residual background 获胜这一宽限制，无法解释这 8 个视图中的大多数拉索错误**。

与之分开，拉索像素的 `p_true≤0.5` 有 **497,901 个（65.089267%）**，`p_bg>p_cable` 有 **489,372 个（63.974294%）**；各图 `p_cable` 中位数在 0.27594–0.78652。raw 混淆矩阵中 cable→background 为 486,246、background→cable 为 187,901；固定 CE 偏置后分别变为 534,879 和 120,589，减少背景误报同时增加拉索漏检。完整 fit/check/all 混淆矩阵、每图分位数保存在 `execution_receipt.json`；`class_mass_summary.json` 只从这些既定计数 CPU 求和，没有新评分/拟合。

这里不能进一步推出“正确拉索子集的质量充足”：近乎不透明的总质量也可能由背景或其它对象的高斯占据。诊断只否定了把**总**低透明度作为多数错误解释的简化假设；它尚未区分高斯分类欠拟合、正确拉索子集的可见质量不足、跨视图硬区域标签与 RGB 表面支撑的不相容。固定 4-bias CE 校准没有缓解原生拉索瓶颈，也不证明所有可能校准目标都无效。保留这个结果，不继续搜索偏置/温度/视图，不据此作 VAL 采用或三维真值结论。

下一项若确需区分两个因素，可预先固定很少的 TRAIN 像素，提取完整实际 compositing 权重 W，先验证 `W·当前每高斯概率+residual背景` 能复现原 p3d；随后仅 CPU 放宽每高斯类别概率为独立 simplex 变量，保持跨像素共享，做 GT 类 margin 的凸可行性检查。有可行赋值而当前场失败，说明这些射线至少允许更好的语义赋值；有可靠不可行证书才支持固定可见性与这些硬标签不相容。不能截取 top-K 并忽略剩余质量后声称不可行，尾部必须完整或按最乐观自由类别分配给出界。这只是后续建议，未选新样本、未提取权重、未拟合或运行 GPU，也不将二维标签当成三维真值。
