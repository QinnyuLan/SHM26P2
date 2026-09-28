# 16 TRAIN 射线：近似权重提案与原渲染器实际验证

此为独立 v2 诊断，现已单次执行完成，实际前向通过，结果记录于末节。v1 的 5e−6 raw/mass 门槛和失败记录保持不变；不会把 v1 失败改写成通过。固定场、相机、16 个坐标和类别目标完全沿用 v1，不增加样本或选点自由度。

v1 的源码审查见 `docs/semantic_compositing_lp_diagnostic.md`：FP32 `alpha=1−T` 再恢复 T 的相对误差可沿颜色反向近似共同缩放权重。v2 将 VJP 明确视作近似权重，以每条射线 `s=forward_alpha/sum(VJP)` 恢复一个公共尺度。原 W、s、原 raw/mass 误差和修正后 W 全部保存；每条射线先保存证据，再执行**原 5e−6 重建门**。修正后不能复现 raw/p3d 或异常零质量时停止，不放宽门、不删除尾部或另选坐标。

LP 和完整 W 双界只针对这个修正代理，保留同/跨 view 重叠诊断。其正/负 margin 都不自动解释成原 renderer 的可行/不可行；只用 LP 产生一次共享类别赋值候选。

为检验候选是否可由现有模型实现，固定 `ε=1e−5`，令 `qε=(1−5ε)q+ε`，避免无限 logits。固定 decoder 类别差矩阵 `D=weight[1:]−weight[0]` 实测 rank4。对 union Gaussian IDs 的原 features f，计算：

`Δf = D⁺[log(qε[1:]/qε[0])−(D f + bias[1:]−bias[0])]`。

这是 CPU FP64 的最小范数改变量，随后转回原 FP32。先检查 CPU 模拟 decoder，再检查实际设备 decoder 的 softmax 与 qε 的最大差均≤5e−6。只在临时加载场副本中修改这些 features；几何、opacity、RGB/SH/background、decoder、refiner 权重、相机完全冻结。

然后使用原 `scene.render` 再渲染相同 8 相机，SH3、`refine=False`，不拦截颜色、不修改 renderer。对原 16 射线计算 `P_GT−max(P_other)`：**全部实际 margin 严格大于 1e−4**才记录“这些固定 TRAIN 射线存在一个可表达的更好共享赋值”。否则结果是 unresolved，不能称真实渲染几何受限。完整 8 图 raw 混淆矩阵前后仅作影响描述，不参与优化或选择，也不改变选中模型。

前后 8 图 RGB 浮点哈希必须一致；赋值过程中所有非 feature 张量哈希必须一致。结束或任意异常时 finally 将 union features 原字节恢复，并核对全部场景张量。只保存诊断数组、feature 候选和 JSON，不生成生产 checkpoint，不训练，也不使用 VAL/真实 RGB。

全流程上限为 16 scene renders（原8+候选8）和16颜色 backwards；LP上限60s，执行外部上限建议180s。现有15项CPU测试/Ruff通过，包括小权重不丢弃、公共尺度显式记录、rank4/秩缺陷、内部概率、最小范数改变量/零空间、实际 margin 门、v1门槛不变。主模型/训练源码未改。

解释仍限于少量已观察错误的 TRAIN probe。即使 actual forward 通过，也不能推出整图高斯分类器能无代价达到更好分数，不能把代理 LP 的负界当 renderer 不可行，更不能把标准尺度校正/线性代数/LP 当学术创新。

独立只读复核无阻断，已完成 CPU 冻结：

- plan：`/mnt/data/SHM2026/runs/semantic_compositing_forward_v2/plan.json`，SHA `4faafcc08e912ed642dd007d0bb74c4142e3a5d55c4958ab6600d93ec13b7dc9`。
- 新入口 SHA：`22d5830cd25390e4585fb4d353643c270dd1166a5e20543b28d6c78d080fedd4`。
- 共32源文件：旧31文件逐字节保留，只新增 v2 入口；16坐标/角色/标签与原 plan exact。实际冻结包 CPU imports、全部输入/源码 SHA 验证通过，记录于 `cpu_freeze_audit.json`。
- feature 审计额外记录目标最小 interior q、CPU/CUDA 实现的最小 q，以及实现误差，方便独立核验内化。

以下为 root 后续明确交接后已执行的一次性命令（原目录不可覆盖，不自动重试）：

```bash
timeout --signal=TERM --kill-after=10s 180s env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/mnt/data/SHM2026/runs/semantic_compositing_forward_v2/source_snapshot OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 uv run --no-sync python /mnt/data/SHM2026/runs/semantic_compositing_forward_v2/source_snapshot/audit_semantic_compositing_forward.py --execute /mnt/data/SHM2026/runs/semantic_compositing_forward_v2/plan.json
```


## 实际结果：16-ray 存在性通过，场景已还原

root 在 Swin6k 自然结束后交接短窗口。执行前核验 plan/runner SHA 与无 compute 客户；session **21980** 在180s外部上限内自然 **exit 0**，回执 completed，内部耗时 **3.007860 s**。实际执行正好 **16 scene renders、16 color VJP、一次 LP**，没有换点/重试。

原始 VJP 的最大质量偏差为 **2.1774561e−4**。仅公共尺度恢复后，16条射线的 raw 最大重建误差 **3.3836017e−7**、p3d误差 **1.7542039e−7**、质量误差 **2.2204460e−16**，均通过未改动的v1门槛。这不追认原始 VJP 是精确权重：代理和原 W 都已保存，实际 renderer 验证才是下面存在性结论的依据。

代理含 **1,211 个 Gaussian、1,684 个正权重项**，完整 LP 用6,056变量/64个margin约束，HiGHS正常收敛；完整W重算的 **LB=0.08242611947774886、UB=0.082426119477749**，gap=1.39e−16。这组双界仍只属于修正代理。

候选 q 的最小内化概率为 **1e−5**，CPU与CUDA实际decoder最小概率均为 **9.99997519e−6**，与目标最大差 **1.2514456e−7**。最小范数 feature 改变量的L2最小/中位/最大为0.1732/5.7344/9.2528，RMS为1.2841；这不是一个已经验证有用的训练更新。

把候选真正实现为临时 semantic features 后，原 `scene.render` 的 **16条实际正确类别margin全部为正，最小0.08242200315**，高于预定 **1e−4**。因此这组少量TRAIN射线确实存在由原decoder/原共享可见性可表达的更好类别赋值：8个原cable→background错误均可纠正，同时保持8个选定正确背景点。这个结论不依赖把代理当精确原renderer证书。

所有8图原RGB浮点哈希前后相同；非feature参数、decoder、几何和相机未变，finally后**全部场景张量逐字节还原**。未写生产checkpoint或更换选中模型；GPU退出后compute列表为空，已明确交还root。

## 贡献重叠与解释边界

同view8对的重叠质量最小/中位/最大为 **0.01245/0.23990/0.90897**，以较小行质量归一后为 **0.01246/0.24486/0.90914**。跨view112对中仅30对存在任意正重叠，质量中位数为 **0**、最大 **0.05473**，归一重叠最大 **0.05827**。因此它主要是一个**16-ray联合存在性检查**，没有强跨view支撑耦合，不能借此声称已排除全局跨视图标签冲突。

可以排除的仅是：“在这16条射线上，固定共享可见性绝不允许更好的类别赋值”。不能进一步推出全场分类已经具备同等可训练性、整图高分可无代价得到、所有几何限制不存在，或这是三维真值正确性。额外feature自由度忽略了refiner共享特征与全图损失约束；没有进行任何后续训练或全场采用。

作为**非选择性附录**，同8TRAIN全图raw混淆矩阵五类mIoU为 **78.918449%→79.517534%**，拉索 **28.426861%→31.914595%**。背景→拉索误判从187,901增至262,934，拉索→背景从486,246减至428,620，显示局部修正伴随其它像素的取舍。这个全图结果没有参与候选选择，也不是VAL成绩或可采用模型。

完整原/修正W、每ray重建、LP、feature内化、实际16margin和全8CM保存在 `/mnt/data/SHM2026/runs/semantic_compositing_forward_v2`；执行回执SHA为 `96a6b391190689de200132c1abfe18911bc18a0ac71c9093ba1cbba9f7fb6217`，`launch_audit.json`记录自然退出与交接。
