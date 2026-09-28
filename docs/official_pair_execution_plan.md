# 原始网格共同评价：固定双模型执行计划

状态：**只完成 CPU 审查和源码锁定，尚未启动真实评价**。v2 RGB30k / semantic8k 自然完成后，由 root 核对空间、输入来源和 GPU 时段再执行。计划不读或哈希正在变化的 v2 `last.pt`，不改原始 GT，不新加 teacher/TTA，不清理文件。

## 已冻结的实现与合同

共同 bundle：`runs/official_common_grid_v1`。两组均使用其中 `source_snapshot/bridge_rgs`，评价及比较入口也使用 bundle 中的 `scripts` 副本；不能让一组读取主源码、另一组读取旧训练快照。

- 源码树 SHA：`de8f1299f2e96c3eaa304317aa4ec1e1f8ee87e8f828f3852da741a5f4ab1c4a`，31 个 Python 源文件，415,422 字节。
- 文件级来源在 `source_snapshot_receipt.json`；机器可读执行计划在 `execution_plan.json`；测试结果在 `cpu_validation_receipt.json`。这些是计划/源码回执，不能当成模型评价已完成。
- 实际从快照导入 8 个核心模块，其 `__file__` 均在快照内；快照内 33 项 CPU 测试通过。未加载真实 GPU scene。
- 原始 reference 固定为 `artifacts/prepared/manifest.json`，SHA `551546979a583d46e840bd485559721f361bc28fa4dca60826374ceb74b315fa`。公共评分与推理 metadata 已分离；当前唯一推理实现仍是 plain。

独立审查确认：RGB 按全部 50 帧取均值并成对 bootstrap；语义按 41 个标注视图采样，每次先汇总混淆矩阵再算 IoU。异质区域测试能区分错误的逐图 IoU 平均，另有无标注 9 帧参与 RGB 的回归。比较方向是 candidate−reference，LPIPS 为负才是改善。缺失类输出 null 而非 NaN，并记录有效 bootstrap 次数。

审查中发现的 receipt 键不一致已由 root 修复：实际键为 `official_metrics_sha256`。比较器现检查 completed 状态、指标原始字节 SHA、receipt 与 metrics 的 scoring/inference 声明、native family 拒绝、完整分母、非负忽略数、pooled CM 和顶层 IoU 一致性。没有修改 native 比较器。

## 固定候选与启动门槛

| 角色 | checkpoint | 模型自身像素协议 | 输出目录 |
|---|---|---|---|
| Reference | `runs/support_split_semantic_coupled/last.pt` | 缺历史字段按 `legacy_mixed_v1` | `runs/official_common_grid_v1/legacy_support` |
| Candidate | `runs/corner_v2_semantic_coupled/last.pt` | `colmap_corner_v2` | `runs/official_common_grid_v1/corner_v2` |

旧模型已完成 receipt 记录 SHA `a329fc70d582972fc338715765304b99da208c821ee34288809db500d5acd6b9`；正式开始前仍需重新核验输入文件。v2 SHA 刻意留空，必须待 RGB30k、semantic8k 自然完成、完成 receipt 生效且不再有写入该模型的进程后才绑定。不得使用中间模型替代最终模型，或依据局部图像选择 step。

预检须验证两份模型自己的训练 manifest、50 个相机仍属 VAL、原始相机与 source 路径相同，以及 checkpoint 已存 manifest SHA。v2 预期训练 manifest SHA 为 `91b41aeecedc352e4882eb80a1bb635d479ae432251a4c96c85a9dc110cbb327`。旧模型不能因 reference 或 v2 数据的 metadata 而升级协议。

两份模型不复制大权重，只读取原 `last.pt`，持续保留输入。正式外层执行 receipt 应补齐：稳定 checkpoint SHA、完成训练 receipt SHA、共同源清单、uv.lock、依赖版本与相同 LPIPS 权重来源；记录准确命令、开始/结束、退出码。若使用独立后台进程，PID 核验必须同时检查完整命令中的精确输出路径。失败产物保留，不覆盖已有非空输出。

## 磁盘与执行预算

固定尺寸均为 1320×989，每模型 65,274,000 像素。每组保存 **50 RGB PNG +50 ID mask PNG**，包括 9 张没有语义 GT 的预测 mask；评分语义仍只有 41 张。没有 probability NPZ、额外 checkpoint 或原 GT 副本。

两组 PNG 的保守规划为 545,944,700 字节（约 520.65 MiB）：按未压缩 RGB3+mask1 字节、行过滤开销、2% 压缩膨胀及每个 PNG 64 KiB 容器余量估计。这是容量预算而非已产生文件大小。总产物保留 600 MiB，覆盖快照、两个 receipt、metrics、日志及比较 JSON；执行前，在扣除所有仍在运行任务的未完成写入预留后至少应有 **1 GiB 可用空间**。已观察到 LPIPS AlexNet 权重缓存存在，预检需确认，不因缓存缺失默默新增未预算下载。不会为满足预算清理任何文件。

GPU 两组顺序执行，同一 evaluator / LPIPS 环境；不并行加载两份 scene。不把 CPU 测试耗时换算为 GPU 耗时。以下命令仅供授权后的执行，当前没有运行：

```bash
PYTHONPATH=/home/sky/workspace/SHM2026/runs/official_common_grid_v1/source_snapshot \
uv run --frozen python runs/official_common_grid_v1/scripts/evaluate_official.py \
  runs/support_split_semantic_coupled/last.pt \
  --output runs/official_common_grid_v1/legacy_support \
  --workspace-root /home/sky/workspace/SHM2026

PYTHONPATH=/home/sky/workspace/SHM2026/runs/official_common_grid_v1/source_snapshot \
uv run --frozen python runs/official_common_grid_v1/scripts/evaluate_official.py \
  runs/corner_v2_semantic_coupled/last.pt \
  --output runs/official_common_grid_v1/corner_v2 \
  --workspace-root /home/sky/workspace/SHM2026

PYTHONPATH=/home/sky/workspace/SHM2026/runs/official_common_grid_v1/source_snapshot \
uv run --frozen python runs/official_common_grid_v1/scripts/compare_official_evaluations.py \
  runs/official_common_grid_v1/legacy_support/official_metrics.json \
  runs/official_common_grid_v1/corner_v2/official_metrics.json \
  --output runs/official_common_grid_v1/comparison.json --repeats 5000
```

命令从项目根执行。每个模型先完整写完所有预测 PNG，之后才哈希/解码原始 RGB 和 annotation；不预读 GT 建立输入或选择预测参数。模型各自按 checkpoint profile 渲染/warp，soft semantic warp 后才 argmax；评分只读交付 uint8 PNG。每组完成后核对实际导入模块 SHA 与共同清单一致，再接受 completed receipt 和指标 SHA。

启动记录：第一次外层预检把XML中明确为`G`的Xorg/桌面/浏览器显示进程也当作计算任务，因而在加载checkpoint和启动scorer之前退出。失败回执、日志与当时runner原文保留在`attempts/01_display_process_guard`。随后仅修正外层占用检查：允许并记录明确的graphics-only进程，仍拒绝`C`、`C+G`及未知类型；18项runner回归和Ruff通过后重新启动。冻结的两组scorer、比较器和模型不受此修复影响。这里的运行隔离指没有其它GPU计算任务，桌面显示进程仍常驻，不能描述为没有任何GPU客户端。

## 固定外层 runner

`scripts/run_official_pair.py` 将以上顺序与门槛固化为一次执行，独立写入 `runs/official_common_grid_v1/pair_execution_receipt.json`，不修改 immutable `execution_plan.json` 或已有源码快照。当前只做 CPU fixtures 测试，**尚未执行真实 runner**。

它先读取训练完成 metadata：两个候选的 training receipt 必须 completed，v2 RGB30k 和 semantic8k 的 stage audit 必须 passed，semantic 审计须确认冻结 RGB/geometry。所有这些条件通过前，不哈希或加载任何 checkpoint。现有 `run_experiment.py` 仅在 train 与 native+LPIPS 子进程都成功返回后写 completed，因此不要求历史回执中不存在的 `natural_exit` 字段。

随后核验模型、训练 manifest、训练 source 与 stage audit、评价源码树、两个入口、uv.lock、既有 AlexNet ImageNet 与 LPIPS Alex v0.1 两个实际本地权重文件。NVIDIA XML 进程表必须可用且无进程（包括计算及图形进程）；不允许缺权重时自动下载。初始可用空间扣除显式 `--reserved-bytes` 后至少 1 GiB，第二组开始前仍须至少 556 MiB（300 MiB 剩余输出预算 +256 MiB 保底）。检查不提供跨进程排他锁，仍依赖 root 事先协调 GPU 交接。

每个子进程记录完整参数数组、工作目录、环境覆盖、PID、日志、退出码、起止时间。每组完成后核对 100 个 PNG 的名字/内容 SHA/尺寸/模式、8 个实际导入模块和入口来源、metric SHA/协议以及由源记录重建的共同 fingerprint；之后才把实际原图与 annotation 字节 SHA 纳入外层绑定。比较完成后重新哈希所有绑定输入，包括模型、源文件、训练回执、评分源码和实际权重。出现失败保留输出与 failed 回执，不覆盖非空产物，不实现自动恢复或清理。

root 明确交接 GPU 且两阶段自然完成后，可从项目根执行下式；`0` 仅在其它任务确实无未完成磁盘写入预留时使用：

```bash
uv run --frozen python scripts/run_official_pair.py --execute \
  --reserved-bytes 0 \
  --gpu-handoff-note 'Root explicitly authorized exclusive official-pair evaluation after v2 completion'
```

CPU 回归覆盖活动训练先拒绝且不读权重、两 stage 审计、GPU 进程/未知查询拒绝、磁盘保留计算、源码/入口篡改、非空输出/失败回执保留、双 LPIPS 文件缺失与 SHA、100 个合成 PNG 交付和 GT 时序/来源检查。测试使用临时文件和 mocked GPU 查询，没有运行 `nvidia-smi` 或加载真实模型。

## 结论边界

两组只有在相同 `official_evaluation_fingerprint` 下才能比较。该指纹在各自全部预测完成后读取真实源数据才产生，不提前伪造；两组 profile 与训练 manifest SHA 仅在来源段记录。不要直接差分旧 native float 分数与新 original-grid uint8 分数。

这是项目定义的官方原始像素网格共同开发评分，不是主办方发布的评分代码，也不是同学未知 Dev30。即使 v2 获益，也只能先表述为这两条完整训练链在共同交付协议下的差别；二者训练源码与恢复历史不同，不能宣称是像素约定的严格单变量因果改进。配对区间只量化这些反复使用的开发视角，不覆盖跨种子、跨桥梁或盲测不确定性。
