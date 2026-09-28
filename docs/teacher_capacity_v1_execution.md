# Teacher capacity v1：固定四阶段执行与完整顺序审计

2026-09-26，root 审阅草案及 CPU 合同测试后授权顺序执行。原草案及四份配置不改写；本文件只记录已授权的执行合同。实验定义、采用门与局限见 `teacher_capacity_v1_plan.md`，结果不预设为正。

## 固定来源与入口

- 输出根：`/mnt/data/SHM2026/runs/teacher_capacity_v1`。
- 源码快照：该目录的 `source_snapshot`；完整 package 为 33 个 `.py` 文件，与草案逐文件相同。此计数不含两个独立入口、配置、锁文件或缓存。
- `launch_manifest.json` SHA256：`f1237a5a89d98c97b1c5044368e21879b948f0fd9292e2997822ef5f731be47a`。
- 整个快照相对路径→SHA 映射摘要：`c005384f5b79371e5f8a7f438a8d68b1b3a1cab7d8e7174a6c8d1391edfa992c`。
- package 映射摘要：`f6f6b6636445467378d5c74cd7d1d78cb6a91ca3d31846dbbff0d3291cd5d221`。
- stage 入口 SHA：`eb4d484ca1400359aca0791e89ac7c08da505c0b79e368f3a55d51abef007c07`。
- sequence 入口 SHA：`2d947eedf73aef77a32b84a5ec7a041a5f241ba0b371be91fa2f8b88b5a5aabb`。
- 未修改草案 SHA：`3ec4f27c3b14f9bebf4a608381cf04d20cf0155e0a91e0f21b345c027b4f42e5`。

唯一启动命令（已运行；不能再次运行覆盖）：

```bash
uv run --no-sync python -u /mnt/data/SHM2026/runs/teacher_capacity_v1/source_snapshot/execute_teacher_capacity_sequence.py --launch-manifest /mnt/data/SHM2026/runs/teacher_capacity_v1/launch_manifest.json --execute
```

执行顺序固定为 H+ real 6000 → H+ render_adapt 2000 → 7B real 6000 → 7B render_adapt 2000。每阶段为独立进程，前一阶段自然 exit 0、回执 completed 且步数符合之后才启动下一阶段。所有阶段使用同一 package/入口，配置读取快照内字节相同的副本；原配置的 SHA 也在前后核验。模型、manifest、训练 mask/validity、域图像来源均前后复核，teacher 实际加载的权重摘要须符合锁定来源。

每个阶段启动前拒绝活跃 compute、C+G 或未知 GPU 进程；允许本机既有明确 type=G 的桌面进程。缺失 GPU/process 状态会拒绝启动。四阶段期间 GPU 独占约定不允许插入其他计算任务。启动时数据盘空余约 784.3 GiB，门槛为 5 GiB；不删除现有数据。

## 完整 trace 的证据范围

包装现有 `teacher.train_teacher` 的调用，不复制训练 loop，不新增随机抽样。每个 `AdamW.step` 写入一行小 JSONL，含该步全部有序事件：

- `ViewReader.read` 的视图名、TRAIN split、RGB 路径、mask/valid 路径、域和尺寸；实际输入必须属于锁定 TRAIN 源。
- `aligned_crop` 或 `aligned_context_frame` 的输入/输出尺寸、协议、NumPy 状态前后摘要及实际 CPU image/label/valid 输出 SHA。输出 SHA 连同状态转换绑定具体裁剪、缩放和镜像结果，不只记录被抽中的视图。
- 每次 `photometric_augment` 的强度、形状，以及调用前后 Torch CPU 和本 GPU RNG 状态 SHA；监督及无标注 weak/strong 调用全部覆盖。
- optimizer 调用前后 RNG 摘要；逐步 flush，每 1000 步 fsync。

终评在包装 `evaluate_teacher` 的明确作用域内排除，不计入 TRAIN 轨迹；此排除只影响审计记录，不改变评价。没有中途 VAL：`eval_every == steps`。阶段末读取固定 `last.pt`，核验其配置、步数、完整 NumPy/CPU/CUDA RNG 与最后训练事件相同。不同容量 head 的参数形状不同，不要求学习结果逐位相同，也不把相同 RNG 当作 CUDA 训练数值确定性的证明。

7B 相应阶段完成后，比较器重新读取持久 JSONL 并计算全序列摘要，逐行比较两容量的完整事件，同时比较终态 NumPy/CPU/CUDA RNG。它还核验终点 checkpoint SHA，拒绝只信回执中宣称的摘要。输出 `real_rng_comparison.json` 和 `render_adapt_rng_comparison.json`。若训练 completed 但配对 trace 不符，训练结果仍保留；单独的配对审计 failed，队列停止，不能把它称作匹配成功。

## 中断与选择

每 1000 步仅滚动保存恢复用 `last.pt`。发生失败不自动重试、不改参数；队列停止并保留日志/回执。授权的技术恢复必须显式 `--resume`，配置全等、总步数不变、checkpoint 处于未完成阶段，且 RNG 等于持久 trace 前缀。每次恢复创建新 attempt：旧 trace 尾部只在重建轨迹时逻辑舍弃，不删除、截断或覆写旧记录。已完成阶段拒绝重跑。第一份 checkpoint 尚不存在时发生异常也不会自动重新抽样启动。

第二阶段只接本容量 completed 第一阶段固定 `last.pt` 的 EMA decoder；核验同源、同计划及 checkpoint SHA，重新初始化优化器。最后一次评价可能同时写 `best.pt` 别名，但所有后续选择只消费固定 `last.pt`，不按分数选 checkpoint。正式官方组合评价由 root 使用与本训练 package 字节一致的独立评价入口执行，不能将阶段内原图/渲染图评价混为最终共同官方网格结果。

## 已完成的 CPU 验证

84 项累计聚焦测试通过，Ruff 通过。核心测试实际执行 tiny teacher CPU loop，证明有/无包装的 decoder、EMA、Adam、NumPy/Torch 状态逐位相同；不同 backbone/head 初始化消耗下完整采样/增强轨迹一致。另覆盖 VAL 排除、异常后恢复原函数、旧 attempt 保留、checkpoint/trace 篡改拒绝、队列遇失败不重试，以及桌面 G/compute/unknown/不可用 GPU 状态门槛。CPU 测试不证明真实 CUDA 数值逐位确定，也不预言更大 encoder 的精度收益。
