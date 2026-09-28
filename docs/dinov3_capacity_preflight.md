# DINOv3 H+ / 7B 有界容量预检

本预检回答同一张 TRAIN 图像上的冻结骨干 BF16 推理、DPT192 两步监督更新是否可运行，以及其显存和耗时；不评价随机解码头精度，不训练正式教师，也不声称 7B 必然提高指标。两骨干顺序运行，各自独立进程、独立随机初始化解码头；不会同时驻留显存。

实现：`scripts/preflight_dinov3_capacity.py`。CPU 合同测试：`tests/test_dinov3_capacity_preflight.py`。全部计划、冻结源码、日志与回执放在 `/mnt/data/SHM2026/preflight/dinov3_capacity_v1`，不复制权重、不保存 checkpoint/概率，也不占用根分区的大容量缓存。

固定输入为 `artifacts/prepared_corner_v2/manifest.json` 中的有标注 TRAIN002，原生 989×1320。只解码这一张 RGB/GT/valid；其他视角及 VAL 不参与。两步分别采用中心 768×768 裁剪（x=276，y=110）与原比例全图上下文（768×1025，再向右补齐至 768×1040），后者明确使用 corner-v2 离散类别重采样，补齐区域忽略监督。使用既有监督损失、AdamW lr=1e-4、weight_decay=.01、clip=1、BF16 AMP；骨干冻结，decoder 参数保持 FP32，EMA 与 Adam 状态保留以反映训练进程内评价时的实际内存。

两步后使用同一个在线 decoder 执行完整 `predict_image`：tile768 / stride512 / flip / context.25 / context_short_side768，共 6 tiles×2 加 context×2，即 14 次骨干前向。统计包括拼接与 CPU 概率传输，不将整个流程称为一次 backbone forward；不按随机头的精度作选择。再尝试一次原生右下补齐 992×1328 的完整前向，此项只推理、不增加第三训练步；若 CUDA OOM，记录该可选项失败，不缩尺寸、不重试，主合同仍独立报告。

每步核验骨干 gradient 全为 None、decoder 有限且确实更新，记录输入/输出/hidden-state 的形状和 dtype。完整骨干 state 在加载后及最后计算字节 SHA，核验保持不变。每个进程执行前后都重新校验 ModelScope 文件、配置、分片 index、下载声明和实际源码 SHA；7B 必须先有 completed 下载回执且 index 的所有分片受同一声明约束，任何 `.part` 文件不能成为权重输入。H+ 保持兼容已有无 `status` 字段的历史下载回执，但仍逐文件验 SHA。冻结包的实际导入路径必须指向计划指定目录。

每骨干进程预算 600 秒，包含完整权重哈希、加载、内存字节校验、两步训练与推理。外部父进程监控并在预算到达后终止；监控采样/子进程查询的短延迟意味着这不是实时硬抢占保证。超时或主流程错误保留回执/日志并停止，不自动重跑。可选原生单次 OOM 单独记录。只允许独占 compute GPU（桌面 G 进程可存在）；父进程约每 0.5 秒采 NVIDIA 设备及当前进程显存，分别报告 torch allocated/reserved/peak，离散 NVIDIA 样本不能冒称连续物理峰值。

容量结果也不等于后续 6000 步收敛/准确率结论，不包含 consistency 或 LoRA 的训练峰值。H+ 与 7B 的 feature 维数不同，DPT 输入投影参数量也会随之变化，回执明确记录两边参数量，不宣称完全等参数模型。两个 optimizer step 不足以测可靠稳态训练吞吐；完整 14-forward 和可选 native 的耗时亦须注明当前冷/热状态与保留的 EMA/optimizer 分配。

准备与锁定为 CPU 操作：先 `uv run python scripts/preflight_dinov3_capacity.py --prepare`，在完成下载与源码稳定后 `--lock <draft_plan.json>`。运行必须使用该计划的 frozen runner、`PYTHONPATH=<source_snapshot>` 和明确的 GPU 执行授权；锁定后不读取当前主源码升级。已有计划/执行结果一律拒绝覆盖。

## 唯一一次实测结果

两 arm 均完成并自然退出 0，无 OOM、无重试、无超时。设备为 RTX 5090，驱动 580.173.02，设备总显存 32607 MiB。H+ 与 7B 的骨干梯度均全为 None、两步 decoder 都有效更新；骨干字节哈希以及所有权重/config/index/source 文件前后保持一致。session98139 完成后 compute 进程列表为空，GPU 已释放。

| 固定项目 | H+ | 7B |
|---|---:|---:|
| 冻结骨干参数 | 840,592,640 | 6,716,035,072 |
| DPT192 参数 | 6,376,326 | 8,539,014 |
| crop768 第一步耗时 / peak allocated | 0.616 s / 2.593 GiB | 1.048 s / 13.686 GiB |
| context768×1040 第二步耗时 / peak allocated | 0.270 s / 2.993 GiB | 0.517 s / 14.168 GiB |
| 完整 14-forward 推理 / peak allocated | 0.919 s / 2.153 GiB | 3.199 s / 13.992 GiB |
| native992×1328 单次前向 / peak allocated | 0.096 s / 2.311 GiB | 0.494 s / 14.703 GiB |
| 全流程最大 peak reserved | 3.891 GiB | 16.559 GiB |
| NVIDIA 当前 worker 离散采样最大显存 | 4596 MiB | 17568 MiB |
| NVIDIA 全设备离散采样最大显存 | 4952 MiB | 17924 MiB |
| 含加载与全部校验的 worker / 外壁总时长 | 8.735 / 10.219 s | 91.165 / 94.090 s |

这说明当前设备能够运行冻结 7B 加 DPT192 的主路径；完整既定推理约为 H+ 的 3.483 倍耗时。这里的 2 步结果不估计可靠稳态吞吐，更不能替代正式长训练精度。预检使用 corner-v2 仅为统一容量输入；正式 H+/7B 容量对照将依 root 计划使用 legacy 数据，以和旧最佳组合直接匹配。半监督 consistency 分支没有在本预检运行，需要继续保留显存余量。

锁定 plan SHA：`2d3ca9ef05416f29718144457c271e564dc4378d7528df07cf732a0631354217`；执行时 frozen runner SHA：`000a74ab1b99f6acc9f4f89511dac6df382f26d4203f891a1a4cc1b3d13571eb`；34 文件 source-tree canonical-JSON SHA：`5a3f432d2a2c65a0053c93281210cc1a2873fc3b74777e8a8e7d8e17f26b971d`。具体回执和采样位于同目录 `execution_receipt.json`、`{hplus,7b}/worker_receipt.json`、`{hplus,7b}/nvidia_samples.json`；独立 CPU 复核为 `posthoc_audit.json`，SHA `02dcb56f0dbddad5fc11dc8397a3be240d185399e613be004458e7739550f8ad`。

发现并保留一个日志缺陷：原 frozen runner 将 `inference.calls` 指向可复用 hook list，native 阶段清空并追加后，最终回执该列表只剩 native 一次的元数据。`forward_calls=14` 是 list 重用前的现场计数和断言；标量时长、显存快照、两步 metadata 均在 native 前记录，不受影响。不能把原 `inference.calls` 描述为完整 14 次明细。原 snapshot/plan/receipt 一律未修改、没有重跑；仅未来主脚本新增深复制，26 项 CPU 测试包含防止此回归的检查。另最后一步 `grads` 列表继续持有 decoder 梯度，推理峰值包含这些残留张量、EMA 和 Adam，较纯推理进程略保守。
