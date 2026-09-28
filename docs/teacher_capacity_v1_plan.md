# H+ 与 ViT-7B 的固定容量对照草案

2026-09-26。**只完成 CPU 准备，没有启动正式训练。** 新的四份完整配置和输入审查位于 [draft plan](../configs/generated_teacher_capacity_v1/plan.json)，SHA 为 `3ec4f27c3b14f9bebf4a608381cf04d20cf0155e0a91e0f21b345c027b4f42e5`。本实验检验更大冻结编码器能否改善当前同场固定组合，属于工程容量对照，不作为论文创新证据。

## 四个固定阶段

| 骨干 | 阶段 1 配置 | 阶段 2 配置 |
|---|---|---|
| H+ | [hplus_real.json](../configs/generated_teacher_capacity_v1/hplus_real.json) | [hplus_render_adapt.json](../configs/generated_teacher_capacity_v1/hplus_render_adapt.json) |
| 7B | [vit7b_real.json](../configs/generated_teacher_capacity_v1/vit7b_real.json) | [vit7b_render_adapt.json](../configs/generated_teacher_capacity_v1/vit7b_render_adapt.json) |

所有 model/manifest/output/warmstart 路径均为绝对路径。输出固定为 `/mnt/data/SHM2026/runs/teacher_capacity_v1/{hplus,vit7b}/{real,render_adapt}`。没有创建这些训练输出或改动旧实验。H+ 骨干仍用工作区的原 ModelScope H+，7B 用新挂载盘的已校验 7B；不复制两份大权重。

| 共同参数 | real | render_adapt |
|---|---:|---:|
| steps / 末次 eval | 6000 / 6000 | 2000 / 2000 |
| recovery checkpoint_every | 1000 | 1000 |
| seed | 20260926 | 20260926 |
| crop / decoder channels | 768 / 192 | 768 / 192 |
| lr / warmup | 1e-4 / 200 | 3e-5 / 100 |
| weight decay / gradient clip | .01 / 1 | .01 / 1 |
| EMA decay | .995 | .99 |
| context 概率 / 开始步 | .5 / 3000 | .5 / 1 |
| context short side | 768 | 768 |
| consistency 开始步 / 权重 | 1000 / .5 | 1000000 / .5（未启用） |
| real/rendered 混合 | real | 严格 1000/1000 步 |
| adapter_rank | 0 | 0 |
| independent_augmentation_rng | true | true |

除上述固定周期/独立增强 RNG/新输出与阶段衔接外，配置取自现有 `teacher_strong_v1.json` 和 `teacher_render_adapt_v1.json`。所有 dataclass 默认值也显式展开，避免两骨干意外依赖不同默认值。

阶段 1 各自创建新头。阶段 2 只读取**本骨干阶段 1 恰好 6000 步的 `last.pt` 中 `ema_decoder`**，将其载入 online 与 EMA decoder，重新创建 Adam、步数与随机数。原代码同时复制 class thresholds，但阶段 2 没有一致性损失，该阈值不参与监督优化。H+ 头不迁移到 7B。最终仅使用阶段 2 恰好 2000 步 `last.pt` 的 EMA 头。

现有训练器在唯一一次末尾评价时仍会写 `best.pt`，其模型来自同一个终点；方案**不读取这个文件、不按分数择优**。每 1000 步只是滚动 `last.pt` 恢复点，不读取验证数据。若中断，恢复必须复用原完整配置和总步数，不从中途恢复点另起学习率日程；现有 trainer 的 resume 校验并未穷尽所有超参数，正式 runner 必须额外要求配置精确一致。

## 实际数据及协议审查

原 manifest 为 `/home/sky/workspace/SHM2026/artifacts/prepared/manifest.json`，SHA `551546979a583d46e840bd485559721f361bc28fa4dca60826374ceb74b315fa`。
阶段 2 manifest 为 `/home/sky/workspace/SHM2026/runs/teacher_render_adapt_v1/manifest.json`，SHA `946987a7c16455182ebb1b92507afee212460423fe26ad0e29d5d0e4c24dd565`。
二者都属于 **legacy_mixed_v1**，没有升级坐标协议。

已实际通过 `validate_image_sources(..., verify_files=True)` 与 `verify_teacher_render_protocol`：

- 相同 350 TRAIN / 50 VAL，其中 TRAIN 有标签 259、无标签 91，VAL 有标签 41；名字、相机、K/pose、mask/valid 路径和其它非 RGB 元数据一致。
- TRAIN real/rendered 与 VAL rendered RGB 的来源回执、文件 SHA、尺寸、renderer checkpoint SHA 均通过核验。标签没有替换为渲染 mask。
- 阶段 2 的 TRAIN 具有 real/rendered 两种输入，VAL 输入只能是 rendered。训练 renderer 固定为 `runs/strong_rgb/last.pt`，SHA `3f57584a785110a70a2b4b52a4b73105b23448b223ca398b1ca160a0c91e1edb`。
- 本次只读审查未解码 VAL 标签/照片；另绑定了 259 张 TRAIN 标签和共享 valid 文件 SHA。完整名单与来源均在 draft plan 中。

教师沿用原来的**带放回随机取 view**，不是学生训练的 350-view shuffle。阶段 1 每步监督取一张 259 标签图；第 1000 至 6000 步共 5001 次再从 91 无标签图取样并执行 weak/strong EMA 一致性。阶段 2 只监督 259 标签图，虽 manifest 保留全部 350 TRAIN，91 无标签图不参加该阶段损失。50:50 是 2000 步中恰好 1000 real / 1000 rendered，不保证每个相机都各占一半。

阶段 2 域计划与 view/crop 的 NumPy RNG 独立，原 seed 固定 20260926；域序列 SHA 为 `48e70f123eb30e475fbe6a7c8c7c797823606c8583869e124cc844bd0aad69e5`。

## 随机数和公平性边界

两骨干使用同一 NumPy `default_rng(seed)`、同样的标签/valid/尺寸、同样的分支开关，因而 view、缩放、类别引导 crop、水平翻转和无标签取样的随机数消费应一致。不同 RGB 数值不参与 crop 选择。`independent_augmentation_rng=True` 的现有实现是在模型构造/warmstart 之后重新设定 torch CPU/CUDA seed 为 `seed+1`，从而消除 7B/H+ 构造消耗随机数不同的影响；它不是另建一个专用 Generator。当前冻结 encoder 为 eval、attention/drop path 为零、decoder 无 dropout，训练模型前向本身不再消费随机数。

同一 seed 不意味着不同 shape 的头参数逐位相同。容量处理同时改变 encoder hidden/layers，以及四个 DPT 输入投影的参数数目（6,376,326 → 8,539,014）；其余 DPT 结构和训练预算相同。比较不是“只增加 encoder 参数而 head 参数完全不变”。

正式 runner 应记录完整有序的 supervised/unlabeled view、domain、context/crop RNG 转移摘要及 photometric 抽样摘要，比较每阶段两臂末尾 NumPy/torch/CUDA RNG 状态。每 100 步的普通 metrics 日志不能单独证明 6000/2000 步的全部样本顺序；这项审计要求已列为启动门，不能提前声称已对正式轨迹验证。

## 固定原图评价和采用门

两臂最终共用 `runs/h3_moments/02_cross/last.pt`，SHA **`22bc8a2ddb260f93cb01b17857c97b2bb0873038efdb9318545cb2bdbb045226`**。相机先渲染同一 RGB/学生概率，教师只接收该场 RGB 的 uint8 量化图；采用固定 tile768/stride512/flip/context0.25/short-side768，教师和学生概率各 0.5。在 pinhole overscan 画布融合后做官方畸变 soft probability warp，再 argmax；绝不把真实 VAL RGB 或标签送进教师。

训练 rendered 图来自旧 strong 场，推理来自 H3 cross 场；这是双方共享、与现有参考一致的固定 renderer 域转移，不是偷偷替换训练图。阶段 1 末尾 real-input 教师成绩、阶段 2 末尾 strong-rendered 教师成绩都只是诊断；正式采用只依据下述同一官方原图组合。

固定 `official_original_grid_v1`、50 RGB / 41 标签，fingerprint 必须是 `21a2f19c5d4d703403a0107402d5dd5e006dcd98e5f8e023820c388c3e69a85d`。输出到各自 `official_final`，不能与 native 指纹差分。

额外强参考为 [/mnt/data/SHM2026/runs/official_selected_ensemble_v1/cross_teacher/official_metrics.json](/mnt/data/SHM2026/runs/official_selected_ensemble_v1/cross_teacher/official_metrics.json)：all5 **95.109016%**、FG **94.036190%**、cable **94.786655%**。其 completed receipt 与 metrics SHA 已绑定。旧教师经历了额外 continuation 与历史 checkpoint 选择，故本次新 H+ 不是旧最佳教师的完整复现。

所有三组比较都报告：7B−matched H+、7B−已选旧组合、matched H+−已选旧组合。配对 bootstrap 固定 5000 次、seed20260926。采用 7B 必须同时满足：

1. 相对 matched H+ **和**已选旧组合，all5 增益均 **≥0.30 pp**，且各自配对 95% 区间下界 **>0**。
2. 相对这两项参考，FG 和 cable 点估计均不下降。
3. 全部 50 张 RGB PNG 与同一 scene 参考逐字节一致。

不达门也报告所有固定结果，保留现有已选组合，不挑权重/层/步数或改 seed 补救。这是该容量工程选项的采用门，不等于已经超过同学 Dev30，也不是独立测试集或多场景泛化证据。

## 预算与当前准备状态

独立预检已由另一任务完成，绑定在 draft 中：`/mnt/data/SHM2026/preflight/dinov3_capacity_v1`。其输入是 corner_v2 TRAIN002，仅测容量/梯度/耗时，不用于正式精度结论。7B 两个监督步骤的 crop/context 耗时分别 1.048/0.517 秒，peak allocated **13.686/14.168 GiB**，完整 14 前向推理 **3.199 秒**；H+ 对应完整推理 **0.919 秒**。第一步含冷启动，不能直接以第一步外推整场，也不能从两个监督步骤保证半监督 consistency 分支的全部峰值。

据此暂规划 H+ 两阶段 **35–55 分钟**，7B 两阶段 **1.5–3.5 小时**，含末次评价和来源校验的余量；正式前若做半监督短预检，应只更新资源 ETA，不能按验证效果改实验。7B 预留 18–22 GiB 的运行预算，并发安排以实测 reserved/进程峰值为准。两骨干可以顺序运行，正式任务尚未启动。

不保存骨干副本。四阶段的 rolling last 和末次 best 头状态合计约 1 GiB 量级，另有原子临时保存、两套官方 RGB/mask、日志与源码；**新预留 5 GiB** 即可覆盖，数据盘当前余量充分。所有正式阶段必须使用同一个完整 immutable source/runner，逐字节绑定模型、manifest、原标签、渲染 receipt 和 scene；草案只记录了当前源码 SHA，尚未伪装为最终启动锁定。

生成入口 [make_teacher_capacity_draft.py](../scripts/make_teacher_capacity_draft.py) 只生成配置/计划，不启动训练。5 项 CPU 测试覆盖两骨干超参数唯一性、绝对 warmstart 路径、固定末态、独立域序列及采用门所有方向/边界，Ruff 通过。生成器拒绝覆盖已有草案，后续审阅修改需明确版本化，不能静默改已锁定计划。
