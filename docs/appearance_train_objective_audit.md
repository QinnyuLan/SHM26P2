# 外观精修的全部 TRAIN 实际目标审计

本检查只回答完成的 3000 步外观精修是否降低了它自身在全部 350 个 TRAIN 视角上的目标。已知原图验证集 RGB 退化，有限方向 CUDA 反传检查也已通过；二者仍不能替代实际训练目标的前后比较。本次不更新任何参数、不引入新候选、不恢复精修训练。

使用 `runs/raw_grid_appearance_v1` 原始锁定的源码和两个完成的最终 appearance delta。每个训练目标只比较原始 base 与它对应的 final：native-target 对 native final，original-target 对 original final，不做交叉目标测试。各 350 个视角，严格共 2×2×350=1400 次渲染。按原先 `training_order` 第一轮 350 个索引枚举，保持原始排序索引与 pose 对应；第一批名称为 152、255、237、075、102。

新脚本 `scripts/audit_appearance_train_objective.py` 直接导入原 frozen runner 的 `load_target`、`render_training_rgb`、`predict_arm` 和 frozen package 的 `appearance_rgb_loss`。这保留了原先 antialiased overscan、SH degree、预先 clamp(0,1)、native crop / continuous-float raw gather、RGB/255、native valid / raw 全图支持，以及完整 7×7 有效窗口 SSIM 的实现。MSE 另行在与 L1 相同的有效支持上计算。所有模型参数 `requires_grad=False`，只有只读前向，前后全场 tensor 哈希不变，delta 与 base 的非外观字段和相机必须精确一致。

每个视角保存 L1、1−SSIM7、0.8 L1+0.2 SSIM loss、MSE 及各自支持计数。主统计为全 350 视图等权均值及 final−base；辅助统计按原先 3000 步的视角采样次数加权，不额外渲染。四个目标分别判断方向，避免把组合 loss 改善误写成 MSE 也改善。报告改善/恶化视图数和差值分位数，不把这个已拟合 TRAIN 检查当独立泛化证据或因果解释。

输入 SHA 绑定原始 plan、completed execution/两 training receipt、base、两个 delta、manifest、uv.lock，以及全部 TRAIN RGB/source-RGB/native-valid 文件。像素读取由精确 allow-list 拦截：语义 GT 标签和任何 VAL 像素均不能打开；不会导出图片或大概率缓存。原实验 35 个冻结 Python 文件原样复制，仅加入新的只读 audit runner。执行验证实际模块 import 路径，不使用之后更新的 main package。

全部输出位于 `/mnt/data/SHM2026/runs/appearance_train_objective_audit`，上限 10 MiB、保留至少 256 MiB 空间。旧正式训练本身 3000 步分别耗时 82.64 / 71.34 秒；本次 1400 无反传前向粗估 45–90 秒，另有输入哈希与加载开销，计划预算 240 秒。循环有协作式 deadline，外部运行再使用 240 秒进程上限；超时失败保留，不改变协议重跑。

13 项 CPU 测试和 Ruff 通过，涵盖完整 TRAIN 集/原始首轮顺序/相机一致性、不可错位配对、两种聚合权重、有效区 MSE/SSIM、拒绝标签或 VAL 像素读取。锁定 plan：`/mnt/data/SHM2026/runs/appearance_train_objective_audit/plan.json`，SHA `69db4e995bcf90b723eef2011b9f4bdf34b1e64d066db4df6ed55e4c9926879d`；runner SHA `33cb83f12eeb855510156add9d430c0cd2d7eeeaf5b94e8dd25206f4de065423`；36 文件 source-tree canonical-JSON SHA `6073e18aa15c2c3450e520ca424c43ff1b5c60a8b807e4df306d916a22536b62`。

## 唯一执行的结果

session31466 / PID1186749 自然退出 0，1400 次渲染全部完成，耗时 40.428 秒，peak allocated 0.477 GiB、peak reserved 0.580 GiB；输出约 1.21 MB。全部 710 绑定输入和源码 SHA 保持一致，所有模型 tensor 字节不变。2100 次像素读取恰落在 701 个 TRAIN RGB/共享 valid 路径上，没有标签或 VAL 像素读取。退出后 GPU compute 列表为空，已向协调 agent 释放。

| 原目标 | 指标（越低越好） | Base TRAIN 均值 | Final TRAIN 均值 | 恶化视角 |
|---|---|---:|---:|---:|
| Native | L1 | 0.01433959 | 0.02221688 | 350/350 |
| Native | 1−SSIM7 | 0.08792999 | 0.11028134 | 350/350 |
| Native | 组合 loss | 0.02905767 | 0.03982978 | 350/350 |
| Native | MSE | 0.00074110 | 0.00134536 | 349/350 |
| Original | L1 | 0.01612021 | 0.02341143 | 350/350 |
| Original | 1−SSIM7 | 0.11905197 | 0.13965856 | 350/350 |
| Original | 组合 loss | 0.03670656 | 0.04666086 | 350/350 |
| Original | MSE | 0.00089465 | 0.00148919 | 350/350 |

结论是训练集自身目标也恶化，不能用“只有验证集泛化变坏”解释。原 3000 步采样次数加权后结论相同：组合 loss 的 final−base 分别为 +0.01076717、+0.00995079。这里比较各 arm 自己的前后变化；native 与 original 的目标和支持不同，不能横向比较其绝对 loss 大小来评优。本检查仍未识别造成恶化的具体原因。

完整逐视角结果 `report.json` SHA 为 `42a24e1501f397430e73842bc70c1e692ecfe4c287a2bd0d5eaa43965f47bcf4`。独立 CPU 重算均值、方向计数、源/输入以及日志定位记录在 `posthoc_audit.json`，SHA 为 `5f8a2d244993c42339c93ac49e4ab4671955224564754f8be96f10ea93bcf03b`；原 plan/report/receipt 未修改。

## 31 条已有训练日志与优化器状态定位

两臂各 31 条已记录日志（step1、100、…、3000）已按 view 名称逐条对齐到本次 base/final 重放，全部条目在 posthoc JSON。两边都是 28 条 logged loss 高于其对应视角的 base、2 条低于、1 条相等；这些是 31 条采样记录，不是对全部 3000 步的直接观测。

step1 的 TRAIN152，组合 loss、L1、1−SSIM7 三个数与 base 重放逐位相等：组合 loss 为 native 0.03350527957081795、original 0.04251156747341156。step3000 的 TRAIN043 在日志记录于 optimizer 更新前；最终 checkpoint 的同视角组合 loss 比该记录分别低 4.05684×10⁻⁶、3.70294×10⁻⁶。这个很小的最后一步差异与本次重放结果一致，不支持“重新加载或审计目标明显错位造成假恶化”的解释，但不等价于已经完成独立的一步更新实验。

原 RGB30k `runs/corner_v2_rgb_full/last.pt` SHA `7fac0df69f34cbb8ecefdd32c34d1ffd4ae8b7f41f54293c75056effaa36951f` 已按其 completed receipt 与原源码快照核验；它的三个外观 tensor 与语义8k base 逐位相同。

| 参数组 | 原 RGB30k 的实际 LR | 新 fresh Adam LR | 原 / 新 eps |
|---|---:|---:|---:|
| SH0 | 0.0025 | 0.00025 | 1e−15 / 1e−15 |
| SHrest | 0.000125 | 0.0000125 | 1e−15 / 1e−15 |
| Background logits | 0.001 | 0.0001 | 1e−8 / 1e−15 |

原 SH/background LR 没有衰减至极低后被提高；新 LR 均为原终点的 0.1 倍。原训练循环指数衰减的是 means LR。原 RGB30k 的三组 Adam step 都为 30000，且一阶/二阶矩非零；语义8k warmstart 后颜色被冻结，对应新 optimizer 无这些状态。因此 fresh appearance Adam 确实丢失原历史，background eps 也发生变化。这些是实现差异，还不是导致失败的证据。

## 下一项最小判断

固定原首个 TRAIN152、原 base、两种原目标，不加其他相机或候选。对完整 masked L1/7-window SSIM 目标计算三个颜色组的解析梯度，分别沿其 absmax 归一化方向测固定 ε=1e−3、5e−4、2.5e−4 的中心差分，保留所有 18 项比较，不挑最有利的 ε。每次 ± 均从同一参数原值构造，恢复原值后才进入下一项；只使用原 frozen renderer、clamp、crop/warp 与 loss。记录 L1 零残差及 RGB clamp 命中比例，明确非光滑点、浮点分辨率可能使差分不具诊断性，不能自动将差异判为 CUDA bug。

然后各目标只用原配置 fresh Adam 做一个临时内存更新，比较同一 view 的更新前/后完整目标，并记录解析 g·Δθ。之后恢复模型全部颜色值并校验整体 tensor 与输入 SHA，不写 checkpoint、不选择 LR、不继续迭代。若完整目标导数与差分不符，先定位完整损失链；若导数一致但单步实际升高，关注有限步长/非光滑性或 optimizer 路径；若这两项都通过，仅排除该固定 view 上的明显一步异常，仍不能证明多视图累积过程正确或断言 Adam history 是根因。

按各目标 1 次带梯度 base、18 次方向探测、1 次更新后前向，预计总 40 次 render，约 10–20 秒含加载/校验，外部 60 秒上限。正式教师当前拥有 GPU；后续已获准完成 [CPU 实现与协议](appearance_actual_objective_diagnostic.md)，尚未启动该诊断 GPU 执行，不产生新模型候选。
