# 条件性全量 TRAIN 外观求解设计

已准备 helper、runner 和 CPU 合同；尚未冻结 fullbatch 实验计划，也未运行该求解器。旧 [40-render 完整目标诊断](appearance_actual_objective_diagnostic.md)的 18 项 FD 偏差为 61.05%–83.68%，当时未通过前置门槛，旧结论与固定回执保留。后续 [SSIM 布局定位、修复及独立新 40-render 证据](appearance_ssim_gradient_localization.md)已出现：两臂基线前向逐位相同，FD 相对差降至 0.00393%–2.50629%，一步实际/预测下降比为 0.998433 / 0.998300；有限 ε 的剩余差异仍保留不确定性。

root 当前优先保持原 Adam 配置的 native/original 两组各 3,000 步重放，以隔离这次梯度实现修复的影响；**fullbatch 仍未放行、不启动**。不改 helper/runner gate 自动接受新回执，不重标旧诊断为通过，后续是否生成新固定计划由独立解释和重放结果决定。本项是标准预条件梯度与 Armijo 回溯的工程应用，不作创新主张；局部导数改善不证明逐视图 Adam 是历史失败原因。

**输入与目标。** 固定 `runs/h3_moments/02_cross/last.pt`，SHA `22bc8a2ddb260f93cb01b17857c97b2bb0873038efdb9318545cb2bdbb045226`；只改 SH0、SHrest、background 三键，498,136 点、三键共 91.211 MiB。SH 阶数固定为 3，SH 系数仍属于可训练三键；几何、opacity、相机、语义及精修器冻结。缺失 profile 仍解释为 `legacy_mixed_v1`，不升级为 v2。按旧 manifest 的 350 TRAIN 名称固定排序；锁定计划必须保存 manifest 原 TRAIN 顺序的完整 names 列表，使用对应索引的 base `training_cameras`，记录与 original pose 的差异，绝不以 nominal pose 静默替换。原图 K、畸变与 legacy warp 使用既有官方原图协议，在其原始畸变网格上定义

\[
F(\theta)=\frac1{350}\sum_v[0.8\,\mathrm{L1}_v+0.2(1-\mathrm{SSIM}_{7,v})].
\]

L1 使用原图全部像素，SSIM 使用完整 7×7 窗口中心；不读语义标签、不加前景权重。按 **legacy profile** 的既有官方映射构建 overscan，先 clamp RGB，再固定浮点四邻 gather；不得沿用 v2 默认网格。与同 profile 官方 float RGB 的前向一致性须在实施前有合同。各视图等权，逐图 `loss/350.backward()` 后释放图；统计 loss 用固定顺序的 float64 标量求和，梯度与原损失计算仍为 FP32。

**必要简化：不用一阶动量。** 建议采用 Adam 型二阶矩预条件，但固定 β₁=0、β₂=.999、eps=1e−8，避免历史一阶动量方向不是下降方向及复杂回退分支。每次完整梯度 g 得到后，仅在临时变量中计算

\[
\tilde v=.999v+.001g^2,\quad \hat v=\tilde v/(1-.999^{t+1}),\quad
d_k=-s_k g_k/(\sqrt{\hat v_k}+10^{-8}),
\]

其中三组固定尺度 s=(2.5e−4, 1.25e−5, 1e−4)，t 只计已接受步。非零有限 g 在精确算术下满足 g·d<0；用 float64 点积检查，否则停止，不更换优化器或方向。该简化不是默认 β₁=.9 Adam，报告须明确。

**固定最多三次接受检查。** α 依次为 1、1/2、1/4，每次从同一个当前已接受 θ 构造 FP32 候选，不累计扰动。记录实际位移 Δ，并要求 g·Δ<0。完整流式重算全部 350 TRAIN，接受条件为 `Ftrial ≤ Fbase + 1e−4 (g·Δ)`，且实测下降至少 `max(1e−7, 1e−5*abs(Fbase))`；后者只是预定浮点保护，不是渲染误差的严格界。拒绝时恢复三键，v/t 完全不动。仅接受后同时提交候选、临时二阶矩与 t+1；三次均拒绝即终止，不降低步长继续搜索。不看 VAL、不挑中间点；每个 accepted baseline 的梯度遍历重新记录目标，若相对上一接受记录异常上升超过同保护量，停止并保留异常，而不宣称单调性已证实。

**预算。** 最多 20 个接受步、最多 40 次完整 350-view 遍历（14,000 renders），优化阶段 360 秒，每 view 前后检查到时；丢弃该次未完整遍历的梯度或候选，任一限制先到即停止于最后一次完整接受；通常不能完成 20 步。一次梯度遍历加最多三次只读回溯为 350–1,400 renders。按已测 1,400 次前向 40.4 秒和旧每步训练耗时，预计完整梯度遍历约 10–15 秒、前向遍历约 8–10 秒；这是旧 v2 场估算，legacy 场速度需记录，不能保证。提交/保存留独立余量。每 view 前后检查 deadline，因此可能超过阈值一个 view 的解码/渲染/反传耗时，而不等待整轮 350 图结束；partial pass 的实际 render 数单独记录、梯度清零、候选回退，不能参与接受。外层硬 timeout 需另留输入校验、finally/保存余量，不把协作式检查称为绝对硬上限。完整 20 步最坏 28,000 renders 会超过十分钟，因此不把 20 步当必须完成的终点。预计六分钟求解加约三分钟固定终评可接近十分钟；输入哈希准备另记。预算耗尽不得保留半次遍历产生的梯度或未验收更新。

不保留 350 个计算图，不建大渲染缓存。原 RGB 可逐张解码，最多可在 CPU 保存约 1.28 GiB 的 uint8 原图；GPU 只保留当前图与共享 warp。三键参数、梯度/梯度副本、当前备份、方向、已提交/临时二阶矩约 0.63 GiB，另保留进程初始全场副本与 double 点积临时量，连同场与渲染暂按 2–3 GiB 峰值、4 GiB 预留估算，不作为实测数字。预算正常结束先撤销未验收候选，再保存最后已接受终点；普通异常/非有限值则 finally 恢复进程初始 base，只留失败记录、不导出候选。硬杀仅保证原文件未改，不伪称内存恢复成功。零接受步不导出重复模型。

**来源、产物与终评。** 实施时绑定完整 base、旧 manifest、350 原 RGB、相机 names/index 映射、官方映射、依赖及源码 SHA；不覆盖旧文件。root 选择单次独立完整推理 checkpoint，不扩张 strict appearance delta 合同。保留原 base 字节及其缺失历史 `manifest_sha256`/profile 声明；缺失 profile 仍是 legacy。新产物的 `appearance_optimization` 命名空间分别记录 base 路径/SHA、当前观测的 manifest 路径/SHA、原 base 的声明值（可为缺失）、源码和新阶段配置，不把观测 SHA 补造为 base 原声明。

只保存一次约 188 MB 的完整场，模型只容许三颜色键改变，其余模型张量和 `training_cameras` 逐位不变。原架构、config、step 与原有协议字段照常保留；实际接受步数/目标/停止原因独立记录。剥离旧 optimizer、RNG、density state 等训练状态，标记 `fullbatch_appearance_inference_or_warmstart`，普通 train resume 明确拒绝；显式 warmstart 可用。原子文件发布不覆盖已存在路径，不保存周期候选；保存前检查完整 CPU state，保存后 CPU 重载审计。零接受步不导出重复模型，普通异常不发布新候选。

固定最终模型做共同原图 50 RGB/41 语义评价及配对区间；VAL 始终使用 original 相机，TRAIN 相机不移植到 VAL。比较未经更新的同一 H3 base，不能拼接另一场的 RGB。尽管语义参数冻结，refiner 和固定 H+ 组合均依赖渲染 RGB，必须分别重新终评，不能沿用旧 mask/95.109% 结果。教师固定为历史 selected H+（本轮 7B 容量对照未通过采用门槛），0.5 融合与推理协议沿用已有固定版本，不根据新 VAL 结果换教师/权重。raw 3D 语义应与 base 完全复现。TRAIN 目标接受只保证所记录浮点训练目标下降，不保证 MSE、VAL 或语义提升；终评不反向选择步数。

准备文件：`src/bridge_rgs/fullbatch_appearance.py` 提供预条件方向、事务性 rollback/accept、遍历预算及完整检查点保存；`scripts/run_fullbatch_appearance.py` 默认只输出拟定规格。未来 `--run` 必须来自新的不可变源码快照，并绑定 40-render 诊断 completed receipt 及 root 对完整差分/单步方向的明确 review；当前没有该实验 plan/snapshot，也没有 GPU 执行。不得用容差自动将差分失败解释为 CUDA bug，或绕过前置诊断。
