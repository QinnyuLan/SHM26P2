# H3 全量外观优化：数值预检与完整结果

2026-09-27。三个独立预检、固定 FP64 full-batch 优化、plain/H+ 官方终评及独立 CPU 审计均已完成。**RGB 三项均有小幅改善，但 PSNR 仅增加0.026024 dB，未达到预定0.15 dB；六门通过五项，不采用，当前 selected 不变。** 首次报告异常与 FP32 15/18 的失败记录完整保留。

本项是固定三组颜色参数的标准二阶矩预条件下降与 Armijo 回溯工程实验，不是新优化方法。FP64 损失链属于数值实现选择；局部梯度验证不等于新的重建机制或性能收益。[FP64 固定协议](/home/sky/workspace/SHM2026/docs/fullbatch_appearance_h3_fp64_protocol.md)与[历史 FP32 协议](/home/sky/workspace/SHM2026/docs/fullbatch_appearance_h3_protocol.md)均保持原样。

## 三次独立尝试

共同输入是 selected H3 `runs/h3_moments/02_cross/last.pt`，SHA `22bc8a2ddb260f93cb01b17857c97b2bb0873038efdb9318545cb2bdbb045226`，legacy manifest、相机和像素协议不升级。固定 TRAIN 名称排序索引 0、175，即 **002.png、205.png**；不是 VAL 的 175.png。每张图原定一次基线、三颜色组各三种 ε 的正负扰动、一次临时 RMS 候选，共20次 scene render；两图共40次。ε 固定 `.001/.0005/.00025`，所有比较使用实际 FP32 参数位移，没有事后换视图或调整门槛。

| 尝试 | 进程结果 / worker 时间 | 已完成调用 | 数值结果 |
|---|---|---:|---|
| 原报告入口 | exit 1 / 2.324 s | 3 render、1 backward | CUDA scalar 被直接传入 NumPy，报告转换异常；未保存完整 FD/RMS 结果，不能判技术门通过或失败。 |
| 仅修复 host-scalar 转换 | exit 0 / 2.873 s | 40 render、2 backward | **15/18 FD 通过**；002 背景组的三个 ε 未过5%门。两 RMS 和 warp 门通过，但不能豁免 FD 门；原 FP32 full-batch 不启动。 |
| 独立 FP64 损失链 | exit 0 / 2.926 s | 40 render、2 backward | **18/18 FD 通过**，相对误差0.000650%–0.050940%；两 RMS、warp、恢复和来源门均通过，仅允许另行审批正式目标检查。 |

首次失败保留原源码、日志和回执。标量修复只先执行 `float(base/plus/minus)`，数学目标及门槛不变；新增 CPU 回归用禁止 NumPy `__array__`、允许 `__float__` 的标量覆盖该错误。第二次失败也原样保留，FP64 是新的明确协议，不是把旧结果改标通过。

FP64 尝试保留参数、原始目标的 FP32/255 数值、renderer、clamp 和固定浮点 gather 为 FP32，只在既有 `appearance_rgb_loss` 前将 prediction/target 转 double，完整 L1、7窗口 SSIM 和标量目标采用 FP64。原 contiguous SSIM 修复及 RMS helper 没有改变；新 wrapper 被 source SHA 和实际导入路径绑定。两图基线 RGB tensor SHA 与前轮 FP32 探针完全相同。

FP64 两图 RMS 实际/预测下降比为 **0.996925、0.997406**；实际下降分别 `3.93124e−5`、`4.37615e−5`，均通过 helper 的 `max(1e−7,1e−5|F|)` floor 与 Armijo 条件。官方 OpenCV float warp 对 gather 的最大差均为 `1.78814e−7`，低于固定 `2e−6`。所有临时候选无条件回退，零提交步、零导出检查点。133项模型 tensor 的初值在 CPU 独立匹配原 H3 checkpoint，最终哈希、requires_grad、module modes、相机及输入/source SHA 恢复核验通过。

## 精度证据与解释边界

002 背景组中，记录的参数方向导数与图像梯度乘预测差分分别只差 **0.014891%、0.014632%、0.029802%**，但标量 FD 相对参数导数差 **5.677750%、9.705178%、8.696842%**。这将该记录中的主要不一致定位到“图像→标量目标”的有限差分比较，而非支持背景 raster VJP 错误的指控；有限步长、损失非线性与浮点计算仍须区别。

固定 CPU 合成例子为64×80明亮平滑RGB，`z=.7+.01x+.006y`、三通道 `[z,z+.015,z−.01]`，target=`base+.01+.001sin(9x)`，方向全 `.01`，全部 valid、完整7窗口、相同三个 ε。端点始终先在 FP32 构造，再送入各自 FP32/FP64 目标，导数与实际端点方向比较。FP32 FD 相对差为 **11.6004% / 5.2767% / 1.8142%**；同公式 FP64 为 **3.40e−11 / 8.16e−12 / 1.48e−12**。原 `E[x²]−E[x]²` 完整窗口方差在 FP32 有1,335个负项、最小 `−2.98e−7`；FP64没有负项、最小约 `1.00368e−7`。这支持对消减敏感的损失计算进行独立精度校准，**不单独证明实际 H3 失败的全部原因**，也没有修改生产 loss 公式。

独立 CPU review 从保存的标量重算 central/one-sided FD、实际导数相对差、对应dtype的ULP、RMS floor/Armijo/ratio，并验证来源及恢复哈希。原始梯度、临时参数数组和预测数组没有持久化，因此 `g·Δ` 与 warp 差仍属于绑定回执中的运行证据，不能声称 CPU 独立重新渲染或重算了全部Jacobian。

## 正式实验与共同原图评价

正式输出为 [h3_fullbatch_appearance_fp64_v1](/mnt/data/SHM2026/runs/h3_fullbatch_appearance_fp64_v1)。固定360秒优化窗口内达到预设20接受步上限，自然 exit0；40次完整350 TRAIN遍历、14,000次 scene render、零部分遍历，全部候选 α=1 被接受。只使用 TRAIN 原始 RGB、三颜色键、β₁=0/β₂=.999、eps=`1e−8`及固定三组rates；没有语义标签或 VAL 参与优化，没有追加步数或搜索超参。

TRAIN 等权完整目标由 **0.036853451969658765** 降至 **0.036680831786791435**。优化耗时 **336.694秒**，外层进程 **340.215秒**；峰值allocated **1,646,458,880 bytes（1.533 GiB）**。终点 CPU 审计核实仅 `splats.sh0`、`splats.sh_rest`、`background_logits` 改变；几何、opacity、语义/精修参数及350原始相机逐张量不变。checkpoint SHA 为 `b2275eb37c7cb15eab8027a265f2307c6782efb45a5ae981b4e378e3a65bf561`，仅作本实验候选，不替换生产场。

最终 fresh plain 与固定历史 H+ `.5` 都重新执行同一官方50 RGB / 41标签评价，mask 均重新预测、没有复用旧 mask。两个新分支的50张 RGB PNG逐字节相同；语义模型参数冻结不等于输出必然不变，因为refiner和教师依赖渲染RGB。主采用比较仍是固定H+组合与旧selected，指纹相同，5000次相机配对bootstrap、seed20260926：

| 指标 | 旧 selected H+ | 新颜色场＋同 H+ | 新−旧，95%配对区间 |
|---|---:|---:|---:|
| PSNR ↑ / dB | 29.40094970 | 29.42697404 | +.02602434 [.01996393,.03244542] |
| SSIM ↑ | .870649924 | .870801982 | +.000152057 [.000133417,.000171306] |
| LPIPS ↓ | .271236874 | .271149900 | −.000086974 [−.000145325,−.000026784] |
| 五类 mIoU / % | 95.10901575 | 95.10608516 | −.00293060pp [−.00583461,+.00007429] |
| 前景 mIoU / % | 94.03618980 | 94.03252102 | −.00366879pp [−.00721483,+.00003447] |
| 拉索 IoU / % | 94.78665499 | 94.78533651 | −.00131848pp [−.00561713,+.00345611] |

三项 RGB 区间均指向小幅改善，不能把本轮概括成“没有RGB收益”。但 **PSNR点估计增益至少0.15 dB** 一项失败；PSNR区间下界为正、SSIM不降、LPIPS不增、五类及拉索各下降不超过.20pp的其余五项通过。六项必须同时成立，所以本轮不采用，不追加颜色步数或调整门槛补救。TRAIN下降和两图局部数值门通过均不豁免系统采用门。plain分支仅作描述：五类94.57818366%、前景93.39517210%、拉索93.83302181%，RGB与新H+分支一致，不据此改选采用分支。

独立终评CPU审计核对新旧plain/H+四组共200张RGB及200张mask的来源，逐PNG重算PSNR，并为四组各41个GT polygon重新栅格化、重算混淆矩阵且完全一致；两个5000次配对区间与六门也独立一致。SSIM/LPIPS沿用绑定的正式评分，没有声称重新计算。这是标准优化与数值精度稳定化的工程结果，不是新机制。FP64消除了这两个固定视角中已测的差分不一致，不能泛化成“FP32解析梯度错误”或其他场景必有增益。

[配对结果图（PNG）](/mnt/data/SHM2026/runs/h3_fullbatch_appearance_fp64_v1/execution/figures/teacher_vs_selected.png)与[PDF](/mnt/data/SHM2026/runs/h3_fullbatch_appearance_fp64_v1/execution/figures/teacher_vs_selected.pdf)对应上表固定H+主比较。

当前选中系统的四项记录保持不变：

| 字段 | 当前 selected，尚未替换 |
|---|---|
| 三维场 | `runs/h3_moments/02_cross/last.pt`，SHA `22bc8a2d…45226`。 |
| 教师 | 历史 `runs/teacher_render_adapt_v1/best.pt` DINOv3 H+。 |
| 组合与协议 | 概率固定 `.5/.5`，`official_original_grid_v1`；fingerprint `21a2f19c5d4d703403a0107402d5dd5e006dcd98e5f8e023820c388c3e69a85d`。 |
| 已选成绩 | RGB **29.400950 dB / .870650 / .271237**；五类 **95.109016%**、前景 **94.036190%**、拉索 **94.786655%**。这是保留的旧 selected 成绩，不是本次候选结果。 |

本轮预定主采用比较为新场+固定H+与上述selected：PSNR≥+.15dB且配对CI下界>0、SSIM不降、LPIPS不增、五类与拉索各下降不超过.20pp。任一门失败保持selected，不挑另一分支、步数或融合权重补救。开发集配对区间不覆盖多轮选择、重训种子或新桥梁泛化；不能称公平超过同学Dev30或学术创新已证明。

## 后续容量观察：独立问题

有限只读核查显示，本项目已完成完整RGB端点均不超过500k：hybrid498,136、mixed499,821、MCMC500,000；MCMC在第4,800步就达到cap。既有协议将500k作为共同工程预算，未找到用户要求、竞赛规则或已证实硬件瓶颈强制这一上限。初始600k配置存在，但没有对应已完成高容量训练证据。同学已完成RGB130/140/148分别652,298／790,065／991,322点；其中近百万RGB148为30k全400训练视角拟合，不可与我们的留出指标直接比较。[RGB148原记录](/home/sky/workspace/SHM2026/project_progress_20260918/sources/RGB-148-STANDARD-GT-REGION-RGB.html)

这支持把约1M容量作为下一轮独立标准参考的性能问题；目前只是CPU设计，不能预报收益或算学术创新。更多点会改变容量轨迹、运行和后续语义成本，也可能拟合浮层；同cap不等于同算力。**这一观察不改变本次20步终点、六项门或未采用结论。**

## 审计入口

- 首次异常：[执行回执](/mnt/data/SHM2026/runs/h3_fullbatch_objective_precheck_v1/execution_receipt.json)、[独立 CPU review](/mnt/data/SHM2026/runs/h3_fullbatch_objective_precheck_v1/independent_cpu_review.json)，review SHA `a4df7a40feb5877d98735ea1e03363efab226e9f0d370e592045b55e94b0b196`。
- FP32 15/18：[执行回执](/mnt/data/SHM2026/runs/h3_fullbatch_objective_precheck_scalar_fixed_v1/execution_receipt.json)、[独立 CPU review](/mnt/data/SHM2026/runs/h3_fullbatch_objective_precheck_scalar_fixed_v1/independent_cpu_review.json)，review SHA `3057b2c6155ba231aa9b04847c9da47857320d613f499aa81af08532f03ddbd1`。
- FP64 18/18：[执行回执](/mnt/data/SHM2026/runs/h3_fullbatch_objective_fp64_precheck_v1/execution_receipt.json)，SHA `6d4a71864ba95d89064fb7f51c90ff37fb4f8fc6c667b211a1f9df7b56686947`；[独立 CPU review](/mnt/data/SHM2026/runs/h3_fullbatch_objective_fp64_precheck_v1/independent_cpu_review.json)，SHA `7bf344e9460a44b167f93e2944670b8abf2dc817e52e8a85b49c8eee772e553b`。
- [固定合成CPU诊断](/mnt/data/SHM2026/runs/h3_fullbatch_loss_precision_cpu_v1/report.json)，SHA `40345f625536854e04e6db8968ce51728365a2806d9c499c4d8f27278b97b194`。
- [当前 selected 官方指标](/mnt/data/SHM2026/runs/official_selected_ensemble_v1/cross_teacher/official_metrics.json)；[selected 结果口径](/home/sky/workspace/SHM2026/docs/selected_ensemble_official_results.md)。

- 正式[训练回执](/mnt/data/SHM2026/runs/h3_fullbatch_appearance_fp64_v1/execution/execution_receipt.json)与[终点CPU审计](/mnt/data/SHM2026/runs/h3_fullbatch_appearance_fp64_v1/execution/cpu_endpoint_audit.json)，审计SHA `abaa890e6a04163cca33dab065fb5bee73836e25e92507da8e9debf27f1714cd`。
- [系统采用门与全部指标](/mnt/data/SHM2026/runs/h3_fullbatch_appearance_fp64_v1/execution/system_adoption_gate.json)，SHA `aaaaa6a57dc6c3d61b8ec99a429ac8f539853d375e638d90b5ef56a95afd1b97`；[H+配对](/mnt/data/SHM2026/runs/h3_fullbatch_appearance_fp64_v1/execution/paired_official_teacher.json)、[plain配对](/mnt/data/SHM2026/runs/h3_fullbatch_appearance_fp64_v1/execution/paired_official_plain.json)。
- [独立官方CPU核验](/mnt/data/SHM2026/runs/h3_fullbatch_appearance_fp64_v1/execution/independent_official_cpu_audit.json)，状态passed，SHA `f24a06e79a01dc3a567072ba280398d93758bb9308dc14c28b8f2453129f9c50`。
