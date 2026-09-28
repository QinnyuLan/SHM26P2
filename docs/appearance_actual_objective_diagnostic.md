# 完整外观目标的方向与一步更新诊断

状态：固定 40-render 诊断已完成，进程自然 exit 0，未重试。两臂单步同视角 loss 均下降，但全部 18 项差分存在明显幅度偏差，不能记作 FD 通过；条件性 fullbatch 求解器尚未放行。此项不是新的精修候选或学习率搜索。

触发证据是已完成的全 350 TRAIN 重放：native 和 original 两个原目标的组合 loss 均在 350/350 视角恶化。该结果与各自第一条日志的 base 重放逐位一致，排除颜色 LR 从原终点低值被提高的说法；fresh Adam 历史丢失与 background eps 改变属于待解释差异。此前平滑 RGB 线性探针的 CUDA 方向检查通过，但没有检验完整 L1/SSIM 目标。

本项固定原始第一训练视角 TRAIN152、原 base、原始两种目标。沿用 `runs/raw_grid_appearance_v1/source_snapshot` 原封不动的 35 个 Python 文件，复用其 target reader、overscan renderer、clamp、native crop / original gather 和 `appearance_rgb_loss`。只增加独立 `scripts/audit_appearance_actual_objective.py`。不读取语义 mask 或任何 VAL 像素，只允许该视角 prepared RGB、source RGB、native valid 三条路径，恰好三次像素读取。

每个目标执行相同顺序：

1. 一次原 FP32 完整目标前向/反向，核对 loss 与原 step1 日志逐位相等，仅 SH0、SHrest、background 获得梯度。
2. 对三个颜色组分别采用其完整目标梯度除以 absmax 的方向。固定 ε=1e−3、5e−4、2.5e−4，从同一原值构造每次正/负扰动，再立即恢复。每目标 9 项中心差分、18 次前向；两目标共 18 项比较，所有 ε 都保留。
3. 从恢复后的同一 base、同一解析梯度构造原 fresh Adam，只做一次临时更新，再对同一 view/目标前向。三组 LR 为 .00025、.0000125、.0001，eps 均为 1e−15、betas=(.9,.999)、无 weight decay；这是原 appearance 配置，不能换成早先 RGB30k 的 background eps。

总数严格为 2×(1+18+1)=40 次渲染、两个临时 optimizer step、零保留候选。每组记录实际 lr/eps/betas/step、参数位移 min/max/absmax/RMS/零位移比例、解析 g·Δθ；记录同 view 完整 loss 实际变化及相对线性预测的余项。不会追加别的视角、优化步数或学习率。

差分继续使用原 FP32 loss，转换为 Python float 做标量减法不改变其原始精度。报告中心/正向/反向斜率、与解析梯度的差异、实际 FP32 参数位移的线性预测、标量 ULP 分辨率，以及 L1 精确零残差、接近零残差和渲染 RGB clamp 命中比例。后者不能覆盖内部 SH clamp 或 SSIM 的全部非线性。数值差异只作为描述，不输出“容差 fail 即 CUDA bug”的判断；若量化/非光滑性明显，保留不确定性。

方向函数、一步 Adam 函数各自用 `finally` 恢复修改过的参数；外层另保存整个 scene state，在 `finally` 恢复全部 parameter/buffer 值与原 requires_grad 标志，清空梯度并逐字节核验。两个目标之间也恢复和核验全模型，防止第一目标影响第二目标。任何正常异常或内部超时均尝试恢复；外部 SIGKILL 不能执行 finally，但只销毁这个独立内存进程，原 checkpoint 从不写入，不能在该情况下伪称已完成恢复。

建议窗口 10–20 秒含加载/校验；内部 55 秒 alarm/循环检查，外部固定 60 秒进程上限。不重试、不缩减协议；源码/输入/plan SHA 执行前后校验。输出位于 `/mnt/data/SHM2026/runs/appearance_actual_objective_diagnostic`，包括完整数字与恢复回执，固定上限 2 MiB，无 checkpoint、图片或概率缓存。

CPU 测试 `tests/test_appearance_actual_objective_audit.py`：13 项通过，Ruff 通过。覆盖固定 18/40 次数、梯度方向、FP32 分辨率而非机械判决、所有 ε 从原值构造、失败/超时恢复、严格一个 fresh Adam step及位移记录、全 state/buffer 恢复、原 first-view 绑定。

冻结计划为 `/mnt/data/SHM2026/runs/appearance_actual_objective_diagnostic/plan.json`；plan SHA 为 `c8db5d457049844b346727ebf83bfc2ac8d65971c0903d7feb33d94cb675b5d3`，runner SHA 为 `06f1b918bdeaf70094d71cd7662bbd10c4e31befbe67f72b7e0bcef030d4fc9e`。36 文件源码树的排序紧凑 JSON SHA 为 `4583345a6cf9b051ac599a4c472cdb15d67815bca0a74df06a6fdbe71465fc54`，绑定 12 项输入。`cpu_preparation_receipt.json` 记录测试文件和 JUnit SHA；原固定计划及源码未修改。实际 [执行回执](/mnt/data/SHM2026/runs/appearance_actual_objective_diagnostic/execution_receipt.json) SHA 为 `149fe9327d288db2d21c18e44c540e61f50736f397d1d1eb6e2a64fbe26208ce`。

解释边界：即使完整目标差分与一步同 view 下降都正常，也仅排除该固定视角上的明显局部异常，不能证明多视图累积过程正确。若一步升高，也需要区分有限步长/非光滑性/optimizer 路径，不能直接将 fresh moments 或某个实现细节定为根因。

## 实际结果与当前判断

恰好完成 40 次 render、18 项中心差分和 2 个临时 Adam step，用时 3.258 秒；三条 TRAIN152 RGB/valid 路径各读一次，未读语义标签/VAL。两个 baseline 均逐位复现原 step1 日志，全部模型张量最终精确恢复，来源/输入 SHA 前后不变，未保存 checkpoint。

| 原目标 | 更新前 loss | 一步后 loss | 实际变化 | 解析 g·实际位移 |
|---|---:|---:|---:|---:|
| 00_native | 0.03350527957 | 0.03347969428 | -2.55852938e-05 | -9.34649961e-05 |
| 01_original | 0.04251156747 | 0.04248578101 | -2.57864594e-05 | -1.07989451e-04 |

两臂 L1、SSIM-loss 分项也都下降；实际总下降只有线性预测的 27.37% / 23.88%。这支持该固定视角上的一步下降，不能代替导数核验。

下表保留全部三个预定 ε（1e−3、5e−4、2.5e−4）的 FD 范围，完整每项数字见执行回执，没有挑选“最佳 ε”。

| 原目标 / 参数组 | 解析方向导数 | 三 ε 的中心差分范围 | 相对幅度偏差范围 |
|---|---:|---:|---:|
| 00_native / splats.sh0 | 0.01045979 | [0.00377744, 0.00391155] | 62.60%–63.89% |
| 00_native / splats.sh_rest | 0.08201439 | [0.02948195, 0.02966821] | 63.83%–64.05% |
| 00_native / background_logits | 0.00321809 | [0.00101514, 0.00105053] | 67.36%–68.46% |
| 01_original / splats.sh0 | 0.00906609 | [0.00335090, 0.00353158] | 61.05%–63.04% |
| 01_original / splats.sh_rest | 0.07108644 | [0.02642348, 0.02647936] | 62.75%–62.83% |
| 01_original / background_logits | 0.00426917 | [0.00069663, 0.00075996] | 82.20%–83.68% |

18/18 差分均与解析方向同号，但幅度偏差为 61.05%–83.68%，没有随 ε 缩小一致收敛。偏差至少为标量端点 ULP 所对应导数量纲的 145.46 倍；按实际 FP32 参数位移修正解析导数后，相对变化最多仅 1.81e−5。因此不能简单归为最终标量分辨率或参数舍入；ULP 指标也不是渲染累计误差的完整界，不能据此直接断言 CUDA bug。

输出 RGB 高于 1 的比例为 2.298e−6，扰动前后相同；L1 接近零残差的通道比例约 5e−5。该统计不覆盖内部 SH clamp 与全部非光滑路径。代码只读检查未发现 baseline/差分使用不同目标、重复累积梯度或显式 grad-mode 前向分支；尚无分项解析梯度核验，不能直接认定 SSIM 是原因。

目前结论是**完整目标导数的局部异常未排除，fullbatch 前置条件未满足**。不归因多视图优化噪声、fresh moments 或泛化，不冻结/运行 fullbatch；下一定位工作限定为独立链式梯度检查，原 40-render 计划和结果保持不动。
