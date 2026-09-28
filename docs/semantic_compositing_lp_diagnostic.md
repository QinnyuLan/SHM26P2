# 固定 16 TRAIN 射线的完整 compositing 权重诊断

固定执行已进行并在重建先决检查处失败，未运行 LP，详见末节。它承接固定 8 TRAIN 原生概率诊断：低总 alpha 只占 GT 拉索像素的 0.58%，不能解释大多数错误；而现有拉索概率同时受类别赋值和可见权重影响。这里固定渲染几何与可见性，检验极少射线是否**允许**一个更好的共享高斯类别赋值。标准 compositing、自动微分和 LP 均不称学术创新。

## 样本在 GPU 前确定

沿用新联合 checkpoint `391a0577…`、corner_v2 native 1320×989、原 8 TRAIN 相机，不读取真实 RGB 或 VAL。每图 E 为 valid 且 GT cable、原始 raw argmax=background 的像素，按 (y,x) 行优先取 `floor(|E|/2)`；另一像素为最近的 valid、GT background、raw argmax=background 像素，平方距离并列时取行优先第一个。不设置距离阈值、不替换 anchor。

坐标是从零开始的数组 `(y,x)`，相机 K/pose 不做任何平移或再投影。GT 3 像素边界只作记录，不影响选择。

| TRAIN | cable→bg `(y,x)` | bg正确 `(y,x)` | 距离 px |
|---|---|---|---:|
| 002 | (303,230) | (302,231) | 1.414 |
| 043 | (342,875) | (333,882) | 11.402 |
| 084 | (553,481) | (504,473) | 49.649 |
| 125 | (628,478) | (613,446) | 35.341 |
| 167 | (567,873) | (508,803) | 91.548 |
| 215 | (370,536) | (360,521) | 18.028 |
| 257 | (617,623) | (608,623) | 9.000 |
| 300 | (443,371) | (321,331) | 128.390 |

这里的“附近”只是最近可用正确背景，有的距离超过 100 px；不会为了提高 W 重叠重新选点。002 的 cable anchor 属于记录的边界带，其余 cable anchor 不在；所有最近背景点都在边界带。

## 直接从同一 compositor 提取 W

复用已完成新语义场的 30 文件 package，场景所有参数冻结且 detach。每图依旧一次 `scene.render(semantics=True, refine=False, degree=3)`；只在 semantic rasterization 调用处把原值完全相同的 `colors.detach().requires_grad_(True)` 作为叶子，保留未归一化输出。对每个选中像素的 raw channel 0 求颜色梯度，`∂raw_0/∂colors_i0=W_i`。其余颜色通道梯度必须为零，所有场景参数不得有梯度。每图两次颜色 backward，共 8 scene renders、16 backwards，包装在 finally 恢复。

不重写 compositor、不截 top-K、不丢弃正的小权重。这里的完整指**实际 renderer 已实现的权重**，包含它原有 early termination，不是对物理光线所有可能贡献的断言。所有 `W>0` 的原 Gaussian ID 合集与 CSR 权重保存；零权重不影响 LP。

每条射线的先决条件：W 必须有限且非负；`sum W` 与同 pass alpha 的绝对误差≤5e−6；用完整 `W @ 当前5类概率 + (1−alpha)·bg_onehot` 复现 raw，接着应用原 clamp_min(1e−7)/normalize 复现 p3d，误差均≤5e−6；当前选点 p3d 还须与此前已存输出相符。任一先决条件失败便停止，不把可疑 W 送入 LP。

## 放宽问题与独立证书

对完整正权重合集中的每个 Gaussian，赋予独立 `q_i∈5-simplex`，同一 Gaussian 在 16 条射线间共享。背景 residual `t_j=1−alpha_j` 固定。对射线 j 的目标 y 与每个竞争类 c，最大化公共 margin γ，约束为：

`γ + Σ_i W_ji(q_ic−q_iy) ≤ b_jc = t_j(1[y=bg]−1[c=bg])`。

CPU SciPy HiGHS 求一次 LP，最多 60 s，无超参搜索。HiGHS 可能忽略微小系数，因此不把 success/fun 作为证书。每高斯 q 先非负裁剪并归一到可行 simplex，再用**完整原 W**重新计算所有 margin，最小值给 primal 下界 LB。

另取任意 `λ_jc≥0, Σλ=1`，本实现将 `−ineqlin.marginals` 非负裁剪并归一；若不可用则使用均匀 λ。按完整 W 计算：

`UB = Σ_jc λ_jc b_jc + Σ_i max_k [Σ_jc λ_jc W_ji(1[y=k]−1[c=k])]`。

这是拉格朗日上界，不依赖求解器是否丢掉小系数。报告 LB、UB、UB−LB、simplex/λ 检查及固定 1e−6 浮点解释容差。LB>1e−6 只说明这些 probe 有正 margin 赋值；UB<−1e−6 才支持该放宽问题不能同时达到非负 margin。落在零附近、界差宽或求解未收敛只报告低 margin/未解决，不武断判不可行。即使存在 q，也不代表全图或跨视图语义已经正确。

现有固定 decoder 的类别差分矩阵 `weight[1:]−weight[0]` CPU rank=4，奇异值约 `[7.0618,2.7962,2.2453,2.0452]`；因此独立可学习的 16D Gaussian features 在不考虑其它损失/正则时可表达任意内部类概率，simplex 边界是其闭包。本 LP 不包含联合 refiner 的特征需求、训练正则、平滑性或整个数据集约束，因此仍是一个很宽松的容量检查。

## 重叠与可解释性限制

额外仅从相同 W 计算所有射线对的 `Σ_i min(W_ai,W_bi)`，并除以较小行质量形成重叠比例，分别汇总同 view 两条射线与跨 view 射线；保存完整 16×16 矩阵，不新增投影或样本。如果跨 view 支撑几乎不重叠，只称 16-ray 联合 LP，不能说检验了强跨视图冲突。

正 margin 只证明这些少量、根据已观察错误选出的 TRAIN 射线允许更好赋类；负 margin 证书只约束这个固定 W 与这些二维硬标签。两者都不是独立三维真值、泛化或性能采用证据。结果不能授权重新选点来获得不同结论。

## 锁定与执行状态

- 源入口：`scripts/audit_semantic_compositing_lp.py`，SHA `17f96c6a6673266b146860576765e395446744b2d770c3291d697ee37ac44349`。
- plan：`/mnt/data/SHM2026/runs/semantic_compositing_lp_v1/plan.json`，SHA `d642e3cfd3b5df4c4a61051539990b30e086c9332bb1a4985e7baf7a80d85fad`。
- 来源：原 completed new-semantic package 的同 30 文件，仅添加新诊断入口；绑定 parent 已完成 8-view plan/receipt、全部 8 个数组/GT/valid、checkpoint/manifest/uv.lock。实际冻结包 CPU import/输入与源码哈希验证通过。
- 10 个 focused CPU tests 与 Ruff 通过，覆盖确定性选点/等距规则、同 compositor 颜色梯度与 detach/finally、完整重建门槛、已知可行/同 W 标签冲突/低 alpha 负 margin LP、零支撑、被求解器可能忽略的小系数对偶界、同/跨 view 重叠。

等待 root 短 GPU 窗口；不与 RGB 30k 训练争抢。执行时仅使用冻结入口和 package，外部建议 timeout 180 s：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/mnt/data/SHM2026/runs/semantic_compositing_lp_v1/source_snapshot OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 uv run --no-sync python /mnt/data/SHM2026/runs/semantic_compositing_lp_v1/source_snapshot/audit_semantic_compositing_lp.py --execute /mnt/data/SHM2026/runs/semantic_compositing_lp_v1/plan.json
```

独立只读复核未发现 VJP 或 LP 数学阻断，补充解释限制：重建先决容差 5e−6 大于证书容差 1e−6。后者是针对**提取 W + recorded alpha 的 FP64 放宽问题**；若 margin 距零只有数个微量单位，不能自动升级为原 FP32 renderer 的严格判定，应同时列实际重建差，并考虑两类差值的约 2×重建误差。最终这类近零情况仍保持 unresolved，不修改容差来选择结论。相关工作另见 `docs/semantic_feasibility_prior_art.md`，其中记录 FlashSplat 的冻结 αT 后最优语义分配先例；本诊断没有原创性主张。


## 实际执行：先决重建失败，未取得可行性结论

root 在 RGB 30k 自然结束后明确交接短 GPU 窗口；启动前已核对锁定 plan SHA 和空 compute 列表。session 67229 使用原冻结入口、原16ray与 OMP/MKL/OpenBLAS=8、外部180s上限，**自然 exit 1**，`execution_receipt.json` 记录 failed，内部耗时 **1.457509 s**。

提取到的一条射线违反预先锁定的重建门：`|sum W−alpha|=4.794933295e−5`，raw 五类最大重建差 `3.616309318e−5`，两者都超过 5e−6。归一化 p3d 的最大误差为 `1.001990956e−7`，虽小，但不能替代 raw/mass 先决条件。**程序未进入 LP，故没有 LB、UB 或 overlap 结果，也不能作分类/可见性瓶颈判断。**

原 plan、source、process.log 和 failed receipt 均未修改；`launch_failure_audit.json` 绑定退出与文件 SHA。当前入口在失败前未持久化 ray index 或原 W 数组，因此不推断具体失败坐标，也不凭该数值模式断言 CUDA backward 的原因。W 整体尺度偏差是可能的解释，但没有本次保存的数据足以验证。没有做归一化补偿、调大容差、换点或自动重试。退出后 GPU compute 列表为空，已明确交回 root。

## 仅 CPU 源码定位：末透射率精度损失的候选机制

已读本机安装的 gsplat 1.5.3，未修改依赖或运行 GPU。`RasterizeToPixels3DGSFwd.cu` 第 105 行以 FP32 累乘 T；153–168 行计算各贡献 `alpha_i*T_i`，173–184 行仅保存 `render_alpha=1−T_end`，而前向背景项仍使用原 T_end。`RasterizeToPixels3DGSBwd.cu` 107–108 行从 `1−render_alpha` 恢复 T_final，195–202 行反向逐步乘 `1/(1−alpha_i)`，用 `alpha_i*T_recovered` 作为颜色梯度。`Rasterization.cpp` 49–58 行使用 means2d 的 FP32 options 分配 alpha；wrapper 保存的是该 alpha，没有另外保存原 T_end。前向文件 174–177 行已有注释，指出小 T 使用 float32 可在反向产生显著梯度差。

这给出一个有源码依据的机制：T_end 很小时，先存 `1−T_end` 再相减恢复，会丢失其相对精度，误差被反向 recurrence 近似共同传播到各颜色权重；反向 reciprocal 的 FP32 舍入还会额外累积。五类颜色都依赖同一透射率，所以原 `model.py` 的 `clamp_min(1e−7)` 后按五类和归一化会消除大部分公共尺度误差。这与 raw/mass 约1e−5而 normalized p3d 约1e−7的模式一致，但失败入口未保存 W/原始 T，不能据此声称已逐项定位了该真实射线。

读取的是既有 16 坐标的已存 alpha：多条 T≈1e−4–3e−4，`0.5×ULP(alpha)/T` 可达约 2.86e−4，足以容纳观察到的相对量级；没有选择新图/新点。

固定 CPU 合成 recurrence（每 primitive alpha=0.1 FP32，沿原 `next_T<=1e−4` exclusive 终止规则，不搜索匹配误差的参数）得到87个贡献项：原 T_end=0.0001044954333，而从已存 alpha 恢复得到0.0001044869423，相对差−8.1241e−5。以恢复值反推的颜色梯度总质量相对 alpha 差−7.9823e−5，raw 五类重建差2.0119e−5，归一化概率差仅2.8290e−8；若仅在这个 CPU 合成链中用原 T_end 做反向种子，质量差降至1.5058e−6。该算例只是展示机制和量级，不是 CUDA/FMA 的逐位复现，更不是已失败真实 ray 的再实验。

完整依赖/冻结 model 文件 SHA、CPU 算例与解释边界保存在 `backward_precision_source_audit.json`。v1 门槛与失败结论保持不变；没有将 W 自行归一化后执行旧 LP，也不据此否定全部已有训练梯度。
