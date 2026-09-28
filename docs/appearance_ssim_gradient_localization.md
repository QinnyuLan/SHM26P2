# SSIM 梯度定位、最小布局修复与固定目标复核

最初的三次渲染把主要不一致定位到 **SSIM 的图像损失链**：在固定 TRAIN152、native 输入、当前基点和唯一 SH0 方向上，参数梯度与像素梯度沿实测图像割线的投影吻合，但 DSSIM7 的像素梯度投影与标量有限差分相差约两个数量级。它不证明所有场景/参数的 gsplat 梯度正确，当时尚未识别具体错误算子或 CUDA kernel。后续无数据合成复现已隔离本机 torch 2.8.0+cu128 的完整 SSIM channels-last 反传路径；入口强制连续 NCHW 后，独立合成梯度回归和真实 40-render 复核均大幅改善。该修复是数值实现修正，不是学术贡献。

以下先保留三次渲染阶段的原始证据，再追加合成布局定位、生产最小修复及独立新回执。此次文档整理仅 CPU 只读；旧回执、旧模型与冻结协议均未修改。

## 来源与边界

- [锁定计划](/mnt/data/SHM2026/runs/appearance_gradient_chain_diagnostic/plan.json)：SHA256 `1b57689905a9ccb5992d6cc51344f804e8604986cd3bcf1f694546cbb1009f48`。
- [完成回执](/mnt/data/SHM2026/runs/appearance_gradient_chain_diagnostic/execution_receipt.json)：SHA256 `83c1cc866fed1a1716f25eb7c0edcb59c253b06fcb8cd7f6de1873872a19839a`；自然完成，3 次渲染、2.413819 秒，无优化器步、无 checkpoint 输出。
- [冻结诊断入口](/mnt/data/SHM2026/runs/appearance_gradient_chain_diagnostic/source_snapshot/audit_appearance_gradient_chain.py)：SHA256 `2867b0f68f124dfd1b00538b644899580bbe0719a362873899b38619cc798a42`。
- [冻结 SSIM 实现](/mnt/data/SHM2026/runs/appearance_gradient_chain_diagnostic/source_snapshot/bridge_rgs/losses.py:6)：SHA256 `76eb8a8c15eea71dd0110f461098f1997620c93095915ca6d0fb771f9a74f25e`。
- [冻结组合损失](/mnt/data/SHM2026/runs/appearance_gradient_chain_diagnostic/source_snapshot/bridge_rgs/raw_grid.py:160)：SHA256 `989476c5940228c0bbd4cda97450e6e29540833ac2b7bfde6d73684c530c2e6d`。

继承 [40-render 诊断](/mnt/data/SHM2026/runs/appearance_actual_objective_diagnostic/plan.json)的 package、参考入口、base、原相机、overscan→clamp→native crop、RGB/valid 和 7×7 完整窗口支持。只解码 TRAIN152 native RGB 和对应 valid 各一次；没有解码语义标签或 VAL 图像。1,290,806 个 RGB 有效像素。组合基线 **0.03350527957081795** 与原 appearance 第一步日志逐值相同。来源 SHA 前后相同，模型全部 state tensor、requires_grad 和 module.training 标志 finally 恢复一致。

## 比较的三个量

令 `P(θ)` 是已完成 clamp/crop 的 RGB 图像，`d` 是基线组合损失 SH0 梯度除以其绝对最大值，其余参数方向为零。固定 `ε=.001`，只渲染 `P0、P+、P−`。FP32 实际参数位移给出 `d_actual=(θ+−θ−)/(2ε)`；图像割线为 `J_sec=(P+−P−)/(2ε)`。

对每个损失组件分别比较：

1. 参数侧：`∇θL · d_actual`。
2. 像素侧：`∇PL(P0) · J_sec`。
3. 标量侧：`[L(P+)−L(P−)]/(2ε)`。

第一与第二之差检查本方向的 renderer→clamp/crop 链；第二与第三之差检查叶图像上的损失链。`J_sec` 仍是有限割线，而非解析 JVP，因此不能仅凭非零差值宣称自动微分错误。FP64 仅将同一批已渲染的 FP32 图像/target 转为双精度重算损失，**不是 FP64 渲染**。

## 实测

| FP32 组件 | 参数梯度 · 实际方向 | 叶图像梯度 · J_sec | 标量中央差分 | 参数–像素相对差 | 像素–标量相对差 |
|---|---:|---:|---:|---:|---:|
| L1 | 0.004640562048 | 0.004640793937 | 0.004640780389 | 0.004997% | 0.000292% |
| DSSIM7 | 0.033736385068 | 0.033735529005 | 0.000324100256 | 0.002538% | 99.039291% |
| .8 L1 + .2 DSSIM7 | 0.010459724618 | 0.010459740128 | 0.003777444363 | 0.000148% | 63.885868% |

两个相对差分别以参数侧量、像素侧量为分母，取绝对值。原 graph 与 FP32 leaf 的每个组件像素梯度差异范数均为 0。也就是说，脱离 Gaussian graph 后，损失链里的主要不一致仍存在；本次不是只凭一个参数侧 FD 猜测 renderer 问题。

| FP64 叶图像组件 | 图像梯度 · J_sec | 标量中央差分 | 像素–标量相对差 |
|---|---:|---:|---:|
| L1 | 0.004640793985 | 0.004640720218 | 0.001590% |
| DSSIM7 | 0.033733019846 | 0.000334487083 | 99.008428% |
| .8 L1 + .2 DSSIM7 | 0.010459239157 | 0.003779473591 | 63.864737% |

DSSIM7 的像素方向导数/标量 FD 比值为 FP32 **104.09×**、FP64 **100.85×**。这是这个固定方向上的比值，不能推广成所有像素或所有参数梯度都放大 100 倍。

FP64 DSSIM7 的前向/后向标量斜率为 **0.000334754424 / 0.000334219743**，都靠近中央差分；对应像素梯度与前向/后向图像割线的 dot 为 **0.033733223810 / 0.033732815882**。实测图像中央割线绝对最大值为 0.152111，意味着 `|(P+−P−)/2|` 最大约 0.000152111。实际参数方向相对名义方向 L2 差为 0.25294%，已通过 `d_actual` 显式纳入参数侧比较。以上强烈不支持“差异只是 FP32 标量 ULP 或未实现的名义参数步长”这一解释；仍不能从单一 ε 严格消除所有非线性/分段行为。

## 布局与组件重组

实际原 graph prediction 是连续 HWC：shape `[989,1320,3]`、stride `[3960,3,1]`、storage offset 0。FP32 与 FP64 的 P0/P±/target 都具有相同连续 HWC 布局。本次实际运行**没有**发生预备合成测试中可能的“非连续 crop→连续 leaf”转换，因此不能把本次差异归因于 leaf 更改了 stride。

冻结 `ssim_map` 的 `permute(2,0,1)[None]` 后，池化输入为 NCHW 形状、stride `[3,1,3960,3]`，普通 contiguous 为 false，channels-last contiguous 为 true。这是具体、可复现的布局事实，**不是已经证实的 avg_pool2d 故障归因**。

损失实现用五次 `avg_pool2d(7,stride=1,padding=3)` 计算两张图的均值、二阶矩与交叉矩，再构成 SSIM；完整窗口 mask 与图像数值无关。代码未包含故意 stop-gradient 或自定义反传。`.8/.2` 重组在 FP32 的标量残差约 7.45e−10～1.49e−9，像素方向导数重组残差 −8.23e−10；FP64 对应标量残差为 0、方向残差约 −3.47e−18。参数梯度重组相对 L2 残差约 4.17e−7～2.75e−6。它们都远小于当前主要不一致，不能解释 0.00668 左右的组合方向导数缺口。

## 三次渲染完成时的结论与下一项定位（历史阶段）

本次证据支持暂停将 appearance 退化仅解释为多视角泛化或优化步长问题，优先验证 SSIM 图像损失算子链。它没有推翻既有 forward RGB/语义评分；受影响的是对使用该训练梯度路径所得行为的因果解释。未使用此 SSIM 梯度的 DINOv3 教师容量结果不能因此一并判无效。

root 已准备的无真实数据最小复现，应在同一数值输入下比较 CPU/GPU、NCHW contiguous/channels-last，以及 `avg_pool2d` 前向和反向的有限差分/显式参考。若只查“梯度非零”，无法检验这里的幅度问题。生产 loss、已完成模型和评分协议在该算子定位完成前保持原样，不应直接把整条 SSIM 公式或所有 gsplat 梯度宣布有错。此文档不新增 GPU 实验或修复。

## 后续合成布局复现与生产修复

[合成报告](/mnt/data/SHM2026/runs/ssim_pooling_backend_diagnostic/report.json) SHA `182e37a648efee436c0a573aed870266d8103226af9517b591d5b9e5aa1ddbf1`：16 个固定比较，0 数据集像素、0 scene render，纯合成 FP64。两种形状下，CPU NCHW、CPU channels-last 与 CUDA NCHW 的完整 SSIM 梯度一致、方向 FD 相对差均小于 5e−11；CUDA channels-last 前向值相同，但梯度不同：

| 合成 H×W | CUDA channels-last 梯度相对 L2 差（相对 CPU NCHW） | 该方向 FD 相对差 |
|---|---:|---:|
| 37×53 | 0.3546922 | 0.0581667 |
| 989×1320 | 0.1109793 | 0.00461376 |

同样布局下，单纯 `avg_pool2d` 线性目标的各条路径正常。因此证据限定为**本机 torch 2.8.0+cu128、CUDA 12.8 下复现的完整 SSIM 表达式/布局反传问题**，不扩张成所有 pooling 或所有 PyTorch/CUDA 版本的故障判断。

[生产修复](/home/sky/workspace/SHM2026/src/bridge_rgs/losses.py:6)只在 `ssim_map` 两个 `permute(2,0,1)[None]` 后显式 `.contiguous()`，固定连续 NCHW；SSIM 公式、7×7 窗口、.8/.2 组合权重和官方评分 SSIM 均未改。修复文件 SHA 为 `1e114069c2fdb181e243d7cb8a986659b2e229d2ec49c34821cb45013d9cf2b2`。

新增 [回归测试](/home/sky/workspace/SHM2026/tests/test_ssim_gradients.py)使用独立 CPU FP64 grouped-convolution 方窗参考，检验 CUDA 完整 SSIM 的 x/y 梯度及固定随机方向 FD，覆盖连续 HWC 与非连续 crop 输入，并核验旧 FP32 前向兼容。8 个新增合同连同 7 个 raw-grid 合同共 15 passed，Ruff 通过；它们没有读取数据集或运行训练。

## 修复后的独立 40-render 复核

新 [固定计划](/mnt/data/SHM2026/runs/appearance_fixed_ssim_objective_diagnostic/plan.json) SHA `36e137d371ffc3e9eb4d69c560e9b898475cadd6172b88ffed2d1eb8db213394`；[完成回执](/mnt/data/SHM2026/runs/appearance_fixed_ssim_objective_diagnostic/execution_receipt.json) SHA `15ef4c0f2541f4ada247adb304d5e5e77a3a43af72af84326a26848df921eb89`。CPU 独立核验全部新快照文件 SHA、实际 import SHA 与旧/新输入模型逐张量 hash：复用 package 仅 `losses.py` 不同，输入模型完全相同。

此次自然完成 40 renders、18 项差分和两次临时 fresh Adam step，3.197 秒；原有两臂 loss、相机、TRAIN152、三个 ε、三组 LR/eps/betas 均保持。两臂 baseline loss 与旧回执逐位相同，0 语义/VAL 像素读取，来源不变、全部模型张量 finally 恢复，无 checkpoint 输出。旧诊断未重标成通过，新证据保存在独立目录。

各组方向仍由**该版本自身的解析梯度 / absmax**构造，故修复前后方向会变化；下面比较的是各自方向的解析–FD 一致性，不是同一个固定向量的跨版本方向导数。

| 原目标 / 参数组 | ε=1e−3 FD 相对差 | ε=5e−4 FD 相对差 | ε=2.5e−4 FD 相对差 |
|---|---:|---:|---:|
| 00_native / splats.sh0 | 0.239139% | 1.125370% | 1.412796% |
| 00_native / splats.sh_rest | 0.004844% | 0.102292% | 0.024932% |
| 00_native / background_logits | 1.646465% | 1.434038% | 1.115081% |
| 01_original / splats.sh0 | 0.150044% | 1.181020% | 0.726285% |
| 01_original / splats.sh_rest | 0.050113% | 0.003927% | 0.198474% |
| 01_original / background_logits | 0.754909% | 2.506293% | 0.412680% |

18 项相对差从旧版 61.05%–83.68%降至 0.00393%–2.50629%，但不是全部精确相等。最大相对差出现在原图 background 的 ε=.0005：解析 0.00063811655，FD 0.00062212348，绝对差约 1.60e−5，是该差分标量 ULP 尺度的 2.15 倍；小信号下 FP32 分辨率确实可见。其他项相差约 0.18–11.75 个该尺度单位，且不随 ε 单调收敛。该 ULP 只描述最终标量分辨率，不包含所有 SSIM 消减、渲染累计舍入、有限步长或 L1/clamp 非光滑误差，**不能把全部剩余差异一概归于量化**。

| 原目标 | baseline loss | 一步后 loss | 实际 Δloss | g·实际位移 | 实际/预测下降 |
|---|---:|---:|---:|---:|---:|
| 00_native | 0.03350527957 | 0.03345740959 | -4.78699803e-05 | -4.79451224e-05 | 0.998433 |
| 01_original | 0.04251156747 | 0.04246669635 | -4.48711216e-05 | -4.49475550e-05 | 0.998300 |

两臂一步后的 L1 与 SSIM-loss 都下降；实际与线性预测下降比为 0.998433 / 0.998300（旧路径为 0.273742 / 0.238787），余项约 7.5e−8。合成 FP64 独立参考、保持基线前向的最小布局修复和真实完整目标结果共同支持修复该局部梯度问题，仍不是全 Jacobian 或多视图收敛证明。

## 后续决策

root 优先执行**保持原 fresh Adam 配置的 native/original 两组各 3,000 步重放**，唯一训练实现改变是上述 SSIM 布局修复，以判断历史颜色精修退化是否随修复改善。不能在该因果重放前把失败归咎于逐视图 Adam，也不立即替换求解器。条件性 [fullbatch 方案](conditional_fullbatch_rgb_solver.md)仍只准备代码，尚不冻结/启动；helper/runner 的 gate 未改为自动放行。未来若有工程收益仍需同场、同原图协议与固定 selected H+ 完整终评，不能由这一个视角的差分结果预报性能。
