# TRAIN002 外观 CUDA 梯度数值审计

2026-09-26，固定v2语义8000步完整模型、TRAIN002相机，执行唯一一次只读数值诊断。**全27组预定有限差分均通过，未发现本测试范围内的CUDA外观反传错误。** 不更新模型、不训练、不调参、不改变原始网格颜色精修的失败/停止结论。

固定base SHA为`a77d304f32de4356c1a4608ed5192258b6a9ced4c67a4297809f439481408d89`，manifest SHA为`91b41aeecedc352e4882eb80a1bb635d479ae432251a4c96c85a9dc110cbb327`。输入只用相机和模型；没有解码RGB、GT或valid图像，没有优化器。

[锁定计划](../runs/appearance_cuda_gradient_audit/plan.json)、[执行回执](../runs/appearance_cuda_gradient_audit/execution_receipt.json)、[完整数值报告](../runs/appearance_cuda_gradient_audit/report.json)和冻结源码全部保留。plan SHA为`25a6c85fabdd0c4a0c098f9da25ad33cf9b1d11ab15a1a327a7d10b8b63c271d`，runner SHA为`620466ade28f41fe40a698aaa78d89ca8528647e1996748daa0efcd2aee3829e`，34文件source/runner tree SHA为`6b694483b20a2dd5d53f7a749f362857769458c3e2c442a49125a24e4c2e4288`，report SHA为`7d91ccf5d44b964cbe4da2b60dc1d975471ac6c4fa6572271819fc89f76c3902`。

## 固定数学合同

三条路径为原生raw RGB、不超过[0,1]的overscan RGB裁出原生网格、同样clamp后的overscan RGB经固定可导raw warp。使用完整SH degree3、antialiased renderer、`semantics=False`，几何/opacity/相机/所有语义参数冻结。三个被临时扰动的参数是`sh0`、`sh_rest`和`background_logits`。

对输出RGB构造固定的空间/通道线性探针：

```
J = mean_RGB(w(y,x,c) * image(y,x,c))
w = (1 + .25 sin(2π(x+.5)/W) + .125 cos(2π(y+.5)/H)) * [.7,1.,1.3]_c
```

乘法与累加均float64，renderer仍使用原FP32参数。对每路径/每参数先求解析梯度`g=dJ/dθ`，固定方向`d=g/max(abs(g))`；方向absmax为1，不搜索方向。报告`sum(g*d)`与`[J(θ+εd)−J(θ−εd)]/(2ε)`，ε固定为.01、.005、.0025，全部报告，不挑最好ε。每次正负扰动都从原参数clone出发，异常finally也恢复原值；另统计FP32实际位移与名义方向的偏差。

固定数值容差为`abs_error <= 1e-7 + .02*abs(analytic)`。它只是本次有限精度诊断的预先界限，不是关于训练收益的标准。SH颜色内部clamp及外部RGB clamp可能引入非光滑点，因此保留各正负输出的越界比例/min/max，而非隐藏某个不利步长。

## 全部27组比较

相对误差列是无量纲比例；例如`1.67e-5`等于`0.00167%`。每行均满足固定容差。

| 路径 | 参数 | ε | 解析方向导数 | 中心有限差分 | 绝对误差 | 相对误差 |
|---|---|---:|---:|---:|---:|---:|
| native_raw | splats.sh0 | 0.0100 | 0.0254146148 | 0.0254146300 | 1.517e-08 | 5.970e-07 |
| native_raw | splats.sh0 | 0.0050 | 0.0254146148 | 0.0254146804 | 6.559e-08 | 2.581e-06 |
| native_raw | splats.sh0 | 0.0025 | 0.0254146148 | 0.0254147601 | 1.453e-07 | 5.717e-06 |
| native_raw | splats.sh_rest | 0.0100 | 0.2170223919 | 0.2170224599 | 6.803e-08 | 3.135e-07 |
| native_raw | splats.sh_rest | 0.0050 | 0.2170223919 | 0.2170224458 | 5.394e-08 | 2.485e-07 |
| native_raw | splats.sh_rest | 0.0025 | 0.2170223919 | 0.2170225502 | 1.583e-07 | 7.293e-07 |
| native_raw | background_logits | 0.0100 | 0.0008030022 | 0.0008029941 | 8.102e-09 | 1.009e-05 |
| native_raw | background_logits | 0.0050 | 0.0008030022 | 0.0008029919 | 1.032e-08 | 1.286e-05 |
| native_raw | background_logits | 0.0025 | 0.0008030022 | 0.0008029941 | 8.116e-09 | 1.011e-05 |
| overscan_clamp_crop | splats.sh0 | 0.0100 | 0.0254146048 | 0.0254146181 | 1.332e-08 | 5.242e-07 |
| overscan_clamp_crop | splats.sh0 | 0.0050 | 0.0254146048 | 0.0254146645 | 5.972e-08 | 2.350e-06 |
| overscan_clamp_crop | splats.sh0 | 0.0025 | 0.0254146048 | 0.0254147545 | 1.497e-07 | 5.891e-06 |
| overscan_clamp_crop | splats.sh_rest | 0.0100 | 0.2170221906 | 0.2170222511 | 6.051e-08 | 2.788e-07 |
| overscan_clamp_crop | splats.sh_rest | 0.0050 | 0.2170221906 | 0.2170222159 | 2.525e-08 | 1.164e-07 |
| overscan_clamp_crop | splats.sh_rest | 0.0025 | 0.2170221906 | 0.2170222452 | 5.457e-08 | 2.515e-07 |
| overscan_clamp_crop | background_logits | 0.0100 | 0.0008030007 | 0.0008029926 | 8.011e-09 | 9.976e-06 |
| overscan_clamp_crop | background_logits | 0.0050 | 0.0008030007 | 0.0008029904 | 1.024e-08 | 1.275e-05 |
| overscan_clamp_crop | background_logits | 0.0025 | 0.0008030007 | 0.0008029926 | 8.034e-09 | 1.001e-05 |
| overscan_clamp_raw_warp | splats.sh0 | 0.0100 | 0.0252491055 | 0.0252491013 | 4.190e-09 | 1.659e-07 |
| overscan_clamp_raw_warp | splats.sh0 | 0.0050 | 0.0252491055 | 0.0252490863 | 1.924e-08 | 7.619e-07 |
| overscan_clamp_raw_warp | splats.sh0 | 0.0025 | 0.0252491055 | 0.0252491479 | 4.238e-08 | 1.679e-06 |
| overscan_clamp_raw_warp | splats.sh_rest | 0.0100 | 0.2192212276 | 0.2192213551 | 1.276e-07 | 5.819e-07 |
| overscan_clamp_raw_warp | splats.sh_rest | 0.0050 | 0.2192212276 | 0.2192213142 | 8.664e-08 | 3.952e-07 |
| overscan_clamp_raw_warp | splats.sh_rest | 0.0025 | 0.2192212276 | 0.2192214378 | 2.103e-07 | 9.592e-07 |
| overscan_clamp_raw_warp | background_logits | 0.0100 | 0.0008061049 | 0.0008060969 | 7.992e-09 | 9.914e-06 |
| overscan_clamp_raw_warp | background_logits | 0.0050 | 0.0008061049 | 0.0008060914 | 1.347e-08 | 1.671e-05 |
| overscan_clamp_raw_warp | background_logits | 0.0025 | 0.0008061049 | 0.0008061003 | 4.583e-09 | 5.685e-06 |

最大相对误差为1.67123996e-05，最大绝对误差为2.10275379e-07。FP32实际扰动方向相对L2偏差最大为4.25244471e-04；报告另保留按实际位移计算的线性化导数。

三个基线探针的标量分别为native raw 0.5920520697、clamp crop 0.5920517329、clamp raw warp 0.5912950191。原始RGB最大1.170344，越界通道比例约4.85134e-6；raw路径确实没有被偷偷clamp。

## 执行、恢复和解释边界

9项CPU回归及Ruff先通过，再按锁定source执行；实际import路径/SHA写入回执。57次真实CUDA render（3次解析梯度+54次中心差分）总计2.476秒，外层`timeout --signal=KILL 60s`与内部55秒截止均未触发。PID1157817 / session48564自然退出0。全部模型tensor最终SHA与运行前逐项一致，每组参数均exact恢复，base/manifest/源码/计划SHA前后未变化。执行前只有桌面G进程，结束后compute client为空，GPU已释放。产物含源码快照约0.49MiB，无概率/图像缓存。

这次验证只覆盖**一个相机、一个平滑线性探针、每组一个解析梯度方向、三个预定步长**；不是完整Jacobian检验，也不验证L1/SSIM等完整训练目标、所有场景/相机或更新后的优化轨迹。它关闭了“该有限设置下颜色参数CUDA image梯度明显错误”的疑点，没有找到需要据此修复的导数bug。

此前CPU源码审计亦未定位RGB目标/相机索引/SH阶数/Adam绑定错误，但新旧appearance实验存在真实配方差异：旧20k场与新30k v2场、十倍学习率差、raw/clamp、SSIM窗口支持、前景加权、不同随机数生成器的shuffle及直接native/overscan渲染。它们不是此次退化原因的实证识别。**原始网格颜色精修仍按失败终点停止，不因数值审计通过而追加训练、改学习率或挑中途点。**
