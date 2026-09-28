# 三维采样尺度约束：CPU证据与固定对照范围

2026-09-27：全量CPU尺度审计已完成；仅准备滤波接口与合同，尚未运行GPU预检或正式训练。唯一候选是在既有 `rgb140_mixed` 配方上增加Mip-Splatting式三维滤波，保留当前二维antialiased渲染。它是已有方法的工程强对照，不是新的结构机制，不另加语义区域、深度、曝光模型或新的增密评分。

## 已确认缺口与先验边界

目前 `model.render` 将 `exp(log_scales)` 和 `sigmoid(opacity_logits)` 直接交给gsplat；`rasterize_mode="antialiased"`只对投影后的二维协方差做opacity补偿，没有限制三维Gaussian本身的最高频率。Mip-Splatting另外按TRAIN采样尺度平滑三维协方差，并配套调整opacity。[gsplat说明](https://docs.gsplat.studio/main/apis/rasterization.html)、[Mip论文](https://arxiv.org/abs/2311.16493)

既有支撑约束分裂只约束拓扑操作当时的位移和收缩，后续优化仍可使轴变得很薄；该修正降低轴比尾部，但完整30k未带来RGB提升。opacity熵、ED/front控制也没有解决整体RGB问题。这些结果没有单独检验持续的三维采样尺度约束，故本候选尚未在本项目350/50划分上被证伪；它们也没有证明当前误差就是混叠。

同学材料已经包含Mip全400比较：`RGB-030-FULL`记载Scaffold相对Mip的完整配方优势，`RGB-130-STANDARD-ALL400`又记录Standard/Mip的opacity LR和kernel不同。这是相关负先验，不能称Mip从未试过；但其全400训练拟合、配方和初始化不同，不等同于本次固定350/50的filter-only对照。详见[RGB030记录](../project_progress_20260918/sources/RGB-030-FULL.html)、[RGB130记录](../project_progress_20260918/sources/RGB-130-STANDARD-ALL400.html)。

## CPU全量尺度审计

输入为已完成的SSIM-fixed corner-v2 RGB场，498,136点。另核验其全部geometry、SH、background及训练相机与语义终点 `391a0577…` 逐位相同。只用350个TRAIN原相机和初始SfM点；没有打开RGB/标签、没有VAL选组、没有render或optimizer step。实际耗时3.27秒，输入SHA在前后均相同。

固定作者源码commit `dda02ab5ecf45d6edb8c540d9bb65c7e451345a9`，文件SHA `58a240615a1d974d374a2bf5a3b9b1b731fb6523f61e0f4ded0134932dbb668f`。复用其 `compute_3D_filter` 思路：float32投影、正深度 `z > .2`、图像边界外扩15%、各点最小有效z及全部TRAIN相机的最大fx；没有有效视锥的点用“有效点最小z中的最大值”回填。这里用真实K的cx/cy；作者写宽高的一半，本数据两者差为0像素。[固定源码](https://github.com/autonomousvision/mip-splatting/blob/dda02ab5ecf45d6edb8c540d9bb65c7e451345a9/scene/gaussian_model.py#L132)

记原始三轴标准差为s，滤波标准差为ρ：

`ρ = min_train(z) / max_train(fx) × sqrt(.2)`

`s_eff = sqrt(s² + ρ²)`，`a_eff = a × sqrt(∏s² / ∏(s² + ρ²))`。

比例及分位汇总使用FP64。本次使用完整原生TRAIN K，不随当前训练图的临时渐进缩放改变ρ。作者估计只检查视锥，不检查遮挡；不能把视锥数量当作真实可见支撑。

| 全场统计 | 实测 |
|---|---:|
| 至少进入一个TRAIN扩展视锥 | 497,401 / 498,136（99.8524%） |
| 使用无视锥fallback | 735 |
| 0／1／2／3轴低于ρ的点数 | 227,264／224,654／41,083／5,135 |
| 至少一轴低于ρ | 54.3771% |
| ρ中位数，世界单位 | 0.000575523 |
| 最小轴／ρ中位数 | 0.850623 |
| opacity补偿系数中位数 | 0.578700 |
| 补偿系数 < .5／< .1 | 44.8492%／16.5366% |
| 按原opacity加权的补偿系数均值 | 0.545342 |

沿用早先固定诊断定义：距任一TRAIN相机小于 `.01 × scene_scale`，且距最近初始SfM点大于同一阈值；本场阈值为0.1626147世界单位。该近相机／远初始点组有 **3004点**，仅38.5486%至少一轴低于ρ，最小轴／ρ中位3.31906、补偿系数中位 **0.952914**。因此这不是专门抑制近相机浮层的证据；作者的最小TRAIN深度本来也可能在近相机处产生很小ρ。全场opacity系数变化很大，则说明不能将给最终checkpoint直接加滤波当作轻微后处理或训练期方法的充分评价。

上述系数、点数与opacity加权值都不是遮挡合成后的像素贡献，也不是可恢复误差。终点统计不能还原历史退化发生时刻。它只证明表示中有广泛低于该采样代理尺度的轴，为有限工程实验提供理由，不保证RGB、细索或共享语义改善。

[完整统计与来源](/mnt/data/SHM2026/runs/mip_sampling_cpu_audit_v1/report.json)，SHA `f7610aef80a4861589d3aeafbd5892f51b66a8a46ff49944ce283e71b7f62698`；[CPU worker](/mnt/data/SHM2026/runs/mip_sampling_cpu_audit_v1/audit.py)。本报告统计底座是较强hybrid场，**不是下面计划的mixed主control**，不把其分位数冒充mixed场测量。

## 固定主次比较与下一步

| 比较角色 | 固定模型／配方 | 共同原图PSNR |
|---|---|---:|
| 单变量主control | `rgb140_mixed`，3D filter关闭 | 29.226362 dB |
| 唯一candidate | 同mixed初始化、相机、采样、30k与硬点预算，仅启用3D filter | 尚未训练 |
| 强系统次级比较 | SSIM-fixed corner-v2 hybrid | 29.467484 dB |
| 当前已选联合系统次级比较 | legacy H3＋历史DINOv3 H+ | 29.400950 dB；其语义95.1090% |

主control与candidate保留相同raw opacity reset、raw尺度／opacity pruning以及原clone/split规则；不一起切换作者的有效opacity reset、增密指标或整套配方。因而应明确称 **Mip-like filter-only参考**，不是作者完整实现复现。开启时所有RGB、语义及相关几何证据通路必须共享同一有效scale/opacity；关闭时不得改变旧checkpoint schema、参数、随机数或render计算。需要锁定刷新时序、完整TRAIN相机列表和filter状态的保存／加载，并由CPU合同和随后一次有界CUDA合同确认，才可冻结长训。系数固定.2，不做强度搜索。

本轮支持普通checkpoint的保存、加载、render与resume，以及冻结geometry的语义暖启；冻结语义阶段继承RGB阶段的ρ和刷新计数，不因新阶段step归零而重算。**旧averaging、appearance-delta和semantic-transfer派生路径暂不支持开启filter的checkpoint**：这些工具的顶层元数据白名单尚未传播filter配置／状态，不能据其旧模型兼容性推断新filter模型也可用。本轮不扩展这些派生接口。

截至CPU复核，`tests/test_mip_filter_math.py`的14项科学合同与集成文件的10项合同合计24项通过，覆盖解析梯度、旋转协方差行列式等价、detached ρ、小轴数值稳定性、非中心主点／全相机最大fx、无视锥fallback、TRAIN来源和固定配置，以及旧off行为与冻结语义resume。它们不替代尚未进行的真实CUDA检查或性能评价。

固定最终checkpoint进入共同原图50RGB评价，无中途VAL选点；报告三项RGB、逐视图配对区间、实际N、Gaussian-step、滤波刷新和总训练成本。先回答相对mixed主control是否改善，再独立与两个更强系统比较。**仅胜过29.226362 dB的较弱control不等于达到用户要求**；不能把较强hybrid称为同底座单变量control。若继续投入门采用“PSNR至少+.15 dB且配对下界大于零、SSIM/LPIPS不退步”，应对较强hybrid也完整核验，正式门槛须在长训计划中固定，不能事后降低。

若RGB确有明显增益，还必须在这一个新场上按固定方案重新训练／评价语义，并验证最终相机输入的完整RGB＋语义输出；不能沿用旧场mask或95.1090%，也不能绕过legacy/v2教师profile保护。即便最终通过，这一轮也只建立更强标准shared-geometry底座。任何后续结构机制仍需在该底座上做独立匹配消融，不能把Mip本身重命名为学术贡献。当前不增加另一个候选、不训练、不扫描强度。
