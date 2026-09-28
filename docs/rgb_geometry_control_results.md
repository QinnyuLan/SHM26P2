# RGB-only 连续位置对照

2026-09-27。此控制在原三臂 TRAIN 结果之后、四状态留出结果读取之前另行固定，不改写原实验。相同起点、259 TRAIN、148 个七图 batch、fresh Adam、固定 q／SH／head 与每点每步 1/64 Mahalanobis 限幅。两个任务的损失和 VJP 都计算，唯一提案差异是只使用 RGB 梯度。协议见[rgb_geometry_control_protocol.md](rgb_geometry_control_protocol.md)。

| 状态 | RGB MSE | raw CE | raw mIoU | raw 拉索 IoU |
|---|---:|---:|---:|---:|
| 共同起点 | .000764064397 | .102152275657 | 80.659340% | 35.367241% |
| RGB-only | .000745636759 | .102151409738 | 80.663534% | 35.343828% |
| 原 joint | .000754606540 | .100634256690 | 81.110159% | 35.745790% |
| 原 PCGrad | .000754731191 | .100625025217 | 81.112166% | 35.747890% |

RGB-only 的 RGB MSE 下降2.411791%，raw mIoU仅+.004194pp，拉索−.023412pp。原joint／PCGrad约+.45pp的TRAIN原生语义变化不能由这次纯RGB续修解释；但这还不是最终部署收益或多种子证据。其固定末态官方原图评价另外执行，参照复用已审计四状态结果，避免重复渲染基线。

148次候选全部提交，2,072训练pairs＋518描述pairs、5,180标准rasters、2,072means VJP、777目标解码，0head／teacher／VAL。内部35.238785s／外层35.733959s自然退出0。该臂训练与末态描述23.220651s，峰值799,628,288 bytes；最大累计Mahalanobis位移2.298316，不把每步限幅写作累计限幅。

83冻结源、541输入及完整恢复检查通过，5冻结CPU测试通过。独立CPU审计1.279s／外层1.466s自然0，核对148次提交、最终Adam状态、累计位移界及259图保存统计，初始描述与原三臂exact，最大复算差5.55e−17。审计不重渲染／解码GT，也不再生成未保存的每步梯度或中间moment。

运行：`/mnt/data/SHM2026/runs/continuous_geometry_rgb_only_v1`。

| 记录 | SHA256 |
|---|---|
| plan | `e79a9b9fd155d6f1b2d8b04fe73d8405675bf625e1999d28e9b7f0b75fe23865` |
| execution | `ebf2f99e09d64bfd1880ec7f385f467aa17a59c423f16ba8e2ffc8fa162e47c6` |
| means-delta | `f9287b460d27bcc1ef59379df978dc0e15876082968b75df0d76f88e655d30c4` |
| independent audit | `bad79a107f20fe54d7e4a403aaba59dfb963fd96b88f8c877c210f87df91dcb7` |

## 单末态官方评价（已完成独立核验）

同50张原图／41张标注，RGB-only固定末态为 **PSNR29.419156dB／SSIM .870582162／LPIPS .271073650**；raw／scene／joint mIoU为 **79.663922%／94.425007%／95.093000%**。相对起点的PSNR+.018207dB，配对区间[+.009849,+.026776]；最终mIoU−.004205pp，区间[−.020048,+.014258]跨零。它没有形成最终语义提升。

原joint相对这个RGB-only控制，raw mIoU **+.362538pp [+.286279,+.440632]**，但scene **−.202679pp [−.287701,−.118018]**、最终joint **−.160187pp [−.219956,−.097261]**；PSNR−.028393dB且区间为负。PCGrad对应raw+.363506pp、最终−.162656pp，同样分别正／负区间。语义梯度对raw有作用的结论因此不只出现在TRAIN；但该作用仍未改善当前部署读出。这里没有辨认具体是哪条head上下文路径导致退步。

仅对新RGB-only状态重渲染50次场／head／DINOv3 H+，700次backbone forward，150新mask与150soft先生成后解码GT，50次RGB评分与123CM，复用旧四态已核验参照完成12组固定配对。内部110.855236s／外111.470783s，自然0；峰值3,552,929,792 bytes。70源／1968输入绑定、4冻结CPU合同通过。run `/mnt/data/SHM2026/runs/continuous_geometry_rgb_only_evaluation_v1`，plan `4ecadb8d513912b82db795d6567dbe176ee4387afe8ad7c925799f16f3018aba`，execution `ae8a63c2a3e3ebe6caacd0a5e23b0d0c9bdc44d2ec0d29bd6596877119007f8f`。独立CPU审计35.595s／外36.321s自然0，70源1968输入及2462项产物SHA一致，重建150 mask、汇总123个已保存CM并复算12组配对，差值0。审计SHA `4cc60cd695eeacb21264262a7fa6ef3800cbd20fc684706d79bb294f9a6dc751`；没有GPU／GT重新解码或独立再生成RGB评分和像素CM。不修改E。

![匹配RGB-only控制的原生、精修与最终读出比较](/mnt/data/SHM2026/runs/continuous_geometry_rgb_only_evaluation_v1/figures/readout_ablation.png)

可导出的[SVG](/mnt/data/SHM2026/runs/continuous_geometry_rgb_only_evaluation_v1/figures/readout_ablation.svg)及[图表数据](/mnt/data/SHM2026/runs/continuous_geometry_rgb_only_evaluation_v1/figures/readout_ablation.csv)均来自已保存配对统计，不涉及新模型选择。
