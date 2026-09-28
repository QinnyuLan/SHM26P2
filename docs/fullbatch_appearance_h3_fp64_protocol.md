# H3 的独立 FP64 图像损失预检与条件性外观优化

2026-09-27。本文定义一个新的数值实现实验，**不改写原 FP32 协议或失败门**，不提出新优化方法。此前两次执行分别保留：首次在第 3 次 render 后发生 CUDA scalar 转 NumPy 的报告异常，未完成技术验证；修复报告转换后的独立 40-render 预检正常完成，但 **15/18 项 FD 通过，3/18 未通过，停止原 FP32 full-batch**，没有正式优化候选或 VAL 评价。第二次 receipt：`/mnt/data/SHM2026/runs/h3_fullbatch_objective_precheck_scalar_fixed_v1/execution_receipt.json`，SHA `483dbdc14a1462046f4f4d318d6a295d02fedb22f93d23c85d2e86fb16854f29`。

## 已知证据与新实验的唯一精度变化

失败项全为 TRAIN `002.png` 的 background logits：ε=`1e−3,5e−4,2.5e−4`，scalar FD 对实际参数方向内积的误差分别 **5.6778%、9.7052%、8.6968%**，超过原 5% 门。令 `A=gθ·实际 FP32 参数中央方向`、`B=g_image·实际 FP32 预测图中央割线`、`D=两端 scalar loss 中央差分`；同三项的 `|A−B|/|A|` 仅 **.014891%、.014632%、.029802%**，而 `|B−D|/|A|` 为 **5.66286%、9.71981%、8.66704%**。这将这些方向上的主要差异限定在图像局部导数与 scalar loss 有限差分之间，不能据此证明所有 renderer 梯度正确，也不能区分全部舍入、有限步长非线性或 L1 kink，更不能归因某个 CUDA kernel。

root 的固定亮平滑图 CPU 数值复现另发现，同一公式只提高图像到 loss 的精度可以显著减小 scalar FD 偏差；该合成证据只支持开展以下独立数值预检，不替代 H3 的真实相机门，也不是 H3 精度收益。

新实验只把 **warp 后 prediction 与原 FP32 target 数值转换成 FP64，然后计算完整 L1、7×7 SSIM、中间乘加/窗口统计和 scalar loss**。target 保留原 `uint8→FP32/255` 的数值后再转换，不借机改变输入目标。数学仍为 `.8*L1+.2*(1−SSIM7)`，SSIM 只计完整窗口中心。scene 参数、SH/raster 输出、clamp、原图 warp、RMS 二阶矩及更新、最终存盘参数均保持 **FP32**；反向经过 FP64→FP32 cast 返回 FP32 图像/参数梯度。FP64 点积聚合与 actual FP32 displacement 沿用原合同。这不是端到端双精度 renderer，也不等同于重新训练或修改旧 loss 全局默认行为。

## 固定 40-render 预检

输入仍为 SHA `22bc8a2ddb260f93cb01b17857c97b2bb0873038efdb9318545cb2bdbb045226` 的同一个 H3 cross 场，legacy 数据与原 TRAIN 相机、原始图像保持不变。仍只测试排序 TRAIN indices **0/175，即 002.png / 205.png**。每 view baseline 1、三个参数组×三个 ε×正负端点 18、一次临时 RMS 更新后 1，共 **40**；无新视图、无额外幅度、无 VAL、无持久模型修改。

方向由该新 FP64 图像损失的梯度按原规则 `g/absmax(g)` 生成；方向的数值可能因此改变，这是精度实现的直接结果，不是独立搜索方向。每次 FD 从同一个恢复后的 FP32 base 参数构造端点，FP64 scalar loss 差分对 `A=gθ·[(θplus_FP32−θminus_FP32)/(2ε)]`，固定全部 **18 项误差 ≤.05**。零方向、相同端点、零解析值或非有限值拒绝；不使用 ULP 宽免门。仍同时要求原 warp 一致性 **≤2e−6**、全模型/相机/flags/modes 与所有输入完整恢复、两个独立新零 RMS 状态的临时一步通过原 floor、Armijo 和 **.9..1.1** 实际/预测下降比值。所有门均未放宽。

新 plan/receipt、独立入口及其源码必须明确 `image_loss_dtype=float64`，绑定实际 FP64 helper、仍相同的 renderer/warp、本文在 run 内的不可变副本及全部来源；原 FP32 gate 不能凭新 receipt 获得授权。只有新预检通过、独立审查、root 明确 GPU 交接后，才允许新的 FP64 full-batch 运行。失败停止，不扫 ε 或改阈值。

## 条件性 full-batch 与最终采用门

预检通过也不保证训练或 VAL 收益。后续若获准，仅将 [原 H3 协议](fullbatch_appearance_h3_protocol.md) 的逐图图像 loss 与跨图标量计算明确设为 FP64；参数梯度仍在 FP32 叶子上逐图累积。原 FP32 helper 强制 scalar dtype，因此新入口必须绑定一个明确允许且要求 FP64 loss 的独立实现，不能把它描述为原 helper 完全未变。三颜色键、350 TRAIN 等权目标、相机映射、无 mask/VAL、rates、β1=0/β2=.999、Adam epsilon、α=`1,.5,.25`、实际 FP32 位移、floor/Armijo/事务回退全部保持原数值。预算仍为 **20 accepted / 40 complete passes / 360 优化秒**，外层 **600 秒**，部分 pass 丢弃、零接受不发布候选、无中途 VAL、没有替代 endpoint。

最终仍重新生成同一场的 plain 和固定历史 H+ `.5` 全部 RGB/mask，不拼接旧 mask；主采用对照唯一是“新场 + 原 H+ .5”对“当前 selected H3 + 同 H+ .5”。共同官方原图协议固定 50 RGB/41 标签，RGB bootstrap 5000 次、seed20260926。六项门保持：**PSNR ≥+.15dB、配对 95% CI 下界 >0、SSIM 不降、LPIPS 不增、pooled 五类 mIoU 与 cable IoU 各下降不超过 .002（.20pp）**。plain 只描述，不作为 VAL 选择的替代分支。任一采用门失败保留当前 selected，不追加优化或调整教师融合比例。

本文件是新实验的预先定义，不是已通过或已执行的证明。旧 FP32 技术失败、历史模型/评分和冻结文档保持原样；是否采用必须依据新数值预检、完整执行来源与独立终点评价。
