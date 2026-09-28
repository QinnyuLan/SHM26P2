# IBGS 抗锯齿兼容性检查

2026-09-27。先检查原始 996009 点、SH3 检查点的前向迁移，再决定是否训练。此前固定16 TRAIN分解为：gsplat AA 32.300961 dB、classic 28.760075、classic near=.2 28.661754、IBGS 28.661841。AA参考已精确重放。这不是新方法的收益，而是检查点与渲染配置不匹配。

本检查只用相同16个TRAIN相机、原检查点、原FP64相机输入与固定缓存评分域；运行16次新前向，零backward、零优化、不加载原图／语义／VAL。先保存所有原始FP32预测，再加载16份缓存目标评分。记录原始及clipped RGB的逐图MSE、PSNR与16图等权均值，并报告与旧AA输出的像素差。不得因相近PSNR而声称逐像素等价。

唯一配置改动：

- 在独立近裁剪后端中，将 `p_view.z <= .2` 改为 `p_view.z < .01`；比较符与gsplat一致。原后端、环境与历史结果不覆盖。远裁剪仍保持上游行为。
- 向IBGS传入 `opacity * sqrt(max(det(C)/det(C+.3I),0))`。`C`由实际均值、激活尺度、已归一四元数和当前相机投影得到，保持IBGS的视场夹取约定。其余IBGS卷积、alpha上限、足迹边界与数值顺序不改变。

这采用已有[gsplat抗锯齿补偿](https://docs.gsplat.studio/main/apis/rasterization.html)，相关先例是[Mip-Splatting](https://niujinshuchong.github.io/mip-splatting/)，**不作为学术创新**。这里也没有实施完整Mip-Splatting的3D平滑滤波。网站只用于公式归属；执行绑定本机既有版本和源码。

补偿由Torch计算，正定内部使用该公式的真实自动微分；非正det比例采用显式零支路，避免sqrt零点反向NaN。gsplat已有手写反向带 `rho+1e-6`，因此不能宣称反向逐位等价。IBGS原后端还保留alpha cap、视场夹取区和median权重回传的局部近似；后续有限差分检查只认证明确的局部范围，不将其扩大为完整融合梯度证明。

16次前向限内部100秒／外部120秒。源码与二进制冻结，使用新的输出目录；保留失败记录。该检查通过只允许继续局部梯度和实际数据训练预检，不据TRAIN分数替换当前最佳系统。
