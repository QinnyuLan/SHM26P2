# IBGS AA 两 TRAIN 视图反向接线预检

固定 002.png、041.png，原 996009 点、SH3、corner-v2 检查点和 TRAIN-only data contract。继承已通过 16 视图前向兼容的冻结包和独立 near `< .01` 二进制；对所有标准光栅调用使用同一 `aa_opacity_rasterizer`。CPU prepare 只读 metadata、哈希文件并复制源码，不反序列化模型、不解码图像、不初始化 GPU。

固定预算为原近邻的 8 次 source depth、2 次目标 render、2 次 backward、10 次 AA hook、0 次 optimizer。无 checkpoint 输出、无 VAL/语义标签。每个目标最多 4 个几何源，使用原 4-buffer/.01 深度阈值；源 RGB 只来自 TRAIN bank。按完整原 FP64 K/pose metadata 创建相机，内部 camera tensor 与既有 adapter 相同。

字段从原始 RGB checkpoint 装入，法向为最小轴、offset=0；融合网络使用固定 seed=20260927 的原始新鲜初始化，不训练、不做 warm-stage 零残差初始化。目标预测完成后才读取对应 TRAIN RGB；这不是所有目标图像完全未被 source bank 看过的保证。损失为 `.5*(raw RGB + fused RGB) + .03*normal + .3*source photo`，通过已完成 warm-training 的 `training_losses(step=6000)` 计算。这里的 step 仅选取完整融合损失，没有运行训练或更新次数。相较早期 `preflight_ibgs_port.py`，继承已修正的 `abs().sum(0)>0` source support，避免带符号特征求和漏支持；权重、valid erosion 和 fresh network 不变。旧脚本与回执不修改。

成功条件仅为两次真实损失和全部 8 类 field 参数、融合网络的梯度存在且有限；field/network 数值保持不变，AA hook 与数值标志恢复，计数完全闭合。记录梯度范数但不新增事后阈值。`status=completed` 与 `numerical_status=passed` 分别表示完整执行与上述有限性接线门。先验必须绑定已自然完成的 AA16 兼容及局部合成 FD 回执、独立 CPU review，且后者与本次 AA module / binary 完全一致。

这不是完整真实场有限差分证明。合成未饱和 FD 通过不能覆盖 backend alpha cap 的已知近似；本次也不宣称前向与 gsplat 逐像素、反向或 trained quality 等价。成功只支持 root 决定是否运行独立冻结训练。

内部 120 秒、外部 180 秒，只执行一次，失败保留；源码、原 field、data contract、像素文件与隔离 backend 在执行前后核 SHA。输出位于新的 `ibgs_aa_port_preflight_v1`，不覆盖历史结果。
