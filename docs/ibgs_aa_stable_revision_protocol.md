# Factored FP64 AA 的固定范围重验证

旧 AA 训练在第 62 步的 004.png 前向失败，保留 step=61 的失败 checkpoint。旧 FP32 Gram 矩阵相减给出负 determinant，而同一已激活输入的 FP64 因子计算为正。这不是据失败改数据人口或放宽 determinant 门。

新 provider 固定为 `bridge_rgs.ibgs_antialias_stable`，renderer profile 为 `ibgs_centered_corner_v2_aa_factored64_near001_v2`。plan 绑定模块 SHA、实际导出的 `PRECISION_POLICY` 以及同一 near=.01 隔离二进制。新计算将实际已激活输入转换为 FP64，用三个 2×2 minor 的平方和计算 Gram determinant，再算 `D+.3*||B||²+.09`；不重做 activation/quaternion normalization，不引入 floor。有效 opacity 回到原输入 dtype；后端自身 FP32 covariance/conic 和近似 VJP 不变。名义 FP64 .3 与 CUDA .3f 也不声称逐位相同。

新薄入口按 mode 单独准备新的输出目录与计划，执行必须使用独立进程。其余 source package 保留旧字节：仅在进程内将旧模块的 `aa_opacity_rasterizer` 属性显式指向新 callable，记录真实 callable 的 module、文件、SHA 和 precision policy，finally 恢复。旧 module 身份和源码不替换；两份模块都受实际 import 审计。

| mode | 复用的冻结逻辑 | 调用与门 |
|---|---|---|
| compatibility | 原固定16TRAIN `run` | 16 forward；全预测保存后读原缓存目标；原指标与状态门不变，不新增相近PSNR门 |
| gradient | 原合成33forward `run` | 33 forward / 3 backward；原4方向主h及容差、zero control不变；cap反例仍另报 |
| preflight | 原002/041 `run` | 8 source depth / 2 target / 2 backward；原有限梯度、参数与恢复门不变 |
| failure | 新固定004回归 | 精确61-step失败态，1 raw forward / 1 backward；0 source、0 GT、0 optimizer |

总计 60 次光栅、6 次 backward、0 次 optimizer。不重复其它早期 backend 测试套件，也不把旧 module 的成功回执当作新 provider 已通过。每个 mode 均保留独立执行/自然结束回执；`status=completed` 与 `numerical_status=passed` 分开，训练准入需四个新回执使用同一 provider、精度策略和二进制。

failure mode 只读取明确绑定的 `full/failure.pt`（SHA `abbe8973ebc7649e99b162f9d878129232fe5475ffe8672f2e4fc9b48c5b5d8f`），验证 status=failed、step=61、arm=full。直接复制其 8 个 field 参数，沿原 raw manifest 相机创建 004；`render_geo=False`、`return_depth_normal=False`，不建 source bank。目标为全 native RGB 的固定线性标量 `mean(RGB·[1,.7,−.2])`，权重先 FP32 再与 RGB 转 FP64 归约。只要求实际 raw 路径的 means/rotation/scaling/opacity/SH0/SHrest 六类梯度存在且有限；normal/offset 不在该路径中，不能要求有梯度。记录原始输出并证明 field 未更新、hook/数值标志恢复，不作质量评估或 FD 认证。

CPU prepare 只解析 metadata、读取绑定字节做 SHA、复制源码；不解码图像、不载入模型、不 GPU。failure 诊断原目录没有 plan.json，绑定它真实存在的 execution/launch/NPZ；不补造历史计划。compatibility 使用缓存目标，preflight 有意使用原两张 TRAIN RGB；failure/gradient 无目标像素读取。各 mode 延续预算：compatibility 100/120 秒、gradient 90/120 秒、preflight 120/180 秒、failure 90/120 秒（内部/外部）。

未来训练另建 v2 计划，两臂均从原 `35b489…` field、原 seed/order、fresh Adam 开始各 6000 步。失败 checkpoint 仅可作为上述回归输入，不允许续训。这仍是既有 AA 公式的数值实现修订，不构成新方法或质量收益。
