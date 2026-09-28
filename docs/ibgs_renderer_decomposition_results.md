# IBGS 起点渲染差异：固定 AA／near 分解

本轮在同一 `35b489fe…078092`、996009 点、SH3 起点场上完成 48 次新前向，
没有训练、参数更新、后端修改、原图或语义标签解码。16 个相机和评分目标均为
既有 TRAIN 缓存，不能视为泛化评价。A 分支的 clipped RGB 在全部 16 图上与
原 AA 缓存逐位相同，最大差为 0。

| 分支 | 设定 | 相同 valid 支持上的等图 MSE | 平均逐图 PSNR |
|---|---|---:|---:|
| A | gsplat antialiased，near=.01 | 0.000885358904 | 32.300961 dB |
| C | gsplat classic，near=.01 | 0.001588202656 | 28.760075 dB |
| N | gsplat classic，near=.2 | 0.001620087126 | 28.661754 dB |
| I | 复用已完成 IBGS 原始起点输出 | 0.001619526471 | 28.661841 dB |

表中均为 clipped RGB 指标；每图先计算 PSNR 再等权平均，不是平均 MSE 的
PSNR。完整 unclipped 指标也已保存。A/C/N 均固定 `eps2d=.3`、far=1e6、
原 FP32 相机输入与 SH3；I 是旧起点原始 FP32 输出，不是后来任一训练末态。

固定顺序的 clipped MSE 差值为：

| 路径 | 等图 MSE 差 |
|---|---:|
| C−A | +0.000702843753 |
| N−C | +0.000031884470 |
| I−N | −0.000000560655 |
| I−A | +0.000734167567 |

三段相加与总差闭合。该固定分解中，从 antialiased 改为 classic 的退化远大于
后续 near 改动。两种 gsplat 模式都保留 `.3I` covariance blur；classic 去掉了
AA opacity compensation，并可能随之改变 opacity 阈值支持和投影范围。
因此此结果比此前“后端总差异”更具体，但仍是有顺序的对照与代数记账，不是
相互独立的物理因果归因。

I 和 N 的平均 PSNR 只差约 +0.00008744 dB，不代表逐像素相等、所有场景等价
或反向传播等价。剩余差异仍可能包含 alpha cap、支撑／半径规则、近面等值边界、
投影和 SH 数值运算次序。不能用裁剪每个 Gaussian 的 opacity 来模拟像素 alpha
cap。本轮没有采用门，不更换 E，也不能把后续恢复起点损失称作新颖收益。

执行自然退出 0：48 scene／48 high raster／48 low raster，16 次 cached target
读取，零 backward／optimizer；hook、数值标志、场参数及 requires-grad／模式恢复
均由原回执记录为通过。内部 8.767719 秒，外层 `/usr/bin/time` 9.12 秒；峰值
CUDA 分配 822,431,744 字节。独立 CPU 薄复核重新读取全部 64 份预测（48 新输出
及 16 份旧 I）、16 份目标缓存和 16 份旧 AA 缓存，以 FP64 分块平方求和重算
raw/clipped MSE、PSNR、每图及总体路径。最大差 **7.105427357601002e−15**，
沿用 `1e−12` 比较容差；AA 16/16 exact。35 个冻结源、59 个输入、10 个安装源
及实际导入绑定均通过。CPU 审计内部 3.804532 秒、外层 3.869174 秒，自然退出 0。
没有 GPU、checkpoint 反序列化或原图／标签解码；runtime 计数和恢复核对的是
源码与记录，不是独立重执行。

产物：

- [计划](/mnt/data/SHM2026/runs/ibgs_renderer_decomposition_v1/plan.json)：`3fdfdd0f45c45d172aee3b6979a6e4b8b72c4c01134d0591a8abd8ed80b06bfa`
- [执行回执](/mnt/data/SHM2026/runs/ibgs_renderer_decomposition_v1/execution_receipt.json)：`997ec5e108a7c69990f064e7c28bad44ceea0152c9b1f57b0c7fb8d9ef0ec1de`
- [原分析](/mnt/data/SHM2026/runs/ibgs_renderer_decomposition_v1/analysis.json)：`87257b11cd3fb26ecdd1cafa2c5fdd945ebcf3bbd1b7e435aa0129d4a8d1d47c`
- [独立 CPU 复核](/mnt/data/SHM2026/runs/ibgs_renderer_decomposition_v1/independent_cpu_review.json)：`26129cf26699fce962790ff84b154e6f191aa23ffecf2d9cdb08d99b878b667b`
- [复核自然退出回执](/mnt/data/SHM2026/runs/ibgs_renderer_decomposition_v1/independent_audit_launch_receipt.json)：`6da2b924b6bf8c99389ad1ffe0cf4f9a8ee8d9c72afcb23e7fc2f499bf161c49`

## 后续 near 改动的只读范围核查

旧 IBGS 的 `cuda_rasterizer/auxiliary.h` 中 `in_frustum` 使用
`p_view.z <= 0.2f`。它同时被 forward preprocess 和
`rasterizer_impl.cu` 的 `checkFrustum`／`markVisible` 调用；RGB 与
`render_depth` 共用该 preprocess。backward 依据保存的正 radii 选择点，没有
需要同步修改的第二处近面常数。

根任务已准备隔离源 `/mnt/data/SHM2026/third_party/ibgs_aa_port_v1/source`，
将此条件改为 **`p_view.z < 0.01f`**，阈值及比较号一起对齐 gsplat；far 不变。
这会影响目标 RGB，也会影响 source depth、遮挡和 fusion 支持，不是只改变
最终颜色。旧包与二进制保留；旧局部数值预检不自动覆盖新纳入的近点。此处仅
记录源码影响范围，不宣称新版构建、实际渲染或训练效果已验证。
