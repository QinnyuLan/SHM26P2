# 固定 TRAIN 源残差传输诊断

状态：CPU 草稿，尚未 prepare 或执行。协议 `residual_transport_probe_v1`；只测工程潜力，无训练、语义、VAL 或采用门。普通 ED 传输是便宜基线，不是 [IBGS](https://arxiv.org/html/2511.14357v1) 的多 Gaussian 交点采样/网络复现。

底座固定已完成的 `rgb_capacity_1m_reference_v1/training/last.pt`，SHA `35b489fe45ad34ad279018609b6d5eaf6493e4c0d82f9c86ec16925268078092`。继承其完整30文件冻结包、原 `prepared_corner_v2` manifest、原 SH3、alpha/geometry 和相机，不混合 MCMC 深度。另加独立 CPU transport helper，资源检查继承已冻结 helper；所有模型/源码/运行库/编译二进制、uv、输入预期 SHA 写入新计划。准备不加载模型、不读取 RGB/valid payload；像素预期 SHA 继承已完成1M计划。

259个 labeled TRAIN 按 name 排序，索引 `floor(i*258/15)` 固定为 002、021、041、059、079、100、118、137、156、176、200、220、241、259、278、300。所有16目标从整个源 bank 排除。每个目标在其余334 TRAIN 中按 `||C_t-C_s||² / manifest.scene_radius² + (SO(3)角距/π)²` 升序、同分按 name 排序；再排除最近1个，取随后4个。不能用RGB、mask或覆盖挑源，不能补救不足覆盖。最多64不同源+16目标，实际名单与计数 prepare 时固定。

渲染仅调用原场 `semantics=False, refine=False`，每相机一次 RGB+ED。实际传入 renderer 的 FP32 K/w2c 同时保存于 NPZ；传输只将这些实际数值提升至 FP64。目标像素反投影使用 `[j+.5,i+.5,1]`，ED是相机z期望，不是射线长度或物理首表面；源投影角点坐标减 `.5` 后 OpenCV INTER_LINEAR 采样。详见 [坐标合同](residual_transport_coordinate_contract.md)。源残差由同网格 prepared 真实RGB减 clamp 后场RGB，绝不使用旧legacy缓存。

固定 true、wrong_residual_correspondence、zero 三路。wrong 仅水平镜像源残差和RGB-valid，源ED/alpha仍在正确几何坐标。两传输路共用有效域：目标/源alpha≥.95、正有限深度、`|z_project-ED_source| / max(z_project,ED_source)≤.01`、正常与镜像RGB-valid插值足迹均全有效。每像素相同源数、均匀权重；无支持处残差为0。不增加视图/改阈值。

前16传输数组全部保存并 SHA 记录后，才允许目标真实RGB的读取、hash和解码；共享 camera valid 可先读。无任何语义标签/annotation读取。源和目标所有数组保存在 `/mnt/data`，包括各相机RGB/ED/alpha/valid、源残差、实际K/w2c，以及目标true/wrong残差、共同支持和source_count，足以CPU复算。

A为固定目标序列偶数索引，B为奇数。true/wrong各拟合A、B两个全局λ（共4个），每图只用另8图拟合的λ；zero固定0。闭式系数 clip 至[0,1]；分子/分母先按每图**完整目标RGB-valid**像素和3通道取均值，再等相机平均，不能仅用传输支持。拟合目标是不裁剪RGB的MSE；另外报告最终clip[0,1]后的完整MSE、共同支持局部MSE和覆盖。不得称λ是clipped目标的精确最优。所有系数/16图/三路等图均值均报告，不选好方向。模型本来见过这些目标，因此只叫源图像排除和系数OOF，不是模型crossfit或新视图泛化。

一次执行，无重试；内120秒、外180秒，含哈希/I/O/CPU传输。记录scene/high/low raster实际调用、源/目标/valid解码、峰值显存、运行数值flags及恢复、场tensor未变。真实RustDesk可存在，未知compute进程拒绝；不终止用户应用。CPU合成测试使用 `-p no:cacheprovider` 与禁bytecode。冻结和唯一GPU执行由root负责；失败原样保存，任何低覆盖/负结果均不重新择源或调系数范围。
