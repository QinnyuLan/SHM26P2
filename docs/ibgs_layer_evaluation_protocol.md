# 固定场三层读出头：官方 RGB 终点评价

本合同仅在 `ibgs_layer_heads_matched_v1` 三臂各 6000 步自然退出 0 后允许 CPU prepare。训练 plan 固定为 `b9a44c95b4e68b955f4fcc27abf27917134b4c5df8f456ed31c04a69b3a41c1a`。所有状态由本轮 plan、render/score execution 和 root 自然退出回执决定；本页不宣称已经训练或评价完成。

三个唯一末态为 `median4_mass`、`top4_mass`、`top4_normalized`，主候选预定 `top4_mass`。头参数从各臂 `last.pt` 严格加载，不选 best、不加入零步或其他候选。场固定为旧 non-AA full/6000（`e977dc1c…aa3ee`），使用原 near=.2 backend（`436b2b55…93adaf`）和已编译 selector（`5847a3c2…a631d4`）；不换成稳定 AA 场。prepare 绑定三头/训练回执、原场、350 TRAIN 数据合同、源 top4 cache 全部数组、原共享 eroded-valid、训练原样 snapshot、binary、旧官方 scoring package/helper 和引用指标。CPU 准备不解码目标照片、不初始化 CUDA。

推理直接导入训练冻结包的 adapter/evidence/fusion；不 import 训练 worker。每臂 50 次独立 source-free fullgeo + selector，共 **150 target raster / 150 selector**。不刷新 source depth；350 TRAIN 源 top4 直接 mmap 原缓存，源照片仅由 `_SourceRGBBank` 从 TRAIN 路径解码。目标相机采用旧官方 `ibgs_warm_evaluation_v1` 的同 50 个原始 distorted cameras，原 double K/w2c 经同 BridgeCamera，实际 source/target 内参一致。邻居按原几何距离/朝向/名称规则选择，最多 4 个，绝不将 VAL 放入源 bank；不足 4 个保留空槽 ID=-1、mass=0、valid=false，分母仍为 4，不复制邻居。

每次复用与训练相同的 contiguous median4/top4 slots、FP32 ref-to-source、相机中心、理想 bilinear same-ID 原质量 support，以及冻结的 MLP/CNN。`mass` 为除以四源槽、`normalized` 为按已有支持总质量归一的显式消融。零支持像素返回 base；整图无支持按训练合同失败。head/evidence FP32、TF32 关闭、无 autocast；无 backward、optimizer 或 field 更新。场参数身份/版本、head 版本、hook、实际 imports 和数值标志都检查/恢复。

评价入口准备前增加被动特征统计，以解释TRAIN日志中部分步的MLP小/零梯度：在原CNN的单次forward之前只读其实际输入，记录32维pool的平均/最大绝对值、全零像素及**有源支持却pool全零**的比例，同时保存ray/base输入幅度。pre-hook返回None且在失败时也移除，不再运行head、不干预源/权重/激活，不读取GT或改变RGB公式；CPU合成验证输出逐位相同、RNG不变、空支持与全零特征区分正确。该观察只能识别实际输入是否退化，非零pool仍不证明CNN使用了有意义的层对应，更不构成机制有效的消融证据。没有改动正在运行的训练快照。

先保存全部 **150 个 FP32 native RGB NPY**，renderer 在旧隔离环境自然完成后主 uv scorer 才运行。scorer 使用旧冻结官方流程：native clip[0,1]、corner-v2 float RGB 反畸变 warp、`rint(*255)` 写 uint8 PNG。全部 **150 PNG** durable/hash/barrier 完成后，才读取 **50 个原 RGB GT 各一次**；三个候选各算 PSNR、SSIM、LPIPS，共 150 次 LPIPS。没有语义 mask、annotation、teacher 或语义指标，不把 RGB 数值冒充完整新系统评价。

固定四个比较，方向统一为主候选减 reference，5000 次按 50 张图配对 bootstrap，seed=20260926：

- `top4_mass − median4_mass`：相同场/头预算/质量规则下的目标层选择比较。
- `top4_mass − top4_normalized`：相同 top4 下的质量归约比较。
- `top4_mass − old_full_fused`：旧 non-AA full/fused 历史工程参考，训练范围与成本不同。
- `top4_mass − E_rgb`：当前 E 的完整 RGB 参考。

前三项报告三个 RGB 指标和区间，不另造机制通过门。**唯一采用门相对 E** 沿原四项：PSNR ≥ +0.15 dB、PSNR 配对 95% 下界严格 >0、SSIM 点差 ≥0、LPIPS 点差 ≤0。无论结果均不自动替换 E；语义路径未重测，不能声称共享几何创新。50 图是反复使用的开发评价人口，不是盲测或选择校正证据。

分离两次 root 执行：render 内限 600 秒/外限 660 秒；score 内限 300 秒/外限 360 秒。输出 fresh `/mnt/data/SHM2026/runs/ibgs_layer_heads_evaluation_v1`。一次固定执行、exclusive started/receipt，不覆盖失败产物。计时明确包含源缓存 I/O 和源照片加载，不能仅把 scorer 时间称端到端 FPS。
