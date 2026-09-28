# 固定旧 IBGS 端点的薄射线缓存导出与 CPU 回放

状态：实现草案，尚未 prepare 或执行。该诊断不训练、不读取 VAL，也不构成采用门。旧 `ibgs_warm_matched_v1/full/last.pt` 固定为 `e977dc1c5c676e930c47ed78e56a5f95c25c8127a5c02f23876ddb8b4e1aa3ee`；二进制固定为原 isolated backend `436b2b55df2b37cddda59e606ec0855ad125eb211058b502197674bd2b93adaf`。使用旧 near=.2、无 AA 的冻结包，与正在训练的新 AA 端点完全分开。

`capture_ibgs_thin_rays.py` 有 `--prepare / --capture / --replay` 三入口。prepare 只读取已自然完成的训练/恢复回执、独立恢复复核及相机元数据、SHA；不反序列化模型或解码像素。复制原 27 个冻结文件（保留其中合法的 `__pycache__/__init__.py`），新增 worker、NumPy 回放库与测试。恢复 run 的 full/raw 16 个数组只用于旧 raw 身份检查，不引入目标 RGB。

固定 16 TRAIN 为 002、021、041、059、079、100、118、137、156、176、200、220、241、259、278、300。每图 32×16 等分格内按固定 name/cell SHA 选择一个中心、加不重叠 3×3 邻域；所有坐标在导出前确定，不按标签、图像误差或质量选择。共 8,192 中心、73,728 射线，原图 1320×989 整数数组坐标，与 corner-v2 相机的 .5 像素约定一致。

GPU 阶段只有 16 次 target render/raster，`render_geo=True, render_depth_only=False, buffer_length=4`。每相机 `nearest_id=[]`，原 renderer 自行构造一张全零 RGB/depth/相对变换占位源。没有 source bank、depth refresh、融合网络、反向或优化；不解码 source/target/valid 图像。原 forward 在源纹理循环之外累加 target median ring、局部权重、low/high 和 median depth；空源只影响源颜色支持，不影响这些 target 摘要及 raw RGB。导出 raw 必须与已审计的旧 full/raw 缓存逐位一致，否则保留证据并停止。旧缓存没有 median 数组，不宣称 median 与旧渲染逐位复现。

hook 仅包 `_C.rasterize_gaussians`，以原 29 个参数调用原函数并原样返回其 13-tuple。保存实际三缓冲区 GPU 地址及 `%128`、原大小、CUB workspace 前的完整 ABI 前缀，以及实际 all_map、背景、view/projection matrix、campos、raw RGB、median depth；若用了 colors_precomp，必须优先保存/使用该实际颜色。每字段 SHA、尺寸、dtype 绑定。按实际 GPU 地址对齐，不能以 CPU 副本地址或 offset=0 代替；image ranges 仅 tile_count 项已初始化，不读取其余为有效 tile。未改 CUDA、未重编译。

CPU replay 只在 capture 自然完成后读取这些缓存；逐 tile 共用实际 point_list、每块最多 32 rays NumPy 重建。按原 power、alpha cap=.99、alpha≥1/255、T<1e−4 提前终止及原 median ring 顺序处理。全部有效 RGB 贡献保存为逐图 ragged NPZ（offsets、ID、ordinal、alpha、incoming T、w、plane z、center z、median/top4 mask 和 slots），另存全射线生产/重建摘要与阈值距离。被采样 tile 引用的 means2D/conic/实际颜色/all_map/depth 必须有限，否则整个诊断失败，不能静默缩减总体。计算得到 inf 的 plane z 保留原 median `z>0` 行为，但 top4 只接受 finite positive z；该射线标 nonfinite。

预设摘要比较：n_contrib、median low/high 必须整数 exact；T、raw RGB、median weight/depth 使用 `abs≤2e−5+2e−4·abs(production)`。这是描述性数值一致标准，非 CUDA 误差上界；CPU `exp`/分离乘加与 CUDA `__expf`/FMA 不保证阈值判定相同。所有 mismatch 原样统计，不改容差。即使摘要一致也仅称“与生产摘要一致的 CPU 重建”，不是 exact CUDA contribution ledger。保存的 inf/NaN 保留在 NPZ，strict JSON 中以 NA 配合明确非有限状态表示。

CPU 阶段可从已绑定旧 target_cache NPZ 懒加载 **仅 `valid` 成员**，绝不加载邻接 `rgb`；GPU 导出仍 0 图像解码。每射线保存 validity，完整总体与原有效域并列报告，绝不删射线或重采样。机制描述限 finite 且 summary-consistent 子集，同时公开全 8,192 中心/73,728 rays 分母及各图有效/不一致数量。

后续纯描述分层预先登记：相对原 median z 近至少 1%；blurred conic inverse 为 SPD 时短轴 std≤2 px、轴比≥4（否则 NA）；全部窄前置群未归一化 w 之和<.5。先报告连续质量分布与每图质量，再报告 median 遗漏且 top4 保留集合，mass≥.01 仅作“非忽略”描述。相同 ID 在一个 3×3 内≥3 条射线和≥2 相机重复分别报告，不能称真实 cable。top4 的质量优势部分来自其定义，不是画质、薄线真值或创新证据。

预设预算：capture 内 300/外 360 秒，CPU replay 内 600/外 660 秒；capture 文件最多 24 GiB、replay 最多 8 GiB，单视图 ABI 前缀最多 2 GiB，prepare 要求至少 34 GiB 可用磁盘。超限/无效输入自然失败并保留产物，不自动加时/重试。时间包含 hash/I/O；峰值显存从模型加载前开始记。root 串行启动 GPU capture，随后主 uv、`CUDA_VISIBLE_DEVICES=''` 启动 CPU replay；当前仅交付代码与合成验证。
