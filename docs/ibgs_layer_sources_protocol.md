# IBGS 源视角逐层查询：固定诊断合同

本次只观察旧 `ibgs_warm_matched_v1/full/last.pt`、原非 AA / near .2 后端与已完成的 `ibgs_thin_rays_capture_v1`。不训练、不改 selector、不编译 CUDA，不读源/目标 RGB 或语义标签。固定 16 TRAIN 目标的 8192 中心全部保留人口记录；既有有效域且 summary-consistent、算术有限的 8131 中心形成查询。每中心取原 median4∪top4 中有限正平面深度的 Gaussian，保留窄前与其他贡献；不依据错误或本次结果选点。

**来源与两阶段。** `--prepare` 绑定原 capture/replay 的自然完成回执、旧端点/源码/后端、相机合同与原四邻居，保存不可变候选 NPZ。64 个邻居互异且均不在 16 个目标内。GPU `--capture` 每源仅一次空源 full-geometry render 与一次真实 `render_depth`，共 128 raster，0 backward/optimizer/fusion。复用原 ABI prefix/address 解析与返回不变的观察 hook。保存原 buffers、all_map、实际 SH RGB、raw RGB、median 摘要与原 Torch CUDA **4 源 batch 矩阵乘法**得到的 `ref_to_src`。源深度使用原 `render_depth_only` 并乘 radius-one eroded-valid；不能以 full-geometry median 代替。仅从已绑定旧 NPZ 读取一个共享 `valid` 成员（不访问邻近 RGB 成员），确认全 TRAIN 共用同一 valid 路径。不创建 TRAIN RGB bank。

CPU `--analyze` 逐源处理，不同时载入全部 source buffers。对每候选把其平面交点投影到源；对目标原 production median 加权深度另作投影。按原 shader 的 FP32 运算顺序重建，但不声称复制 CUDA FMA。texture 的 bounds 先判断原 projected UV；再先作 FP32 `uv+.5`，以该 texture 坐标生成四个 clamp taps。固定同时报告理想 bilinear 和小数 8-bit、nearest-even 量化读出（包括恰好 midpoint 标记）。CUDA 文档规定了 8-bit 小数权重，但此 CPU tie/归约实现不是逐位 production texture 证书。[CUDA linear filtering](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#linear-filtering)

**三个不同量。**

- `G_M`：原整源门的 CPU 重建。目标原 median4 加权点在源的深度相对误差严格 `<.01`、正深度/边界有效，且原 median4 各槽经自身投影后 `sourceDepth>0` 支持的权重和正。保存未乘最后支持条件的 depth-only predicate，避免混同。权重来自既有 summary-consistent 重放，不伪称 CUDA 内部原值。
- `G_F`：单个候选层的**反事实**同形式 `.01` 深度门；原程序没有这项逐层一致性门。两种插值判定不同或存在量化 midpoint 时标 ambiguous，仍保留误差、margin、坐标、无效 z/bounds/valid、候选和中心分母。不得把 `G_F` 不通过直接叫遮挡。
- `W_F`：候选投影四个源像素中同一 Gaussian ID 的原 RGB `alpha*T`，按两种 texture 权重插值。全量合法 RGB 贡献都可匹配，不限制该 ID 是否属于源 median/top4，亦保存源 plane z 以分清 RGB 支持与合法层。每个源 tap 重放须对照实际 T、n_contrib、RGB、median low/high/sum/depth；原容差 `2e-5 + 2e-4*abs(production)`，离散量必须相同。失配全部保存；只有所有正插值权重 tap summary-consistent 的查询可作主要 W 描述。重复 clamp tap 按原权重自然累加，不重复算成额外射线。

固定描述分层是 all / narrow-front / **omitted-narrow-front** / other。窄前定义沿既有 `.01` 相对前移、blurred conic 短轴 std≤2 px、轴比≥4、SPD；不更改阈值。因为候选已限定 union，omitted-narrow-front 恰是 top4 找回子集，**不覆盖所有遗漏薄层**。各层公开完整候选数、无效/ambiguous/tap 失配数、G_M×G_F 计数、可解释 W 连续分布及同 ID 正 W 与两门的联合计数。0.01 质量层、重复 ID 或 top4 的定义性质量增益不成为训练/采用门。一个模型 ID 的跨视图贡献也不是物理索、正确源纹理或泛化收益证明。

**预算与输出。** GPU 内 300/外 360 秒；CPU 内 600/外 660 秒；总新输出≤40 GiB；时间含本阶段校验/IO。单源完整保存后释放，保留失败回执，不重试。入口 `scripts/diagnose_ibgs_layer_sources.py --prepare --output …`，之后只运行冻结快照的 `--capture`、`--analyze` 且传 `--plan` 与 `--expected-plan-sha256`。候选表、每源 capture、query NPZ、四 tap 完整重放 traces、`analysis.json` 足以追踪分母；无自动科学通过门。GPU 仅由根代理在代码审查/冻结后启动。
