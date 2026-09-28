# 固定旧场的 350 TRAIN top4 源层缓存

本入口只准备下游固定场证据传输的源缓存，不训练、不推断改进。执行仅由 root 启动，状态以 fresh `/mnt/data/SHM2026/runs/ibgs_layer_cache_v1` 的 plan、execution 和自然退出回执为准。当前不执行 prepare 或 GPU。

场固定为旧 `ibgs_warm_matched_v1/full/last.pt`，full/6000/996009/SH3，SHA `e977dc1c5c676e930c47ed78e56a5f95c25c8127a5c02f23876ddb8b4e1aa3ee`；原 non-AA near=.2 backend SHA `436b2b55df2b37cddda59e606ec0855ad125eb211058b502197674bd2b93adaf`。逐字继承该训练包的 official renderer、BridgeCamera、warm loader，不使用 AA 新末态。selector Python SHA `fd20717c1e8a64f29d0ea827c03e738a9d165634154cafa4bf2471a9bb79fc2a`，已编译二进制 SHA `5847a3c2595edc147a40469dc9c3f09e732e056dd72333a9dfb946e440a631d4`；绑定 `ibgs_layer_select_check_v1` 的 plan、passed execution、自然退出与分析，不触发重新编译。既有数值检查不等于所有 CUDA ledger 的逐位数学证明。

数据为固定 data_contract SHA `599f6cbf09a16f56bef6f37d0e0f8ce0a3f24449376213963ee033fc2ea9d84b` 内按 name 排序的全部 350 TRAIN 相机，1320×989，原 raw-double K/w2c 经原 BridgeCamera 生成实际 FP32 矩阵。每相机一次 source-free fullgeo render、一次 selector，严格 **350/350**。原渲染器仅接收全零源占位，源 RGB bank 不实例化、不读取 TRAIN 照片、VAL 或语义标注。共享 TRAIN undistortion-valid 可以读取一次，并复用原 `erode_valid(radius=1)`，图像外按 invalid 处理。

新 `ibgs_layer_adapter.py`（SHA `450a95687eeece23066bf0a950caaa84d43d29c6aaff1f925ad14f4c47d198b1`）只承接冻结渲染器 live buffers 与已编译 selector，不导入 evidence/fusion；缓存与后续融合独立。

selector 返回原始、未归一化的 alpha×T top4。每图独立保存三个无压缩 `.npy`，可 mmap：`ids:int32[989,1320,4]`、`depth:float32[989,1320,4]`、`weights:float32[989,1320,4]`。缺失槽严格 −1/0/0；非缺失 ID 必须在原 field 行范围内、同 ray 不重复，深度和质量有限且正；质量≤1，总质量只容许 8×FP32 epsilon 的求和舍入，不 clamp、不重新归一化、不静默删坏点。所有 selector status 必须为 0，输入有限性与 T 的有效算术由已检查 selector 状态检查，缓存再核验实际 ID/深度/质量。失败即完整缓存不可复用，保留已有文件与失败回执。

全图层数组不乘 valid，单独保存一份 `shared_eroded_valid:bool[989,1320]` 供消费者应用。只缓存源 top4，不保存 RGB、target median/top4、base、ray、几何/binning/image buffers；源图片由以后 trainer 的 `_SourceRGBBank` 按已绑定 TRAIN 来源解码。纯数组约 21.93 GB，输出上限 30 GiB（含 NPY header，元数据另少量），内限 300 秒、root 外限 360 秒，无重试。字段全部 `requires_grad=False`，0 backward/optimizer；前后检查原 Parameter 身份及版本不变，恢复数值标志和 backend 观察 hook，末端再次检查所有 source/input/runtime SHA。

输出接口固定：`plan.json` 含 `source_snapshot/source_hashes`、`checkpoint{path,sha256}`、`data_contract{path,sha256}`、350 `views`、原 `neighbors`、`backend{path,sha256}` 与 `selector{source_sha256,binary,check_run,check_plan_sha256,check_execution_sha256}`。成功的 `cache_manifest.json` 含同一来源身份、`shared_eroded_valid{path,sha256,shape,dtype,bytes}` 及350 `records`；每条为 `name/camera/actual_camera/ids/depth/weights/selector_status_nonzero/cache_validation`。`actual_camera` 保存实际 backend FP32 view/projection matrices、camera center、由实际 FP32 tanFoV 得到的 focal、array principal=(W−1,H−1)/2 及尺寸/模式。三个数组字段各自是 `{path,sha256,shape,dtype,bytes}`。

消费者必须同时检查 execution `status=completed`、`cache_status=safe_complete_350`、root natural exit0 与 `cache_manifest_sha256`，不能仅看到 manifest 的 completed 就使用。该文件保留 `full_grid_unmasked_selection=True` 和原邻居；缓存不选择新源，不提供真实表面/缆索标签，不把 mixed source RGB 当作单 Gaussian radiance。
