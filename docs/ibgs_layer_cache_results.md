# 350 TRAIN 源层缓存完成记录

固定旧 non-AA 场的源 top4 缓存已自然退出 0，execution 为 `completed / safe_complete_350`。这只是后续层证据传输的输入准备，没有训练或质量收益结论；采用范围见[冻结协议](ibgs_layer_cache_protocol.md)。

| 实际项目 | 结果 |
|---|---:|
| TRAIN 相机 / source-free fullgeo / selector | 350 / 350 / 350 |
| 每图 top4 IDs、depth、weights | 各 `[989,1320,4]`，共 1,050 个 NPY |
| 共享 eroded-valid | 1 个 bool NPY，解码原 valid 1 次 |
| 数组总字节（含 NPY header） | 21,933,504,008 |
| worker / root 外层耗时 | 46.4270 / 46.81 秒 |
| GPU 峰值 allocated | 1,497,400,320 字节 |
| RGB / target / 语义标签解码、VAL 读取 | 全部 0 |
| source-depth render / backward / optimizer | 全部 0 |

[cache_manifest.json](/mnt/data/SHM2026/runs/ibgs_layer_cache_v1/cache_manifest.json) 保存全部 350 条相机、实际 FP32 view/projection/focal/principal、原邻居及数组路径、SHA、shape、dtype。IDs 为 int32，depth/weights 为 FP32，均为独立无压缩、可 mmap 的 NPY。top4 保留原 alpha×T 质量，不归一化，也不乘 valid；共享 eroded-valid 单独供消费者应用。原邻居列表中 344 个相机有 4 个邻居、6 个不足 4 个，缓存没有补造或复制邻居。

来源固定为旧 `ibgs_warm_matched_v1/full/last.pt`（6000 步、996009 Gaussian、SH3，SHA `e977dc1c5c676e930c47ed78e56a5f95c25c8127a5c02f23876ddb8b4e1aa3ee`），使用原 non-AA near=.2 backend `436b2b55…93adaf` 与已检查 selector binary `5847a3c2…a631d4`。这不是稳定 AA 新末态。执行回执记录 35 个冻结源、13 个输入和 9 个 runtime 源前后不变、实际 imports 绑定正确、selector status 全零、缓存 ID/空槽/有限深度/质量合同通过；8 个 field 参数的身份和版本不变，hook 与数值标志恢复。

本页只读取完成后的回执和 manifest 元数据，没有再次扫描约 22 GB 数组或独立重放 CUDA。上述数组安全与运行恢复结论来自 producer 的实际检查，不应扩展成真实表面识别或渲染数学的独立证明。消费者仍须联合核对自然退出、execution 完成状态及 manifest SHA。

| 完成产物 | SHA256 |
|---|---|
| [plan](/mnt/data/SHM2026/runs/ibgs_layer_cache_v1/plan.json) | `447a83a57ab08a4050ff2f095039a0d0a9034e4e54fb591212b24dc67cef722f` |
| [execution](/mnt/data/SHM2026/runs/ibgs_layer_cache_v1/execution_receipt.json) | `d8b410934d637ca17e3766301e430a7f21d0a9e3032998fe2c3be96fa2ea11ef` |
| [root launch](/mnt/data/SHM2026/runs/ibgs_layer_cache_v1/launch_receipt.json) | `69334fba65c351b87cdfc43307b6ad87a94b1d707b2e2403a2a0a53ad7dde520` |
| [cache manifest](/mnt/data/SHM2026/runs/ibgs_layer_cache_v1/cache_manifest.json) | `5d920b7e9a314cc3c49a2c0bf6ea714e3181c78ea11a9eda3280622a8335d004` |
