# 分层头技术预检与正式运行状态

2026-09-28：技术预检 v2 已自然退出0；正式三臂各6000步和统一评价现已完成，见[结果](ibgs_layer_heads_results.md)。主候选不采用，E保持。以下预检仅验证训练可执行，不能证明质量收益或新颖性。

固定旧 non-AA full6000 场、相同66,095参数网络，三个 fresh seed42 head 分别在002、041各更新一次。三个初始 head 的逐参数哈希相同；全部六次梯度与更新后参数/Adam状态有限，场参数版本不变且没有梯度。首步末层零初始化令MLP梯度为零，第二步三个MLP/CNN均非零。

| 对照 | 更新数 | 002／041有效支持比例 | 第二步MLP梯度绝对值和 | 峰值CUDA分配 |
|---|---:|---|---:|---:|
| median4_mass | 2 | .89149／.67325 | 8.324e−6 | 12,656,867,328 B |
| top4_mass | 2 | .96469／.89829 | 1.446e−5 | 12,656,741,888 B |
| top4_normalized | 2 | .96469／.89829 | 4.906e−4 | 12,656,741,888 B |

有效支持不是正确颜色比例；梯度大小不同也不是质量证据。002首步三个损失相同来自零残差初始化，不是多种读出等价。

六次target raster/selector、六次head更新；10张TRAIN照片经缓存各解码一次，原valid解码一次；0VAL、语义标签、teacher、几何更新。首次步耗时.789秒，其余约.300–.406秒；含输入校验的worker总25.937426秒、外层26.41秒。缓存准备的46.81秒和21.93GB另见[缓存结果](ibgs_layer_cache_results.md)，不能从单步时间抹去这项成本。

v1因选择器返回 `[H,W,2,4]` 的跨步长模式切片、证据接口要求连续张量而失败，1次raster/selector、0更新，自然exit1，内13.895005／外14.37秒。v2只把已选ID/深度/权重显式转为连续布局，CPU回归确认数值与层选择不变；原失败未覆盖。其余源模块、q、损失和三个对照保持原计划。

- v1：[失败回执](/mnt/data/SHM2026/runs/ibgs_layer_heads_preflight_v1/execution_receipt.json)，SHA `0e6a11983c8d9d998ac89788054c6bb69f0f2ad2564b17589a3b7a04290f1866`。
- v2：[执行回执](/mnt/data/SHM2026/runs/ibgs_layer_heads_preflight_v2/execution_receipt.json)，SHA `544372635c1b05c2ac2a53d61c45b43937a191a5f16f96f21544afcf1e0ff262`；[自然退出记录](/mnt/data/SHM2026/runs/ibgs_layer_heads_preflight_v2/launch_receipt.json)。
- 正式[冻结计划](/mnt/data/SHM2026/runs/ibgs_layer_heads_matched_v1/plan.json)：`b9a44c95b4e68b955f4fcc27abf27917134b4c5df8f456ed31c04a69b3a41c1a`，worker `5dd48b57ce31be61ca3ce4e4b67a61c083d4e6ead16ef3077f10ce9fa4987c43`；最终是否完成以运行回执和root观测的自然退出为准。

下一固定评价见[评价协议](ibgs_layer_evaluation_protocol.md)。旧IBGS训练过几何、head与source depth，新实验只训练head；历史分数差不能被解释为单一模块的匹配因果作用。现三臂分别隔离目标层选择和归一化，共同same-ID支持的作用仍未隔离，相关[先例](ibgs_layer_transport_prior_art.md)和额外[配对控制](ibgs_layer_controls.md)的限制保留。
