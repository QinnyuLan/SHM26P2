# 同配方 1M 容量对照：SSIM／LPIPS改善，PSNR投入门未通过

2026-09-27。固定一次30k训练、native终评及共同原图终评均自然exit0，训练终点与官方结果独立CPU审计均通过。**容量从实际498,136增至996,009点后，官方SSIM和LPIPS的配对区间明确指向改善；PSNR仅+.028664dB，区间跨零。预定四门通过2/4，不启动该场8k语义训练，不扫描容量或补训，当前selected不变。** 这是工程容量对照，不是新增密机制或学术创新，也不等于同学RGB148／RGB150的精确复现。

## 固定比较与执行

对照为已完成的 SSIM-fixed corner-v2 500k RGB参考。本次直接复制其实际30个Python模块，逐文件SHA相同，原始配置仅改变 `max_gaussians: 500000→1000000` 与输出目录。相同seed42、350 TRAIN／50 VAL、v2初始化、原相机、30k步、渐进分辨率、优化器、损失及分裂／剪枝／重置日程均保留；从相同62,000初始点重新训练，没有从500k终点续训。该配方仍使用TRAIN mask的前景RGB权重和初始化prior参与容量分配，不能称完全不使用标签的RGB基线。[锁定协议](rgb_capacity_1m_protocol.md)

原增长窗口没有延长：223次有效增密调用，最后一次23,900步；线性预算的最终值分别为498,136和996,009，不是精确500k／1M。新场183次调用结束时填满当时预算，最终及记录的field峰值均为996,009。训练与native外层于04:43:59 UTC自然结束，没有超时、重试、中途VAL选点或新增smoke。714个绑定输入、30个源码模块、350相机逐位一致性、模型／Adam／density／RNG终态合同均通过CPU核验；该合同检查不是一次实际CUDA resume重放。

## 原生与共同原图分开报告

原生评价使用同一corner-v2准备网格和50张验证图，fingerprint为 `f5c4419d3d81183310ae322b2c6343406a3e00a3e903a5f9a6e3d549bb63fca9`。以下是描述性点估计，不替代预定官方投入门：

| native RGB | 原500k：498,136点 | 新1M：996,009点 | 新−旧 |
|---|---:|---:|---:|
| PSNR ↑ / dB | 30.17230614 | 30.20042561 | +.02811947 |
| SSIM ↑ | .903149970 | .904981747 | +.001831777 |
| LPIPS ↓ | .223477727 | .215460618 | −.008017109 |

共同原图采用两组相同评分源码、原始畸变像素网格、最终uint8 RGB PNG及50个相机；fingerprint为 `21a2f19c5d4d703403a0107402d5dd5e006dcd98e5f8e023820c388c3e69a85d`。这是本项目为官方原始网格定义的共同开发评分协议，不是已知主办方完整评分实现。评分模块使用独立快照，未向旧30文件训练包添加模块。两组均plain推理，没有教师或TTA。未训练语义头的mask不进入结果排名或采用门。

| 共同原图 RGB | 原500k | 新1M | 新−旧，95%相机配对区间 |
|---|---:|---:|---:|
| PSNR ↑ / dB | 29.46748412 | 29.49614836 | +.02866424 [−.12255218,+.18880335] |
| SSIM ↑ | .870854827 | .872983084 | +.002128257 [+.000757714,+.003660311] |
| LPIPS ↓ | .270773566 | .263819912 | −.006953654 [−.010023306,−.003483995] |

固定5000次相机配对bootstrap，seed20260926；候选减对照，不翻转LPIPS差值符号。PSNR增幅至少+.15dB、PSNR区间下界严格大于0两项失败；SSIM不降、LPIPS不增两项通过。**不能把这轮写成RGB完全没有改善，也不能因两项改善而改动四门全过的投入条件。** 这些是反复使用的同场开发相机、单seed结果，区间不包含多轮研究选择、多种子或跨桥泛化的不确定性。

## 实际容量与成本

| 成本口径 | 原500k | 新1M |
|---|---:|---:|
| 最终／记录field峰值点数 | 498,136 | 996,009 |
| 主训练render次数 | 30,000 | 30,000 |
| 主训练Gaussian-step求和 | 9,581,235,400 | 18,318,918,900 |
| 平均主render点数 | 319,374.51 | 610,630.63 |
| 额外诊断render次数 | 2,350 | 2,350 |
| 主训练＋诊断Gaussian-render求和 | 10,236,505,068 | 19,548,237,372 |
| 主训练＋诊断render像素求和 | 27,231,229,440 | 27,231,229,440 |
| 最后训练日志耗时 / 秒 | 1,115.77 | 1,169.87 |
| 含native的外层实验耗时 / 秒 | 1,136.57 | 1,193.93 |
| 记录allocated峰值 / GiB | 1.463 | 2.302 |
| 完整训练checkpoint / bytes | 467,917,643 | 935,418,507 |

新场主训练累计Gaussian-step为旧场1.912倍，包含诊断的Gaussian-render为1.910倍；按相同训练日志计时口径，耗时增加约4.85%。这不等于同算力、线性FLOPs或受控硬件性能基准：Gaussian投影覆盖、排序、CPU工作、缓存及计时范围都会影响墙时。新场训练worker记录1172.765秒，包含其加载／保存等边界，不能与旧训练日志1115.77秒直接混作同口径。新场native worker为16.176秒；独立官方子进程为25.104秒。

新成本来自仅读取N／尺寸的外层render观察器，并由原源码与每100步post-update日志独立重建一致；旧成本只有源码＋日志重建，不能把旧值称为运行时观察trace。两组分辨率与render像素总量相同，点数轨迹不同。新训练峰值allocated为2,471,874,048 bytes，峰值reserved为23,666,360,320 bytes（22.041GiB）；两者都不是设备总显存占用，reserved也不是活跃张量大小。点数峰值不覆盖临时grow/prune张量。

## 审计与失败记录

首次CPU终点审计错误地要求旧冻结包导入 `bridge_rgs.data`；该历史包实际通过 `bridge_rgs.io` 加载数据，train/native都已经正确导入losses。故这是**审计模块名合同错误，不是训练、评价或losses缺失失败**。原审计脚本和root报告的失败说明保留在 [audit_attempts/001_import_contract_failure](/mnt/data/SHM2026/runs/rgb_capacity_1m_reference_v1/audit_attempts/001_import_contract_failure/failure_record.json)，没有伪造未保存的原始traceback。仅修复workspace审计器为阶段对应的真实核心模块，所有实际导入模块仍逐路径及SHA核验；16项CPU合同与Ruff通过。训练／评价／冻结源码均未重跑或修改。

修订终点审计passed。后续独立官方CPU审计也passed：核验新旧100张RGB PNG与原图字节，逐图PSNR重算最大差 `7.1054e−15`；独立5000次配对重采样的三项区间及四门与原记录完全一致。SSIM／LPIPS沿用已绑定评分值，未声称CPU另算这两项；未解码或排名语义mask。

| 关键产物 | SHA256 |
|---|---|
| 训练plan | `40191687627d54da91b78c8b819e83bc04b04b986ae6568efe042c6dbcd447f3` |
| 30k checkpoint | `35b489fe45ad34ad279018609b6d5eaf6493e4c0d82f9c86ec16925268078092` |
| CPU终点审计 | `2e52df6b4ab9193d4a1b2c07d32bd246e6f84a398453ec97dec9c7e5b1c349e1` |
| 官方evaluation plan | `0c86330eaf27d82740350953d448c920186984a356c1ceae59998b2cd7639842` |
| 官方metrics | `0b4b90fca81ad4f220794b9656ce308cca4dc68a8efe83769efbfde2cd51100e` |
| 官方配对报告 | `a7307a49816188110dfb61921c0f3e3f31c9005e14ae5983ce113a13142a0f83` |
| 四项投入门 | `c822424182a2bbceb6d90cc0d6cdb78c7a6aeb97a3cf925b45b2a3203549f400` |
| 独立官方CPU审计 | `7ecc15ef9f33a3800723c8eb37bb6b38599373a6c381abdb0b0d67e5c417f2d9` |

全部产物位于 [run目录](/mnt/data/SHM2026/runs/rgb_capacity_1m_reference_v1)，其中[终点审计](/mnt/data/SHM2026/runs/rgb_capacity_1m_reference_v1/cpu_endpoint_audit.json)、[官方配对](/mnt/data/SHM2026/runs/rgb_capacity_1m_reference_v1/paired_official_rgb_minus_500k.json)、[投入门](/mnt/data/SHM2026/runs/rgb_capacity_1m_reference_v1/semantic_investment_gate.json)及[独立官方核验](/mnt/data/SHM2026/runs/rgb_capacity_1m_reference_v1/independent_official_cpu_audit.json)均保留。

**研究与采用决定：**不追加这个1M场的8k语义、不复用legacy H+到新v2场、不扫容量／步数／权重。selected仍是原H3＋固定DINOv3 H+组合（官方五类95.109016%、拉索94.786655%，RGB29.400950／.870650／.271237）。不能把新1M的RGB与旧场语义拼成联合成绩；SSIM／LPIPS的真实容量收益也不构成学术创新或对未知peer划分的公平胜出。已绑定协议与计划保持原样。
