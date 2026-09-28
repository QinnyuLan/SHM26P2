# 固定 H3 的全量 simplex 优化：目标微降，未收敛

2026-09-27，固定计划已自然退出0，独立CPU审计通过。只接受了 **1次更新、12个完整259 TRAIN遍历**，随后 `backtracking_exhausted`。这次有界数值诊断没有充分求解属性子问题，不能据此认定原生语义低分来自几何容量不足；没有新VAL结果或模型采用。

底座为H3 `22bc8a2d…5226`、legacy网格、498136个Gaussian。几何、opacity、RGB、feature、classifier、refiner和相机均冻结，直接优化五类simplex概率；目标是每视图等权的加权CE，正仿射噪声 `δ=5e−7`，不把生产clamp/归一化加入凸目标。它与旧391a/corner-v2一次EM诊断使用不同可见性算子，不能直接比较两次CE绝对值。

| 全259 TRAIN指标 | 初始 | 唯一接受终点 | 变化 |
|---|---:|---:|---:|
| 仿射噪声目标 F | 0.103494906859 | 0.103402089840 | 相对下降0.089683% |
| raw五类mIoU | 80.318479% | 80.328298% | +0.009818pp |
| raw拉索IoU | 34.630569% | 34.665557% | +0.034988pp |
| 冻结head五类mIoU | 98.822226% | 98.811723% | -0.010503pp |

head只替换p3d及log prior，其余原context保留。表中初始值是归一化master q重渲染的读数；初始化FP32概率最大变化1.78814e−7，F相对原classifier直接渲染仅差4.08754e−10，但不声称逐位相同。终点近似Frank–Wolfe gap为 **0.0862922413**，`convergence_certified=false`。FP32算子/梯度加FP64归约不是严格全局证书；回溯耗尽不等于收敛，CE也不提供IoU表示上限。raw的小增益同时伴随冻结head小降，不能推出改q会改善最终系统。

实际成本来自[执行回执](/mnt/data/SHM2026/runs/raw_simplex_fullbatch_v1/execution_receipt.json)：内部186.945959秒，包含初始化、载模、全部遍历、q输出与来源校验；外层[自然完成回执](/mnt/data/SHM2026/runs/raw_simplex_fullbatch_v1/launch_receipt.json)187.396149秒。3108次scene、6216次gsplat加3108次直接q shader，共9324次raster；777次VJP、1036次head、518次mask/valid解码，0部分pass视图。allocated/reserved峰值为1,846,232,576/2,204,106,752字节。计时不是部署FPS。0 teacher、0 VAL、0 RGB载荷读取，未写生产checkpoint；原模型状态、flags、梯度和相机恢复，源与输入不变。单独保存的q产物仅供诊断，不支持普通续训。

[独立CPU审计](/mnt/data/SHM2026/runs/raw_simplex_fullbatch_v1/independent_cpu_review.json)用0.418228秒核对61源、281输入、保存的每视图标量、CM汇总/IoU、调用预算、q的simplex误差及Armijo记录，状态passed。它没有重新渲染或保存的逐像素预测/梯度，故未独立重算Wᵀ、像素CM或gap；Armijo中的梯度点积及恢复/计数仍依赖绑定源码和运行回执。这些限制不因审计通过而消失。

来源：plan `c19aa6bde8190e5899475b56d02676ded38e8f16632731553c8ddfbf448972d0`；[analysis](/mnt/data/SHM2026/runs/raw_simplex_fullbatch_v1/analysis.json) SHA `1ed72861cfdc6f60d431a1211b85d1e657c5f9d3847b0f18bff987d8037ddf97`；执行回执 SHA `987162b413afcf8d6f4fe8e729b860e7e793b3f8808741cf6f5d6b6175ed2a3a`；独立审计 SHA **`dc4f4ec9e62990e7e2a990fef74e8d1b616c47702cfe1cbe1cea1222ccdcdaf3`**。原[固定协议](raw_simplex_optimization_protocol.md)及预检失败记录保留；本轮不改变工程E，不证明新学术机制。
