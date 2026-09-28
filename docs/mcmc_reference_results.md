# MCMC RGB 工程参考：运行记录

2026-09-27。完整 30k、native 与官方原图评价均已自然 exit0，训练终点 CPU 审计通过。**PSNR 存在单项进展，但未达到整体采用门槛；现有 selected 联合系统保持不变。** 本项是已有 MCMC 方法的本地工程配方参考，不是学术创新消融。

固定方案见 [执行协议](mcmc_reference_protocol.md) 与 [文献边界](mcmc_reference_decision.md)。主对照是 SSIM-fixed v2 hybrid 的同 50 视角官方 RGB，另报告 mixed 和 selected。只有官方 PSNR 平均提高至少 .15 dB、配对 95% 区间下界大于零、SSIM 点估计不降且 LPIPS 点估计不增，才投入新场语义训练。

## 已完成的接入验证

- CPU：106 passed / 2 CUDA skipped；覆盖新策略与训练接线及相邻 scope、resume、mixed、Mip 合同。
- 独立实际 CUDA 小样：强制低 alpha 噪声、重定位、增点、Parameter/Adam 替换和一次随机状态恢复均通过。仅此小样恢复的参数与 Adam 逐位相同，不能推广为完整 30k 的确定性保证。
- 实际数据 610 步 smoke 自然 exit0，无 VAL 评价。第 600 步重定位 18,811 点、新增 3,100 点；末步 65,100 点。610 是独立 LR horizon，不是 30k 轨迹前缀。
- CPU 终点审计与另一份独立核查一致：350 原始相机逐位一致，语义 prior 为零，参数/Adam 形状和有限值、独立视图 RNG、冻结源码及输入哈希通过。
- smoke render Gaussian-step 37,851,000，noise Gaussian-step 37,854,100，像素 796,342,800；wrapper 77.384 秒包含 CUDA 编译。以上不是精度评估。

证据：`/mnt/data/SHM2026/runs/mcmc_cuda_contract_v1.json`，`/mnt/data/SHM2026/runs/mcmc_reference_smoke_v1/`。smoke plan SHA `ea57b59ef577bfb197284276a6083389dc6060b514328201a3aa730fdc9ecc8c`；CPU 审计 SHA `fa479dc6794bdac733de6bf803fa2230a3251d1b749a56b3fe43f030dc5057f1`；独立核查 SHA `37a92d4d370d4cd3ef4d5c1cc2e4e152812fcf9a70fbb65b546b14ec45de8122`。

## 完整运行

目录 `/mnt/data/SHM2026/runs/rgb_mcmc_reference_500k`；plan SHA `bc7687cca5552aa5c39fa37dab64f83f40ff886164fe66ffa4c16c35dadfa35e`。固定 350 TRAIN / 50 VAL、全程 native、seed42、30k、500k 上限，训练中不评价 VAL，最终一次 native 与共同官方原图评价。计划与源码已经冻结；本文件只记录状态，不是可变的训练输入。

### 同一官方原始网格的 50 视角 RGB

| 端点 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| 已有 mixed 参考 | 29.226362 | .870529 | .271269 |
| 较强 SSIM-fixed v2 hybrid | 29.467484 | .870855 | .270774 |
| 当前 selected 联合系统的 RGB | 29.400950 | .870650 | .271237 |
| 本轮 MCMC RGB 参考 | **29.651371** | .869256 | .278660 |

MCMC 尚未训练语义，不把旧场的 95.109% mIoU 放入新场一行。上述是完整配方比较：初始化 alpha/scale、分辨率日程、正则、随机过程和拓扑机制均有差异，不能将差值归因于单独的 MCMC relocation。

主要比较 MCMC−较强 SSIM-fixed，5000 次固定相机配对 bootstrap，seed20260926：

| 指标差 | 点估计 | 95% 区间 |
|---|---:|---:|
| PSNR / dB | +.183886 | [−.196999, +.609540] |
| SSIM | −.001599 | [−.004781, +.001237] |
| LPIPS | +.007887 | [+.001889, +.014137] |

固定四项投入门仅“PSNR 点估计至少 +.15 dB”通过；PSNR 区间下界、SSIM 和 LPIPS 均未通过。**不启动该场 8k 语义，不按 VAL 扫 noise/初始化/正则，也不替换 selected。** 不能把 PSNR 点估计上升写成可靠的整体优势，更不能写成已验证的学术机制。

保留真实的次级进展：相对较弱 mixed，PSNR **+.425009 dB**，区间 **[+.062155,+.831562]**；同时 LPIPS **+.007392**，区间 **[+.003879,+.011221]**，体现取舍。相对 selected，PSNR +.250421，区间 [−.126136,+.644355]；LPIPS +.007424，区间 [+.001065,+.013798]。这些区间条件于已选择端点和开发集视角，未经多轮模型选择校正，不代表多种子或外部桥梁泛化。

Native 指标另列为 **30.387738 / .901760 / .231396**；不能与上表官方原图数值相减。native 同样呈现 PSNR 上升、SSIM/LPIPS 退步。

### 实际计算量与可复核性

- 30,000 次训练 RGB+ED 渲染，全程 native，共 **39,164,400,000** 像素，比 mixed 的渐进分辨率多 **44.78%**。
- 第 4,800 步达到 500,000 点；244 次 refine 均发生重定位，累计重定位 **3,665,195** 次点实例，允许同一点多次出现，不是独立点个数。
- 渲染前 Gaussian-step **13,517,466,800**，比 mixed 多 3.56%；重分配后噪声 Gaussian-step **13,517,904,800**。相对较强 hybrid，主渲染 Gaussian-step 多 41.08%，但该 hybrid 另有诊断渲染，不能将这些计数当总 FLOPs。
- 训练记录 **962.29 秒**，训练+native wrapper **980.820 秒**，官方 subprocess **22.583 秒**；峰值已分配显存约 **1.123 GiB**。相同 cap/步数不是等计算量消融。
- 350 原始 TRAIN 相机逐位相同，prior 为零；参数、Adam、density 与策略状态有限且形状一致；独立视图 sampler、244 次增长/重定位日程、训练/噪声点数和像素累计均通过 CPU 审计。比值>51 的官方 relocation 内部截断触发次数未作无侵入观测，不声称为零。

checkpoint SHA `3796c4c73ce2189ce74153dd7b1209d88cc2d9a1bf57c9299366977cde8c876b`；CPU 终点审计 SHA `7e8614524a13c589d23a97d8ad9a019b50c2041caec328e734ed3fabaef6ea19`；官方 metrics SHA `ac22b1b3e0a60abac20998a62042916b5f95f29d8a6d1bdf427c5c01fb259a73`；官方计划 SHA `319b5810df894de200d51e76ad065d068e509730b1c2c21d91255c09ff7a8c2a`；官方比较 receipt SHA `652e3944c78ecc591f743b5b472c464d5711671ebf547ee37711ae700a336cf7`；门报告 SHA `f7c4d8d86a690b2f295aa0029d983eb0a6a21792253f3c714ba3b0a26e0d00f8`。

独立官方 CPU 核验也已 exit0、passed：50 RGB/50 schema-required mask 的尺寸与字节、来源和计划绑定一致；逐 PNG 重算 PSNR 最大差 `3.55e-15`；独立 NumPy multinomial-count 形式的三组 5000 次配对区间最大差 `1.11e-16`，四项门完全一致。SSIM/LPIPS 沿用已冻结的正式评分值，未声称独立重算；mask 仅核格式，不参与性能结论。实际加载的 9 个项目模块均来自计划快照且 SHA 对应。报告 `independent_official_cpu_audit.json` SHA `a178d28ced7538a8e320aefb7393153033905e21d874d1cc64ddddfc0b5d6aa3`。

## 未执行的条件式语义预案

若未来另有候选通过门，最短合法后续是复制已完成 `ssim_fixed_corner_v2_semantic_coupled` 的 8k 配方，仅改 warmstart/output，复用相应新 snapshot。冻结新场 geometry/RGB，用 259 个 TRAIN 标签训练语义，再比较 v2 plain 联合结果。必须检查新语义模型与其自身 RGB 终点逐张 RGB PNG 一致。本轮 MCMC 未通过门，没有创建或执行该训练。

当前 selected 使用的历史 H+ 教师绑定 `legacy_mixed_v1`；新场是 `colmap_corner_v2`，不能改标签绕过协议检查或拼接旧场 mask。若继续采用 DINOv3，需真实 v2 输入的新训练/适配与独立记录。
