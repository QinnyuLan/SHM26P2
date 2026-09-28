# 固定组合 RGB 的 H+ 迁移结果

2026-09-27。两臂按[冻结协议](rgb_teacher_transfer_protocol.md)自然完成：旧 selected RGB 与真实 1M/MCMC 各半 RGB，均通过同一显式原图→legacy 教师画布→原图适配器，使用同一个 H+ EMA、tile768/stride512/flip/context0.25，纯教师权重1。全部 **100张新 mask** 完成并hash后才读取41份标注，没有拼接历史mask。预定联合门仅 **4/7通过**，当前 selected 保留，停止这个固定迁移方案，不扫TTA、融合权重或边界填充。

| 共同官方原始网格结果 | 全5类 mIoU (%) | Cable IoU (%) |
|---|---:|---:|
| 当前 selected 完整系统（历史 H3＋H+） | 95.109016 | 94.786655 |
| 同适配器旧RGB纯H+控制 | 94.172488 | 94.316904 |
| 实际组合RGB纯H+候选 | **94.386572** | **93.123743** |

| 差值：候选−参照 | 全类差值，百分点 [95%区间] | Cable差值，百分点 [95%区间] |
|---|---:|---:|
| [同适配器控制](/mnt/data/SHM2026/runs/rgb_teacher_transfer_v1/paired_minus_same_adapter_control.json) | +0.214084 [−0.296584, +0.870290] | **−1.193161 [−2.401976, −0.359063]** |
| [当前 selected 完整系统](/mnt/data/SHM2026/runs/rgb_teacher_transfer_v1/paired_minus_selected_complete_system.json) | −0.722444 [−1.515616, +0.177241] | **−1.662912 [−3.068324, −0.682707]** |

同适配器对照已经显示，替换为新RGB后出现缆索退步，不能将全部损失归于去除H3语义读出；对比度、平滑或教师输入域变化的具体机制仍未识别，单帧可视化不能替代因果证据。全类均值的小幅增加也不能掩盖缆索下降。区间是固定50 RGB／41语义开发视图的5000次配对bootstrap（seed20260926），不是独立桥梁、种子或选择校正后的盲测结论。

候选RGB逐PNG字节保持既有真实组合输出，因此 **PSNR 30.035661、SSIM 0.874420736、LPIPS 0.264741845** 的RGB结果仍成立。本轮在字节核验后复用该评分，没有重新读取RGB GT或运行LPIPS；语义由实际新RGB重新预测、重新评分。[七门结果](/mnt/data/SHM2026/runs/rgb_teacher_transfer_v1/system_gate.json)继承此前相对500k参考已通过的四项RGB门，而相对完整selected的三项语义门——全类≥+0.20pp、配对下界>0、Cable≥−0.10pp——全部失败。

本次 **99.279959秒、峰值CUDA allocated 2,181,768,192字节**只计缓存RGB的迁移、教师推理、评分与I/O，0场渲染／0训练更新。新视角仍需两个RGB场共1,496,009个Gaussian的两次渲染；纯二维教师读出不等于共享几何的原生语义GS，也不构成学术创新或精确peer公平胜出。重采样与固定黑边属于两臂共同适配器；没有第三场补边、没有绕过checkpoint像素协议、没有自动启动此前停止的1M 8k语义训练。

[计划](/mnt/data/SHM2026/runs/rgb_teacher_transfer_v1/plan.json) SHA `2df49d471db41b4c0323d6a9b9a955529a93ee6530352b0417a8e0580470bbe9`；[完成回执](/mnt/data/SHM2026/runs/rgb_teacher_transfer_v1/execution_receipt.json) SHA `b205a3727e14d951104ef309fc0e8cb24c1d9704c430001b985d8057ff593dbc`。源与绑定输入未变，预测完成时间早于GT评分开始。前置fingerprint多name兼容修复发生在冻结前的CPU准备阶段，没有失败GPU尝试；不得将其写成推理或训练故障。

[独立CPU复核](/mnt/data/SHM2026/runs/rgb_teacher_transfer_v1/independent_cpu_review.json)已通过，SHA `4ea3b9a3f177e99a496b9b31accd6aa2922c690ef205a402baf682583210d881`：100张RGB与输入逐字节一致，100张mask的SHA／尺寸／类别ID合法；独立栅格化41份annotation后，GT mask SHA和两臂82个逐图混淆矩阵全部一致；两组5000次配对bootstrap与七门结论一致。该复核没有运行Torch、GPU或教师推理，也没有读取真实RGB GT；RGB数值在字节一致后继承，未重新像素评分。预测先于GT的证据来自原时间戳、预测回执与已审控制流，不宣称独立运行期I/O跟踪。
