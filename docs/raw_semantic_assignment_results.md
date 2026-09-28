# 单轮 raw semantic assignment：TRAIN 结果

2026-09-27。预定单次诊断自然完成，未追加迭代、未读取 VAL、未替换生产模型。**近似 EM 的真实 TRAIN 目标通过预先固定的数值下降门，但收益很小；Adam 的 CE 下降更大而拉索 IoU 下降。两者都降低了冻结 refiner 的最终 TRAIN 成绩。** EM没有整体优势，不作为采用方案，也不称为固定几何下的内层最优解。结果不能证明几何无优化空间，也不证明任一方法已经收敛或可以泛化。

| 固定阶段 | affine-noise raw CE | 归一 raw CE | raw五类IoU % | raw拉索IoU % | final CE | final五类IoU % | final拉索IoU % |
|---|---:|---:|---:|---:|---:|---:|---:|
| 共同q0初始 | 0.104296700 | 0.104296497 | 79.92396 | 35.25168 | 0.003703828 | 98.60123 | 98.55902 |
| 一次近似EM | 0.103996088 | 0.103995874 | 79.98782 | 35.29202 | 0.005467771 | 98.03567 | 97.85385 |
| 259步feature Adam | 0.103492442 | 0.103492250 | 80.30479 | 34.88681 | 0.004366312 | 98.29980 | 98.24413 |

三行均为同一 259 张有标签 TRAIN、334,318,754 个有效像素。IoU来自 pooled CM；CE先对各视图有效像素取均值，再对259张等权。固定class weight power=.25。final仅描述冻结refiner受feature/prior变化的影响。这里的共同q0经过预定ε内化，不是原base的完整成绩；没有额外测base完整目标，不能将初始化计作优化收益或与既有VAL成绩拼接。

EM真实noise CE下降0.000300611747，超过固定门max(1e−6,1e−4×初始CE)=0.000010429670。它仅通过数值下降检查，不是质量显著性检验或模型采用门。EM的raw五类IoU仅增0.063857个百分点、拉索增0.040347个百分点；Adam分别增0.380829、降0.364870个百分点。冻结final五类IoU分别下降0.565557、0.301427个百分点。CE与IoU不是同一目标；这些方向差异不构成程序错误证据。

原raw拉索→背景像素13,555,969，EM为13,563,792、Adam为13,721,940；背景→拉索原3,681,293，EM为3,627,018、Adam为3,478,222。EM的微小拉索IoU增加并非拉索召回整体提升；Adam进一步减少误报的同时漏报增加。单轮试验仍没有解决raw拉索瓶颈，不能据此归因为不可改变的几何限制。

EM对456,825个有责任Gaussian更新，41,311个零责任Gaussian features逐位保留共同起点。所有理论active targets满足q≥1e−5、行和1；GPU实现最大概率误差3.247e−7。实际最小q=9.9999561e−6；798,690个分量略低于理论ε属于所报告FP32实现偏差（最深约4.4e−11），没有重新夹紧。Adam实际最小q=1.1010e−9，1,144,141个分量小于ε；两者可行域确实不同，不称纯optimizer消融。两臂均无FP32零分量。

所有评分阶段均无“零alpha且GT为前景”像素，也没有噪声责任≥0.5的像素。噪声责任的按视图平均约1.31e−7（Adam1.36e−7），最大值初始0.01198、EM0.00908、Adam0.11342。固定δ稳定项未被用来筛除任何像素；这些统计也不等于已排除其他可见质量瓶颈。

完整预算为1,036次scene render、2,072次rasterization、259次颜色backward、259次Adam更新及一次M-step。E-pass16.995s、EM终评12.859s、Adam更新7.799s、Adam终评12.850s；worker52.851s，含启动的队列54.154s，自然exit0（600s硬限内）。峰CUDA allocated约1.717GiB、reserved2.088GiB。两臂相同数据遍历不意味着相同更新次数或算力，未作充分收敛比较。

CPU独立后审通过：完整四段259视图顺序/camera index/逐步noiseCE、全部259视图跨四段RGB hash、pooled CM合计和IoU均复核；source/input与NPZ SHA一致。重新执行water-filling得到逐位相同targets，再由固定decoder在CPU重建active候选features逐位相同；零责任features保持逐位不变。CPU解码概率与保存的实际GPU概率最大差初始4.17e−7、EM5.36e−7、Adam3.58e−7。所有geometry/RGB/decoder/refiner/camera状态不变，finally恢复完整原state和flags。没有生产checkpoint。

结果及完整CM：[execution_receipt.json](/mnt/data/SHM2026/runs/raw_semantic_assignment_v1/execution_receipt.json)，[CPU独立审计](/mnt/data/SHM2026/runs/raw_semantic_assignment_v1/result_audit.json)，[完整逐次轨迹](/mnt/data/SHM2026/runs/raw_semantic_assignment_v1/view_trace.jsonl)。诊断NPZ共195,770,294字节，含共同初始化、EM责任/目标/候选features、Adam候选features，可复核但不是部署模型。

冻结plan SHA `091aa8a89fae986c148dc79de55ba3e351abd42b5460399b77ee097c0b583e5a`；执行receipt SHA `b4e64db24faf0110c274f2d69ce73558784e0e69869ff6829df6a7c33ca752a6`。GPU已明确交回协调者。本轮结束，无追加训练或验证集评测。
