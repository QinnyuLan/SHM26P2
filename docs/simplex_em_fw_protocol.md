# 固定可见性的有界 EM/FW 属性诊断草案

固定有界诊断已自然完成，20次EM与2次FW全部接受，26个完整pass按原日程停止；raw TRAIN改善但冻结head下降，近似gap仍未达停止值，未收敛。独立CPU审计通过，见[实际结果](simplex_em_fw_results.md)。起点为 `simplex_direction_diagnostic_v2` 的唯一已确认候选，不是原H3；本轮无VAL、无模型采用，不延长solver。下文保留原固定协议。

固定原 H3/legacy、259 TRAIN、350 原训练相机、原类别权重和有效区域，目标仍是逐 view 等权的 affine-noise raw CE（δ=5e−7）。几何、RGB、decoder、features、refiner 与 W 均冻结。保留 FP64 master，renderer 只取 FP32 cast；不把 cast 再归一回 master，不加入概率 floor。标准 EM 提案为 `u=q_master*(-g)`、`q'=u/sum(u)`，零责任行保留、零类别锁定；非有限、正梯度及 helper 不支持的下溢是数值错误，停止而非裁剪。梯度实际在 q32 上计算，因此精确算术 EM 单调定理不是这里的证书。

固定两个 block：每个最多10次 EM，之后唯一一次 FW。每次 EM 用一次完整259视图 pass 同时得到实际 F 和下一完整梯度；仅严格 F 下降才提交 master。完整 finite 但不降或 cast 不变时，跳过该 block 剩余 EM 槽位并执行其预定 FW，不做 EM 回溯。部分 pass、异常、超时直接失败，不转 FW。逐 view 的 FP32 VJP 先转 FP64 累加再除259。

每个完整梯度 pass 暂存 target raw a 及有效类别，接受后才替换 current cache；拒绝只释放本 run 的 trial cache。因此 EM 提前结束也有同一 current q/g/a，不额外重求梯度。FW 使用该梯度唯一 LMO 顶点、一次 vertex 前向形成 b，复用已验证 A/B32/B64 门（abs2e−7+rel5e−4，负且信号>2e−6）和64次固定导数二分。只在门通过后做一次实际候选完整梯度/F pass。缓存预测下降与实际下降仍满足同一误差门，且真实严格下降。FW cast 停滞、数值门失败或真实候选未确认即结束整个诊断，保留 last accepted，不进入下一 block 重复方向。仅 FW 接受且两种报告 gap 的最大值>1e−5才继续；小 gap 仅是近似停止原因，绝不追认收敛。

最多26个完整 pass：baseline 梯度+评分1、20 EM候选、2×(vertex前向+FW候选梯度)4、final 梯度+评分1。终点 pass 必须预留，baseline/final 同时报告 raw 与冻结 head 的 TRAIN CM；0 VAL/teacher，无 feature 映回、无生产 checkpoint。每个梯度点均报告完整近似 FW gap，包括拒绝候选；所有拒绝原因和实际下降保留。只保存独立 q delta，不能普通 resume。

预算内1500秒/外1560秒，不延长/重试。已有约23秒梯度 pass、8秒vertex pass、146秒CPU缓存二分给出约900–1000秒估计，余量覆盖 I/O、模型加载、终点评分与 hash；估计不是完成保证。current/trial/vertex target cache 只存本 run 临时数据，预计峰值<5GB，加模型/标签缓存和 FP64工作量预留额外内存16GiB；不保存所有逐view梯度，不删除旧 run 缓存。每次替换/删除自产临时缓存需计数并保存路径/内容摘要。新 source 从 direction-v2 冻结包继承，只新增本 helper/controller/适配入口及测试，冻结 pytest 必须禁 cacheprovider，前后源 hash 一致。

可支持的结论限于固定 W 下有界预算内属性目标还能否实际下降，以及终点 TRAIN raw/head 的关联变化。无下降或未收敛均不能直接证明几何容量不足；TRAIN loss 下降也不是新视图收益或学术创新。
