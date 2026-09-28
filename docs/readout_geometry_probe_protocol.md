# 部署读出几何梯度接线检查

固定 TRAIN 002/041，原 H3 加已审计的 EM/FW q；不读 VAL、不训练 head 或教师。既有 raw 优化的最终读出退步之后，本检查只验证一个工程问题：保持直接 q 部署前向的同时，能否让冻结 H3 head 的实际输入路径全部对 means 求导。

新 adapter 使用两次标准 gsplat，显式合成 q、features、z/z²/zf；refiner 和深度矩的 geometry_grad 选项默认关闭，只有新 adapter 打开。旧 direct-q 调用作前向参照，禁止 straight-through 替换前向结果。沿原训练数值设置，不把旧 3/12 FD 检查追认为通过。

每图保存旧／新 baseline 与一个新候选的 RGB、raw、scene 概率；记录逐通道最大差、mask 差及实际三个损失。三个损失为同 valid RGB MSE、原 weighted affine raw CE、同定义的 scene CE。仅 means 开梯度，分别保存三个完整 FP32 VJP；每图 fresh Adam 沿 scene CE 产生一个实际 FP32、每点 Mahalanobis≤1/64 的候选，LR及其余参数复用原事务。候选仅做描述，随后完整回滚 means 和 optimizer，0次训练提交。两个相机独立从相同原 means 起步。

固定 2 旧 direct-q 调用、4 新 adapter 调用、6 head、12标准 gsplat高层调用、2 custom shader调用、6 means VJP、6目标解码、2候选与2回滚。低层通道分块 raster另行计数。内限120秒／外限180秒。保存完整初始means、候选位移、三组梯度及对应输出，便于CPU复算有限实际点积与目标。

新旧前向的差异完整报告，不预称exact，也不通过调阈值或再采样消除差异。非有限梯度、断链、意外参数梯度或状态未恢复是实现失败；实际候选是否改善仅是这两个TRAIN相机的描述。没有数值认证、泛化、自动训练采用或创新门。
