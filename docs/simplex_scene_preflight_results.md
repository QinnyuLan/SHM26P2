# 直接类别概率算子：两个真实 H3 相机的预检结果

2026-09-27。修复报告序列化问题后的v2自然完成，全部原定数值条件通过；**没有训练或新精度指标**。仅支持继续检查固定H3可见性下的直接概率优化，不是全算子精确伴随证明或严格最优性证书。

固定002/118两个TRAIN相机，只读模型与相机元数据，0真实图像/标签解码、0优化步。完整51个bridge包文件继承已审查的旧分区快照。两端传入同一个q leaf、固定系数`[0,0,1,−1]`，用其求和VJP检查固定W；不使用条件投影、OT或固定类别总量helper。详细[预定协议](simplex_scene_preflight_protocol.md)保留。

| 检查 | 002 | 118 |
|---|---:|---:|
| 初始未归一raw与原gsplat最大差 |0|0|
| 初始生产p3d最大差 |0|0|
| alpha最大差 |0|0|
| 中点仿射关系最大差 |7.78586e−7|8.71718e−7|
| 三个伴随探针最大绝对差 |7.51624e−8|2.40336e−8|
| 主h=.01的CE方向FD相对误差 |.014093%|.014045%|

六个伴随探针均满足信号强度门。CE用合成棋盘类标签、δ=5e−7、单位类别权重；未读取真实标签。FD与VJP点积使用实际独立舍入后的FP32中央位移，主h预先固定，其他步长未用于挑选结果。报告中的分辨率下界仅针对FP64标量/归约，不涵盖全部FP32前向/atomic误差。raw不做clamp或归一；生产p3d的一致性单独检查。

两次scene、四次原gsplat raster、18次新shader、8次VJP，共22次栅格化。内部3.5337秒、外层4.4125秒自然exit0，peak allocated为1,060,147,712字节；启动时没有检测到其他compute进程。全部模型、相机及RGB/深度/特征/context不变，requires_grad、训练模式、已有梯度和TF32等设置完整恢复。该小预检耗时不是训练或部署FPS。

v1外层4.4532秒后失败：数值调用已完成，但`fd_result`中的NumPy布尔值无法严格JSON序列化，留下部分analysis文件。失败目录保持原样，未追认通过。v2只转换原生JSON标量并先序列化后exclusive写入；公式、相机、阈值、步长、预算和调用数不变。新增回归实际复现旧失败并核对修复前后各数值相同。v2共13项冻结CPU测试通过，root另独立复跑13项；独立代码审查未发现阻断。

| 记录 | SHA256 |
|---|---|
| [v1失败执行](/mnt/data/SHM2026/runs/simplex_scene_preflight_v1/execution_receipt.json) |`6fe0be39ee350431e66c50ac72827fd7fa09ea90c7133b79a55b1e7a35f06173`|
| [v2计划](/mnt/data/SHM2026/runs/simplex_scene_preflight_v2/plan.json) |`8b511295e5a9e1522b35d70e9789355fa90d6a4b2cf06f5aeb877b3acf5c971c`|
| [v2执行](/mnt/data/SHM2026/runs/simplex_scene_preflight_v2/execution_receipt.json) |`236d1d8828e3d833d6cafe63a8bf03f8b7a15e3d9ffde9968be225d585f62303`|
| [仅序列化修复差异](/mnt/data/SHM2026/runs/simplex_scene_preflight_v2/serialization_revision.json) |`0b37815aca027d3b579cdca77027ff893ea55b2d90d99e1089fd27ea27237bae`|

全量优化必须另行冻结[完整TRAIN目标、接受规则和预算](raw_simplex_optimization_protocol.md)。生产E继续保持30.035661dB/95.109016%；本预检没有验证集、teacher调用或新模型采用。
