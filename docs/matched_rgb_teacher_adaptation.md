# 新组合 RGB 的匹配渲染域适配

2026-09-27。本轮检验工程假设：旧 DINOv3 H+ 教师未经新 RGB 域训练，是否限制了双场 RGB 改善向语义指标的转化。不是学术创新实验，不预设适配一定有效。

此前真实新组合 RGB 的纯 H+ 五类 mIoU 为94.386572%，拉索93.123743%；这来自 `rgb_teacher_transfer_v1` 的共同原图重评，不是数值相近的历史 native 成绩。与同适配器旧RGB控制相比，拉索下降1.193161个百分点。事后空间诊断显示错误并不主要落在黑边的直接回投足迹内，但不能排除网络上下文传播；诊断不用于删图、改变标签或选择超参。

本轮两臂均从同一个 `teacher_render_adapt_v1/best.pt` 的 EMA 初始化（SHA `00f5b84ac9a56c39512c5b8e43f70110397923feea1a2b4c78bdffd1f5524bff`），冻结 ModelScope DINOv3 H+ 骨干，只训练原解码器2000步。初始化、损失、AdamW、学习率3e-5、100步warmup、EMA .99、768 crop／50%全景课程、seed20260926及每步视角／裁剪／增强随机序列保持一致。两臂都使用精确1000步真实TRAIN图和1000步渲染TRAIN图：

| 组 | 渲染训练输入 | 最终评价输入 |
|---|---|---|
| selected 控制 | 旧 selected H3 场的原图 RGB，经同一显式 legacy 适配器 | 新双场组合 RGB，经同一适配器 |
| composite 候选 | 1M 与 MCMC 两场的原图 uint8 RGB 各半平均，经同一适配器 | 同上 |

缓存先从原始相机生成三场各350张TRAIN RGB，共1050次场渲染。另固定001相机对三个场各做一次RGB逐像素复现，共1053次。各场保留自己的legacy／corner-v2声明，先按各自正式导出约定生成原始畸变网格RGB，再以 `initUndistortRectifyMap`、INTER_LINEAR、黑边映射到legacy教师画布。组合使用float32等权后round-to-nearest-even uint8，和已评测RGB完全相同。缓存生成不读取真实照片、mask或valid像素。

原始legacy manifest的400视图、划分、人工mask、valid及相机元数据不变；新schema明确记录多个renderer及坐标转换，不能冒充单个legacy场。旧教师的warmstart须验证旧派生manifest与新派生manifest都沿用同一原始manifest，不能通过改tag绕过检查。训练仅消费259张有标签TRAIN图，350缓存中的91张无标签图不参与本轮监督更新。训练内关闭所有VAL评价；每500步只保存故障恢复状态，最终只使用固定2000步 `last.pt` EMA，不选best。

两组训练自然完成后，在新组合的全部50个相机上生成100张mask，再读取41份语义标注评分。RGB保持实际组合PNG，字节一致后继承已有评分，不把另一场mask与新RGB拼接。使用既定5000次配对视图bootstrap（seed20260926），完整报告候选减同预算控制、减未经本轮训练的冻结教师、减当前完整selected三组差值。

联合采用沿用之前的条件：既有四项RGB门全部通过，候选相对完整selected全类mIoU至少增加.20个百分点、配对区间下界>0、拉索不下降超过.10个百分点。独立的域匹配证据门要求候选相对同预算控制全类至少+.20个百分点且区间下界>0；它与联合采用分别报告。若没有通过，不追加步数、换best或扫描混合权重。本轮开发集已反复使用，区间不覆盖选择偏差、多种子或跨桥泛化。

缓存固定内限570秒／外限600秒；每个训练臂内限900秒／外限960秒。历史同配方2k为376秒（含验证），只是预算依据。大文件写入 `/mnt/data/SHM2026/runs/matched_rgb_teacher_adaptation_v1`，不覆盖历史权重。执行状态、精确来源、实际成本和结果以该目录回执为准；本文尚不声称任何训练收益。

缓存已自然退出0：1053次scene调用、319.758726秒、峰值CUDA allocated 1,869,171,712字节。复用正式相机预测函数会同时计算其语义分支，但这里只保存RGB；1053不是仅RGB raster调用次数。三个001检查均逐像素一致，生成1050张成员TRAIN RGB、两域各350张legacy训练输入以及50张新组合legacy验证输入，没有打开真实GT像素。[完成回执](/mnt/data/SHM2026/runs/matched_rgb_teacher_adaptation_v1/cache/execution_receipt.json) SHA `432aa05baed9416a23e8280b2428028977e463e9f66a96d62aaac60085c44dff`，外层自然退出证据见同目录 `launch_receipt.json`。

两臂[训练计划](/mnt/data/SHM2026/runs/matched_rgb_teacher_adaptation_v1/plan.json) SHA `ec4f827a1729251a0b3b407859ff59599a769498a8afedc1043addafed5b2d99`，绑定46份源码、1331项输入。工作区82项相关CPU测试与冻结副本14项测试通过；两臂真实旧EMA来源链在CPU准备阶段已核验。该固定训练计划不自动宣称最终精度，需自然完成及独立共同原图评价后报告。

后续两臂均自然完成2000步，逐步随机轨迹及终态审计通过。第一次终评因新增RustDesk CUDA桌面上下文在资源检查处退出，0预测／0标注／0 CUDA allocated，原失败目录保留。单独 `evaluation_v2` 只允许经 `/proc/<pid>/exe` 确认的真实RustDesk与已有图形进程，仍拒绝未知计算进程；全部模型、输入、指标与门槛保持相同，不关闭用户桌面程序。工作区36项评测／适配器测试与冻结副本30项测试通过。恢复后的终评自然完成，独立CPU复算通过；其103.597秒属于共享桌面GPU下的缓存图像推理与评分时间，非独占GPU或双场端到端时延。

实际候选五类94.493989%、拉索93.623609%，未通过联合采用及整体域匹配证据门。相对同预算控制的拉索增加.388491pp、基础下降.341717pp，不能概括为适配完全无效或全面改善。完整表格、配对区间和来源见[结果报告](matched_rgb_teacher_adaptation_results.md)。
