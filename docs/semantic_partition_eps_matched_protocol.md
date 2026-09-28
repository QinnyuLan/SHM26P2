# Semantic partition：固定 epsilon 匹配对照

状态：代码与 CPU 合同准备，尚未创建计划或启动训练。协议 `semantic_partition_eps_matched_v1`，建议目录 `/mnt/data/SHM2026/runs/semantic_partition_eps_matched_v1`。

旧 `semantic_partition_matched_v2` 的失败门保持原样。其独立审计通过；完成后统计发现 integrated 内/外端点的 Adam epsilon attenuation 中位数为 0.000933/0.000789，端点 TV 中位数约 2.11e−6。固定 16 TRAIN 的同端点读出干预也测到少量内部空间作用，但精修后差异很小。这支持检验标准优化配置的假设，不证明 epsilon 导致预测错误，不把旧失败改称成功。参数计数统计不等于可见像素加权，也不是另一条优化轨迹。

## 唯一数值变化

完整继承旧 v2 的冻结 `bridge_rgs` package，包括已通过真实预检的 projection/Cholesky transport；不取当前主树的新模块。四臂仍为 `refiner_only`、`marginal`、`point`、`integrated`，均从原 H3 `22bc8a…45226` 重新加载，fresh Adam，不续训失败终点。

实际 Adam 分组明确保存：`head` lr 3e−4 / eps 1e−8；`endpoints` lr .01 / eps 1e−15；`shape` lr .001 / eps 1e−15。后两组是本轮唯一优化数值变化；β=(.9,.999)、weight decay=0 保持。三 field 臂均为 15N 新参数，marginal 法向不参与前向，允许其 Adam state 缺失；不能将正常 unused 参数说成训练失败。降低 epsilon 也可能放大微小数值梯度，不能以更新量变大直接证明效果。

其余全部固定：259 labeled TRAIN、seed42 独立逐 epoch shuffle、每臂 2000 更新、全原生 legacy 网格、原始场 RGB/geometry/features/cameras 冻结、final weighted CE + .2 Lovasz + .001 residual² + .25 raw weighted CE、无增强/中途 VAL/teacher 训练/新 Gaussian。仅固定 `last2000`，不挑 best。源码、350 相机、历史 mask/valid SHA、权重与初始 head 来源沿用原计划。

## 固定评价和解释

保留原 E RGB 字节/分数以及同旧 H+ 的固定 .5 软概率融合。先重现原 E 的 50 张 mask；四臂共 600 joint/scene/raw mask 全部完成后才读取 41 张官方 annotation，输出原 12 组指标。原四个主比较仍是 integrated 对新 refiner-only/marginal/point 及原 E；all5 增益分别至少 .15/.10/.10/.20 个百分点且对应 95% 配对区间下界 >0，索点差不得低于 −.10pp，其余类不得低于 −.20pp。所有门必须通过，仅具备后续评审资格，非自动采用。

完成四个主比较后，额外输出 marginal/point/integrated 各自“新 epsilon 终点 − 旧同臂终点”的三组配对结果，5000 次、seed20260926、同 50/41 相机。它们仅描述优化配置影响，不替代主门，不选择获胜臂。若三个新 field 臂共同改善，也不因此证明 CDF 积分相对 point 的独立价值。开发集已反复使用，非新盲测、非无偏 selection-adjusted 泛化证明；标准 epsilon 配置不是学术贡献。

## 执行边界

准备时必须核旧自然 exit0、完整四臂/评价、独立 CPU audit passed、旧 plan/source/input SHA；旧 gate 不要求 passed。绑定旧三臂 joint metrics 与 plan/exec/launch/audit 字节。新 checkpoint format 为 `semantic_partition_eps_matched_delta_v1`，仍只保存 head/new field 与独立本阶段状态，不支持 ordinary resume。

沿用整轮内部 3600 秒、外部 3660 秒，四臂顺序执行、0 retry；共享显示/RustDesk 环境，时间不是独占部署 FPS。CPU 测试包括实际参数组、微小梯度下 epsilon 的更新差异、marginal unused direction 与采样/目标/主门不变。冻结合成测试使用 `-p no:cacheprovider`。本文件和源码须经 root 审阅后才可准备/冻结；GPU 由 root 显式调度。
