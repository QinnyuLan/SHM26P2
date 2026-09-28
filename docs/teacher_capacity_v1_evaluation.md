# H+／7B 固定终点评价入口

正式四阶段训练已经开始，结果尚未产生。评价代码在读取这轮任何终点评分之前单独冻结，训练快照没有修改。

- 评价计划：`/mnt/data/SHM2026/runs/teacher_capacity_v1/evaluation_preparation/plan.json`，SHA `be5a47bd4ace31f796c6d69cc9ed0f4c769a52278d3aa103a74e668da6f53f69`。
- 正式训练 launch manifest SHA：`f1237a5a89d98c97b1c5044368e21879b948f0fd9292e2997822ef5f731be47a`。
- 评价快照包含与训练逐字节相同的 33 个 package 文件，以及既有评价、配对、采用门、轨迹核验入口和环境锁文件，共 39 个文件；没有新写一套评分公式。

执行前要求四个阶段均有完成回执，`last.pt` 分别对应 real 6000 步、render_adapt 2000 步；阶段 2 必须读取本骨干阶段 1 的最终 EMA。两阶段的 H+／7B 完整采样增强 trace 与最终 RNG 比较都必须通过。恢复段由冻结的 `compare_stage_receipts` 重建，不能只信 advertised digest。任何来源不符或轨迹不匹配都保留并报告，不能悄悄换 checkpoint。

两套终点共用 H3 cross 场、固定 0.5 教师概率权重、768 tile／512 stride／水平翻转／0.25 全图上下文。教师只读取该场渲染的 RGB；所有相机预测完成之后，评分器才读取原图真值。使用同一原始畸变网格的全部 50 RGB／41 标签，fingerprint 为 `21a2f19c5d4d703403a0107402d5dd5e006dcd98e5f8e023820c388c3e69a85d`。

固定报告三组：7B−匹配 H+、7B−旧已选最强组合、匹配 H+−旧已选最强组合。配对 bootstrap 为 5000 次、seed20260926。逐文件核对三套输出的全部 50 张 RGB PNG，不能用 RGB 平均指标相等代替。

7B 的采用门保持[原方案](teacher_capacity_v1_plan.md)：相对两个参考，五类增益都至少 0.30 个百分点且各自区间下界大于零；前景及拉索点估计不下降；全部 RGB PNG 一致。旧强组合原图五类为 95.1090%。不通过也完整报告固定结果，保留旧组合，不搜索新权重、种子或终点。这是单场开发验证的工程容量比较，不是同学 Dev30 的公平复现或学术创新证明。
