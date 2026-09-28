# 语义分区 epsilon 复验：参数更新改善，最终指标未提高

2026-09-27。四组各2000步、600张新mask及独立CPU复核均已自然完成。**预定 integrated 联合 mIoU 为95.022657%，未超过同轮对照或原 E，19/28采用条件通过，当前配方停止。** E 的模型与导出接口保持不变。

## 单因素与结果

相对[上一轮](semantic_partition_matched_results.md)，endpoint/shape 的 Adam `eps` 从`1e-8`改为`1e-15`；head仍为`1e-8`。学习率分别`.01/.001/.0003`，betas`.9/.999`，无weight decay。四组从原H3独立初始化，fresh optimizer、seed42、相同259 TRAIN与2000项采样序列；没有中间VAL评分或checkpoint选择。原RGB、几何、opacity、语义特征/classifier、相机固定。三组field均7,472,040参数，head为564,757参数。[执行前协议](semantic_partition_eps_matched_protocol.md)保留。

完整bridge包继承上轮冻结源码，仍用Cholesky投影；新OT及固定类别总量helper没有混入。下表为41张有标注验证图、官方网格pooled CM，单位%。joint是scene与冻结旧DINOv3教师各半后映回原图；RGB明确复用E的30.035661dB/.874420736/.264741845，不是本模块收益。

| 组 | raw五类 | scene五类 | joint五类 | joint拉索 |
|---|---:|---:|---:|---:|
| 仅head |79.312262|94.566872|95.036611|94.433513|
| marginal |79.368130|94.543829|95.021868|94.414291|
| point |79.475666|94.543541|95.023887|94.418582|
| integrated，主候选 |79.446399|94.543260|95.022657|94.407133|
| 原E |—|—|95.109016|94.786655|

固定5000次配对视图bootstrap、seed20260926。以下为integrated减参照，单位百分点：

| 参照 | joint五类差 | 95%区间 |
|---|---:|---:|
| 仅head |−.013954|[−.037628,+.005925]|
| marginal |+.000789|[−.016885,+.017538]|
| point |−.001230|[−.019737,+.019601]|
| E |−.086359|[−.381195,+.119929]|

四个最小增益门与四个正区间门全部失败。相对E拉索为−.379522pp，区间[−1.656233,+.498492]；点估计超出预定−.1pp保护界限，但不能称统计确认的退步。其余类别保护通过，合计19/28，决策`stop_fixed_partition_candidate`。

预登记的次要描述性比较（新eps同组减旧eps同组）：marginal −.012245pp，区间[−.029303,+.002969]；point +.012630pp，区间[−.009424,+.032486]；integrated +.010132pp，区间[−.007306,+.028614]。均未显示可靠最终改善。未改配置的head-only跨运行也由95.026534%变为95.036611%，相差+.010078pp；不可声称bitwise重放或把这个量级的变化全归因于epsilon。次要比较不进入采用门，也未作多次开发选择校正。

## 更新确实发生，但不能代替质量收益

所有训练、评分、独立审计完成后，同一个CPU只读脚本检查Adam状态，0新图像、渲染或优化。integrated非零二阶矩分量的`√vhat/(√vhat+eps)`中位数：

| 参数 | 旧eps1e−8 | 新eps1e−15 |
|---|---:|---:|
| inside logits |.000932838|.999578358|
| outside logits |.000788823|.999487443|
| direction |5.11090e−7|.994921479|

端点概率TV高斯中位数从2.11446e−6升至1.27781e−5，p99从.0975825升至.2354852；法向偏转中位数从1.49e−7rad升至.0999456rad。因此不能继续把本轮无收益简单解释为原epsilon使参数不动。此比值只描述各自保存moments中epsilon的影响，不是可见性加权诊断或反事实训练轨迹。

此前16 TRAIN的[质量/排列干预](partition_mass_arrangement_results.md)只显示raw +.079303pp、scene +.002146pp。结合本轮近零最终效果，以及尚无证据将损失归因于类别总量漂移，**暂不追加固定qbar训练**；这不证明约束必然无效。其[数学组件](semantic_mass_constraint_helper.md)完成32项CPU测试和独立一阶梯度检查，修复了部分路径可返回不完整二阶导数的接口漏洞，现明确拒绝高阶梯度图。组件未接入模型，没有贡献指标。标准求根、条件积分及epsilon调整不作为原创贡献。

## 成本、审计与范围

worker1452.924秒、外层1454.898秒自然exit0；四组训练181.240/396.241/394.312/381.318秒，最大allocated5,727,755,264字节。终评98.976秒、250次scene/150次partition shader、0新teacher调用。200份新soft、600份mask完成后才读取41份原标注；50张原E语义逐字节复现。GPU与RustDesk共享且已有JIT缓存，跨运行墙时差不代表算法加速或部署FPS。

独立checker35.907秒、外层36.686秒自然exit0，无CUDA。56源文件、591输入、2034绑定产物不变；12份metrics、492张新CM与41张旧E CM、400份soft到mask、全部600mask、4组主比较及3组次比较的5000次bootstrap和28条件，最大数值差0。raw soft未保存，独立审计对交付hard mask评分，未重跑shader；RGB为E复用，未新跑LPIPS。

| 记录 | SHA256 |
|---|---|
| [plan](/mnt/data/SHM2026/runs/semantic_partition_eps_matched_v1/plan.json) |`c3974d205afaa3c897cb6412a086879019e9b999fd284cac20c7d2dad58afffb`|
| [执行](/mnt/data/SHM2026/runs/semantic_partition_eps_matched_v1/execution_receipt.json) |`1b120b089cf3ddfd77052d4dcca7a65188ac1d9463f6ef739f782efd6c576e6f`|
| [独立复核](/mnt/data/SHM2026/runs/semantic_partition_eps_matched_v1/independent_cpu_review.json) |`be20503edbe2eb73bd003d25ed6710b3c8f5996e5d3a65c6e8c29bc89085b986`|
| [优化器描述诊断](/mnt/data/SHM2026/runs/semantic_partition_eps_matched_v1/optimizer_endpoint_diagnostic.json) |`fb9967eeca5676a9245dfd2cbd8d9fbe63638251e1036f87ba161356abc24944`|

这仍是同一桥、同一反复使用的开发划分。区间不涵盖seed、跨桥变化或选择不确定性，也不是同学Dev30精确复现或官方盲测。明确提升与有效学术机制的联合目标尚未完成。
