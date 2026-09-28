# 真实轨迹冲突与 H3 剩余错误：固定 CPU 关联诊断 v1

2026-09-27，在读取此轮预测数组/生成关联统计前冻结。上一轮 `train_track_semantics_v1` 已证明保存 TRAIN 对应存在严格区硬标签冲突；当前只检验它与 H3 final 剩余错误的条件关联。没有新模型、训练、渲染、RGB/annotation/mask 解码、VAL读取或官方分数；现有工程 E 不变。

## 人口、输入与查询

全量参考来自已完成的 60,000 tracks／457,102 observations NPZ，绑定旧 plan、执行/收集回执及独立审计。仅使用原格 known 标签、positive depth、重投影 ≤1px、primary EDT>10px 的 strict observations。按全部350 TRAIN name排序的 floor(index×4/350) 分组，与上一轮相同；第3组无已知标签，明确不当作第四个有效标注组。

查询相机固定为旧 `cross_renderer_authority_v1` 的16个TRAIN：002,021,041,059,079,100,118,137,156,176,200,220,241,259,278,300。其S为同一个SHA `22bc8a2d…045226` H3检查点自身RGB/几何/完整head的final概率，prepared legacy 1320×989 HWC FP32；不是raw Gaussian类别，也不是E的H3＋旧教师组合。先核旧plan/receipt/source/hash与当前manifest相机，再取缓存中的S，不新增渲染或复活失败JS门控。

查询条件：上述strict、属于固定16相机、legacy-valid且legacy标签已知。采用已保存的legacy_index作整数数组索引，断言primary/legacy标签相同。原缓存FP32转FP64计分，不裁剪/重新归一；事前断言范围[0,1]与和偏差≤2e−5。背景也作为正常第五类计分。报告重复(image_id,x,y)查询数，不能把重合像素视为独立证据。

prepare只读取既有JSON/代码元数据并哈希文件字节，不load observations或soft数组。execute才加载已存在的数值数组。全量概率来源的旧检查点身份由已完成收集回执与代码链证明，本轮不加载模型。只绑定16份S，不消费其它教师预测。

## 排组参考与目标

每query仅从同track、**其它相机组**的strict已知观察构造参考。拒绝重复track/image记录。需要至少3个不同参考image_id；没有基于纯度的接受筛选。参考使用全部strict人口，不额外受legacy-valid限制。无跨组支持保持NA，不能当一致观察。

连续exposure固定为 `1 − reference_count[target_GT] / reference_count_total`。主分组为 exposed(exposure>0) 与 compatible(exposure=0)；另描述参考多数类与target不同（平票argmax取最小类）和continuous exposure。exposure明确使用查询GT类别，**不是部署分数或OOF预测**，不得宣称剂量关系或因果解释。

错误为 `argmax(S)!=target`；五类Brier为 `sum_c(S_c−onehot(target)_c)^2`，理论范围[0,2]，缓存FP32求和舍入只容许上述范围公差。先分别在每个side内对同track查询求平均，再对含该side的tracks等权。两个side可以共享track，也可以有不同track人口；这是条件均值差，不是轨迹内配对效应。

报告所有可查询strict与跨组eligible的人口、逐类5×5混淆矩阵、错误/Brier；eligible按side、五类、参考数3–5／6–9／≥10分别描述，保留空格NA。主差值 = exposed−compatible，正值表示冲突侧错误更高。整体做2,000次track bootstrap，seed20260927，共同从所有eligible unique tracks重采样，每个抽取携带该track全部两侧query；不能两侧独立抽，也不能把缺侧均值当0。每次按各side有效track重计分母；无某侧的draw记NA。按类、参考数、相机组只作描述，不搜索新区间/阈值。

## 预定必要条件与敏感性

主分析错误覆盖需同时满足：≥1,000 eligible queries、≥200 eligible tracks、实际预测错误涉及≥20个distinct tracks且≥4个image_id。任一不满足为 `inconclusive_error_coverage`，不能以很大的条件比率取代事件数量，也不能据TRAIN拟合良好否定未来VAL机制。

覆盖足够时，条件关联信号须同时满足：整体Brier差的track-bootstrap95%下界>0；背景与拉索每类的两侧均各≥20 distinct tracks且各类Brier差>0；至少2个有效target相机组的Brier差>0。有效组预先定义为两侧各≥20 distinct tracks；全部组都报告，无标签组不算。满足仅为 `conditional_error_association`，否则 `not_supported`。这不是模型采用门、统计因果结论或保证性能可提高。

预定敏感性：由上一轮duplicate metadata删除所有曾触及duplicate pair的整track，包括它的全量参考和query，然后重新构造reference/eligibility并报告同一统计。敏感性不执行新的bootstrap/gate，不替换主人口/主结论。

## 限制与行动

H3已拟合全部TRAIN相机，只有参考票排组；这不是基础模型OOF、未见视图泛化或E最终错误。类别、轨迹长度、几何位置、相机困难度、遮挡和标签定义都可能混杂；背景/拉索与参考数分层不能消除全部混杂。轨迹稀疏且只采strict内区，不能推断全像素错误上限、边界问题或VAL像素增益。

若覆盖不足，明确停止用当前16缓存判定该解释，不新增视角、阈值或GPU救援；若无关联，不把该现象当已识别H3误差瓶颈。若信号存在，也只允许继续事先锁定的对应/可见性调查，尚不直接加方向语义模型。普通view-dependent特征、方向核或跨视图一致性没有自动学术新颖性。

uv CPU单次执行；内部120秒/外部180秒，禁止覆盖已有执行。保存查询prob/row、逐观察reference/exposure/error、完整汇总、源码与输入hash；先合成测试/代码审查，后冻结执行，再以独立CPU实现复算。当前最佳交付保持30.035661dB／95.109016%，用户要求的创新消融增益仍未达成。
