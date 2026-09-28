# 下一共享几何假设：当前不推荐新增方法

**后续状态：**固定尝试已经执行；最后报告序列化失败，保存数组经CPU恢复及独立核验后不支持该干预，见[结果与证据边界](shared_visibility_intervention_results.md)。下文保留形成计划时的设计状态，运行时冻结的是run内proposal副本。

2026-09-27。本轮只读既有文档、生产代码和已完成诊断的说明；未打开活动checkpoint、未使用GPU、未实现新实验。下述设计在Mip-like终点评价前形成，不修改已绑定的Mip材料。协调者随后报告Mip已完成终评、固定投入门未通过；结果见[Mip结果](mip_filter_reference_results.md)。这不是下述命题的验证，也不意味着所有采样约束均无效。

**决定：当前证据不足以推荐一个同时改善RGB与共享几何语义、且具有明确学术差异的新机制。** 这不是断言共享几何没有改进空间；而是尚未把剩余两类误差定位到同一可干预的几何原因。继续增加语义分裂、贡献门控或频率权重，容易将尚未识别的原因包装成方法。

## 现有证据排除了什么，仍没有排除什么

| 已完成证据 | 对下一机制的约束 |
|---|---|
| H1的clean/mild配对均未显示增量RGB收益；相机恢复诊断有响应 | 排序时补偿残差的当前实现没有收益证据。它不等于已经检验所有场梯度消元；后者的独立方向校准仍未过门，不能跳过数值问题开始pilot。见[H1](pose_stress_results.md)、[校准](pose_profile_numerical_calibration.md)。 |
| H2投影范围、H3交叉矩的主要对照均不支持独有增益 | 不能由原生语义差就继续增加不确定性或上下文输入，也不能把这些负结果解释为“只能改几何”。见[H2](h2_multiscale.md)、[H3](h3_depth_moments.md)。 |
| support-constrained分裂修复降低极端轴比；front控制显著降低目标前侧质量，却损害整体RGB和索类 | 形状／密度代理改善并非任务改善。不能用“浮层看起来少了”代替完整RGB和语义证据；这也未排除其他几何干预。见[分裂审计](floaters_and_split_audit.md)、[front配对](sparse_front_pair.md)。 |
| 16条固定TRAIN射线在几何不变时，实际renderer正确类margin均为正，最小.082422 | 已否定这16条射线必须改变几何才能共同分对的强前提。但跨视图权重重叠很弱，不能推广到全场、强耦合射线或三维真值。见[实际前向证据](semantic_compositing_forward_diagnostic.md)。 |
| 259 TRAIN单轮赋值：EM的CE下降小于feature Adam；两者冻结final均退步 | 没有获得类别内层充分优化或剩余几何难分证据；一轮收益小不等于到达固定几何极限。见[单轮结果](raw_semantic_assignment_results.md)。 |
| 修复SSIM后的固定颜色续训相对base有约+.11 dB，但raw相对native仅+.0047 dB；完整hybrid相对mixed的RGB区间跨零 | 不能概括成“所有外观优化均无效”，也不能声称已有显著RGB创新优势。当前较高final分数与较低raw分数并存，不自动构成共享几何损坏证据。见[颜色重放](ssim_fixed_appearance_replay.md)、[RGB参考](rgb140_reference_results.md)。 |

这些实验的底座、像素协议与预算不同，不能把跨实验相关现象合并成一个因果结论。当前共同原图可比较性解决的是评分网格，不会使不同训练链变成单因素实验。

代码检查与上述边界一致：`model.render` 的RGB和语义使用同一套means/shape/opacity及启用时的有效Mip参数；语义主调用默认隔离几何梯度，显式语义几何分支另受门控。强语义阶段冻结几何/RGB是设计合同，不是无意断图。raw概率与16维特征还共同供refiner使用，因此单独改类别赋值后的final变化包含特征分布改变，不能归因于几何。生产代码没有实现“充分消除外观／类别欠优化后，再比较结构动作”的条件求解；此前[条件结构方案](conditional_structure_decision.md)已保留它，但启动证据不成立，不能把它再命名为本轮新方向。

## 唯一仍可保留的必要命题，以及为什么尚不成为新方法

必要命题是：**存在由不同TRAIN视角共同支持的有限共享几何改动，在控制颜色／类别重新赋值的影响后，仍能减少真实RGB误差和原生语义误差，且收益在未参与该局部适配的TRAIN视角保留。** 它没有被当前实验全面否定，但本质上正是旧条件结构方案尚未证明的前提。本轮不提出第二个变体，不把“未否定”当成“值得长训”。

最强先例冲突仍是：固定可见性下优化语义赋值已有[FlashSplat §3.2](https://arxiv.org/html/2409.08270v1)；以语义冲突梯度分裂已有[GradiSeg §3.4](https://arxiv.org/html/2412.00392v1#S3.SS4)；按投影边界改变高斯形状已有[SAGD §III-D](https://arxiv.org/html/2401.17857v4)。把这三类方法接起来、加入RGB不退步门或换成LP/EM，并不能直接产生学术差异。先前提出的“同预算有限动作、重新赋值后跨视角验证”至多是更严格的动作诊断，需要先有可测净收益才值得定位贡献。多视图频率／可见性方向的重合另见[五篇定向核查](visibility_sampling_prior_art.md)，本轮不扩大检索。

## 最小必要条件诊断：固定近相机集合的双向opacity干预

**状态：仅可审设计，没有新runner、没有执行授权，不因Mip结果自动启动。** 它先检验更窄的命题：在当前已训练参数下，一个仅由TRAIN几何定义的集合是否存在可重复的共享可见性干扰——降低它的opacity能在同一批TRAIN像素同时改善RGB和raw语义，并保全图RGB。这个前置检查不声称已经控制了类别／颜色欠优化，不等价于上面的充分条件结构命题。

### 固定输入与干预

- 场：已完成的`/mnt/data/SHM2026/runs/ssim_fixed_corner_v2_semantic_coupled/last.pt`，SHA `391a0577450f7f458b75cb9e2fcf926953142b516728ea3600ef2ef064d72c13`，498136点、无Mip。复用已完成`raw_semantic_assignment_v1/source_snapshot`中的加载／render接口，不从活动主源导入新模型。CPU准备时重新绑定该既有源码、checkpoint、manifest、init和相机字节，不在本文预造新plan SHA。
- 相机：复用既有raw-semantic-mass计划的8张TRAIN：`002/043/084/125/167/215/257/300.png`。全部native1320×989、原保存TRAIN相机、SH3、`refine=False`。不换视图、不读VAL；RGB、valid与mask路径只来自这些TRAIN记录。
- 集合G：点到最近350个TRAIN相机中心距离小于`.01*scene_scale`，且到原始init SfM点最近距离大于同阈值。阈值固定为`.16261470794677735`；既有CPU统计为3004点。重新计算必须复现点数并保存排序IDs与SHA，否则停止输入合同错误，不调阈值。规则来自既有全局几何描述，不使用RGB、mask或当前误差选点。
- 三状态：原场0；仅G的opacity logits减`log(2)`（odds乘.5，记−）；仅G加`log(2)`（odds乘2，记+）。每个状态均从原logits独立构造，不顺序累加；保存实际FP32 opacity差、原始opacity质量和饱和比例。means、scale、rotation、SH、background、feature、decoder及camera全部不变。**这不是删除点，也不是将alpha直接乘.5／2。**

G不是已证明的浮层。尤其旧[VAL235深度集合](floaters_and_split_audit.md)是另一场上的1215点，屏蔽它已损害TRAIN234/236的RGB；不能重新把该失败集合包装为本次新证据。这里是不同的全TRAIN相机／init距离规则与391a场，仍须披露它受既有浮层问题启发，不是独立发现的假设。

### 固定贡献支持、测量与成本

先在原场每view额外做一次完整场的双通道前向：颜色为`[1(i∈G), 1]`、背景0，所有点继续参与原排序、alpha和transmittance。得到`m_r=Σ_{i∈G}W_ri`及全点质量；不是只渲染G子场，不用曾出现缩放误差的颜色VJP。全点质量须复现同次alpha（max误差≤5e−6），`0≤m≤alpha`允许同数值容差，否则整个诊断`inconclusive_render_contract`。所有后续贡献加权指标都固定使用这个原场m，不能按干预后的m重新选支持。

每view依次独立渲染`0,−,+,0,−,+`，保存RGB、未归一raw语义通道和alpha到CPU；所有8view预测完成后才解码这些TRAIN GT。主指标用FP64聚合：未夹紧float RGB的逐像素三通道MSE；以及现有诊断的affine-noise raw CE（δ=5e−7，同259 TRAIN固定class weights，精确复用既有plan的FP32值，约为`[.45604560,.92277277,.85304344,1.19358921,1.57454896]`，不把p3d归一化梯度当counts）。二者在相同`valid & known_label`上按固定m加权、除以Σm；同时报告8view等权的完整valid RGB MSE、完整语义CE、pooled raw CM/IoU及每类FP/FN。IoU和图例是描述项，不代替预定连续主指标，不运行final refiner或教师。

可测覆盖预先定义为每view共同有效支持上`Σm≥64`个等效全alpha像素，且`N_eff=(Σm)²/Σm²≥256`。这是有限诊断的覆盖资源门，不是物理可见性真值或数值噪声界。少于2张预定view达到它，结果为`inconclusive_coverage`；仍完整报告8view，不补相机、不扩大G。不要求原方案并未保证的6/8覆盖，正结果也只能外推到实际可测的这些view。

成本固定为**48次scene render（RGB+semantic共96次raster）+8次双通道贡献raster=104次raster，0 backward、0 optimizer、0 VAL、0保存新模型**。源码／输入绑定在CPU准备阶段完成；若以后获执行授权，worker加载到恢复总限时120秒，超时保留`inconclusive_timeout`、不降分辨率重试。所有状态和buffer在finally逐位恢复，核checkpoint输入SHA及全scene张量；原场重复也用于检查临时干预没有残留。

### 事先固定的判读门

对每个连续标量，数值容差取三状态独立重复之间的最大绝对差与`32*eps_FP64*max(1,各状态绝对值)`的较大者，再乘10。报告原值和容差，不把重复稳定当作完整渲染正确性证明。三状态指标各取两次重复的均值，所有增益定义为`原场loss−候选loss`。为避免仅凭极小但可重复的变化启动后续研究，贡献加权两项主指标另设相对改善至少0.1%的资源门；它不是数值误差界，也不是普遍有意义的精度阈值。

只有同时满足以下条件，才记录`necessary_interference_signal_present`：

1. 至少2张view覆盖可测；在其中至少`max(2,ceil(.75*M))`张，−状态的贡献加权RGB与语义loss**同时**下降并超过对应容差；可测view等权的两项增益均须超过容差且达到各自原场loss的0.1%。
2. 同一统计上的−状态两项增益均大于+状态增益，差超过容差。+是固定反方向控制，用于排除“任意改一次opacity都好”的简单解释；不要求有限非线性干预严格反对称。
3. 8张全view等权完整RGB MSE不得增加超过容差，完整raw语义CE须下降超过容差。额外报告每view完整RGB损害，不能只呈现贡献加权裁片。

覆盖够而上述门未满足，记`specified_intervention_not_supported`：**只停止这个固定G、固定±log(2)、固定8view的有限干预，不推断所有几何调整无用。** 覆盖不足或数值合同失败为不可判定。TRAIN全图不退只是本次寻找“当前参数下无明显TRAIN代价的共同干扰”的资源门，**不是改善VAL泛化的数学必要条件**；有用正则化可能牺牲训练拟合，本诊断负结果不会排除它。没有VAL选择、没有尝试第二个幅度，也不根据cable指标挑选正结果；8个视角不是独立场景样本，不给伪总体显著性。

### 能承接什么，不能承接什么

可直接承接已通过的共享RGB/semantic原renderer前向、原始raw通道捕获、完整场颜色渲染和全张量恢复接口；贡献量直接由前向合成获得。它不调用means梯度、pose有限差分J、stop-J代理、SSIM反向或VJP权重恢复。因此不触发也不修复[pose-profile方向校准](pose_profile_numerical_calibration.md)的失败门，不能据它批准被冻结的profile训练。

正结果只证明**当前固定颜色／类别赋值下**存在一个可测的共享可见性干预方向。它没有重新拟合赋值，没有等预算拓扑控制，也没有区分“G确是错误几何”和“G的颜色／类别尚未优化”。没有匹配非G的opacity控制，因而更不能声称近相机规则具有独特性。负结果也只否定该有限方向。标准opacity pruning／透明质量正则及已有front控制是最强直接重合；这个必要条件诊断本身没有新方法的学术差异。

此前16-ray正margin已经否定那16条射线“必须改几何”的强命题；本设计不推翻它。两点四步动作探针的检出力不足也仍成立，不再复活。下一步唯一可决定的事项是：协调者是否认为上述104次纯前向足以检查一个尚未被直接检验、范围非常窄的共同干扰命题。即使过门，也先重新评估原因可识别性，不自动追加训练。现阶段仍没有同时成立的精度收益与新学术机制，不能以一个局部诊断替代它。
