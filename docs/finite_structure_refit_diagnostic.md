# 有限共享结构交换：同预算属性重拟合后的动作选择

**后续实际状态：**五态对照与独立审计已完成，四候选 A raw CE/RGB 均劣于 keep，两个选择器都选 keep，未支持必要信号。见[完整结果](finite_structure_refit_results.md)。以下保留制定时的规格；运行内冻结副本未改动。

2026-09-27，待实现的单次 TRAIN 诊断规格；本页没有启动训练、读取新像素或冻结运行。有限动作本身用于取得缺失的证据，**不以事先证明几何瓶颈或 q 收敛为准入条件**。它不改变此前负结果，也不宣称新的分裂、EM 或条件优化公式。

## 场、相机与读取边界

固定原 H3 `22bc8a2ddb260f93cb01b17857c97b2bb0873038efdb9318545cb2bdbb045226` 的geometry/SH，498136 点、SH3、legacy native 1320×989。所有状态的q均从已完成 `raw_simplex_em_fw_v1` 的同一最终 **FP64 master** 初始化，renderer取其原样FP32 cast；不能拿原classifier softmax当精确归一的master，不能另加floor。该q及其上游额外优化成本共同继承，不把它称为原H3属性。源码继承已审计EM/FW冻结包及其direct-q预检祖先；仅新增显式有限交换helper/worker，不取当前主树其它功能。

协调者已报告本轮匹配head终评完成：new-q adapted联合95.012857%、old-q adapted95.024143%，均未超过E95.109016%；raw官方改善未传至最终输出。因此本探针只检验共享结构的有限条件作用，不以prior/head适配成功为前提，也不把raw改善当最终采用证据。

沿已固定16张 labeled TRAIN 名单，偶数序号为 A，奇数序号为 B：

- **A 拟合/选动作：**002、041、079、118、156、200、241、278。
- **B 一次检验：**021、059、100、137、176、220、259、300。

相机严格取 checkpoint 对应名字的 TRAIN camera 与 manifest K。B 的相机元数据可用于预先检查视锥；B 独有的 RGB、mask、annotation **字节及散列均延后读取**，只继承已完成来源中的预期 SHA（prepared RGB 可追到 `matched_rgb_teacher_adaptation_v1/plan.json`）。明确例外：本manifest的A/B共用 `artifacts/prepared/valid/camera_1.png`，A拟合必须读取这个相机共享valid；它不包含B独有的图像或语义内容，不声称此前从未读过B所引用的共享valid。先完成所有 A 拟合，写下两个不可更改的选择及其依据/SHA，再开启 B 独有像素。B 曾参与基础场训练，仅是本轮局部重拟合的留出相机，不是 base OOF 或新场景泛化。无 VAL、教师或 head 更新。

## 四个候选、一个共同 keep

候选生成只读原场参数、A 像素及 A/B 相机元数据。A 原场渲染中取实际投影均值/协方差、正半径；用中心、±两个一标准差主轴、±两条标准差对角线，共9个固定位置，按 `floor(x),floor(y)` 查询 native known/valid 标签。这只是找候选的离散足迹探针，不作遮挡或真实三维语义证明。

parent 合格条件：opacity 在 [.005,.95]；至少2张 A 中中心入图、正半径、屏幕最大标准差在 [1,32] pixels，且9点同时出现背景0和拉索2；至少2张 B 中原中心正深度且在视锥内（仅元数据，不声称无遮挡）。按混合 A 视图数降序、各混合视图 `min(n_bg,n_cable)` 之和降序、原Gaussian ID升序取256个不同 parent。数量不足时报告候选覆盖不足，不换相机、扩大尺度或改标签规则。

四个候选依排序位置模4分组，每个 **K=64** 个 parent。共同可编辑集合 U 固定2048行：先包括全部256个 parent，再按到任一 parent 中心的最小欧氏距离、原ID排序补足。共同64个 donor 从 U 中非 parent 行选取，按 A 投影的 `opacity × Σ sqrt(det(Σ2D))` 最小优先、ID打破并列；出图/非正深度视图计0。该量是未计遮挡的几何保留代理，不能叫实际可见贡献。所有候选删除完全相同的 donor，不能各自挑最便宜的删除。

每候选明确删除64 parent及64 donor，再追加128 child，N始终498136。使用 [finite_structure.py](../src/bridge_rgs/finite_structure.py) 的纯显式交换：稳定 source_indices；方向固定为 parent 最大主轴；`support_constrained_split_geometry` 给 ±0.5 Mahalanobis 单位位移；仅该轴尺度除1.6；每child opacity为 `1−sqrt(1−alpha_parent)`，沿helper既定clamp。SH、features、q及所有逐点属性均由 parent 复制。**不调用会全局prune的 densify 函数**；不改变未选行、相机或背景，不做后续geometry/opacity优化。新结构同用于RGB与raw语义，没有semantic专用几何。

该opacity规则只在理想重合、未缩放足迹情形保持中心透射率，不保证实际AA alpha/RGB守恒。记录每parent实际世界/屏幕位移、零步alpha/RGB变化及donor质量代理。K=64和显式半标准差位移提供有限干预，不依赖四次极小means更新。动作也不是“充分适应后的最优几何”。

## 每态相同的有限 q/SH 重拟合

keep和4候选分别从原场独立构造；U经 source_indices 映射后每态仍恰2048行。只更新这2048行的直接q及SH0/SH-rest；其他q、SH、background、features、decoder、refiner、geometry、opacity全部固定。q通过同一叶子 `q_in=q_out` 的已测direct shader合成，几何改变后重新取得该态自己的tile ledger，不能复用keep的W。

每态固定 **2个block，每block10个EM槽位＋1个FW槽位**，共20EM＋2FW；FW的simplex LMO仅改变U中的行，允许原本精确为0的类别重新获得质量。使用既有EM/FW数学：EM `u=q*(-g)` 逐行归一、零责任行原样保留；FW沿唯一LMO方向，以实际vertex前向和固定64次导数二分选步，沿用已审计方向数值检查和实际目标接受规则。不加floor、不扫步长，不把EM-only zero-lock留给新child。

SH在20个EM槽位各做一次fresh Adam提案，源 `model.optimizers` 参数固定为SH0 LR=.0025、SH-rest=.000125、betas=(.9,.999)、eps=1e-15、无weight decay；FW槽位不更新SH。每view的FP32梯度先转FP64累加再除8，实际SH仍FP32。

两个独立目标均逐view等权：q精确复用已审计 `affine_raw_ce`、完整FP32 class weights及δ=5e−7，不经production clamp/renorm；SH为 `clamp(RGB,0,1)` 对原prepared RGB/255的完整valid三通道MSE，FP64归约。这里明确用简单MSE作局部颜色拟合，不冒充旧完整L1/SSIM配方。固定几何下q不影响RGB、SH不影响raw，因此同次提案pass分别检查两目标：只有严格下降的对应提案提交；拒绝则恢复该参数及其optimizer moments/step，且保留该分量原梯度供下一槽位。实际trial pass同时得到各分量的新梯度；不能把被拒参数处的梯度用于已恢复参数。

固定26个完整A pass：baseline梯度1、20个EM/SH实际提案梯度pass、2×(FW vertex前向＋实际FW候选梯度)4、最终确认1。为保持实际预算匹配，finite但未下降的EM槽位仍占原pass；无可行下降FW方向的保留槽位执行当前q的no-op完整pass，明确记作未接受而非优化。非finite、非法simplex或有信号方向的数值检查失败停止，不能以no-op掩盖数值错误。各态不提前增加槽位、换幅度或延长；分别报告真实接受数与26pass轨迹。

最终确认pass核实际联合末态；q与SH的提交/梯度缓存分离只是利用目标独立性，不改变目标。这是相同**尝试/渲染预算**的近似重拟合，不保证相同接受步数、条件最优或消除全部属性欠优化。保留零步/末态，允许结果显示重拟合仍不足，不因此续训。仅保存诊断delta/trace，不生成可采用生产模型。

## 两个选择器与 RGB 保护

普通控制在原geometry的A上用相同起始q32及raw CE，取得标准gsplat五通道概率合成的 `means2d` 有符号梯度范数（按width/2、height/2归一）；每parent跨A取均值，候选分数为64 parent之和。q作为显式colors输入，不经feature逆映射或原classifier替换；背景仍为固定背景类别，raw前向须与direct shader对齐。它是常规一阶增密启发式，**不是有限对称split的精确方向导数**，不把它当W伴随证书。普通梯度仅排名；不用于执行几何优化。所有候选使用同一q/同一目标计算分数。

对两个选择器共同先施加**相同的拟合后A RGB可行集合**：相对keep重拟合末态，完整A等view MSE不得增加超过 τ；任一A图严格要求 `candidate MSE ≤ 1.001 × reference MSE`，**不另加τ**；固定局部支持内等view MSE同样不得增加超过 τ。局部支持为所有256 parent及64 donor在该view的原3σ椭圆并集，生成后固定，不能按动作/预测误差缩小；每view记录实际二值支持的SHA256，各态零步/末态/重复必须一致，CM逐GT类行和也须相同，RGB valid支持由外层绑定。keep-repeat专指拟合后keep末态在A/B的重复测量，两次模型state SHA必须相同；绝不用零步与拟合后差值估计噪声。τ=`max(10×abs(mean_keep−mean_repeat), 1e−6×mean_reference, 1e−12)`，对应完整或局部的同一视图集合。某view局部支持为零时所有态必须一致，local MSE记None且只跳过该view的局部均值；全图与逐view保护始终保留。B对gradient参照仍使用同一keep-repeat的噪声项，但relative floor及逐view .1%取当前gradient参照的MSE。这是本次资源投入保护门，不是物理误差界。

- **gradient choice：**在共同可行候选中选上述普通梯度分数最大者；ID打破并列。没有可行候选则选keep。
- **refit choice：**在同一集合中选末态A raw CE最低者；仅严格低于keep才选动作，否则keep；ID打破并列。

两种规则实际共享**全部相同候选的重拟合与RGB筛选结果**，普通控制不使用重拟合后的语义值来排序。这样隔离“用有限重拟合语义收益替换梯度排序”这一决策，而非给新规则额外算力/额外RGB过滤。共享这项成本是机制对照，不能宣传普通gradient本来需要枚举成本。所有数值/选择写盘后才读B；B不用于回选第二名。

## B 的一次输出与判读

B上对keep和四候选的零步、末态都实际重渲染，完整报告；主比较只为事先锁定的refit choice−gradient choice及refit choice−keep，其余是描述项。报告全图及上述固定支持内的RGB MSE/PSNR、raw CE、5×5 CM/各类IoU、cable TP/FP/FN/precision/recall，以及逐view差；不要把像素数当独立重复，8视图只给配对原值与方向一致数，不造总体泛化显著性。

必要信号定义：两个选择不同；refit choice在B的raw CE与cable IoU均优于keep及gradient choice；且相对两者，完整B和固定支持MSE均不增加超过同公式τ、每图MSE增幅≤.1%。同时报告FP/FN权衡，防止仅靠拉索扩张获得表面收益。若同选择，记“该库没有区分选择器”；若动作收益仅在A、RGB保护失败或B cable未改善，记该固定实验未支持。没有动作可行亦是有效结论，不能救援幅度/区域/轮数。这里不要求证明全局几何瓶颈，也不设VAL采用门。

## 成本与可实现边界

拟合本体为 `5态×26pass×8view=1040 scene`；全部零步/终点B80、普通梯度A8、keep重复A/B16，合计约**1144 scene**，实现上限1200（最多约3600 raster）。q梯度最多 `5×24×8=960` 次VJP，RGB为baseline及20个候选梯度pass，最多840次反向；head/teacher为0。候选筛选复用原A渲染信息，不能另起全场大量权重缓存。每态串行，完整场保留，只有2048行优化器状态；FW只缓存8个A的target通道，不保存全部相机梯度。

已完成EM/FW为6734 scene/20202 raster、1047.97秒，但含FW CPU搜索与I/O；直接按总量折算1144 scene约178秒，仅作参考。此处新增RGB反向、少量拓扑JIT而减少全场q更新/缓存，估计**250–420秒**，不作速度保证；建议内部600/外部660秒，预留B和恢复，不足即完整记录未完成，不减少样本或延长。真正执行前只需小helper合同与来源/调用计数接线检查，不再要求先跑一次证明几何瓶颈的实验。

正结果最多支持这套候选库、有限属性拟合和共享RGB保护下的动作排序；它不是新split/EM公式、三维真值证明或已超过E。若值得后续完整训练，仍需同预算head适配和同一官方RGB/语义终评；本轮不自动开展这些工作。
