# 固定 H3 可见性、直接类别概率的全量优化诊断

真实算子预检 v2 已自然完成且全部数值门通过，完整状态与数值 flags 恢复；固定有界的259 TRAIN诊断已自然完成，但仅接受1次更新后回溯耗尽，未收敛；见[实际结果](raw_simplex_fullbatch_results.md)。绑定预检 plan `8b511295e5a9e1522b35d70e9789355fa90d6a4b2cf06f5aeb877b3acf5c971c`、execution `236d1d8828e3d833d6cafe63a8bf03f8b7a15e3d9ffde9968be225d585f62303`；v1报告序列化失败原样保留，不作为通过证据。目的仅是辨别原生语义的优化空间，不称学术创新、不自动采用、不读取VAL。新任务使用当前H3/legacy，与旧391a/corner-v2的[一次EM实验](raw_semantic_assignment_results.md)不是同一W，CE或IoU不能直接作跨实验差。

## 目标与证据边界

固定原H3的全部几何/opacity/RGB/features/classifier/head和相机，直接优化每Gaussian的五类概率 `q_i∈Δ_5`。渲染的未归一原始五通道是 `R(q)=Wq+T_bg e_bg`。用固定正仿射噪声目标

`F(q)=mean_259_views mean_valid_pixels −weight_y log((1−δ)R(q)_y+δ/5)`，`δ=5e−7`。

采用既有H3训练类别权重。只有这个未clamp/renormalize的目标对q为凸；生产用`p3d=normalize(clamp(raw,1e−7))`、head与IoU不属于上述凸性结论。梯度是`−Wᵀ[weight_y(1−δ)/p̃_y]`，没有逐点softmax的目标类责任乘子。q不每步映回feature，原refiner的context全部保留，只在终点评估时替换其p3d与log prior。

完整梯度下 `G(q)=Σ_i(g_i·q_i−min_c g_ic)` 为Frank–Wolfe gap，精确算术满足`F(q)−F*≤G(q)`。实际前向/VJP为FP32、跨视图归约为FP64，故报告的是数值近似gap，不是严格区间证书。即使该CE近最优而IoU仍低，也不排除存在CE略高而IoU更高的赋值，不可据此宣布几何表示上限。

## 先检查算子

只用预定002/118两TRAIN相机，不读图像/标签。继承已审查的冻结partition shader，将同一q leaf传给两端、系数固定`[0,0,1,−1]`；前向两端相同使gate消失，反向两端累加给同一q。此处不使用条件投影或其OT替代。

原classifier概率前向需同时核对未归一raw与生产p3d，防止归一掩盖整体缩放。有效simplex的两组固定q之间，用独立前向差分与VJP检验伴随恒等式；另对合成标签的正仿射CE做固定主步长方向差分，其他步长只描述敏感性。不能只验证梯度非零、或用同一错误反向互相比对。精确测试及容差以`preflight_simplex_scene.py`冻结SPEC为准；失败保留、不选择通过的差分步长。

## 固定的唯一优化规则

预检通过后再冻结完整runner、输入及以下预算；不扫描LR、正则或多起点。原classifier FP32概率转FP64并显式归一一次作为master q，记录这次初始化的最大变化和基线差异；simplex投影不加概率地板。master q以FP32提供给shader，记录cast后simplex误差，不隐藏修补。

每次完整259 TRAIN累计梯度后，计算`d=ΠΔ(q−ηg)−q`。初始`η=1/max_i(range_c(g_ic))`；全部切向梯度为零则停止。随后每轮最多先尝试上轮接受步长的2倍，再固定乘.5回溯。用完整259 TRAIN目标作Armijo，系数`1e−4`，最多8个试探；实际FP32位移与FP64梯度的点积必须为负，真实F必须下降。零切向梯度行逐位保持q，不能因为共同梯度偏置而移动。

拟预算为最多20次接受更新、最多80个完整数据pass（预留初始/最终读数），内2700秒/外2760秒。只有完整pass后才能接受；部分pass不更新q。每轮保存真实F、gap、步长、试探数、位移、累计调用及耗时。gap≤1e−5可记为预定数值停止，仍不是严格全局证书；预算结束且gap大只能判未充分求解。回溯耗尽或浮点停滞需明确报告，不能当收敛。

最终单独汇报raw与冻结head的TRAIN CE/CM/IoU，以及gap和实际成本。不训练head、不调用teacher、不读VAL、不改生产checkpoint。该诊断无最终采用门，也不能以TRAIN拟合声称超过E或同学方案。是否有后续泛化实验取决于实际效应和数值充分性，不能从本草案推定必然投入。
