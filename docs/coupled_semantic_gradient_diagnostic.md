# Coupled终点的TRAIN梯度干扰诊断合同（CPU设计）

2026-09-27。仅定义一次有限诊断；未实现worker、未执行GPU或批准新训练。对象固定为 `runs/support_split_semantic_coupled/last.pt`，真正coupled的8k终点，而非随后head-only的H3阶段。来源／权重采用[data的CPU核验](/mnt/data/SHM2026/runs/support_coupled_source_cpu_audit_v1/report.json)，SHA `8e9aea80c1bf77bc1f77e677268bfa682ed746d7c888dd81f012df770f8396ce`。17个实际源码模块、原legacy数据和350相机不升级。

## 唯一对象与三个梯度

按名称排序完整259个labeled TRAIN各一次、原生分辨率；仅002／205各增加一次独立forward作为重复误差，重复不加入主均值。只允许16维Gaussian特征F有梯度；decoder、refiner、geometry、opacity、RGB均冻结，没有moments或教师。源训练曾同时更新decoder和refiner；现在冻结它们的终点条件诊断不能重构过去8k的训练历史。

**冻结前数值约定：**本轮独立梯度校准显式设置 `torch.backends.cudnn.allow_tf32=False`、`torch.backends.cuda.matmul.allow_tf32=False`、`torch.set_float32_matmul_precision('highest')`，保持 `cudnn.benchmark=False`，禁用AMP／autocast，不引入GradScaler。参数、renderer、head及原损失公式仍为FP32，只有已规定的统计／差分聚合采用FP64。root在CUDA未初始化时实查本机当前默认是cudnn TF32=True、matmul TF32=False、precision=highest、benchmark=False；卷积输入的低精度量化可能影响小步长FD，因此在尚无GPU结果或锁定plan时统一采用上述设置，而不是结果后改精度救门。

worker必须记录设置前／实际执行中的这些flags及autocast状态，退出时恢复进程原flags。该选择**不声称逐位复现旧8k训练的默认TF32数值执行**，也不单独证明旧梯度有错。下述两分支前向逐位相等只要求在本次统一数值设置内成立；未来若有控制也须采用相同设置。固定h、公式、数值门和渲染预算均不变，不因禁TF32另改阈值或追加尝试；禁TF32不保证整个CUDA渲染逐位确定性。

调用原snapshot的 `semantic_loss`，CE分母是有效且非ignore像素数，不是class weights之和。raw／final的共同权重固定为 `[.4560448825,.9227690101,.8530434370,1.1935913563,1.5745513439]`，不再次根据本次结果调权：

- `gR = ∂F [.5 × weighted CE(p3d)]`。
- `gF = ∂F [weighted CE(final) + .2 Lovasz(final) + .001 mean(residual²)]`。
- `gA`采用与gF完全相同目标，但所有p3d通路停止梯度，只保留rendered feature通路。

用只读refiner prehook捕获原features、RGB、depth、alpha、p3d，立即移除hook；额外一次head forward将p3d输入（包括派生熵）和最终log(p3d)基底都detach，features保持连通。两路residual／final概率／损失须逐位相同。不能使用 `refinement_grad_to_field=False`，因为它还会切断features。使用 `autograd.grad`分别取三个梯度，不更新任何参数。

## 主统计与解释

固定本coupled decoder的 `D=W[1:]-W[0]`、FP64正交投影Prow，Pnull=I−Prow；rank4/null12已由来源报告确认。逐view及等view平均梯度分别报告total／row／null平方范数、夹角、内积的正负贡献质量；不以负内积Gaussian占比作为主结论。

必须同时报告 `mean_v <gRv,gFv>` 与 `<mean gR,mean gF>`，二者不能互相替代；报告 `||gR+gF||`、`||gR+gF||/(||gR||+||gF||)` 和原权重下raw的一阶变化 `−<gR,gR+gF>`，区分减少raw下降量、预测raw上升和接近多目标平衡。投影实现检查：`||gR Pnull||/||gR||≤1e−5`，`||(gF−gA)Pnull||/(||gF||+||gA||)≤1e−5`；零梯度单列，不用除零或改方向补救。

负内积在正常加权多目标驻点也可能成立。尤其总梯度接近重复误差时，不能称病态干扰。非零null梯度的存在也不自动证明有用信息或收益；只有经过数值校准的方向才有局部可测意义。逐坐标Adam的预条件和历史moment不保留此正交分解，故本轮不模拟或应用Adam，更不承诺null梯度更新后raw不变。

## 固定有限差分与停止规则

只在002／205校准。方向是各自基线的 `dR=−gR/absmax(gR)`、`dN=−(gA Pnull)/absmax(gA Pnull)`，h固定`.001/.0005/.00025`。每个FP32端点F±hd独立从同一原F构造；不可测或零方向不替换、不增幅。**主非零校准只有三项：raw沿dR、真实完整final沿dN、feature-only final沿dN**，共两view×三h×三项＝18项。第三路的端点函数必须把**head p3d和log基底固定为基线值**，不能每端点重赋值后再detach，导致FD检验了另一个函数。

记录 `A=<g,(Fplus−Fminus)/(2h)>` 和 `B=(Lplus−Lminus)/(2h)`。原损失前向保持FP32／原公式，实际参数位移、点积及两个已计算标量之差以FP64聚合。每项主校准先要求abs(A)超过10倍数值floor：重复梯度对应dot差、重复标量loss差/(2h)、对应FP32损失2 ULP/(2h)、FP64点积舍入界中的最大值；未达到仅标inconclusive，不说信号不存在。可测主项要求 `abs(B−A)/abs(A)≤.05`，所有固定h均保留。该floor不是全部GPU误差的严格上界；Lovasz排序、clamp分段和有限步长均可能造成不一致，失败不直接归因CUDA，也不更改旧pose-profile门。

完整final沿dR另作交叉导数描述：可测时报告相对FD误差及5%条件，近零记`cross_inconclusive`，不使上述18项主门自动失败。raw沿dN复用已有端点，是预期零的控制，不要求超过可测下界，也不算非零下降校准。令实际FP32中央方向为d_act，记录其row泄漏及A／B；绝对零一致性容差固定为 `10*floor + ||gR_row||*||d_act_row|| + ||gR_null||*||d_act_null||`，后两项是量化／梯度泄漏的一阶Cauchy界，不是非线性误差严格上界。分别报告abs(A)、abs(B)是否落入该容差；不满足记零控制未解决，不把它错误计为非零主门失败，也不声称该方向保持raw。任一主方向本身为零时不另选方向，保留不可测／无可测null余量结论。

预算为259主scene＋2重复＋24 FD＝**285 scene／570 raster／783次field VJP**；不额外对FD反传，无optimizer、无新checkpoint、无VAL。每scene的原head之外至多一次detach-path head；重复和FD不改变主259平均。原参数每个端点后恢复，整个state／camera／flags／modes在finally逐项恢复。内部210秒、外层240秒，包含载模与记录，不重试；若未完整遍历，或来源／恢复／前向等价／投影合同失败，不作全TRAIN机制判断。18项主非零FD未全部满足时，不宣称完整梯度与可用null方向已共同校准；完整原始统计仍保留。交叉导数近零和预期零控制独立报告，不偷换上述主门。

这轮不设性能采用门，不由负cos、零空间代数或FD通过自动启动训练。最多确定当前固定分类器与头下是否存在可测竞争、总梯度余量和局部null方向；历史伤害、可训练补偿、跨视角泛化及机制收益仍未识别。
