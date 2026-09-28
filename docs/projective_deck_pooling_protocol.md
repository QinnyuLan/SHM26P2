# 单桥面轴投影 pooling：四臂匹配训练合同

2026-09-27。**本文件是可冻结的执行合同；是否已冻结、启动或完成，以对应 plan 与 receipt 为准。** 前一固定双轴诊断的塔轴不稳定、删组覆盖不足，因此该双轴实现已经停止。桥面轴通过了该诊断；本轮是在已知该结果后提出的**自适应单轴探索**，不是原双轴门的通过、盲测或创新证明。工程 E 保持当前版本，任何臂均不能自动替换它。

## 固定来源与四臂

共同完整起点为 `runs/h3_moments/02_cross/last.pt`，SHA `22bc8a2ddb260f93cb01b17857c97b2bb0873038efdb9318545cb2bdbb045226`；legacy manifest SHA `551546979a583d46e840bd485559721f361bc28fa4dca60826374ceb74b315fa`。350 TRAIN 原相机中固定259有标签视图用于训练；只用其prepared mask/valid监督，不用VAL、teacher伪标签或真实RGB作head输入。

固定桥面轴来自 `/mnt/data/SHM2026/runs/projective_structure_axes_v1/execution_receipt.json` 及其独立CPU审计：候选3382、可用线点2309、30块，轴 `[0.39810321798792697,-0.012510913979839057,0.917255310619157]`，无向符号不重要。原计划/结果/独立审计及输入init点都需在新plan绑定，不重估、不用本轮loss或VAL重新选轴。原点与语义票条件于TRAIN SfM及二维标签；不称独立三维真值或OOF。

| arm | panorama的横向strip分支 | 其余部分 |
|---|---|---|
| `original` | 原始整行均值与原投影模块，直接原forward | 不变 |
| `per_camera` | 当前相机几何投影场dyad的最佳常方向，固定有限线采样 | 不变 |
| `projective` | 固定桥面轴在当前相机的dense投影线场 | 不变 |
| `wrong` | `R_wrong=Rz(90°) @ R`产生的线场；仅pooling方向使用错误R | 不变 |

scene渲染的K/R/位姿始终真实。竖向strip、所有pyramid分支、merge、FPN、边界细节分支、H3 cross moments、残差上限6均保持。wrong采用确定camera-frame roll，适用于任意新相机，不用camera ID表；不沿用前次诊断的16视图循环映射。

新三臂均在1/16特征格上用17个对称距离 `linspace(-256,256,17)`（原画布像素），相同bilinear kernel和边界归一。feature中心为`16j+.5`；先转feature index `(p-.5)/16`，再用 `align_corners=True` 对应的归一网格。三种新方向每query、每offset的有效mask取共同交集，各臂按同一mask的sum/count取样。空交集位置的pool输入以原row mean填充，避免零占位改变GroupNorm；投影模块之后，该位置严格回退原分支输出。GroupNorm仍在完整dense图上计算，不能把非空位置解释为相互独立。

设 `B_old=strips[0](x.mean(width)).expand`、`B_sampled=strips[0](sampled_mean)`，二者**共享同一个strip卷积/GroupNorm/SiLU参数**。在完整branch输出之后做 `(1-alpha)*B_old+alpha*B_sampled`，不是先插值均值再过非线性。训练迭代为0…1999，`alpha=min(iteration/500,1)`；iteration0直接走旧forward且逐位相同，iteration500起完全新分支。original永远走旧forward。所有其他branch维持原值；新几何无Parameter、无axis梯度、不增加可训练参数。

## 已核训练接口及最小调用

不要直接复制旧 `configs/generated_h3_moments/02_cross.yaml` 的暖启开关：旧配置从coupled场给off头添加cross-moment路径，所以 `warmstart_add_depth_moments:true` 有效；本轮起点已经是cross头，必须 `warmstart_add_depth_moments:false`、`warmstart_reset_refiner:false`，并保持完整 `refiner_config={type:multiscale,channels:64,residual_bound:6,context:pyramid_strip,depth_moments:cross}`。当前train在同架构仍要求add-moments时会报错；配置变化或reset则会构造替代头，不能当作保留旧头。

最薄独立runner可复用冻结包的 `load_scene`、`apply_parameter_scope`、view sampler和 `semantic_loss`，不修改 `src/bridge_rgs/train.py`，也不通过会保存完整scene和自动终评的通用pipeline。顺序为：

1. 每臂独立 `load_scene(base)`，核全模型初始tensor/相机/profile/SHA一致；冻结所有参数后，仅令`scene.refiner`可训练。`attach_projective_context(scene, {'mode':arm})`只替换运行行为，不更名或复制权重；scene/refiner state_dict键与初始值需完全一致。
2. 用保存的350相机按manifest TRAIN name排序映射，核original poses，再建立259标签视图population。plan用fresh NumPy `default_rng(42)`、shuffle且耗完再排列，预先保存完整2000 view indices，各臂复用同一顺序；Python/Torch种子42。每臂实际2000个names日志须与plan重建结果相等；7轮完整259+第8轮187，共2000。不要求未保存的sampler终态，也不声称delta可恢复训练。`set_step(iteration)`在forward前调用，不消耗sampler RNG。
3. 每步fullnative1320×989、SH3、无progressive/crop/flip，调用 `scene.render(K,pose,W,H,degree=3,semantics=True,refine=True,geometry_grad=False,refinement_grad_to_field=False,absgrad=False)`。保留原RGB/depth/alpha/features/p3d/cross-moment输入，无真实照片替换。新几何网格只来自渲染调用的相机及固定轴。
4. fresh `Adam(scene.refiner.parameters(),lr=3e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)`，固定LR，无scheduler、旧Adam恢复或额外梯度裁剪。这个eps是现`heads`默认值，不是Gaussian单独Adam的1e−15。只做目标反传与这一个optimizer.step，2000步唯一终点。
5. 不调用 `train()` 的camera优化、grow/prune/reset、伪标签、fusion、opacity/depth辅助loss、RGB优化或中途evaluate。最终保存独立refiner delta；卸载adapter，核非head/相机状态逐位未变。日志可每100步，但完整采样序列必须落盘。

有效可优化目标与现refiner-only head部分一致：

`L = weighted_CE(final_p, mask, valid) + .2 * Lovasz_present(final_p, mask, valid) + .001 * mean(residual**2)`。

weighted CE先`clamp_min(1e-7).log()`，NLL逐像素乘类别weight，再以有效且label≠255像素数为分母，**不是按类别weight总和归一**。Lovasz只平均当前图存在的类别；residual²对全部H×W×5平均，不裁valid。`semantic_weight=1`。当前代码没有独立boundary-loss；所谓边界分支是架构。train原本还加`.5*raw_CE`和RGB loss，但在本scope下均对可训练头为常数，本runner省略这两项并明确不再把日志总值称原train total。

复用既有259TRAIN类权重，power=.25，依序为背景/桥面/索/塔/基础：

`[0.45604488253593445,0.9227690100669861,0.8530434370040894,1.1935913562774658,1.5745513439178467]`。

来源是已完成 `coupled_semantic_gradient_diagnostic_v2/plan.json` 中同manifest的固定权重（weights SHA `293335de8920192faabd54225db289c47bc9bfd2eb6b6965ad7792740417b3b7`）。原函数按259 prepared mask中ID<5计频、频率floor.002、逆频率^.25、除五类均值后cap3；这一步不使用valid筛频，实际CE仍使用valid。新plan绑定数值及来源；runner可按相同原函数从259TRAIN mask重算并核对上述数值，不重新调整权重。

## 独立delta与不变量

不能复用只支持SH/background的appearance delta，也不能把pooling metadata丢进普通旧checkpoint。新格式明确为 `projective_deck_refiner_delta_v1`，至少含：`refiner_state`（完整refiner tensor）、`base_checkpoint_sha256`、`mode`、`step=2000`、原refiner architecture/profile、轴/采样/alpha规则、源/plan/manifest绑定。不得包含冒充完整scene的`model`键，不保存或替换生产场。加载必须先核22bc完整base，再strict加载refiner_state，attach对应mode、set_step(2000)；缺mode/轴/配置或错误base一律拒绝。初版是推理/显式新stage暖启格式，不允许普通strict resume；技术续跑若将来需要，必须独立保存并验证Adam+sampler+iteration+adapter合同，不能借base旧optimizer。

每臂初始全部原tensor逐位相同、freshAdam空状态；终点所有非`refiner.*` tensor（含geometry、SH/background、sem_features、classifier、priors）及training cameras逐位相同，只有head允许改变。记录所有参数finite及实际head更新。相同相机下RGB/depth/alpha/raw p3d与base不变，新增方向sampling不改变shared Gaussian visibility。成本按各臂完整2000步与最终推理计，不用同参数数目冒充同算力。

## 唯一终点评价与缓存边界

四臂均为原图共同50相机/41标注协议，固定旧H+ `00f5b84a…524bff` 与每臂新H3概率在legacy canvas上0.5/0.5混合，之后唯一畸变soft warp再argmax。禁止hard-mask融合、挑best、中途VAL或搜索权重/TTA。scene-only final与raw语义可并列报告，但不能替代完整系统采用门。

本轮可复用已完成multifield A教师soft缓存，前提是逐文件SHA、五类/profile/grid、教师来源均核实，且**每臂每图新渲染的H3 canvas量化RGB及原图PNG均与缓存来源逐字节一致**。四臂各50次新scene调用，共200；不产生新的teacher调用。E的RGB直接引用已验证newmean PNG及对应每图RGB指标，不重新渲染1M/MCMC，不读取真实GT RGB重评分。因此明确报告“新语义渲染+SHA验证的教师/E-RGB复用”，不宣称本轮重新运行完整三场系统或用本次缓存耗时代表部署FPS。已有E继续保留。

所有200新joint、200新scene-final及200新raw mask（共600）和对应soft预测先完成并写prediction receipt，之后才读取41份annotation，统一栅格评分；不能在每个arm结束后先看其GT再继续下一臂。旧E metrics/receipt/plan和50人口均绑定，复用RGB每图记录只能来自字节完全对应的E输出。source GT只作固定终评分，未参与pooling输入、alpha或轴选择。

固定5000次、seed20260926的view-paired bootstrap，语义每次按抽到的41视图CM pooled后计算五类mIoU；这不是单桥多seed或盲测置信区间。projective必须同时：

- 对`original`：all5增益≥0.15个百分点，paired95%下界>0。
- 对`per_camera`和`wrong`：all5各增益≥0.10个百分点，paired95%下界分别>0。
- 对旧E：all5增益≥0.20个百分点，paired95%下界>0。
- 对上述四个参照各自：索IoU点估计不下降超过0.10个百分点；其余每类分别不下降超过0.20个百分点。

RGB完全沿用E，不能再把旧RGB进步记为本次方向模块收益。任一门失败，停止这个固定单deck实现，不扫邻域、采样距离、alpha日程、LR、步数、seed或融合权重补救；不把其它臂的偶然最高值直接选为新生产模型。

## 学术解释

[Bundle Pooling](https://openaccess.thecvf.com/content_CVPR_2020/html/Zeng_Bundle_Pooling_for_Polygonal_Architecture_Segmentation_Problem_CVPR_2020_paper.html)已有消失线束pooling，[VPSeg](https://arxiv.org/html/2401.15261v2)已有VP引导语义信息；这里是单TRAIN轴来源与Gaussian渲染读出的组合探索。即便通过上述工程门，也只支持这一固定桥梁/协议的剩余增益，不证明首次投影pooling、物理三维语义改善或跨桥泛化。有限采样和原整行strip不同，原始续训与同相机强常方向均必须保留；wrong-pose控制共同有效支持也不能消除全部表示或优化混杂。
