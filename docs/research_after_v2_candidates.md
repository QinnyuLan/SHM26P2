# v2 之后的研究候选：先检验配对渲染误差是否值得专门处理

2026-09-26；状态：**仅完成文献、代码和实验设计审查，未运行本文件中的诊断或训练。** 本文件不改变既有模型、缓存、协议或其他实验计划。建议只保留下面一个有条件候选，不为凑够两个方向再增加模块。

候选问题是：**在固定三维证据和固定精修器下，同相机的真实—渲染 RGB 差异是否包含一种空间对应特有、且可以通过训练消除的有害敏感性？** 如果成立，才值得做严格匹配的配对训练；如果普通真实/渲染混合或一般扰动一致性已经解释收益，则归为已有域适配工程，不宣称新机制。这里的目标是改善最终相机渲染分割，不要求 raw field 的分数同步提高。

## 1. 不能忽略的现有证据

| 已有结果 | 对下一步的限制 |
|---|---|
| [H2](h2_multiscale.md)：final KL 相对 GT 为 −0.00204 pp；projection−fixed 为 −0.00453 pp，两者区间跨零 | 不能再以换一个可靠性名称、放宽门槛或增加多视角聚合为默认方向。 |
| [16 TRAIN 教师审计](teacher_signal_audit.md)：教师纠错 11,844 像素，反向错误 95,639；低于 0.8 的 2.09% 像素贡献 88.12% KL | 不能因低置信度遗漏纠错就直接取消阈值。缓存教师不是这些拟合样本上更可靠的像素监督。 |
| [H3](h3_depth_moments.md)：cross−variance −0.0384 pp，95% 区间 [−0.2049,+0.1796]；约增加 15% 渲染延迟 | 不再加入额外矩、容量或融合头，然后把普通优化波动解释为几何信息收益。 |
| [固定教师融合](teacher_renderer_transfer.md)：cross+teacher 95.0231%，但相对 variance+teacher 的区间跨零 | 只证明固定高成本组合在该开发集上存在互补性，没有证明 H3 或可压缩性。 |
| [翻转 TTA](refiner_flip_tta.md)：flip 模型增加 0.0704 pp，推理约 35.93→66.63 ms | 是小幅工程收益，不足以支持新结构主张。 |

这些结果涉及不同冻结场，不能拼接 RGB 和语义成绩。[旧研究备忘](next_research_hypotheses.md) 的表面对应/区域标签假设不在本轮候选内；不构建语义 support surface，不把标注与物理表面的错配当成核心创新，也不新增相机 ID、逐视图模板或 VAL 图像输入。

已有 teacher 的 real/render 域差距及 2k 域适配结果不是这里的待发现结论。本候选只检验**当前学生/refiner 对固定 E 下 RGB 干预的响应**；不再跑同一 DINO gap 后宣布诊断通过。旧 16 TRAIN 学生错误率仅 0.1201%，所以本候选的最小审计以连续 margin/NLL 与定向响应为主，精度翻类是完整报告的次要结果，不能依赖 TRAIN 尚有大量可改正错误的假设。

## 2. 与一手工作重合到什么程度

本次核查以论文、作者项目及官方实现为依据；下表是方法边界，不表示完整覆盖所有文献或已经证明空白。

| 一手工作与官方实现 | 已经覆盖的内容；本候选必须额外证明什么 |
|---|---|
| [Tangent Prop，NIPS 1991](https://proceedings.neurips.cc/paper/1991/hash/65658fde58ab3c2b6e5132a39fae7cb9-Abstract.html)；[VAT 论文](https://arxiv.org/abs/1704.03976)、[作者代码 vat.py](https://github.com/takerum/vat_tf/blob/master/vat.py) | 指定变换方向的导数约束、局部分布平滑均已有基础。将方向换成 RGB 残差，不能声称发明 Jacobian 正则或不变性学习。 |
| [AugMix，ICLR 2020](https://arxiv.org/abs/1912.02781)、[官方训练实现](https://github.com/google-research/augmix/blob/master/imagenet.py) | 官方实现对原图监督，并对原图及两次增强的概率施加 JS 一致性。配对图像再加 JS 本身不是新损失。 |
| [SHADE，ECCV 2022](https://arxiv.org/abs/2204.02548)、[官方仓库](https://github.com/HeliosZhao/SHADE)、[训练入口](https://github.com/HeliosZhao/SHADE/blob/master/train.py) | 风格变化下的一致性以及利用已有真实世界表征限制过拟合均已存在。必须区分普通域混合收益和同相机残差对应的独立作用。 |
| [RobustNet，CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Choi_RobustNet_Improving_Domain_Generalization_in_Urban-Scene_Segmentation_via_Instance_Selective_CVPR_2021_paper.pdf)、[官方代码](https://github.com/sunghac/RobustNet)、[选择性 whitening 实现](https://github.com/sunghac/RobustNet/blob/main/network/instance_whitening.py) | 根据原图/光度变换的统计变化识别并抑制风格敏感成分已有先例。不要把本候选换名为“渲染不确定性解耦”。 |
| [RT-GS2，BMVC 2024，§4.3](https://arxiv.org/html/2405.18033v2)、[官方仓库](https://github.com/mbjurca/RT_GS2) | 已融合视图相关信息和可渲染的三维特征，并研究跨场景分割。RGB+三维证据+二维网络不是本项目可独占的架构创新。 |
| [Feat2GS，CVPR 2025](https://fanegg.github.io/Feat2GS/)、[作者链接的代码](https://github.com/fanegg/Feat2GS) | 已用轻量读出探测基础模型特征中的几何/纹理信息。换用 DINOv3 或增加 feature readout 不能构成这里的机制贡献。 |
| [RogSplat，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Kong_RogSplat_Robust_Gaussian_Splatting_via_Generative_Priors_ICCV_2025_paper.pdf) | 已比较训练图与渲染图的 DINO/DIFT 特征以定位不一致，再用生成先验修复。它的目标是鲁棒重建，不是本候选的固定分割器干预，但足以说明“比较 real/render 差异”不新。未把另一个同名人体 RoGSplat 仓库误认成其官方实现。 |

本候选的研究机会只能是更窄的经验结论：在冻结共享场、固定模型容量和完整匹配控制下，识别哪种**实际渲染残差的空间对应**导致最终分割不稳定，并证明针对这一对应的训练优于普通混合和通用扰动。即使在本桥场景得到正结果，也只是场景内证据；论文级主张仍需要额外场景/独立划分，不能由一次视角 bootstrap 代替。

## 3. 唯一保留的候选与最小干预

对 TRAIN 相机 v，记冻结场输出为 `E_v = (F_v, depth_v, alpha_v, p3d_v)`、渲染 RGB 为 `R_v`、真实 RGB 为 `I_v`。固定精修器输出为：

```
p_v(X) = softmax(log(p3d_v) + h_theta(E_v, X))
d_v = I_v - R_v
X_v(t) = (1-t) R_v + t I_v
```

**机制假设**：在相同 E 下，实际配对路径 `X(t)` 对错误的影响不同于幅度相近但空间对应破坏的残差；而且这种敏感性可以在只更新原精修器的条件下被抑制，改善推理时的 `p_v(R_v)`。前半句可以用固定模型干预排除“增加容量”解释，后半句必须由训练对照检验，不能靠一次真实图替换直接宣布成立。

真实 RGB 在这里是 TRAIN 干预/训练输入，**不是可部署预测**。最终仍仅使用相机→同一冻结场→原精修器；不把 `p_v(I_v)` 计入正式分数。`d_v` 也不能直接称为纯外观噪声：它可能含重建细节缺失、曝光、遮挡和残余配准误差。本诊断正是要避免预先把这些因素混成一个有益正则方向。

### 3.1 固定 16 TRAIN、无新模型的诊断

在 v2 RGB/语义固定终点及共同 official-grid 评价完成后，再决定是否登记执行。诊断只用匹配 v2 checkpoint、v2 manifest、v2 像素协议；既有 `runs/strong_rgb/train_render` 是旧 renderer/legacy 数据，不能直接配给 v2。v2 的 16 张 render 可当场生成、逐张释放，不需要导出 350 图或新的概率缓存。

1. 仍按有标注 TRAIN 名称排序取 `floor(i*(N-1)/15)`，N 应为 259。预期名单为 002、021、041、059、079、100、118、137、156、176、200、220、241、259、278、300；实际运行必须从 v2 manifest 重新核验集合、尺寸和 profile，不能凭名称假定字节相同。计划先锁定 checkpoint/manifest/源码/16 RGB/16 GT/valid 的 SHA 与名单。
2. 每相机只 render 一次 `refine=False`。原生完整网格分别计算 `p(R)`、`p(X(.25))`、`p(X(.5))`、`p(I)`。这是固定诊断路径，不从中挑最好的 t 部署。所有 F/depth/alpha/p3d tensor 不变，替换 RGB 后必须同步重算 RGB 梯度派生通道；当前 `evidence_tensor` 的 RGB detach 不妨碍这种有限差分前向。不得只替 RGB 三通道却保留旧梯度图。
3. 两个预先固定的错位残差为 `roll(d, (floor(H/3), floor(W/3)))` 及反向 roll；比较 `clip(R+.25*roll(d))` 与 `clip(R+.5*roll(d))`。未 clip 前 roll 精确保留每通道残差的范数、直方图和傅里叶幅值，但打乱像素对应。分别记录 clip 后范数差、饱和比例和 wrap 接缝区域；这些量不再精确匹配，不能隐去。不得依据 GT/error 选择 roll 大小或残差位置。
4. 加一项相机错配控制：目标 v 固定取这16张名单中下一张相机的 `d_next`（最后一张循环到第一张），每通道做 RMS 比例缩放到 `d_v` 的 RMS，再计算 `clip(R_v+.5*d_next_scaled)`；零 RMS 通道写零并记录。它匹配幅度、不匹配频谱，和 roll 是互补控制，不可写成同时控制所有因素。只使用同 v2 renderer 的残差。按环形相邻相机流处理，最多暂存两相机证据及首张残差，不落全量缓存。
5. 加一个低成本色调控制：在固定像素网格上仅由 `(R,I)` 最小二乘拟合每 RGB 通道的全局 affine 变换，三个通道各两个系数，不读 GT；将修正后的图输入同一头。若其解释了几乎全部效果，优先归为普通色调/曝光域差异。该控制不允许 dense flow、不校正标签、不增加图像恢复网络。
6. 每张图的全部干预预测完成后才解码 GT/valid。按全部像素、五类、固定 GT 2px 边界统计：双向纠错/引错数、总 CM、每视图 CM、NLL 和正确类相对最强错误类的 margin 变化。主连续量是每视图内各出现 GT 类等权的 margin 增量，再对16相机等权汇总；同时记录单位实际 RGB 扰动能量的输出 JS、中心化 log-probability 变化以及32×32固定块的扰动 RMS/clip 比例，避免把相同全图能量误写成相同局部难度。GT 只评分/分组，不选视角、定义扰动或产生 gate。报告原模型正确/错误、边界/非边界分层，以免背景高置信度或几个高梯度像素主导结论。

十种干预需要 16 次三维 render、16×10 次原生头前向（原图路径四次、两个 roll×两个 t、相机错配一次、affine 一次）。实现另增加每相机一次原 `scene.render(refine=True)` 只读复现检查，因此实际合计32次 render、176次头前向；这16次校验不构成额外候选干预。无优化、无 DINO 推理，只保存汇总 JSON，不保存全尺寸多通道概率。

实现约定另见 [诊断运行说明](paired_render_sensitivity_audit.md)：`R` 是原始浮点渲染 RGB，可能超出[0,1]；baseline和两条配对半路径不clip，roll/错配/affine按[0,1]clip，统计所有实际幅度差异。不能用PNG或clamped RGB替换baseline再称为原模型复现。

### 3.2 诊断能够和不能够识别什么

固定同一 θ/E 的前向差异排除了新参数、训练更久、teacher 标签和集成的解释；实际残差与 roll 对照检验的是**空间对应是否有特殊作用**，affine 对照检验是否只是全局色调。但它们不能证明真实图是纯 label-preserving nuisance，不能证明图像中补回的细节能够从 E 恢复，也不能证明未见相机泛化。错误与 RGB 残差大小的相关性本身不够。

必须披露两个非对称性：既有头只在 render 域优化，真实图可能是其分布外输入；反之，完整真实图可以补回 render 根本不存在的语义线索。因此仅 `p(I)` 更好、只有 t=1 更好、或头对任意图像变化都敏感，都不支持“可消除的渲染误差敏感性”。**即使半路径有效，也仍可能只是逐渐补回真实细节，不能单独排除证据替换解释。** roll 和相机错配还可能产生比实际误差更不自然的局部纹理，即使胜过它们也不能排除增强合理性的解释。

以下是**未来取数前固定的资源门槛**，不是来自文献的理论阈值，也不是已满足的结果：

- 主诊断固定 t=.5；上述每相机类别等权 margin 增量相对 `p(R)` 的配对 bootstrap 95% 下界大于零，NLL 方向不恶化；t=.25 同方向，不能只靠完整真实图终点。TRAIN all5/双向翻类完整报告，但不要求饱和 TRAIN 精度再跳一个大幅度。
- t=.5 的实际配对相对两个 roll 及相机错配控制的 margin 增量差分别具有正的配对下界，且不只是 affine 已能解释的变化；同时输出响应/局部幅度分布必须可解释。各类、边界和 clip 后实际幅度完整列出；若幅度/接缝/离流形难度失配使结论不可识别，停止，不追加一串更有利的扰动。
- 由于 TRAIN 学生已经高度拟合，这些门槛可能没有统计功效。失败应写作“本有限审计不足以支持继续投入”，不能升级为对所有 renderer/场景的否定；同样不能转去看 VAL 的 realRGB oracle 来挽救假设。

## 4. 只有诊断通过时才值得登记的训练对照

训练不是本次任务的一部分。若后续获准，四组共同使用 v2 固定终点场及原 refiner warmstart，259 GT-only、seed42 独立 view shuffle、原生完整图、相同 3000 optimizer steps、Adam 和 refiner LR=3e-4。只更新原 557,413 参数的 multiscale64/off 精修器，冻结几何、SH/background、语义 field/classifier、相机；原始 raw 监督、teacher、fusion、TTA、crop、flip、schedule 及其他额外 loss 全部关闭。若最终 v2 主头配置不等于此架构，则先修订和锁定完整计划，不在执行中暗中转换。

共同定义 `L_gt=.5*L(p(R),Y)+.5*L(p(I),Y)`，其中 L 完全沿用预注册的既有语义 GT 损失；`J(a,b)=.5 KL(a||m)+.5 KL(b||m)`，`m=(a+b)/2`，按 valid 已标注像素平均，概率 FP32，epsilon=1e-7，温度1。JS 系数固定 .1，不搜权重。两侧均反传，GT 在每一训练步维持监督地位。

| 组 | 有效目标 | 作用 |
|---|---|---|
| 00 继续原 GT | `L(p(R),Y)` | 区分额外训练的影响。 |
| 01 普通成对域混合 | `L_gt` | 经典真实/渲染混合适配强控制。 |
| 02 实际配对一致性 | `L_gt + .1 J(p(R),p(X(.5)))` | 检验实际渲染误差路径的约束。 |
| 03 错位残差一致性 | `L_gt + .1 J(p(R),p(clip(R+.5*roll(d))))` | 保留一般扰动正则，破坏同相机空间对应；按 step 奇偶固定使用两种 roll。 |

四组每步都执行相同一次场 render、三次同头前向；不生效项以明确的零权重保持可审计的计算流程。00/01 的第三输入使用实际半路径；00 的真实 RGB 支路不参与有效损失。记录实际前反向次数、有效 GT loss 权重、clip 后扰动统计、优化器状态和计时，不能只声称“都 3000 步所以成本相同”。同一批次顺序、相同初始化/源码及非 refiner tensor 的逐位不变均需验证。

主要机制比较是 **02−03**，同时必须满足 **02−01**。若 01 已达到同等效果，就只有标准 DA 收益；若 03 同等或更好，就只有一般扰动正则收益。02 比 00 好不能单独支持配对机制。四组均使用单个原头的 `p(R)` 推理，不用真实图、teacher、额外头或预测平均，所以不把集成容量收益带入。

主要终点是固定 last 的共同 official-grid all5 mIoU，全 50 相机/41 标签一次评价；同时保留 native 诊断、各类 IoU、索边界、raw CM、RGB 三指标和成本。所有组/配对都报告，不按中途 VAL 选点。预先采用实际投入门槛：02 相对 01、03 **以及未经本轮继续训练的原始 v2 semantic warmstart 终点**均至少 +0.20 pp，配对 95% 下界均大于零，缆索/基础点估计不下降超过 .20 pp；这些是本项目的继续投入规则，不是学术显著性的通用定义。只恢复00/01/03继续训练造成的退化不能称为突破，避免重演H3 zero对照退化后的误解。一次固定种子通过后仍不能宣称稳定创新；需另批资源做多种子及其他场景验证。

**停止条件**：诊断未过则不训练；训练未同时超过 01 和 03 则停止定向机制，不调 JS 权重、t、roll 距离或重跑到正结果。RGB/原生 raw CM 不一致、profile/SHA 混用、存在非 refiner 更新属于实现失败，必须先修复并清楚区分于负结果。

## 5. 预算和数据边界

- 诊断预设300秒协作式窗口（单次调用不能抢占，详见运行合同）；无需重新下载/运行H+，没有新350图缓存。现已按固定计划一次完成，预测/逐视图CPU评分99.298秒、CUDA峰值2.192GiB，之后仅CPU汇总；GPU已释放。
- 若进入四臂训练，总共 12,000 optimizer steps、12,000 场 render、36,000 头前向。已有单头 3k 的并发实测约四分钟不等于本方案耗时；按三次头前向保守预留总 GPU 时间 90 分钟（含评测），先用两步功能预检与固定短计时确认。超预算即在正式启动前报告，不暗中降分辨率/改终点。torch threads=8，uv 管理环境。
- CPU 实际构造现有 multiscale64/off 头得到 FP32 state 2,229,652 bytes（约2.13 MiB）；参数及 Adam 两个 FP32 moment 的主体约6.38 MiB/臂。为避免复制整个冻结场，独立 runner 可保存“严格 base checkpoint SHA + 头 state + optimizer/RNG + 完整配置/协议”的研究增量产物，需专门验证严格装载和与完整 scene 前向一致后使用。它不是当前训练 checkpoint 格式，尚未实现，不能冒充已支持 resume。
- 按每臂四份上述量级的 model/optimizer 临时与固定产物，加 4×75 MiB 最终 RGB/mask/metrics 及80 MiB源/日志，粗预算小于0.5 GiB；空间必须在正式执行前按实际 v2 输出、atomic 临时文件和其他排队任务重新核算，保留至少256 MiB余量，不删除旧结果。RGB可逐图比较共同base后只保留可审计散列，但若执行协议要求每臂PNG，则按上述完整输出预算。
- 所有原图来自已允许的 TRAIN；不存在 real VAL 输入。既有 TRAIN 渲染和场均拟合过这些视角，诊断偏乐观/饱和，且不是从零留出场景。不要把其中16张称为“未见标签验证集”：现有 warmstart/field 已看过它们。

## 6. 为什么不保留第二个候选

另一个看似自然的提案是从同一 teacher 的 real/render 输出取 logit 差，试图消去共享教师偏差，再蒸馏其变化。但差分不会自动消掉与输入相关的教师错误；现有 real-cache teacher 与 render-adapt teacher 又不是同一个 checkpoint，直接相减还会混入模型变化。新的 v2 teacher 目前只有成本计划，不能跨 profile 复用旧头来制造“免费”证据。

更关键的是，[Cross-corruption distillation，ETRI Journal 2026](https://onlinelibrary.wiley.com/doi/10.4218/etrij.2026-0159) 已明确对齐 teacher/student 在原始与受扰输入之间的 logit 差异。其任务是文本指令，不验证桥梁分割，但已经覆盖宽泛的“蒸馏条件响应而非绝对输出”思想；本次没有核实到可归属的官方实现，因此不据此声称其完整工程可直接复用。结合本项目已有 KD 负证据，暂不把该方向列为第二项训练候选。

如果唯一候选也不能通过门槛，应保留已验证的工程成绩和负结果，完成 v2/共同网格比较，而不是给新 gate、attention 或辅助网络起名。当前没有证据保证“再加一个模块”就能同时满足明显提升与学术创新。


## 7. 固定诊断的实际结论：不进入四臂训练

2026-09-26，v2语义8000步与共同official评价completed后，按固定16TRAIN名单和十个干预完成一次审计；[完整结果、SHA和数值限制](paired_render_sensitivity_audit.md)。原始raw float baseline概率/证据逐位复现，模型和相机未变，全部预测先于GT解码。未实现或启动本文件第4节的训练。

实际半路径的类别/相机等权margin相对raw增加+0.00356447，95% CI [+0.00183090,+0.00522646]，但相对next-camera幅度匹配控制为−0.00015963，CI [−0.00251612,+0.00207942]。半路径NLL相对raw增加+0.00001578（方向变差，CI跨零），TRAIN五类mIoU由98.739774%变为98.738433%；纠错940、引错1138，净多198个错误。它胜过roll的margin响应不等于通过相机错配控制，roll局部clip最高超过82%也使难度匹配存在混淆。

因此**预先登记的继续投入门槛失败，停止该候选的四臂训练**。不调整比例、roll距离或一致性权重，不按VAL真实图救援，不用拉索/基础某个分层的局部纠错替代总体门槛。这仅是该固定模型、饱和TRAIN和有限16相机上的负证据，不是对所有域适配或其他场景的否定；当前没有可据此声称成立的创新机制。
