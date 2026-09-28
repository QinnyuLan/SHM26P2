# H3 原生语义与精修的表征读取差异：CPU 审计

2026-09-27。已完成固定 H3 场的 CPU 代数核验；没有运行 GPU、解码图像/标签、优化参数或保存新模型。**raw 与 final 并非对相同信息作强弱不同的分类：raw 只看四维类差，精修器还读取对这些类差不可见的特征分量及 RGB/空间上下文。** 这是一项尚未完成因果干预的读取诊断，不是新方法或信息上限结论。

对象为已完成 `runs/h3_moments/02_cross/last.pt`，SHA `22bc8a2ddb260f93cb01b17857c97b2bb0873038efdb9318545cb2bdbb045226`，498,136 点、每点16维特征。使用它自己的冻结 model/refinement/depth_moments/train 源码；缺失历史 pixel metadata 按原合同解析为 legacy，不升级网格。没有读取本轮活动的1M训练检查点。

## 假设与现有证据

唯一假设：**冻结精修器用于补全索区域的一部分有效输入，位于原生分类器的 softmax 不可见特征子空间。原生类别概率丢弃了这部分表征，因此 raw cable 低分不等价于整个共享 Gaussian 特征场完全没有索区域信息。** 若移除该分量、保持 raw/RGB/深度/alpha 不变后，原有索区域纠正基本不受影响，则不支持这个固定精修器依赖该通道的解释。

实际模型先计算每点 `softmax(Wf+b)`，再按共享透明度权重合成为 raw p3d；精修器同时读取 `Σw_i f_i`、RGB、depth、alpha、p3d、空间梯度，以及 H3 的深度—特征矩。H3 的3k阶段只更新head，field和共享decoder冻结。因此H3 raw保持底座约30.8% native cable，而cross最终约93.8%；固定DINO组合在另一个官方原图口径达到94.79%。不能跨这两个网格直接相减，也不能把DINO增量、H3矩增量和基础refiner收益混为一因。

已有控制分别排除了部分解释：Rdetach未提供总体优势；低总alpha只覆盖原8 TRAIN中少量GT索像素；4-bias校准、单轮EM/Adam与共享opacity干预没有解决raw瓶颈。16-ray实际前向则说明那些固定点存在更好的赋类，不能据raw低分断言几何不可表达。这些均未测试本项class-invisible特征依赖。H3 cross−variance无增量也不检验全部16维普通特征的作用。

## 可复核的 CPU 数值

对固定 decoder 定义 `D=W[1:]-W[0]`。在FP64做SVD，令 `P=V₄V₄ᵀ` 为D行空间的正交投影；唯一候选为 `f'=FP32(FP64(f)P)`。在实数代数中 `D f'=D f`，两组logits只相差每点公共偏移，softmax不变。这里的12维包含softmax的公共logit不变方向，不能简单用 `rank(W)` 的零空间替代。

| 核验 | 实测 |
|---|---:|
| D奇异值 | 6.996987、2.894677、2.027827、1.958096 |
| FP64秩阈值／rank／不可见维数 | 2.48583e−14／4／12 |
| `max abs(D(I−P))` | 9.57104e−16 |
| 被移除分量占全部feature平方能量 | 74.6432% |
| 被移除分量占去全局均值后变异能量 | 54.4094% |
| FP32实际重算中心化logits最大差／RMS | 4.76837e−6／5.08588e−7 |
| FP32每Gaussian概率最大差／RMS | 4.17233e−7／1.11241e−8 |
| 498,136点argmax变化 | 0 |
| FP64概率最大差 | 2.44249e−15 |

能量比例依赖当前feature坐标和欧氏度量；对feature作可逆线性重参数化、对decoder作对应逆变换即可改变能量比例而保持读出，所以它**不是内在信息量、结构容量、互信息、贡献比例或“54%语义被丢弃”**。rank4足以让独立16维feature实现任意内部五类概率，不能把以上结果说成decoder秩缺陷。CPU FP32概率近等也不是CUDA p3d图像逐位一致证明，后续必须单独核真实render。

报告：[report.json](/mnt/data/SHM2026/runs/h3_semantic_readout_cpu_audit_v1/report.json)，SHA `dde4019996075d4db3a81942976b35df746dac866c4173d04cf26d94730fce50`；[只读CPU入口](/mnt/data/SHM2026/runs/h3_semantic_readout_cpu_audit_v1/audit.py)。输入checkpoint SHA前后不变，三项读取tensor哈希不变，CUDA未初始化。

## 唯一最小前置诊断：设计，尚未执行

仅固定两张有标签TRAIN：按原manifest全部350个TRAIN名称排序的索引0和175，即 **002.png、205.png**。不按误差挑视图、不追加样本。两状态分别为原field与上述唯一投影field，共4次 `scene.render`，原生1320×989、相同原相机/SH3、`refine=True`。没有DINO教师、真实RGB输入、优化器或新模型存盘；GT/valid只在预测结束后作TRAIN评分。

从 **Gaussian sem_features 源头** 临时应用投影，不在head入口只修改渲染平均feature。这样普通特征和 `z*f` 同步变化，depth-feature covariance在`asinh`之前随之投影，不留下cross-moment零空间旁路；全部feature派生量重新从对应状态渲染，不复用旧feature或moment缓存。RGB、geometry、opacity、decoder及head均不改。直接对已经`asinh`压缩的context作线性投影并不等价，禁止这样替代原路径。

前置数值门固定为：两状态的RGB/depth/alpha逐位相同；每view全图独立核 `max abs(p3d_original−p3d_projected) ≤ 1e−6`，另报raw argmax变化和最大误差位置。若失败就停止依赖解释，不放宽门或换另一种投影。完整state/flags/modes/camera在finally恢复；捕获原/新p3d、最终概率、残差及三个保持不变输入的SHA。

通过数值门后，只描述两view的固定类别加权final CE、pooled/逐view CM、各类IoU，特别列出“原raw错为背景、原final纠正为索”的像素在投影后保留/丢失数，同时报告新错和反向纠正。不按变化重新选像素，也不把这两张TRAIN当未见视角。若最终输出和纠正不变，则否定该组输入中的显著依赖；若明显退化，只支持冻结head依赖这部分输入。这是普通feature通路及其相关cross-moment通路的联合依赖，不能由此区分两条路径。它仍可能编码位置、区域上下文或训练集记忆，不能据此命名为物理三维索语义。

直接投影还移除了null分量的全局均值：若 `f_null=μ_null+r_i`，合成输入中的被删部分包含 `alpha*μ_null`。这可以是可见性/覆盖编码，未必是新语义信息；即便head另外读取alpha，冻结权重也不保证自动补偿。保持原先唯一投影，不为此另做去均值版本。

**分布外限制必须保留：**零空间投影改变feature幅度/相关性以及GroupNorm前激活，冻结head可能因输入分布改变而退化。因此单次下降不证明该信息对重新训练的head必需、不证明raw理论上限，也不证明增加feature维度或新loss有效。现阶段不训练补偿头、不接teacher、不引入第二投影或扫阈值；这是表征可读性与输入依赖检查，标准线性代数不作创新。

已限定检查 `runs/h3_moments_preflight_v2`、`runs/h3_moments/02_cross` 与 `artifacts/diagnostics`，未发现对齐的H3每像素feature/moment缓存。已有391a赋值NPZ是另一个场的每Gaussian特征，不能冒充H3渲染证据。本轮因此只保存CPU报告，没有生成新feature缓存或执行上述4次render；GPU仍需独立授权。

独立设计复核：novel已只读确认类差投影与源头同步moment的数学合同，没有发现阻断；强调全图逐view p3d门、全部feature派生量重渲染、联合通路/OOD解释边界。该复核不构成GPU执行授权。
