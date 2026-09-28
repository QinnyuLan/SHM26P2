# SEM386/RGB150 的本协议近似复现：范围与优先级

仅审阅 `project_progress_20260918` 已有页面与登记；不寻找 Dev30 原清单、源码或权重，不下载、不运行实验。结论是：**可以建立“SEM386 风格的独立参考”，但不宜优先于正在完成的 H+/7B 容量对照，更不能把它叫同学正式实验复现。** 最多保留以下五项关键差异。

| 项目 | 已披露、可借用 | 在我们 350/50 上必须明确选择或保留的缺口 |
|---|---|---|
| 1. 任务与数据域 | SEM386/384 用冻结 RGB150 的渲染输入训练二维 Mask2Former；270 个有标签 TRAIN、30 个 holdout；SEM384准备仅访问370个 RGB 训练角色的渲染。不是 DINOv3 教师，也不是三维语义高斯联合拟合。 | 我们固定350 RGB TRAIN/50 VAL、259/41有标签；按自己的相机/像素协议制作输入与标签。选择“在当前冻结场上比较语义架构”还是“另建近似RGB150整链”，两者须命名分开。不能沿用对方93.741796%作为共同协议的目标基线。[SEM384](../project_progress_20260918/sources/SEM-384-RGB150-DEV270.html)、[SEM386](../project_progress_20260918/sources/SEM-386-RGB150-DEV270-CONTEXT50-MEMORY.html) |
| 2. RGB 重建 | 可借其前驱RGB140的 Standard 3DGS/AbsGS思路：大高斯absolute split阈值.0004，小高斯signed clone .0002，30k，500–15000增密、间隔100、opacity reset3000，固定相机。 | RGB140这些是明确披露；RGB150仅被引用为Standard-AbsGrad-Dev30，本包未展开其完整resolved重建配置，不能自动视为所有参数逐项继承。若独立实现还要指定初始化来源/点数、distortion与像素约定、renderer/antialias、优化器与各项默认行为；我们现有gsplat/support-split场并非其原Standard补丁的等价实现。最低增量成本应先复用我们自己的冻结场，而非同时重做RGB。[RGB140](../project_progress_20260918/sources/RGB-140-STANDARD-ABSGRAD-ALL400.html) |
| 3. 语义模型与优化 | SEM384从fresh Swin-L ADE预训练开始，6000次成功更新，四个backbone阶段均训练；SEM378声明沿用作者optimizer/loss/poly配方。旧SEM349另披露batch1、AdamW LR1e−4/backbone×.1、poly.9无warmup、100query、class/mask/dice=2/5/5、no-object=.1、deep supervision/12544采样点，并非逆频率CE。 | 上述SEM349细节可作为近似配方依据，不能冒充SEM386完整resolved YAML逐项证据。新实现仍需锁定Swin/decoder完整配置、五类输出头如何从ADE初始化、AdamW参数组/weight decay/clip、point sampling及AMP/FP32分工。仅换成HF同名模型不保证等价。[SEM378](../project_progress_20260918/sources/SEM-378-RGB140-ABSGRAD-ALL300.html)、[SEM349](../project_progress_20260918/sources/SEM-349-MASK2FORMER-FINAL.html) |
| 4. 采样与内存 | seed20260805，6000成功步；奇数native768裁块，偶数完整1025×768上下文，各3000步；源1320×989，Pillow bilinear RGB/nearest label，等比例、无letterbox，resize后flip；四阶段Swin activation recomputation。 | 旧parent采样轨迹只能按披露规则在新259训练标签集合重新生成并锁定，不能声称同轨迹。需自定uniform视图/裁块坐标/flip抽样的精确RNG及AMP重试是否重用样本；这些选择都仅由TRAIN制定。context策略可借用，但本项目已有裁块/上下文路线，不能把这一项另称创新。[SEM386](../project_progress_20260918/sources/SEM-386-RGB150-DEV270-CONTEXT50-MEMORY.html) |
| 5. 推理与评分 | 768 tile、512 stride滑窗；原生FP16 score零质量像素仅用相同logits在FP32恢复，非零像素保持原样，最后输出原始五类。 | 精确tile边界/padding、重叠累积、query score组合/归一化、argmax时机未在这份页面完整展开；推理flip不能由训练flip推断。可独立选稳定的全FP32累积，但须声明差异。两架构都应在我们的同一原始网格评分指纹下评价并报告推理成本，不能跨对方Dev30、我们的native与official协议比数字。[SEM386](../project_progress_20260918/sources/SEM-386-RGB150-DEV270-CONTEXT50-MEMORY.html) |

**建议次序：先完成当前 H+/7B 容量对照，再决定是否需要一个 Mask2Former 架构参照。** 前者复用已锁定的模型加载、训练/评价与来源核验链，新增实现选择较少；近似同学路线同时改变监督预训练、可训练backbone、decoder/loss、渲染输入和推理，不能回答“DINO容量是否不足”这一个问题。SEM386相对其SEM384仅+0.043190pp五类mIoU，且其报告自行限定单划分单seed，不能由此推断上下文混合或完整复现必有实用优势。

若之后做，优先仅建**同一现有冻结RGB场、同259标签、同公共评分的SEM386风格Swin-L Mask2Former参考**，不同时重建RGB150，不使用真实VAL RGB作模型输入。它能比较我们协议下两类语义路线的效果/成本，不能解释同学正式分数差、替代其缺失产物或证明新颖性。profile两臂备忘仍只是独立proposal，不因本次审查改变或获执行授权。
