# 连续位置候选三臂训练：联合优化改善 TRAIN，接受控制未显示优势

2026-09-27。三臂均完成预定148次候选。**普通联合与PCGrad在完整259 TRAIN上同时降低RGB MSE和raw CE，原生语义mIoU约提高0.45个百分点；实际接受控制收益更小，且完整TRAIN RGB略退。** 这还不是留出视图或最终mask的提升，也不支持接受控制的学术增益。

运行目录：`/mnt/data/SHM2026/runs/continuous_geometry_proposals_v1`。原数值预检仍是3/12通过；本次是已明确另立的局部近似VJP候选实验，不能追认梯度认证。推导与范围见[协议](continuous_geometry_proposals_protocol.md)、[链分解](geometry_chain_results.md)。

## 固定末态的完整 TRAIN 描述

所有行使用同一个H3、已完成EM/FW的固定q、同259图和原valid网格；仅means变化，无head或teacher参与。

| 方法 | 接受／候选 | RGB MSE | RGB MSE相对起点 | raw CE | raw五类mIoU | raw拉索IoU |
|---|---:|---:|---:|---:|---:|---:|
| 共同起点 | — | .000764064397 | — | .102152275657 | 80.659340% | 35.367241% |
| joint | 148／148 | .000754606540 | −1.237835% | .100634256690 | 81.110159% | 35.745790% |
| PCGrad | 148／148 | .000754731191 | −1.221521% | .100625025217 | 81.112166% | 35.747890% |
| actual RGB acceptance | 12／148 | .000764412762 | +.045594% | .101988506395 | 80.710321% | 35.397677% |

joint／PCGrad／acceptance 的raw mIoU变化分别为+.450819／+.452826／+.050982pp。PCGrad与joint的点估计仅差约.002007pp，没有多种子优势证据。没有best-step挑选，表中全是固定最后末态。

接受控制检查的是当次七张图。即使12次提交都通过各自batch的RGB和CE条件，也不保证259图平均不退；本次完整TRAIN RGB略升就是这一边界的实际例子。拒绝后回滚参数及Adam状态的实现合同通过CPU测试；尚不能仅凭表格独立证明每次状态恢复。

## 执行与状态

三臂同批次顺序、初始means、fresh Adam与实际候选渲染成本。444 attempts，6,216训练view pairs加1,036描述pairs，共14,504 standard rasters、6,216 means VJP；777次缓存目标文件解码，0head／teacher／VAL／生产checkpoint覆盖。内部80.931356s／外层81.395623s，自然退出0。36项冻结CPU测试通过，79冻结源、536输入及完整场／数值状态恢复检查通过。

joint／PCGrad／acceptance 的最大累计Mahalanobis位移分别2.303060／2.303183／.187495；每步1/64限幅并非累计限幅。各臂allocated峰值799,629,312／826,067,968／826,379,776 bytes。这里是此卡、此环境的实际耗时，不能当跨方法同墙时优势；原q优化与H3训练成本仍另计。

| 记录 | SHA256 |
|---|---|
| plan | `61c66187e63325fa0d23e8ea9b8522165f9335b93424f3daa54b612e4553b588` |
| worker | `06ec5655e6a6d31c3babe25a63a557a07ba111d20216f5b6b4b7909ebfcf91c2` |
| execution | `5a7e9ed582cbd530123d48f50cd394bbf939b28567fd270d958fbf4dc7a29242` |
| joint means-delta | `6a60b4ddad8a6fda4c3a6befa8215cae144f330b3fc23b17c4357ec2100f4bd4` |
| PCGrad means-delta | `33fdfb53005098d1c952778a84aeef910db02a80aba2bca37d34bb3e40f76dee` |
| acceptance means-delta | `7fe09181180786ffa02163c82871aa4d1433f6dc38709ece921c549e47ae45d9` |

独立CPU审计已完成，内部1.524s／外层1.717s自然退出0。79源、536输入、444次决策、端点与最终Adam步数、累计位移上界和259图均值／混淆矩阵均一致，最大复算差2.22e−16。报告SHA `8f9f9a8c24becec9e46894a3521b7afd3b137b583724d1a0cf32beef34509844`。未保存每步高维梯度或完整中间Adam状态，因此不独立重算每步VJP或每次moment回滚；不重新渲染或解码GT。

另对已保存444行做事后只读计数：三臂全部148个候选都降低各自batch CE。joint和PCGrad各147个候选通过batch RGB条件；语义单独提案的acceptance臂只有12个通过该条件，147个通过逐图条件。因此本轮大部分拒绝发生在batch RGB条件，不能归因于逐图阈值。该计数不触发阈值调整或补训。两控制各有1次RGB实际变化与保存的梯度内积异号，acceptance臂有7次；这是本条轨迹上的描述，不是全局梯度精度证书。

本轮也没有额外RGB-only训练臂，故不能把joint的全部TRAIN改善归于新增语义梯度；标准联合优化作为完整方法取得的点估计与单一机制的因果增益必须分开。

## 后续固定评价（已完成）

在查看三臂完整结果前，已指定共同起点及三个固定末态全部参加相同50张原图／41份标注的评价。每个末态重新产生自己的H3 RGB、raw、原冻结head输出，并对自身渲染RGB重新运行旧H+教师，固定各半融合。无新增head适配、teacher训练或组选择。基线须在新GT读取前复现旧optimized-q-zero的输出；所有候选输出完成后再统一评分。

主报告会记录各自H3的真实RGB指标。工程E的主RGB来自另外两个场，不能把E的30.035661dB拼作本次means修正的RGB收益；E只单列参考，当前部署与95.109016%联合mIoU均不变。以上标准联合梯度／PCGrad／实际接受算法本身不构成新公式；是否有可迁移增量由后续完整评价决定。

留出评价已完成并独立核验：raw约+.36pp没有传递为最终收益，joint／PCGrad最终各退约.16pp，详见[完整留出结果](continuous_geometry_evaluation_results.md)。额外RGB-only控制已另立协议并完成训练，后续报告单列，不改写原三臂设计。
