# 冻结场精修头的水平翻转工程对照

新增 `refiner_horizontal_flip_probability`，缺失或0保持原流程；预定工程对照使用固定0.5，不搜索翻转概率，也不声称学术创新。实现位于 `src/bridge_rgs/refiner_flip.py`，`train.py`只新增配置/strict-resume检查和训练分支。没有修改prepared、相机、geometry、渲染器、已有H3快照或模型结构。

启用条件：`parameter_scope: refiner_only`，冻结geometry/RGB，labelGT-only TRAIN，独立view RNG，禁止crop、teacher/pseudo、fusion、相机优化、增密和geometry/opacity辅助loss。已有scope将field语义参数、classifier、SH、background和相机全部冻结；flip helper进一步detach所有渲染输入，只有refiner参数得到最终loss梯度。

每步只按 `(seed, step, 704291)` 初始化局部NumPy SeedSequence/Generator，决定是否翻转。它不消耗全局NumPy、Python、torch/CUDA或view sampler状态。step为当前stage步骤，恢复checkpoint继续使用原step；strict resume不允许变概率或启用状态，活动翻转也不允许变seed。改这些参数必须新stage warmstart。概率0的旧checkpoint缺字段保持兼容。

翻转步只做原相机的一个3D渲染，暂不在scene.render内部执行head；之后把HWC的features、RGB、depth、alpha、p3d prior及可选depth moments全部沿宽轴翻转，再调用同一个refiner。GT和valid同步翻转，不交换类别ID，不改K/pose。最终CE/Lovasz和residual正则使用翻转后的输出；原RGB和raw p3d监督继续用原始全幅result及GT。非翻转步沿用原调用路径。推理、验证保持原相机/原图，无TTA。

配置增量例子（现有GT-only refiner_only stage基础上）：

```yaml
refiner_horizontal_flip_probability: 0.5
parameter_scope: refiner_only
freeze_geometry: true
freeze_rgb: true
train_labeled_only: true
independent_view_rng: true
refiner_crop_probability: 0.0
pseudo_dir: null
pseudo_weight: 0.0
pseudo_refiner_weight: 0.0
multiview_fusion: false
multiview_weight: 0.0
```

其余学习率、warmstart、步数、class weights、sampler和native grid由控制实验固定。本次未生成或启动formal训练，未指定/改变选中模型。若用于后续正式实验，应先在immutable snapshot完成真实CUDA短预检并核验非refiner参数逐位不变；现CPU检查不冒充该端到端预检。

CPU验证106项通过，2项已有CUDA scope测试因显式 `CUDA_VISIBLE_DEVICES=''` 跳过。新增测试包括：所有输入/GT/valid及moments对齐；ignored输出无直接loss梯度；4种moment模式仅head有梯度；原RGB/raw loss数值不变；概率0不分配RNG及前向逐位一致；400步真实view sampler顺序/状态一致；断点后的stateless选择一致；strict resume拒绝变概率/活动seed；实际GaussianScene参数scope与optimizer上2个CPU更新仅改变refiner（使用合成rendered evidence，不调用GPU renderer）。相关Ruff通过。同伴独立只读审查未发现阻塞。

- 最小train diff：`artifacts/refiner_flip_support/train_integration.diff`
- CPU审计与source SHA：`artifacts/refiner_flip_support/cpu_review.json`
- CPU复现：

```bash
CUDA_VISIBLE_DEVICES='' uv run pytest -q tests/test_refiner_flip.py tests/test_refiner_crops.py \
  tests/test_depth_moments.py tests/test_semantic_schedule.py tests/test_training_scope.py
```

此增强检验当前单桥精修头的视图适应能力，不保证提升held-out mIoU，也不将开发过程中的适应性选择称为盲测。

## 固定正式候选与真实预检设计

在正式启动前，已固定两组各3000步的工程候选。共同warmstart为 `runs/support_split_semantic_coupled/last.pt`（131个model张量），保留原multiscale64/bound6 head，moment off，`warmstart_reset_refiner:false`、`warmstart_add_depth_moments:false`。259个有GT TRAIN视图，shuffle seed42、independent RNG、原生1320×989、fixed head LR .0003，save/eval_every=0仅最终保存/正式runner的最终评价；crop、teacher/fusion、geometry/opacity/depth辅助loss全关。唯一科学变量为pflip0对.5，输出目录不同不属于训练变量。

配置与预定3000步view/flip序列位于 `configs/generated_refiner_flip/flip_plan.json`、`00_control.yaml`、`01_flip.yaml`。前8步flip固定为 `[true,false,true,false,false,true,false,true]`；没有按实际loss或验证结果选择序列。

源码来自 `runs/h3_moments_preflight_v2/source_snapshot` 的完整目录复制，仅覆盖train.py并增加refiner_flip.py。逐文件SHA断言只有这两项变化，未copy正在改动坐标接口的main其它模块。base tree SHA为 `1da936d3d5173b319391c88b9c477fc17555e6d65a3f0c1e626767e32477f499`；新tree为 `2e0ad4dbfc78a727e9d515c6f0cb4adb3ae914f6ab3f17f4f905bc4880aa3efb`。各Python文件SHA及两项delta在 `runs/refiner_flip_preflight/source_provenance.json`。

真实预检设计为新快照p0/.5各8步，及旧H3v2快照与新快照的off各2步。旧off使用相同原始head、相同训练配置但无新增flip字段，并非把H3的zero-moment架构当成原head。worker在真实trainer优化器创建处核验初始化131张量逐位同源及fresh Adam，每步记录实际view、flip决策、render里的head开关、原生field输出SHA。最终检查仅refiner变化、field/classifier/cameras与元数据不变，比较两臂sampler和全局随机状态，记录旧新off参数及Adam的逐张量数值误差，不预先假设CUDA逐位相同。

只读保存hook仅在末步完整校验后写出 `audit_state.pt`（refiner、Adam、RNG），明确不是可推理/恢复的场checkpoint。它不改变训练运算，避免存4份完整场；正式runner仍使用原checkpoint保存。预备时新增总预算约680MB，含4个紧凑审计、两组正式末权重及评价，预留768MiB安全余量；没有删除既有文件。

```bash
# 已执行一次；拒绝覆写固定设计
CUDA_VISIBLE_DEVICES='' uv run python scripts/preflight_refiner_flip.py --prepare
# 明确授权且GPU空档后执行；只做短预检，不会启动formal
uv run python scripts/preflight_refiner_flip.py --run-gpu
# 两组formal需另行授权，再复用同一source_snapshot，不使用当前main。
```

## 真实预检结果与默认路径数值复核

两臂各8步、旧/新off各2步均完成。每次真实初始化的131个张量逐位同源、fresh Adam；训练仅改变120个refiner张量，所有field/classifier/cameras及元数据逐位不变。8步的实际view/flip均与计划相同；两臂sampler与NumPy/torch/CUDA RNG相同，每步RGB/depth/alpha/features/p3d的完整数组SHA相同。GPU峰值控制4.947GiB、翻转4.997GiB。只保存compact审计，没有预检完整场last。

旧/新off的CUDA更新未逐位一致，因此按事先约定如实保留误差，随后获得授权做同新版快照独立newB两步重复。没有改变确定性开关、seed、学习率或任何阈值。

| 两步对照 | head最大绝对差 | head平均绝对差 | head相对L2差 | Adam最大绝对差 |
|---|---:|---:|---:|---:|
| newA vs oldA | 1.33322e−4 | 1.15667e−7 | 6.40340e−6 | 4.06799e−7 |
| newB vs newA（同源码重复） | 1.12236e−4 | 1.16867e−7 | 6.42157e−6 | 2.23677e−7 |

三次step1总loss完全相同：.0962995291；step2的newA−oldA总loss差2.23517e−7，newB−newA为−1.49012e−8。RGB loss始终相同，sampler与全部已保存RNG完全相同。同源码重复造成同量级的head漂移，符合CUDA backward/reduction及Adam近零梯度敏感性的解释，但本小检查没有定位具体kernel，也没有估计3000步后的指标方差。不能声称旧/新CUDA训练轨迹逐位相同。

额外只做了旧/新源码各一次原始warmstart首TRAIN视图的前向探针，二者的RGB/field、probabilities、residual SHA与newB实际首步全部相同。最初A预检只保存了field SHA和逐步loss，没有保存head forward buffer；不把这个新探针冒充对A历史head缓冲的追溯恢复。数值详情见 `runs/refiner_flip_preflight/legacy_numerical_detail.json`、`repeat_comparison.json`；原预检的 `requires_review_before_formal:true`保留，后续复核不覆盖历史记录。

冻结快照上的额外36项flip CPU测试通过，脚本Ruff通过。所有短GPU进程自然退出。上述为正式启动前记录；正式两臂必须复用同一新快照、共同输入与固定设计，并诚实保留这些数值限制；不为了让某一臂匹配而单独打开确定性开关。


## 固定3000步正式配对结果（2026-09-26）

两臂按计划顺序运行，均自然退出0并完成原生1320×989的50个RGB相机、41个有GT语义相机以及LPIPS评价，含235；没有中间验证/选checkpoint。控制训练231.04秒，翻转231.98秒；训练峰值allocated GPU分别4.958/5.005 GiB，时间不包含最终评价。翻转的3000步中预定1524步翻转，每个259标签TRAIN视图访问11或12次，两臂完整像素预算相同。

| 固定终点 | all5 mIoU % | FG mIoU % | cable IoU % | tower IoU % | foundation IoU % | cable 2px boundary F1 |
|---|---:|---:|---:|---:|---:|---:|
| 原始warmstart | 94.51176 | 93.31972 | 93.54931 | 93.94700 | 88.78790 | 0.885672 |
| control p=0 | 93.18450 | 91.78772 | 88.31813 | 93.67142 | 88.42940 | .754015 |
| flip p=.5 | 94.55065 | 93.36205 | 93.99743 | 93.80983 | 88.69844 | .900031 |

固定两臂主对照flip−control：all5增加1.36615个百分点，按41验证视图配对重采样5000次、seed20260926的95%区间为[+.70468,+2.16967]；cable增加5.67930个百分点，区间[+2.68370,+9.41281]。重采样先聚合抽样视图的混淆矩阵再计算IoU，并非简单平均每视图IoU。

同时必须看源模型：control−warmstart all5下降1.32726个百分点，区间[−2.09217,−.66915]；flip−warmstart仅增加.03889个百分点，区间[−.05911,+.15764]，cable增加.44812个百分点，区间[−.10001,+1.13482]。因此本次augmentation主要防止无增强继续训练的退化，不能说它明确超过原始warmstart，all5仍未达到95%。tower相对源下降.13717个百分点；不隐去类别权衡，也不因此搜索翻转概率或追加训练预算。

三者RGB均为PSNR30.10240955、SSIM.9029809678、LPIPS.2240065262；50张RGB PNG逐文件SHA相同。原生3D all5=79.19865875%、cable=30.80808275%，所有41视图的原生3D混淆矩阵逐项相同。最终语义变化完全来自refiner，不属于3D field的改善。

终点审计通过：131个model张量的keys/shape/dtype和架构元数据保留；11个非refiner张量（包括全部geometry/RGB/semantic features/classifier/prior）与training cameras逐位同源，只有120个refiner张量改变。冻结优化器无状态，head Adam步数3000、LR固定.0003；真实8步预检已核验初始化fresh Adam且不reset head。两臂torch/CUDA/全局NumPy RNG精确相同，sampler的末次order/cursor/RNG与预先锁定3000步序列相同。正式仅step1及每100步有日志，这些实际view/flip全部匹配；完整3000步顺序由不可变采样代码与末态重建，不把稀疏日志冒称逐步直接观测。

配置文件SHA与实际receipt.config逐值绑定；两臂source/input SHA相同，唯一训练配置差为flip概率。全部运行使用预检NEW snapshot，与后续main像素协议修改隔离，没有改prepared/camera，没有删文件。`launch_receipt.json`记录启动前磁盘和源码核对。标准水平增强属于工程对照；该方向在查看开发错误后选定，不能声称盲预注册、跨场景泛化或新机制。单seed且CUDA更新存在已记录漂移，视图区间不代表训练种子间不确定性。

可复核文件：
- `runs/refiner_flip/flip_report.json`：冻结、配置、RNG、采样、RGB/raw3D与数值汇总。
- `runs/refiner_flip/paired_flip_minus_control.json`：主对照。
- `runs/refiner_flip/paired_control_minus_warmstart.json`、`paired_flip_minus_warmstart.json`：相对源的完整配对结果。
- `runs/refiner_flip/{00_control,01_flip}/last.pt`、各自`evaluation_native/metrics.json`和`experiment_receipt.json`：固定终点权重与native评价。

```bash
CUDA_VISIBLE_DEVICES='' uv run python scripts/report_refiner_flip_pair.py
```

独立只读复核发现的配置→receipt绑定增强已加入；最终报告脚本Ruff通过。尚未执行任何refiner翻转TTA；后续若实施，属于另外锁定的推理工程对照，不能混入这次训练augmentation结果。
