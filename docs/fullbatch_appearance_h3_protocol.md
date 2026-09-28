# Selected H3 的条件性 full-batch 外观优化协议

2026-09-27。这是**标准二阶矩预条件下降与 Armijo 回溯的工程验证**，不提出新优化数学或共享几何机制。先完成本场专属的 40-render 数值预检；只有预检通过、独立复核且取得 GPU 交接后，才执行一次下面的固定 full-batch 实验。本文只固定协议，不能代替执行 receipt；旧 probe、旧门与旧结果原样保留。

修复 SSIM 后，旧 corner-v2 场的 3,000 步原图颜色 polish 相对其 base 获得 PSNR **+.118266 dB**、SSIM **+.001671**、LPIPS **−.002036**；它支持继续检查颜色优化，但未通过原先 raw 对 native/base 的方法采用门。该场、像素约定和优化器不同，**不预报本次 H3 的收益**。此前 full-batch 求解器只有 CPU 准备，未执行；本轮不使用 MCMC 新场、不重启结构覆盖路线。[已执行证据](ssim_fixed_appearance_replay.md)、[原 full-batch 设计](conditional_fullbatch_rgb_solver.md)。

## 固定场与禁止改变的部分

- 输入：`runs/h3_moments/02_cross/last.pt`，SHA `22bc8a2ddb260f93cb01b17857c97b2bb0873038efdb9318545cb2bdbb045226`，498,136 个 Gaussian，SH3，历史 selected H3 cross 场。
- 数据：`artifacts/prepared/manifest.json`；当前观测 SHA `551546979a583d46e840bd485559721f361bc28fa4dca60826374ceb74b315fa`。checkpoint 缺失的历史 manifest/profile 声明不事后补造；缺失 profile 依原合同解释为 `legacy_mixed_v1`，不迁移到 corner-v2。
- 只允许三个键改变：`splats.sh0`、`splats.sh_rest`、`background_logits`。所有 means、quaternion、scale、opacity、语义特征、decoder、refiner、buffers、TRAIN 相机逐位冻结；无增密、删除、相机更新、语义 loss 或教师训练。
- 350 TRAIN 按名称排序遍历；姿态从 base 的 `training_cameras` 按 manifest **原 TRAIN 顺序**建立名称映射，不能静默替换为 original pose。固定原图 K/畸变与 legacy overscan→clamp→浮点 gather。训练读取所有 350 张原始 TRAIN RGB；不读取 TRAIN mask 或任何 VAL 像素。
- 新实验 plan 在执行前绑定 checkpoint、manifest、相机名称/映射、350 RGB、依赖、数值预检报告、实际 runner/helper/renderer/loss 源码 SHA，以及 run 内的本文不可变副本。不得把 mutable workspace 文档当运行期间可编辑的绑定材料。

## H3 专属 40-render 技术前置门

固定相机为排序后 350 TRAIN 的 **index 0、175（零基）**，即 `002.png`、`205.png`。不是文件名 `175.png`，也不按误差挑相机。两相机分别使用独立恢复的同一 base，只测试原图目标；每相机 20 次渲染：1 次 baseline、三参数组×三个 ε×正负端点的 18 次有限差分、1 次临时 RMS 更新后前向，总计 **40**。ε固定为 `1e-3, 5e-4, 2.5e-4`；每组方向为该组完整目标梯度除以其绝对值最大值，不反馈搜索方向或幅度。

技术门全部满足才允许进入后续实验：

1. 每相机三组×三步长的 **9 项**中心有限差分全部通过，两个相机共 18 项。实际解析量为 `A=g·[(θ_plus_FP32−θ_minus_FP32)/(2ε)]`，中央差分为 `D=(F_plus−F_minus)/(2ε)`；固定相对误差 `abs(D−A)/max(abs(A),1e−30)≤.05`。点积用 FP64 聚合；端点仍是实际 FP32 参数，原 loss 仍 FP32，不能把重新求和称为全链 FP64 验导。方向不有限、零方向或不可分辨必须停止，不换相机或放宽 ε。
2. 同一 clamp 后渲染 canvas 的既有官方 OpenCV 映射与训练固定浮点 warp，最大绝对 RGB 差不超过 **2e−6**；必须先核完整原图支持，不能填 GT 或丢边缘来过门。
3. 每相机从 base 及新零二阶矩生成一次 `β1=0` RMS 候选，使用下节 helper 的原 α=1 与三组 rates。令实际 FP32 位移为 Δ，`P=−g·Δ`，实际下降为 `D_step=F_base−F_trial`。要求 `P>0`、helper 的 floor 与 Armijo 同时通过，且 **`.9≤D_step/P≤1.1`**。这是单相机一步检查，不是 full-batch 接受预算，也不能保证后续 350-view 平均梯度或 VAL 改善。
4. 全模型张量与 buffer、TRAIN 相机、参数 flags/身份等约定状态完整恢复，输入 SHA 未变。所有候选只在内存临时存在，不保存优化后生产 checkpoint。

预检不通过就停，不能拿旧 v2 数值结果替代、不调方向／步长／容差，也不把偏差直接归咎于某个 CUDA kernel。通过也只约束这两个相机、三个组方向及有限步长。新的 H3 receipt/plan 必须被后续 runner 显式绑定；不改旧诊断的 `failed` 或旧执行门结论。

## 仅一次 full-batch 运行

保留已审核 helper 的全部数值：β1=0、β2=.999、epsilon=`1e−8`；三组 rates 分别 `2.5e−4 / 1.25e−5 / 1e−4`。目标为 350 张图各自 `.8*L1 + .2*(1−SSIM7)` 的等权均值：L1 用原始网格全部 RGB，SSIM7 只计完整 7×7 窗口中心；不量化为 PNG，不加 mask/类别权重，不更换 SSIM 实现。固定顺序逐图 FP32 计算并累加 `loss/350` 梯度，标量损失以 Python float / FP64 聚合；只保留当前图计算图。

每次完整梯度得到后临时更新二阶矩并生成方向。依次试 **α=1、.5、.25**，每个从同一个最近已接受状态构造，记录实际 FP32 Δ；用完整 350-view 前向判定：

`g·Δ<0`，`F_trial≤F_base+1e−4*(g·Δ)`，且

`F_base−F_trial≥max(1e−7,1e−5*abs(F_base))`。

这是标准接受规则，floor 是固定浮点保护而非严格误差界。只有完整、按时接受后才提交参数、二阶矩和接受计数；拒绝或异常回退，三个 α 均失败即停止。**.9..1.1 的比值门只用于数值预检，不额外加入 full-batch 的原 Armijo 协议。** 不因为一次拒绝更换优化器、追加 α 或改预算。

上限为 **20 个接受步、40 次完整 350-view 遍历、360 秒优化阶段**，任一先到就停；梯度遍历和 trial 前向均计完整 pass，最多 14,000 次完整-pass scene render，另如实记录中断 pass 的实际 render。每 view 前后检查时间，超时丢弃未完整的梯度或候选，可能越界一幅图的耗时；不能把此检查称为绝对进程硬限时。加载/哈希/最终保存另记成本，外层硬超时及保存余量必须在锁计划时固定。普通异常不发布候选；正常预算结束只保留最后完整接受状态，零接受不导出重复模型。没有中途 VAL、不挑接受轨迹中的“最佳”点、不重跑。

终点保存独立完整推理 checkpoint；剥离旧 optimizer/RNG/density，不伪装可从旧训练状态 resume。原 config/step 留作 base 元数据，新接受步数、目标、停止原因及观测来源写独立 namespace；普通 resume 明确拒绝、显式 warmstart 可用。原 checkpoint 始终保留不写。保存前后 CPU 逐位检查所有非三键模型张量和相机；进程 finally 恢复其原始内存场，硬杀不能伪称恢复成功。

## 两种终点评价与唯一系统采用门

只评价这一个固定终点，使用共同原始像素网格 `official_original_grid_v1`，指纹 `21a2f19c5d4d703403a0107402d5dd5e006dcd98e5f8e023820c388c3e69a85d`：50 RGB、其中 41 标签图，uint8 PNG 后评分。不得混 native 数值、训练 float loss 或同学未知 Dev30。

1. **plain**：新颜色后的同一个场重新运行 renderer/refiner；与原 H3 plain 对照，作为颜色优化对系统的描述比较。
2. **固定历史 H+ .5**：同一新场重新生成教师输入 RGB、原场分支概率及融合 mask。教师仍为 `runs/teacher_render_adapt_v1/best.pt`，SHA `00f5b84ac9a56c39512c5b8e43f70110397923feea1a2b4c78bdffd1f5524bff`，历史 ModelScope DINOv3 ViT-H+，不换成 7B。固定 tile768、stride512、flip、context weight .25/short-side768，教师/scene 概率 **.5/.5** 在原图 warp 前融合。

两者使用完全相同的新场 RGB，应核输出 RGB PNG 完全一致。颜色虽不改变 geometry 或 raw 类别参数，refiner 和教师都依赖渲染 RGB，**必须重算所有最终 mask，不能拼旧 mask，也不能继承旧语义分数**。固定教师在新 RGB 分布上的表现未经验证，本轮不再域适配。

**主采用比较唯一固定为“新终点 + 历史 H+ .5”减“当前 selected H3 + 同历史 H+ .5”。** plain 不作为备选采用分支，不根据 VAL 在两个系统中选择。对照：

- `/mnt/data/SHM2026/runs/official_selected_ensemble_v1/cross_teacher/official_metrics.json`，SHA `1ec5b8ac8e05fdd30de7c247d6f76daa0c8acc15ecccc171ca56470ca5916f1a`；completed receipt SHA `f22e34da98a6a8b9354640ef55a3986ccea30981ce342ef8da7bcbcf47a28074`。
- 现值：PSNR `29.40094970107756`，SSIM `.8706499242782593`，LPIPS `.2712368741631508`，五类 pooled mIoU `.9510901575393396`，cable IoU `.9478665498969027`。这些是既有开发集观测，不是本轮结果。

所有六项同时满足才采用：

| 固定条款 | candidate−selected 阈值 |
|---|---:|
| 50 视图平均 PSNR | ≥ **+.15 dB** |
| PSNR 配对 95% bootstrap CI 下界 | **>0** |
| 平均 SSIM | ≥0 |
| 平均 LPIPS | ≤0 |
| 41 图 pooled 五类 mIoU | ≥ **−.002**，即最多下降 **.20 个百分点** |
| 41 图 pooled cable IoU | ≥ **−.002**，即最多下降 **.20 个百分点** |

RGB 配对按固定同名 50 相机，5000 次 bootstrap、seed20260926、双侧 percentile 95%；语义先汇总 41 图 confusion 再求 IoU，不把逐图 IoU 均值当 pooled 分数。语义容差使用点估计，不伪称非劣性已被统计证明；同时报告语义配对区间作描述。容差继承历史颜色优化的固定 `.20pp`，不是看新结果再放宽。

技术预检通过、TRAIN 目标下降、最终系统采用是三件不同的事。任一采用条款失败就保留现有 selected，不扫描步数、LR、teacher 权重或按 VAL 选图重做；不能把“full-batch 有下降”称为 RGB 已超越。即使系统采用成功，也只支持这一个场和固定工程配方，不构成学术创新或多场景泛化证据。
