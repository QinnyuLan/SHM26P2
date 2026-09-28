# TRAIN 实际轨迹语义一致性诊断：冻结协议 v1

本协议在新逐观察标签统计之前登记。它执行 `next_research_hypotheses.md` 中假设 A 的必要条件检查，不训练模型，不评价 VAL，不替换当前交付 E。无新性能或学术机制结论。

## 人口与数据边界

仅使用 legacy `artifacts/prepared/init_points.npz` 保存的 60,000 个实际 SfM track ID、457,102 个 point/image 观察对；保持原 CSR 顺序。不是高斯克隆、最近点继承或全部未使用 SfM 点。恢复原始 COLMAP **实际观测关键点**；同一 track/image 多个关键点取最后一个，与原准备代码一致。每点保存的 image_id 必须严格递增。所有 350 TRAIN 视图参与元数据及分组，其中 259 有标签、91 无标签；无标签是 NA，不当背景。非 TRAIN 观察行跳过，不解析坐标。

原 SfM 相机估计使用全 400 图，点重三角化使用 TRAIN。本检查条件于既有相机、TRAIN 点筛选与保存的内点，不能称独立 SfM 或 OOF。原点构建曾要求至少 3 视图、正深度、误差不超过 3 原生像素；新检查不改变观察集合、不重新三角化。保存 XYZ 为 FP32，新误差以 FP64 复算，微小超过旧阈值也原样报告。

prepare 只读元数据、点数组与代码，并通过 SHA256 读取 TRAIN JSON/mask/valid **文件字节**绑定输入，不解码标签/图像、不统计冲突。冻结包含脚本、统计模块、测试、此协议、运行库版本与 uv.lock；绑定原 H3 的 prepare/data/geometry 源码作为来源证据。execute 才解码 259 TRAIN JSON 和对应 legacy mask/valid。没有 RGB 载荷、VAL JSON/mask 载荷、模型、预测或 GPU 调用。

## 两套采样，各自独立报告

主诊断使用原始畸变图像的官方 LabelMe 多边形栅格：1320×989，背景 0，deck 1，stay_cable 2，tower 3，foundation 4，未知类 255；按 JSON 顺序 PIL polygon 覆盖。COLMAP 坐标原点在左上角，原数组像素索引固定 `floor(raw_xy)`。

敏感性诊断使用旧 prepared mask/valid：原 observed keypoint 经源相机 `cv2.undistortPoints`，由 manifest K 投到 prepared 格，再 `np.rint`。这是复现旧 lookup，不修改既有训练数据/检查点，也不把新主诊断与旧模型成绩合并。分别报告两个完整统计及共同可用观察上的标签差异；只以主诊断作下面的必要条件判断。

误差：原始 pose 下 saved XYZ 投影，与实际 observed keypoint 去畸变后的原生内参坐标之欧氏距离，单位 native px。必须正深度，否则停止而不默默剔除。`viewing_ray_world` 是 camera→observed ray；`point_direction_world` 是 camera→saved XYZ，两个字段不混用。

边界距离：采样像素中心到其他类别、ignore、无效支持或画布外的欧氏像素距离。每类区域外补一圈 false 后计算 EDT。不是到连续多边形线段的距离。原格全部画布支持；legacy 还须 prepared valid。

固定分层：重投影误差 ≤1、(1,2]、>2；边界距离 ≤3、(3,10]、>10；报告全部 9 个交叉层。strict = 误差 ≤1 且边界距离 >10。每层先筛观察，再重新计每点观察和类票，不借用层外标签。低误差与内区是代理，不能据此命名为可靠物理表面；遮挡、误配、区域定义和相机误差仍可能存在。

## 统计与唯一必要条件门

对至少 2 个可用观察的 track，报告全 5 类计数，以及：

- 观察等权固定硬标签冲突：`sum(n_t - max_c n_tc) / sum(n_t)`。
- 每 track 等权冲突：`mean(1 - max_c n_tc / n_t)`，作为主统计。
- bg/cable 同时存在的 track 数、观察 image_id 覆盖；含 cable 的全部可用 track 数单列。

这只是每轨迹一个固定硬标签的最小观察错误，**不是 3DGS mIoU 上限**：track 不等于 Gaussian，视角变化的 alpha 合成允许固定点类别产生不同像素输出，二维上下文头更不受此硬标签假设约束。

LOO 共识只作描述：target 不进参考票，参考与 target 同层，至少 3 个其他不同 image_id，最高类占比 ≥0.8。报告全 5×5 混淆矩阵、各类/整体 coverage、接受后的观察等权和 track 等权错误。接受依赖参考标签，可能偏向少数类 target（3A1B 中 holdout B 接受而 holdout A 不接受），因此不以 LOO 接受错误作 gate。

all 与 strict 各做 2,000 次 track bootstrap，seed 20260927，报告两种冲突的百分位 95% 区间。9 个交叉层不做区间，不选有利层。区间条件于已选轨迹与共享相机，不是独立相机/新场景显著性。所有 350 TRAIN 按 name 排序，以 floor(index×4/350) 分 4 组；分别删除一组的观察后重计支持，报告 strict 的覆盖与点估计敏感性，不重三角化、不重采样相机、不新增门或删组 CI。

主 strict 的覆盖必须同时满足：≥1,000 个至少 2 观察的 track；≥50 个含 cable 票的此类 track；其 cable 观察覆盖 ≥8 个不同 image_id。信号必须同时满足：全 5 类每 track 等权冲突 bootstrap 下界严格 >1%；≥50 个 bg/cable 混票 track。

- 覆盖不足：`inconclusive_coverage`。
- 覆盖足够但信号条件未全满足：`not_supported`。
- 两者满足：`conditional_conflict_signal_only`。

最后一种状态仅支持已保存 TRAIN 对应上的条件性硬标签冲突值得进一步调查；不证明方向关系、三维不可表示、当前性能主瓶颈或新模型有效。方向只保存 ray 并描述 strict 轨迹间 `acos(mean pair cosine)`；不拟合 classifier，不选择方向 gate。当前不挑图、不人工改标签、不搜索阈值。

## 执行与后续

以 uv 虚拟环境 CPU 执行，CUDA_VISIBLE_DEVICES 为空；单次固定人口、禁止覆盖旧输出，内部 900 秒/外部 960 秒。输出逐观察 NPZ、collection_receipt、主/敏感性 analysis 和 execution_receipt，绑定输入/源码 hashes。合成测试与独立代码审查在冻结前进行；结果由独立 CPU 实现复算核心统计。若执行失败，保留失败现场，仅登记明确实现错误修复，不能按统计结果改门。

任何结果都不改变当前 E 的验证成绩或交付选择。只有诊断支持一个可区分的解释，才另行登记与已有工作区别明确的模型控制；不能把 view-dependent semantic SH、跨视图一致性或更换 DINOv3 自动称为创新。
