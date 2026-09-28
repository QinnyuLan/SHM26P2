# 真实 TRAIN 轨迹的传统 BA 留出观测检查

本协议在运行之前固定。它补上目前缺失的“实际持久相机修正”几何对照，不重新开启 pose-profile pilot，不把标准 BA 当作创新。当前 E、prepared、任何模型检查点均不覆盖。

## 数据与隔离

只载入已独立核验的 `train_track_semantics_v1/observations.npz` 中七个几何数组：`point_positions, point_track_ids, observation_offsets, image_id, raw_xy, undistorted_native_xy, saved_track_rms`。不载入其中标签/边界数组，不解码 RGB、mask 或新 keypoint，不读取 VAL 观察。manifest 提供 TRAIN 位姿、相机编号与 source K。保存的 keypoint 已是各 image/track 的原始最后一个观察；其重复选择及既有过滤局限保持不变。

将全部 60,000 track ID 按 UTF-8 字符串 `ba_track_holdout_v1:track:{track_id}` 的 SHA-256 字节序排序；散列相同时按整数 track ID 排序，前 500 条为 fit。按同一顺序扫描其余轨迹，取最先 5,000 条保存观察数至少为 6 的轨迹为 check。资格过滤不会补入额外轨迹。两个集合严格不交。

每条 check 的观察按 `ba_track_holdout_v1:obs:{track_id}:{image_id}` 的 SHA-256 排序，散列相同时按整数 image ID 排序；前 `ceil(2n/3)` 为 support，剩余至少 2 条为 score。support 单独在原位姿下 DLT 三角化；原位姿下得到有限点、所有该 track 观察具有正深度、support 的点到相机方向最大夹角至少 0.5°，才进入固定主要分母。这些条件在候选 BA 求解前计算。该分母不使用修正后误差做筛选。

原数据已按全条 TRAIN 轨迹重投影筛选，提供的标定/对应又来自上游全部 400 视图 SfM。因此这里的留出只针对本轮拟合和重三角化，是**有条件的额外一致性检查**，不是从原始数据开始完全独立的 OOF，也不是相机物理真值。

## 修正与评分

把恰好 500 条 fit 构成 `TriangulatedCloud` 交给现有 `bounded_bundle_adjustment`，源码不改，参数固定 `max_points=500, max_nfev=20, seed=42`。原有两锚相机、固定内参、旋转/平移/点移动边界、soft-L1 与先验全部保留。20 是最大函数求值次数，不声称 20 次迭代或收敛；单独报告 solver success/message/nfev。原 helper 根据 fit 重投影 RMS 接受位姿，此接受不是留出检验通过。

每条 check 的 support 分别在原位姿和候选位姿重新 DLT 三角化，再各自预测 score 观察。误差使用连续的原 native 去畸变坐标：`undistorted_native_xy` 和 source K，单位原像素；不混入 distorted raw xy 或 legacy 整数采样。score 不参与两次三角化、BA 或接受决定。候选支持点非有限，或该轨迹任意观察深度非正/投影非有限，则固定分母失效，主要量标为不可定义且继续门失败，不能删点求漂亮均值。候选 parallax 降至 0.5°以下单独报告，不新增筛选。

主要误差为每条轨迹 score 观察径向误差的均值，再等轨迹平均。保存所有 baseline/candidate 点、预测、残差、深度、布尔资格与布局，供独立实现复算。报告径向均值、中位数、RMS、p95、每相机和四个相机名组；四组按全部 350 TRAIN 相机名字排序，`group=floor(rank*4/350)`。组统计也先取该轨迹在本组 score 观察的均值，再等轨迹平均，不与主要量混用观察权重。报告实际被优化相机覆盖，不能把评分相机覆盖当优化相机覆盖。

配对区间：从固定 eligible tracks 有放回抽样 2,000 次，NumPy RNG seed 20260927，对 candidate−baseline 的等轨迹误差求 2.5/97.5 分位数。相机和几何共享依赖仍存在，不能把此 track 区间当独立多桥重复试验。

## 固定继续条件

- 覆盖：至少 1,000 条基准 eligible tracks，至少 200 个各有不少于 5 条 score 观察的相机，四个相机名组各至少 100 条 score 观察。
- 修正：原 BA 接受、所有基准 eligible tracks 的候选仍可评分。
- 信号：主要误差相对下降至少 1%，candidate−baseline 的配对区间上界小于零，四组至少三组的等轨迹误差点估计下降。

所有条件满足只允许考虑另一个预先定义、匹配成本的固定相机/传统修正训练比较。没有直接采用、RGB/语义提分、profile 放行或创新声明。未通过则停止这次固定 500 点/20 求值修正，不扩大点数、换抽样、改阈值救援；结果不能证明所有 BA 方法无效。

## 执行与来源

使用现有 uv 环境，固定 NumPy/SciPy/OpenCV/Pillow 版本和三项 CPU 线程变量为 1。冻结 runner、核心、原 geometry/data/coordinates、测试、协议和独立 checker。计划绑定既有 manifest、init、观察 NPZ、原诊断及独立复核回执、uv.lock 的 SHA-256。计划生成不载入 NPZ 数组。

CPU 内部截止 600 秒，外部截止 900 秒；只有一次执行，先独占创建 `execution_started.json`，超时/错误留回执、不重启。输出位于新的 `/mnt/data/SHM2026/runs/ba_track_holdout_v1`。先保存布局和全部输入几何数组，再 BA、留出评分和统计；执行后复核所有源码/输入哈希。独立 checker 不调用实验核心或重跑优化，独立实现 DLT、布局、误差、bootstrap 和判定，并核验实际位姿、固定锚点与边界。
