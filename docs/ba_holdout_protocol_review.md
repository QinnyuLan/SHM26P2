# 受限 BA 留出对应诊断：独立协议评审

2026-09-27。仅审查拟定协议及 `geometry.py` / collector 源码；未打开真实观察数组、图像、标签或检查点，未执行 BA、训练或 GPU。

**结论：协议可作为一次条件性相机一致性筛查，没有原则性的 fit/check 泄漏或估计量阻断。** 它补的是“实际累计位姿改动能否帮助未参与 BA 的对应”，不是新算法、完整独立 SfM 验证或 RGB/语义采用证据。旧 H1 与 profile 数值门保持原结果，不借本实验重新解释为通过。

## 冻结前须明确的实现合同

1. **点与观察的分离。** 所有原轨迹按 `sha256('ba_track_holdout_v1:track:{trackid}')` 排序，碰撞以整数 track ID 排序；首 500 条为 fit。然后在余下轨迹中筛选长度至少 6，按同一顺序取首 5,000 条 check，不因之后的 eligibility 失败补样。原 BA 接收的 cloud 必须恰为这 500 条 fit tracks：其内部 seed42 随机选择在 500 条上选全部，不能重新从 60k 中抽 500。check 点的坐标、观测和残差均不能进入 BA 目标、锚点选取、scale、先验或接受判断。fit 相机可以同时含有 check 观察；这里留出的是对应，而非相机。
2. **检查点只从 support 重建。** 每条 check 的观察按给定 observation hash 排序，碰撞以 image ID 排序；support 取 `ceil(2n/3)`，其余至少两个为 score。两组无重复 `(track,image_id)`。baseline 和 candidate 都采用相同 support、相同固定 DLT，不能利用 score xy 改点、剔除异常或重选 support。保存原 XYZ 只作来源信息，不替代 support-only 的 X₀/X₁。
3. **像素单位必须一致。** 从缓存 `undistorted_native_xy` 和原 source camera K 恢复连续归一化坐标；评分是该原生去畸变坐标系的二维欧氏像素误差，不是每坐标 RMS、原始畸变 raw_xy 误差或 legacy 缩放网格误差。连续观测不做 floor、rint 或半像素平移，也不读取语义采样列。原 BA 的 residual 为乘 fx/fy 后的两个坐标分量；其训练 RMS 与本次平均径向误差名称要分开。
4. **固定人口和异常。** 先以 baseline support-only X₀ 的有限性、所有该轨迹观察相机下的正深度，以及 support 最大两射线夹角至少 0.5° 确定 eligibility；固定此集合及 score 分母后再运行 BA。candidate X₁ 有非有限值、投影非法或任一原观察非正深度，即记录失败，不能从分母丢弃、clip depth 后照常评分或补点。无法定义的总体误差/CI 应为 NA；`all_candidate_valid=false` 阻断正结论。candidate 视差下降可描述，不另设事后筛选。

原函数固定 fit 相机中两台相距较远的相机、内参不变；该规范和先验照旧。须保存 anchors、参与/实际优化相机、位姿改变量及未被 fit 覆盖的 scoring 相机数。拟定 coverage 不保证 200 台 scoring 相机均被优化，不能将静止相机的覆盖称为校正覆盖。`max_nfev=20` 是原求解器的函数求值预算，不是 20 次收敛迭代；同时报告 `success/message/nfev`，不临时增加预算。保持原 `accepted` 定义，不偷偷把 optimizer success 变成另一道门。

## 估计量、门和不确定性的解释

每条 eligible check track 先对 score 观察求平均径向 native-pixel 误差，再对 tracks 等权平均。差值为 candidate−baseline；相对下降为 `(mean_old−mean_new)/mean_old`，不是各轨迹百分比的均值。paired bootstrap 对这些轨迹共抽 2,000 次，seed20260927，区间同样针对平均差值；旧均值为零时相对量 NA，不能报成功。

四组按完整 350 TRAIN 名称排序后的 `floor(4i/350)` 划分，包含无标签相机。组内统计也先对每个 track 在该组的 score 观察求均值，再对有该组 score 的 tracks 等权。每组覆盖另报 score observations、tracks、scoring cameras；同一轨迹可出现在多个组，四组不是独立重复实验。

沿用拟定必要门：至少 1,000 eligible tracks；至少 200 台各有 ≥5 score observations 的 scoring 相机；四组各 ≥100 score observations。覆盖足够后，须同时 BA accepted、candidate 全有效、总体相对 MAE 下降 ≥1%、配对差值 CI 上界 <0、至少 3/4 组均值差为负。其余量仅描述，不用中间结果或组别替换主门。门是固定资源决策，不是 BA 新颖性、相机真值准确率或后续训练收益保证。

## 科学限制与执行后的分支

原 saved tracks 已按包含所有其 TRAIN 观察的误差筛选，且提供的 SfM 来自全部 400 照片。再划 support/score 不能消除这层选择；因此是**既有 SfM 条件下的对应留出筛查**，不能称干净 OOF 或独立相机标定。共同相机与轨迹网络导致 tracks 相关，track bootstrap 只是这个固定条件样本上的描述区间，并未传播相机、内参、匹配或完整 BA 后验不确定性。已知 duplicate-keypoint 的 last-observation 约定也仍保留，不能据小像素误差证明物理对应真值。

正结果只支持随后审议“固定相机 vs 标准 BA/交替优化”的匹配工程对照：support 重三角化同时改变检查点与相机，改善代表跨观测几何一致性，不单独识别哪一相机更接近真值。覆盖不足为 inconclusive；覆盖足但必要门未过则不推进此固定 BA 配方，不通过改点数、迭代、先验、锚点或像素阈值追结果。即使通过，也不自动允许新模型、语义损失、残差门控或 profile 训练。
