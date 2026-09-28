# 数据、相机与验证协议审计

## 实际输入

已检查本地 `Dataset`，不是按 README 示例文件名假设输入：

| 内容 | 实际结果 |
| --- | --- |
| 已标注照片 | `Dataset/images` 中 300 张 PNG |
| 未标注照片 | `Dataset/unlabeled_Images` 中 100 张 PNG，目录中的 `I` 大写 |
| 语义标注 | 300 个 LabelMe JSON，3811 个 polygon，无其他 shape 类型 |
| 原图和 JSON 尺寸 | 全部为 1320 × 989 |
| 类别 | 0 background、1 deck、2 stay_cable、3 tower、4 foundation |
| polygon 数量 | deck 656、stay_cable 1109、tower 542、foundation 1504 |
| 相机 | 400 个 image poses，共享 1 个 `SIMPLE_RADIAL` 相机 |
| 缺失的 SfM 文件 | 没有 `points3D.txt/bin`；`images.txt` 保留了有 track ID 的观测 |

共享内参为 `f=925.70161892457077, cx=660, cy=494.5, k1=0.008987863345268233`。
`images.txt` 中四元数按 `(qw,qx,qy,qz)` 读取，平移为 **camera_from_world**，不是相机中心。
相机中心计算为 `C = -Rᵀt`；坐标约定为 OpenCV 的右、下、前。

LabelMe 按原始 polygon 绘制顺序栅格化，后画的区域覆盖前画的区域，不进行类别优先级重排。
stay_cable 是官方 polygon 区域标签；不能未经论证把它变成单条缆索中心线真值。

## 从同学探索中吸取的经验

补充核查：资料还包含 SEM-384/386 的 **270训练/30留出 Dev30** 实验，不能将同学全部结果概括为训练拟合。SEM-386 留出五类mIoU为93.7418%、前景mIoU为92.3300%，数值高于当前模型；划分仍不同，尚不能直接算方法增益。详见[完整对照](peer_comparison.md)。

`project_progress_20260918` 是离线结果展示包，含图片、视频和 HTML 实验记录；README 明确说不含 checkpoint。
因此没有把缺失的训练代码或模型当成可以恢复的基线。

- `RGB148 + SEM380` 的 95.306% 五类 mIoU 是 **全 300 标签拟合**，RGB 也使用了全 400 照片。它不能当作新视角验证成绩。
- 历史 74 图经历多次开发审计，且与 RGB/相机流程并非整条链的独立盲测。本项目新建验证划分，不对这些分数宣称公平超越。
- 同学的 DINOv3 实验是冻结 **ViT-B，85,660,416 参数**，不是此次要求的高容量版本。提高 SFP 学习率至 `1e-4` 有收益；加入空间先验后仍低于同期 Swin-L 全量拟合结果。只替换骨干不能保证更好。
- RGB148 采用桥梁结构类别并集、原生 4 px 膨胀、额外区域 RGB 监督；结构区域很小，全图损失容易由背景主导。
- RGB140 的大高斯 split 用 AbsGS 的绝对屏幕梯度，阈值 `0.0004`；小高斯 clone 保留 signed 梯度 `0.0002`。细结构增密有依据，但应同时报告点数预算。
- 展示包清楚区分：评价图片采用原生 PINHOLE 网格，参考轨迹视频采用官方畸变网格。这里也显式保存网格与原相机参数，避免比较不同像素位置。

原始证据：`project_progress_20260918/data/content.json`、`README.txt`，以及
`sources/RGB-148-STANDARD-GT-REGION-RGB.html`、`sources/RGB-140-STANDARD-ABSGRAD-ALL400.html`、
`sources/SEM-383-DINOV3-VITB-SPATIAL-SFP-ALL300.html`。

## 准备流程与不泄漏边界

`uv run python -m bridge_rgs.prepare --dataset Dataset --output artifacts/prepared --max-width 1320 --max-points 60000`

1. **先划分相机，再读图像内容与构建轨迹。** 相机中心 PCA 主轴排序，默认每 8 个相机留出一个；随机种子仅确定确定性的偏移。标签和 RGB 不参与划分。
2. 原图按 `SIMPLE_RADIAL` 去畸变至相同原生 PINHOLE 内参，保留完整大小，可按目标宽度等比例缩放。RGB 双线性插值；语义最近邻插值。
3. 保存有效采样区域。边界外的语义存为 255，加载数据时按有效图设为 ignore；未标注照片保存 `mask_path: null`，不伪造背景标注。
4. 从 **train views** 提取已有 track 的二维观测，用 OpenCV 去畸变为归一化坐标；任何 val 观测均不进入当前三角化、BA、颜色和语义点初始化。
5. 多视角 DLT，必要时随机双视角假设取得一致内点，再重拟合；要求至少 3 个训练视角、正深度、至少 0.5° 视差、每个保留观测的原生重投影误差不超过 3 px。长轨迹最多取 32 个训练观测；随机打乱 track 后限制总点数，避免只取 SfM 编号靠前的区域。
6. 在训练照片上采样中位数点颜色、训练标签类别计数。保存每个点的观测图像 ID 和 offsets，能直接复核是否含有验证图像。
7. 保存原始 `w2c_original`、使用的 `w2c`、原始 camera model 和参数；后续可以原样转换到官方请求的相机。

默认 seed=42 得到：259 张有标签训练图 + 91 张无标签训练图，41 张有标签验证图 + 9 张无标签验证图。
split SHA-256 为 `f24cf4d859dd…`，完整值在 manifest/audit 中。语义评价应使用 41 个有标签验证视角；RGB 评价可覆盖全部 50 个验证视角。
它衡量**相机轨迹内插的新视角表现**，不是空间外推或跨桥泛化。`--val-every 0` 明确表示全量最终拟合，不能输出“held-out”成绩。

一个不能消除的上游条件：官方 COLMAP 内参、姿态和 track 关联本来是用全部发布照片估计的。
因此严格表述是“在提供的相机标定条件下，图像与标签监督留出验证”；不能称为从训练照片开始重建 SfM 的全链条归纳盲测。
若要后者，必须额外对训练照片从头匹配与 SfM，并用仅基于训练地图的定位为验证相机定姿。

## 可选离线 BA 与不确定性

`--bundle-adjustment --ba-max-points 500 --ba-max-nfev 20` 启用带稀疏 Jacobian 的 SciPy robust BA。
固定两个相距较远的原始训练相机以约束相似变换 gauge，共享内参保持固定。只优化选中点与其训练相机，旋转和平移有边界及保守先验。
仅当训练重投影 RMS 不变差才采用结果，然后用新相机重新三角化全部初始化点；不按验证表现选择相机。
默认实际产物关闭 BA，以便先建立可复现控制组。

点协方差是固定相机的条件 Gauss–Newton 近似；相机协方差是固定点的条件 Gauss–Newton 近似，带先验并以训练残差设置噪声量级。
验证相机只有先验，不计算验证残差。**这些是质量代理量，不是经过覆盖率校准的完整 SfM 后验**：它们忽略相机/点相关、共享内参相关和系统误差。

manifest 的 `pose_covariance` 顺序为 `[tx,ty,tz,rx,ry,rz]`，对应左乘 camera-frame SE(3) 微扰。
每个 view 和 manifest 顶层都有 `pose_covariance_order`。几何内部 audit 记录内部旋转优先顺序，导出时显式重排矩阵两个轴。

## 文件接口

- `manifest.json`：`views` 含 `name/image_id/image_path/mask_path/valid_path/width/height/K/w2c/w2c_original/split/camera_id/pose_covariance`，路径为绝对路径。
- `init_points.npz`：`points[N,3]`、`colors[N,3]`（RGB，[0,1]）、`covariances[N,3,3]`、`track_ids`、`reprojection_error`、`num_observations`、`observation_image_ids`、`observation_offsets`、`semantic_counts[N,5]`、`scene_radius`。
- `geometry_audit.json`：划分、原生重投影误差、BA 设置、协方差含义和验证访问审计。
- `BridgeDataset` 延迟读取指定 split；`image` 是 CHW float、`mask` 是 HW int64（缺失/无效为 -1）、`valid` 是 HW bool。

已运行的数据单元/集成测试包括：COLMAP 空观测行与四元数约定、将验证观测全部改成 NaN 后点云严格不变、训练误匹配剔除、确定性相机划分、去畸变中心与 mask 覆盖顺序、BA 固定 gauge、完整小型数据准备与泄漏凭据。
`uv run python -m pytest tests/test_data.py -q`：7 passed。

实际 660 px smoke 数据保存 50,000 个点，训练原生重投影误差 mean=0.478 px、median=0.327 px、p95=1.438 px。
这些是初始化几何的训练残差，不是 RGB/语义重建比赛指标。
