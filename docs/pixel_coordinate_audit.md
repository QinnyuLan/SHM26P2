# 相机像素中心约定：只读 CPU 审计

确定存在坐标接口不一致，但不能概括为“全部图像偏了半个像素”。旧 ED / 多视图融合在采样渲染图时，每轴偏 **+0.5像素**；prepare与官方导出则是未做两端坐标转换，其native零畸变误差相消。本数据实际native畸变映射误差中位仅约 **0.00284px**，最大约 **0.01475px**。直接制备660宽数据的缩放误差更大，必须另列。

本审计只写本报告、`artifacts/pixel_coordinate_audit.json` 和其CPU复现脚本。没有修改任何生产源代码、prepared文件、模型、训练配置或运行中的H3输入；没有启动GPU、读取VAL像素、删除文件或重新选择阈值。JSON记录源码/输入SHA、软件版本、所有数值、确定性断言与分级结论。

## 坐标合同与解析对照

COLMAP透视模型使用角点原点，数组左上像素的中心为 `(0.5,0.5)`。`SIMPLE_RADIAL`的公式为归一化射线乘 `1+k*r²` 后再乘焦距、加主点；现代码把k写入OpenCV的k1、其余畸变系数置零，**这部分正确**。[COLMAP官方相机说明](https://colmap.github.io/cameras.html)

OpenCV `remap` 的坐标是源数组索引，`(0,0)`取左上数组元素。`initUndistortRectifyMap`对目标整数数组网格反求源位置。[OpenCV变换文档](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html)、[去畸变公式](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)

本地gsplat 1.5.3投影保留输入K，rasterizer的像素位置为 `(j+0.5,i+0.5)`，所以传入COLMAP原K正确；不应全局将manifest/gsplat的主点减0.5。[官方对应版本CUDA源码](https://raw.githubusercontent.com/nerfstudio-project/gsplat/v1.5.3/gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu)

记 `p=(j,i)` 为数组坐标、`h=(.5,.5)`、`W`为角点坐标中的任意warp，则数组warp应为：

`W_array(p) = W(p+h) − h`。

prepare要调用OpenCV栅格接口时，可仅对临时输入/输出K的cx、cy各减.5；manifest仍保留corner K。一般误差不是常数，局部约为 `(J_W−I)h`。同分辨率零畸变时W为identity，误差为零；相同尺度的常量平移也完全相消。独立同伴只读审查确认了这个共轭推导。

相反，稀疏连续关键点 `cv2.undistortPoints(xy_colmap, K_colmap, d)` 只运算数值坐标，xy和K一致即可，**不用平白减.5**。将xy和K的主点同时减.5得到完全相同归一化射线。CPU真实TRAIN小样本验证最大差为0；三角化、其重投影RMS和条件位置协方差的这个接口没有半像素bug。已记录的全点重投影中位0.32674px不因本审计被改写。

## 定位与分类

| 接口 | 审计结论 | 影响范围 |
|---|---|---|
| [data.py:190](/home/sky/workspace/SHM2026/src/bridge_rgs/data.py:190) / prepare的cv2 maps | **确定约定bug**：栅格两端直接用了corner K，缺少上述共轭 | native为小畸变残差；直接低分辨率prepare还有缩放相位差 |
| [ray_support.py:93](/home/sky/workspace/SHM2026/src/bridge_rgs/ray_support.py:93) | **确定约定bug**：corner UV作为数组中心索引传入旧ED采样 | depth、alpha筛选、valid footprint每轴+.5px |
| [reliability.py:267](/home/sky/workspace/SHM2026/src/bridge_rgs/reliability.py:267) | **确定约定bug**：projection_jacobians输出corner UV，采样器却按整数数组中心处理 | fusion概率/置信度/depth/alpha、可见性与接受率；480网格的+.5px不能写成native+.5px |
| [prepare.py:73](/home/sky/workspace/SHM2026/src/bridge_rgs/prepare.py:73) | **确定约定bug**：corner UV直接rint，而最近数组像素应取UV−.5后的最近索引（非边界等价floor(UV)） | 初始化颜色、semantic_counts人工票；不会移动三角化点 |
| [evaluate.py:207](/home/sky/workspace/SHM2026/src/bridge_rgs/evaluate.py:207) 官方PNG导出 | **确定约定bug**：arange直接做undistort，未将目标中心+.5、源数组位置−.5 | 畸变warp；k=0直接输出路径正确；overscan平移本身不改变该推导 |
| [io.py:37](/home/sky/workspace/SHM2026/src/bridge_rgs/io.py:37) 缩放mask/valid | **确定的重采样相位差**：legacy nearest和RGB面积/线性半像素中心策略不同 | scale<1的边界监督及有效网格；scale=1无此项。实际指标影响未测，不将所有nearest选择都当数据损坏 |
| [tests/test_export.py:26](/home/sky/workspace/SHM2026/tests/test_export.py:26) | **确定测试合同缺口**：RayScene用j而非j+.5生成射线 | 现测试证明了legacy整数中心自洽，未验证与真实gsplat合同相同 |
| 新ray_termination采样 | 对renderer buffer的corner→index转换正确 | 仍继承旧ED target前置valid gate，之后再追加correct-corner gate；不等于已统一全链 |
| [export.py](/home/sky/workspace/SHM2026/src/bridge_rgs/export.py) PLY | 无像素重采样，不受此接口问题直接影响 | 已训练语义/几何可能含历史训练影响，不能由此声称3D内容绝对正确 |

`grid_sample(align_corners=False)`的corner UV应映射为 `2*UV/[W,H]−1`。旧fusion额外加.5，相当于采样UV+.5；旧ED虽用align_corners=True，其公式同样把UV当数组索引，错误等价。[PyTorch网格定义](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.grid_sample.html)

## CPU数值实验

全部误差为二维位移范数。prepare统计在两版本共同有效区域，export在全部官方输出位置；所有纠正版本只在内存构造，未写回prepared。正确OpenCV map与独立SIMPLE_RADIAL解析式在真实尺寸上的最大分量差为6.11e−5px，符合float32精度。

实际官方相机：1320×989，f=925.7016189246，cx=660，cy=494.5，k=.008987863345。

| 纯坐标实验 | 位移中位 | 最大 | 单位/说明 |
|---|---:|---:|---|
| native prepare：legacy vs正确共轭 | .002839 | .014751 | 原生源像素 |
| native官方export：legacy vs正确共轭 | .002843 | .014471 | 原生pinhole源数组像素 |
| 直接prepare660×494 | .713106 | .737130 | 原始1320×989源像素 |
| 同一prepare660差异乘目标scale | .356373 | .368380 | 660×494目标像素单位近似 |
| 合成k=0、native等尺寸 | 0 | <1e−14 | identity抵消，否定普遍半像素平移 |
| 合成k=0、精确半尺寸 | .707107 | .707107 | 源网格每轴−.5；目标每轴−.25 |
| 合成常量平移(+1.25,−.75) | 0 | 0 | 共轭与legacy相同；解析线性场remap误差0 |

固定畸变k=.1与−.12的合成相机同样产生随位置变化的残差，完整结果在JSON。它们只验证数学路径，不代表本数据畸变强度，也没有用于选择任何训练参数。原生导出的解析射线颜色测试中，包含OpenCV线性插值相位量化后最大等效误差约.03522px；它与纯坐标约定误差分开记录，不把插值误差都归为半像素bug。

8像素棋盘验证：本数据native prepare的两种map在共同有效区域，线性重采样平均差0.13658/255，最近邻类别差0.05779%；直接prepare660分别为24.7164/255与9.5103%。合成零畸变半尺寸棋盘恰好对齐棋盘块，两版本像素可相同，虽然解析坐标差非零；说明不能单靠某一棋盘视觉检查断言无偏差。

对存放在gsplat像素中心的解析线性场 `F(x,y)=1+x+2y`，正确采样3点为 `[2.5,12.4,21.5]`，旧ED/fusion为 `[4,13.9,23]`，正好多1.5，证明每轴+.5。1像素交替棋盘的两点正确值 `[0,1]` 被旧ED取成 `[.5,.5]`。此项是清晰可重复的接口bug，**不是根据验证分数猜测**。

## 有效网格与下采样

现prepared原生valid可由legacy map逐位重现，存储和代码未脱节。正确共轭下valid总数仍为1,290,806，但40个边界位置不同（两侧各20）。评分代码在RGB、mask、valid的同一prepared数组上评价，SSIM还腐蚀11×11支持；未发现评测阶段又单独施加额外.5偏移。它继承prepared的微小相机/栅格约定残差，不能称与官方原始畸变网格完全等价。

纯合成零畸变缩放还观察到map接近0的负浮点尾差被严格 `>=0` 判为无效，一列可被剔除。这是边界容差问题；当前真实畸变native的40个差异不应全部归因该合成尾差，也没有据此放宽现有评分mask。

native prepared后由`load_view`再缩小，是不同于“直接prepare660”的路径。corner K整行按实际宽高比缩放，与RGB像素面积中心兼容；不应在这个K缩放上再补一次主点偏移。不过mask和valid使用OpenCV legacy `INTER_NEAREST`，其采样相位不同。CPU索引坡度测试中，989→494时legacy和nearest-exact在全部494行选取不同源行；相对理想源footprint中心，legacy平均偏−1.0源像素。989→360为−1.37222源像素。nearest-exact平均偏差均约0。PyTorch官方也区分nearest与nearest-exact并说明与OpenCV legacy的关系。[插值文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.interpolate.html)

这不意味着mask被双线性混成新类别，而是边界取哪一侧的策略不同。旧progressive训练、低分辨率评测、fusion缩放valid可能受影响；当前fullnative H3和最终scale=1评测没有新增这个缩放项。H3 refiner内部统一buffer的卷积/插值不属于外部corner UV采样，无证据把本问题归因于moment通道。

## 小规模真实TRAIN检查

只取预先固定SfM射线计划中的TRAIN `014/159/240`，分别均匀取378/512/512个已有合格观测；不看VAL，不按结果换视图。用原prepared TRAIN RGB与mask比较两种采样规则，不更改任何prior或伪标签。

| TRAIN视图 | 观测数 | bilinear RGB平均绝对差（0–255） | legacy rint vs corner floor语义类别变化 |
|---|---:|---:|---:|
| 014 | 378 | 2.9125 | 3 |
| 159 | 512 | 4.0202 | 0 |
| 240 | 512 | 3.5830 | 0 |

共3/1402个最近邻类别改变；这批SfM观测偏向稳定纹理且不是随机全图边界样本，不能外推全部prior准确率，也不是3D真值校验。此结果不足以解释索扇形标签与物理表面之间的大差距，亦不支持推翻此前投票纯度/票数不足的总体审计。

## 对已有结论的边界

1. 同一prepared/native/fingerprint下的成对PSNR/SSIM/LPIPS/mIoU数值仍是该固定协议的有效观察，没有VAL泄漏新增。不能擅自把历史分数换成新网格再沿用同一实验名称。原生小畸变残差本身不足以证明235大浮层的成因；浮层跨越大量像素，因果归属仍需独立对照。
2. 所有使用旧ED的深度控制，以及H2/fusion和TRAIN投影jitter诊断，确实包含+.5网格像素的采样偏差。两臂共享偏差保持工程配对，但其作用可能与不确定性采样/阈值交互，故不能外推“纠正坐标后机制仍无效”。原有负结果应保留并注明legacy协议，不用本审计替负结果寻找保证翻盘的解释。
3. front的新增CDF采样已按corner约定；其共同ED底座仍旧约定。这削弱“全链严格同射线”的表述，但不抹掉固定两臂只增加front项、RGB损失和低alpha增加的实测取舍。此前“不推广front”结论维持。
4. 当前H3已锁相同native输入、关闭这些额外采样loss，可继续原定配对；不宜在运行中改map、prior或像素中心。直接prepare660 smoke与native训练不能仅通过宽高比例假设像素完全同源；progressive与fullnative比较本来也不隔离单一变量。
5. 官方PNG导出的现有测试使用错误的整数中心替身，需未来修订；PLY格式导出没有该重采样问题。对官方评分器具体栅格定义仍以赛方实现为准，本审计依据其提供的COLMAP模型及文件图像数组，不捏造未见的评分脚本。

## 建议的后续修正，尚未实施

优先在独立版本定义 `corner_colmap` / `array_index` 接口合同。manifest和gsplat K保持corner；OpenCV栅格map两端临时转array K；连续稀疏点与K一起使用原坐标。ED/fusion/custom sampler统一消费corner UV，nearest prior改按对应数组中心，边界/valid footprint一起审查，避免只减一处.5。

mask/valid缩放可另行显式采用nearest-exact策略，并与teacher/fusion一致；这属于训练协议变化，不能偷偷重写旧mask或旧实验输入。官方输出改为输出中心+.5、undistort、再减.5进入cv2源数组，保留overscan整数平移；对应CPU测试必须用gsplat的j+.5射线替身，覆盖identity、常量平移、畸变、缩放、首末像素。

任何实际修正应输出新的prepared版本/manifest SHA，保留legacy评分与已有checkpoint历史，用单独受控实验测影响。本审计不预测RGB/语义收益，不建议中断H3，也不把坐标修复包装为学术创新。

复现（只运行CPU审计，不修改生产文件）：

```bash
CUDA_VISIBLE_DEVICES='' uv run python artifacts/pixel_coordinate_audit_support/reproduce.py
```
