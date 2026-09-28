# TRAIN 稀疏射线前侧质量诊断

实现和短测可行，但当前 16/32 个全视图 log depth 阈值非常粗：32-bin 的上下质量区间中位宽仍为 **.6231**。现输出是严格向下选边的离散中心深度累计质量下界，不能称精确 ray CDF、真实空域错误量或已解决浮层。本节的初始诊断没有训练集外像素或参数更新。当时未接入训练损失；后续固定两臂工程控制已完成，见文末补充。

这是经典射线深度约束的工程诊断。DS-NeRF 已使用稀疏 SfM 深度与深度不确定性约束射线终止分布。[DS-NeRF 项目页](https://www.cs.cmu.edu/~dsnerf/)。URF 的式17直接惩罚测量表面前的终止权重，其消融指出孤立的空域损失可能损伤重建，需注意近表面约束的作用。[Urban Radiance Fields，§4.2/5.5](https://arxiv.org/html/2111.14643)。因此不把该组件包装为创新，也不预言继续优化必然有效。

## 渲染量与不等式

在现有 gsplat 的中心深度排序、投影 footprint 和 alpha-compositing 约定下，令每像素有效 opacity 为 a_i，透射率 T_i=∏_{j<i}(1−a_j)，终止权重 w_i=T_i a_i。给每个高斯的第 k 个特征颜色设为 `1[z_center_i < edge_k]`，背景设为零，则渲染特征为

`M(edge_k) = Σ_i w_i 1[z_center_i < edge_k]`。

它是未归一化质量，不除总 alpha。完整几何参与一次32通道渲染，不删除任何高斯、不按旧 opacity/early termination 预先缓存或裁减候选；使用 renderer 本身正常的 clipping、alpha 阈值和提前终止近似。gsplat 支持多维特征及按深度排序 alpha 合成。[gsplat 官方 API](https://docs.gsplat.studio/main/apis/rasterization.html)。本地调用复用 scene 的 antialiased 模式、near=.01/far=1e6。

SfM 观测目标 D 的条件位置深度方差为 `R_z Σ_position R_zᵀ`，不是 Gaussian shape 方差。定义

`b = D − max(3 σ_z, .02 D)`。

在该视图所有合格且 b>near 的目标中，先固定32个从最小到最大 b 的 log edges；16-bin 是32-bin的嵌套子集，保留两个端点。对每个目标选择最大的 `edge≤b`，所以 `M(edge_low)≤M(b)`。上邻 edge 只提供诊断上界，不上插、不作为训练目标。若处于首 edge 以下，下界为零；超过末 edge，上界为总 alpha；b≤near 是未检验的空自由段，不能算作已经证实没有浮层。

该不等式只针对同一 rasterizer 的**中心深度离散质量**。高斯有空间厚度，真实表面也可能由跨越多个深度的高斯表达；它不是 ray–ellipsoid 首次交点，也不是连续体密度 CDF。3σ 来自条件 SfM 位置协方差的启发式裕量，未校准为真实三倍标准差的覆盖概率。测得正质量不等于确定找到一个错误高斯。

几何、K、pose、深度门控、edge、SfM target/confidence 均 detach。opacity 经 sigmoid 和完整 transmittance 保留梯度；相机、位置、shape、SH、语义参数没有梯度。本次仅对第一视图计算一次梯度检查，**没有 optimizer step**。

## 坐标约定限制

COLMAP 的首像素中心是 (.5,.5)。[COLMAP 官方相机说明](https://colmap.github.io/cameras.html)。本地 `RasterizeToPixels3DGSFwd.cu` 同样使用 `(j+.5,i+.5)`。新 `sample_corner_pixel_features` 将角点原点 UV 映射到数组坐标 `UV−.5`，等价于 `grid_sample(align_corners=False)` 的 `2*UV/[W,H]−1`。CPU 首像素测试验证 (.5,.5) 恰好取数组[0,0]，不会再偏半个像素。

已有 `ray_support._sample` 把同一 UV 当作整数像素中心。本诊断没有修改已跑 ED 协议、fusion、K 或 prepared 数据：先保留已有 SfM/valid 质量筛选，再追加 corner sampling 的 eroded-valid footprint 检查。`cv2.initUndistortRectifyMap`、原 K、COLMAP 观测和图像数组的整体约定仍需要系统审查；不能仅补新 sampler 后声称全链完全统一。双线性采样本身也只是已渲染网格的近似，不是任意原始观测射线的精确积分。

## 固定输入与结果

冻结 `runs/strong_semantic_coupled/last.pt`，498,136 高斯。复用此前仅按 TRAIN 相机元数据固定的16个 anchor，全部属于 TRAIN；只读 SfM 原观测、位置 covariance、相机参数和 TRAIN valid mask，不读 RGB、语义 mask 或任何 VAL 像素。目标门槛沿用已审计 `SparseDepthSupport`：至少3视图、track reprojection RMS≤1px、当前观测误差≤2nativepx、条件相对深度std≤.1等。原始 observed UV 从 COLMAP 恢复，未以高斯中心反投影替代。

共 **19,956** 个目标均有有效 floor edge。全部 confidence 是质量权重而非正确率。总 alpha 的 min/median 分别 .2507/.99986；alpha<.5 的14个目标仍保留，不沿用 ED loss 的 min_alpha 门槛将它们丢弃。

| 量 | 16-bin | 32-bin |
|---|---:|---:|
| 质量权重加权前侧下界均值 | .16150 | .17728 |
| 前侧下界中位数 | .11079 | .12504 |
| 前侧下界90分位 | .36522 | .38901 |
| 下界>.01比例 | 93.776% | 95.069% |
| 下界>.1比例 | 53.763% | 58.870% |
| 上下质量区间宽度中位数 | .77592 | .62307 |
| 上下质量区间宽度90分位 | .94713 | .91097 |
| cutoff 到 floor 的相对深度间隔中位数 | 10.482% | 4.948% |
| 相对深度间隔90分位 | 18.526% | 8.971% |

每个目标均验证32-bin下界不低于16-bin、32-bin上界不高于16-bin，允许浮点容差；质量随 edge 单调且不超总 alpha。阈值裕量 `max(3σ,.02D)/D` 的中位数2%、90分位3.52%。相比这个裕量，bin间隔仍偏粗；相邻 edge 很容易跨过主要表面质量，使上界跃升，不能把区间中点当成可信估计。

第一视图159的加权32-bin下界为 .21341；opacity 梯度有限，norm=.008973，4987个非零项。其他参数没有梯度；结束后全部 scene 张量与起始场逐位一致。没有新的训练任务，也没有用这16视图结果选择验证阈值。

## 下一步边界

目前可以用它作为保守前侧质量的诊断量，不能直接根据约 .177 的均值判定17.7%都是漂浮物。若后续授权试训练，应先解释深度边离散误差、位置/表面厚度差、SfM离群和采样坐标约定；仅降低前侧 opacity 有损坏真实近表面或形成透明空洞的风险。RGB 配对控制不可省略，不能只以自身正则项降低为成功标准。本次未实现损失接入，也未扩展到 near-surface 学习或修改现有几何。

## 复现

```bash
# CPU：固定16 TRAIN相机、SfM目标、位置std、嵌套edges和输入SHA
CUDA_VISIBLE_DEVICES='' uv run python scripts/audit_ray_termination.py
# 短GPU：读取已固定plan，不改变任何参数
uv run python scripts/audit_ray_termination.py --measure
```

产物位于 `runs/ray_termination_diagnostic/`：`plan.json`、`targets.npz`、`mass_results.npz`、`report.json`、`summary.json`。11项CPU测试覆盖质量非归一化、遮挡合成、ED前后补偿反例、严格向下选边、嵌套细化、边界外区间、坐标半像素和梯度权限。

## 后续固定工程控制

初始诊断之后，已按固定32边方案实现可选 `sparse_front_weight`（默认0无额外渲染），并完成RGB+ED共同底座的opacity-only两组2000步。该后续工作及全50/41评测、同16 TRAIN目标复测完整记录在 [sparse_front_pair.md](sparse_front_pair.md)。front相对控制令PSNR下降.5464dB、索IoU下降.4098pp、final all5增加.2201pp；目标下界大幅下降但区间变宽，不能称浮层已解决。上述“本次未接入”的语句描述初始零训练诊断阶段，非当前代码能力。
