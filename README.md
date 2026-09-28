# Bridge-RGS：桥梁 RGB 与语义联合高斯重建

本项目实现 IC-SHM 2026 Project 2 的相机输入 → RGB／官方类别 ID 渲染流程。语义教师使用 **ModelScope DINOv3 ViT-H+/16**；项目 Python 环境由 **uv** 和 `uv.lock` 管理。当前交付入口是下面的工程版本 F，原单场共享几何流程保留供研究和历史复现。

## 当前可用版本 F

**最新可用工程版本 F：PSNR 30.236593 dB / SSIM .875024123 / LPIPS .259300123 / mIoU 95.109016%。**top4_normalized替换前一版本E的AA成员，与原MCMC各半，语义保留H3＋DINOv3。52个相机实际重渲染、50/41原图重新评分及独立CPU复核通过。相对本地同协议SEM386-style Swin-L参考，PSNR **+.835644 dB**、mIoU **+.790633pp**，配对95%区间分别为[+.564120,+1.139829]、[+.136189,+1.653615]；SSIM／LPIPS也改善。这是不同训练历史和推理增强的完整系统比较，不能称同学Dev30的精确复现或同预算优势。

F仍需要 **1,994,145个高斯、三套独立场和DINOv3 H+教师**，另需66,095参数RGB头、350张TRAIN来源照片和21.93GB层缓存；不查询目标照片或既有预测PNG。固定官方传感器的改名和位姿偏移均已实际验证。相对E的PSNR再提高.200932dB，语义混淆矩阵完全不变。完整指标、成本与失败记录见[新联合结果](docs/ibgs_joint_bundle_results.md)。组合是工程措施；四角消融支持归一化条件下的层选择收益，独立学术贡献仍需进一步验证。

### 直接导出新相机

使用公开源码中的通用渲染入口：

```bash
uv run bridge-rgs render \
  --checkpoint checkpoints/bridge_f_v1/semantic_or_rgb.pt \
  --cameras /absolute/path/to/cameras.json \
  --output /absolute/path/to/new_output_directory
```

`--output`必须尚不存在。F 版本的多场部署资产、组件哈希和获取说明见
[F release manifest](release/bridge_f_v1/README.md)；大型训练检查点不进入普通 Git
历史。相机 JSON 使用官方坐标系的`name,K,w2c,width,height,distortion`。

| 目的 | 入口 |
|---|---|
| 当前 F 的 camera-only RGB／mask 交付 | 上面的冻结导出命令与[部署说明](docs/ibgs_joint_deployment.md) |
| 当前 F 的 50／41 同协议结果、成本与审计 | [新联合结果](docs/ibgs_joint_bundle_results.md) |
| 前一版本 E 及教师对照 | [多场结果](docs/multifield_h3_teacher_results.md)、[旧E导出](docs/multifield_deployment.md) |
| 单场模型渲染、训练、评价 | 下文标注 **legacy** 的 CLI；不是 E 的重训命令 |
| warm-IBGS 工程参考与稳定 AA 训练 | [原端口结果](docs/ibgs_warm_matched_results.md)、[稳定 AA 结果](docs/ibgs_aa_stable_matched_results.md)；使用独立 uv 环境 |
| 历史结论与逐次执行记录 | [实验历史](docs/experiment_history_20260927.md)、[优化日志](docs/optimization_log.md) |

## 当前实验状态

**固定RGB成员替换和完整联合交付已通过，采用工程版本F。**相对E的PSNR+.200932dB、配对95%区间[.111521,.307454]，SSIM/LPIPS也改善；50张新mask与E精确相同，并已从41份原JSON重新评分。见[先行RGB实验](docs/ibgs_member_replacement_results.md)、[完整联合结果](docs/ibgs_joint_bundle_results.md)。以下为先前实验的当时结论，均保留负面结果与选择历史。

**归一化对应关系控制已完成。**正确 top4_normalized 相对匹配训练的特征置换控制提高 **.045201 dB PSNR**（配对95%区间[.020287,.068611]）和 **.000828 SSIM**（[.000615,.001070]），LPIPS差异−.000191且区间跨零。控制在训练和评价都执行，实际改变约86.2%–99.3%的有效特征向量；不替换F，也不将该有限机制证据写成首次创新。见[控制结果](docs/ibgs_correspondence_results.md)。

**继续优化的两个候选均未改写 F 默认。**为 F 建立 350 张 TRAIN 渲染域缓存并各训练 2000 步的 selected/composite DINOv3 教师，50/41 评价 mIoU 分别为 94.415452%/94.414633%，均低于 F 的 95.109016%，详见[渲染域教师结果](docs/f_domain_teacher_adaptation_results.md)。固定成员的 α=.6 RGB 加权候选为 30.252777/.875538013/.257947664，SSIM/LPIPS 区间改善但 PSNR 工程门失败；权重又是在同一开发集探索，不能当盲测提升，详见[加权候选结果](docs/f_rgb_weight_candidate_results.md)。

**与同学目录的全量对照已补齐，但不替代留出门。**在 300 个有标注视角上，F 的全量对照为 PSNR 31.095728 dB / SSIM .876151 / mIoU 96.590043%；相对 RGB148 + SEM380 的 23.305737 dB / .896586 / 95.306455%，PSNR 和 mIoU 更高、SSIM 更低，且该表未计算 LPIPS。协议、回执和边界见[对照报告](docs/project_progress_comparison.md)。

**已完成的 warm-IBGS 尚未替换 E。**两臂各 6000 步，预定 full/fused 在共同 50 视角原图上为 **29.846757 dB / .872022326 SSIM / .250069213 LPIPS**。相对 no-source 控制，PSNR **+.510873 dB**，配对 95% 区间 [+.433969,+.589825]，SSIM／LPIPS也改善；相对 E，PSNR −.188904 dB、SSIM −.002398410、LPIPS −.014672632，未满足整体采用条件。它是已知 IBGS 的工程端口变体，本轮没有新的语义结果。见[结果与成本](docs/ibgs_warm_matched_results.md)、[端口验证](docs/ibgs_validation_results.md)。

**稳定 AA 两臂训练和评价已完成，仍保留 E。**两臂各6000步，full/fused 在共同50视角为 **29.825302 dB / .875525202 SSIM / .252678474 LPIPS**。相对 no-source 控制 PSNR +.258703 dB，配对95%区间[+.202423,+.319489]；相对原AA 1M三个RGB指标均改善。但相对E，PSNR −.210359 dB（区间[−.467715,+.045332]），SSIM +.001104466、LPIPS −.012063372，未达到预先规定的整体采用条件。这是已知IBGS及标准AA的工程修订，无新语义结果或学术创新。见[稳定AA结果](docs/ibgs_aa_stable_matched_results.md)。

**固定 E＋稳定 AA 各半组合也未采用。**50视角为30.079030 dB/.876132687/.257035658，相对E仅+.043369 dB，95%区间[−.092072,+.179029]，通过2/4门。保留SSIM与LPIPS改善事实，但未达到明显整体提升，且增加场和源照片缓存成本。见[固定组合结果](docs/ibgs_e_fixed_blend_results.md)。

**四组分层对照已完成：归一化条件下层选择有收益，整体尚未替换E。**各6000步后，新增median4_normalized为29.806118dB；top4_normalized为29.933965dB，相对它提高 **.127846dB，95%区间[.095144,.164561]**，SSIM与LPIPS也改善。交互说明该选择效应依赖归一化。原top4_mass主假设仍失败，不能追认为绝对质量或细索收益；同ID来源支持的独立价值也尚未隔离。新median4相对E采用门仅1/4通过，top4归一化也仍低于E的PSNR/SSIM。见[四组结果与后续决定](docs/ibgs_factorial_results.md)、[原三臂及41图类别RGB诊断](docs/ibgs_layer_heads_results.md)。

首版AA在61次更新后因FP32行列式消减失败，现由不加floor/clip的FP64因子表达修复；原失败保持可追溯。修复不改变后端已知近似VJP，不能当作完整梯度认证。见[修复验证](docs/ibgs_aa_stable_revision_results.md)、[失败诊断](docs/ibgs_aa_determinant_diagnostic.md)。固定TRAIN射线及[对应来源查询](docs/ibgs_layer_sources_results.md)、新CUDA选择器检查和350个源层缓存均已完成，协议与失败记录保留。

三臂的[固定评价](docs/ibgs_layer_evaluation_protocol.md)和[41个已标注视角的类别RGB误差诊断](docs/ibgs_rgb_class_diagnostic_protocol.md)已自然完成。评价入口12项、类别统计7项CPU检查通过；独立重算保存均值与配对统计最大差0，类别SSE重建逐图PSNR最大差7.11e−15dB。类别RGB误差不等同语义mIoU。

已关闭的 ED 传输、几何路径、结构交换、语义分区、教师适配和容量对照原文已移至[实验历史](docs/experiment_history_20260927.md)，其中保留全部数字与审计链接。早期 native 指标另见[实测汇总](docs/measured_results.md)，不能与上述共同原图指标直接相减。

## 已有单场模型入口（legacy）

使用已有单场检查点：

```bash
uv run bridge-rgs render --checkpoint runs/support_split_semantic_coupled/last.pt \
  --cameras Dataset/camera_parameters --output runs/render_demo --grid official --max-views 3
```

运行 legacy 同场 H3＋DINOv3 组合（不包含 E 的双场 RGB）：

```bash
uv run bridge-rgs render --checkpoint runs/h3_moments/02_cross/last.pt \
  --teacher-checkpoint runs/teacher_render_adapt_v1/best.pt \
  --cameras Dataset/camera_parameters --output runs/render_teacher_combo --grid official
```

该接口先从同一场渲染RGB，再执行固定滑窗/翻转/全图教师预测及0.5概率平均，最后映射回官方畸变网格。没有输入测试照片；回执记录两模型、骨干权重及训练/推理渲染场的来源。新接口在原生001相机上逐像素复现已测组合的RGB和mask，见[复现回执](runs/render_ensemble_smoke/reproduction.json)。95.023%属于完整原生开发验证协议，不能直接当作官方畸变网格或未知测试集成绩。

## 单场研究链路（legacy）

```text
官方相机 + train 图像轨迹 → 去畸变 / 训练专用三角化 / 可选受限 BA
                                          ↓
ModelScope DINOv3 H+ → DPT 式多层解码器 → 软语义概率
       冻结编码器       RGB细节分支 + EMA       ↓
                           投影协方差 / 遮挡 / 类别一致性融合
                                          ↓
                共享几何高斯：RGB球谐 + 三维语义属性 + 有界精修
                                          ↓
            归一化AbsGrad + 相机补偿剩余误差 → 预算分配 → 克隆/结构分裂
                                          ↓
           测试(K,T,H,W) → RGB PNG + 0..4语义 PNG + 语义Gaussian PLY
```

- **高容量教师**：冻结 840,592,640 参数的 DINOv3 ViT-H+/16；不是把已有 ViT-B 换名。固定 ModelScope 仓库提交和权重 SHA-256。新解码器使用 `1e-4` 学习率、多层特征、原生裁剪、RGB细节、边界损失与弱/强光度一致性。详见 [教师说明](docs/teacher.md)。
- **独立三维语义**：每个高斯有 16 维语义特征，由共享分类器转成五类概率，按与 RGB 相同的遮挡关系合成。无需读取测试真实照片。精修头的 logit 残差有界，RGB/深度输入停止梯度；三维语义与精修结果分别监督。
- **投影可靠性**：相机和点的位置估计协方差传播到像素椭圆；结合教师熵、遮挡与跨视角一致性形成软监督。Gaussian 空间形状不充当位置误差协方差。几何梯度还需通过多视角可靠性、RGB边缘和不透明度门控。
- **相机补偿残差增密**：固定场景，在低分辨率对六维相机增量构建有限差分 Jacobian，求受限带先验的最小二乘修正，重新渲染核验。hybrid模式结合逐帧AbsGrad和受不同视图支持的剩余误差排序；未经残差诊断的点仍可由梯度路径入选。PCA推断局部切向，统一预算下执行小点克隆/大点分裂、剪枝与Adam状态迁移。轻微位姿扰动配对暂未证实归因收益，详见优化记录。
- **相机更新默认关闭**：相机诊断用于残差归因。受限离线 BA、受限光度相机更新分别可启用，只有原始验证相机表现改善时才纳入最终配置。

实现范围和数学定义见 [方法说明](docs/method.md)。这里的协方差是条件近似与质量代理，尚不是经过覆盖率校准的完整 BA 后验；残差诊断也是局部近似。核心新机制需要消融支持，DINOv3、DPT、EMA、gsplat、绝对梯度本身属于已有方法。

## 环境与数据

当前已在 RTX 5090 32GB、CUDA Toolkit 12.8 上运行；PyTorch 2.8.0+cu128、gsplat 1.5.3、Transformers 4.57.6 记录在锁文件。第一次渲染会编译 gsplat CUDA 扩展。CPU 可运行数据和数学单元测试，3D训练需要 NVIDIA GPU。

```bash
cd /path/to/bridge-rgs
uv sync --python 3.11
uv run bridge-rgs prepare --dataset Dataset --output artifacts/prepared \
  --max-width 1320 --max-points 60000 --pixel-protocol legacy_mixed_v1
uv run python scripts/download_dinov3.py
```

数据不被修改。真实目录为 `Dataset/images`、`Dataset/json` 和 `Dataset/unlabeled_Images`。

上面的legacy参数用于复现本文既有实验。新的独立数据准备默认使用修正像素中心的`colmap_corner_v2`，应写到新目录，例如`--output artifacts/prepared_corner_v2 --pixel-protocol colmap_corner_v2`；它不会覆盖已有数据目录。新的训练配置需同时指定该manifest、新output/pseudo/teacher目录及`pixel_protocol: colmap_corner_v2`。旧检查点保留原导出约定，跨协议warmstart/评分会被拒绝。新v2数据已准备并核验，旧705个文件保持原样；独立两阶段训练与[共同原图网格对照](docs/corner_v2_official_comparison.md)均已完成，未支持整体提升。两套native协议仍不能混报性能。见[独立准备](docs/corner_v2_preparation.md)、[像素审计](docs/pixel_coordinate_audit.md)及[迁移设计](docs/pixel_coordinate_migration_plan.md)。
官方唯一相机为 `SIMPLE_RADIAL`，1320×989。没有 points3D 文件，因此由 `images.txt` 中已有的训练视图 track 观测重新三角化。

默认划分为 350 个训练视角（259有标签+91无标签）与50个验证视角（41有标签+9无标签）。这是按相机中心排序的轨迹内插验证。验证像素不参与新三角化、BA、点颜色/语义初始化、教师训练或高斯训练。
官方相机/轨迹关联在发布前已经用全部照片估计，因此这是**给定官方标定条件下的监督留出验证**，不声称整条 SfM 流程完全独立。详细统计和同学实验中的经验见 [数据审计](docs/data_audit.md)。

## 单场研究训练（legacy）

以下流水线用于原单场研究配置，不会直接重建 E 的三套已选模型。顺序执行全流程，支持失败后显式恢复：

```bash
uv run python scripts/run_pipeline.py --resume
```

或者分阶段运行：

```bash
uv run python -m bridge_rgs.teacher train \
  --manifest artifacts/prepared/manifest.json \
  --model-dir models/dinov3-vith16plus --output runs/teacher \
  --steps 6000 --crop-size 768 --lr 0.0001

uv run python -m bridge_rgs.teacher predict \
  --manifest artifacts/prepared/manifest.json \
  --checkpoint runs/teacher/best.pt --output artifacts/pseudo

uv run bridge-rgs train --config configs/bridge_rgs.yaml
uv run bridge-rgs train --config configs/bridge_rgs.yaml --resume runs/bridge_rgs/last.pt
```

主配置先 RGB 预热，3000步接入语义，4000步开始归因增密，6000步接入精修；22000步后冻结点数，最终恢复全分辨率。全图 RGB 始终受监督，结构区域仅增加有限权重。相机不因语义误差移动。

软标签导出只覆盖 train，目录需完整且来源 SHA 与 manifest、教师检查点和类别顺序一致，3D训练会在开始时验证。不要把不同划分、不同去畸变网格或旧教师的 NPZ 混用。完整软概率缓存会占数 GB 磁盘，检查点也随高斯数量增长。

训练保存 `last.pt`、`train.jsonl`、配置与原始/优化训练相机，定期保存带完整验证指标的候选模型。主配置不伪造竞赛复合分数：规则给出了视觉/语义各占50%，但未提供 PSNR、SSIM、LPIPS 的具体归一化公式，应同时比较各项与成本。

## 单场评价与导出（legacy）

```bash
uv run bridge-rgs evaluate --checkpoint runs/bridge_rgs/last.pt \
  --manifest artifacts/prepared/manifest.json --output runs/evaluation --lpips

uv run bridge-rgs render --checkpoint runs/bridge_rgs/last.pt \
  --cameras /path/to/test_camera_parameters --output runs/submission --grid official

uv run bridge-rgs export-ply --checkpoint runs/bridge_rgs/last.pt \
  --output runs/bridge_semantic.ply
```

`render` 接受含 `cameras.txt/images.txt` 的 COLMAP text 目录，也接受 JSON `{"views": [...]}`。每个 JSON 视角只需 `name,K,w2c,width,height`；如需官方畸变网格，另提供 OpenCV `distortion` 系数或使用原始 COLMAP 目录。本项目 prepared manifest 自带 `source_cameras`，使用 `--grid official` 时会恢复原始尺寸、内参与畸变，即使训练准备图像经过缩小；`--grid pinhole` 保留 manifest 网格。负畸变需要的画面外射线采用扩大画布渲染后映射。`w2c` 始终为 camera-from-world，不能填 c2w。渲染路径不加载图像文件、不优化测试相机、不使用测试视图专属嵌入。

输出 `rgb/<name>.png` 与 `mask/<name>.png`。mask 为单通道 uint8：0背景、1deck、2stay_cable、3tower、4foundation；可视化颜色不写入提交mask。PLY 保留标准 Gaussian 参数，并带 `semantic_id/semantic_confidence/semantic_p_0..4`，可以读取真正的三维语义属性。

验证默认使用未经训练调整的原始相机，在去畸变 PINHOLE 网格计算 PSNR/SSIM/可选LPIPS；同时报告四个前景类别 mIoU、包含背景的五类 mIoU、每类 IoU、三维语义未经精修的 mIoU、点数和渲染耗时。图像无标签时仍计算RGB，跳过语义。LPIPS 首次使用会下载公开 AlexNet 权重。

开发选定配置之后，运行 `uv run python scripts/run_pipeline.py --all-data --resume` 可重新准备全400 RGB / 全300标签最终拟合。它在 `prepared_all/pseudo_all/teacher_all/bridge_rgs_all` 等独立目录重新训练，教师使用最终 `last.pt`，关闭验证评分；不能沿用留出验证分数作为全量模型的新结果。可用 `--gaussian-config` 指定配置，流水线按配置中的 manifest、pseudo 和输出路径组织各阶段。

## 单场测试与消融（legacy）

```bash
uv run pytest -q
uv run bridge-rgs train --config configs/smoke.yaml --output runs/smoke_new
uv run python scripts/make_ablations.py
uv run bridge-rgs train --config configs/ablations/06_full_method.yaml
uv run python scripts/summarize_results.py runs/ablations --output runs/results.csv
```

消融配置固定数据划分、教师来源和高斯上限，逐步比较普通伪标签、点采样融合、投影不确定性、相机补偿残差、结构分裂、相机更新与精修。`absgrad` 对照使用绝对屏幕梯度来选点、共享本项目预算控制器，**不是官方 AbsGS 完整复刻**。相同最大点数不代表实际点数/耗时完全一致，论文仍需匹配实际预算，并报告多种随机种子。

另建议受控相机扰动与几何欠表达实验，检验误差归因是否按预期变化；数学单元测试覆盖了线性可解释残差和不可解释残差的分离，不等价于真实桥梁上的完整实证。

## 来源

- [DINOv3](https://github.com/facebookresearch/dinov3) 与 [ModelScope ViT-H+/16](https://modelscope.cn/models/facebook/dinov3-vith16plus-pretrain-lvd1689m)。权重原始许可保存在 `models/dinov3-vith16plus/LICENSE.md`。
- [gsplat](https://docs.gsplat.studio/) 提供抗锯齿、多通道和可微 Gaussian 渲染。
- [AbsGS](https://arxiv.org/abs/2404.10484) 提供绝对屏幕梯度相关思路。
- [SPARF](https://github.com/google-research/sparf) 是相机与辐射场优化的相关工作，不能将“优化相机”本身当作新贡献。
- [UniMatch V2](https://github.com/LiheYoung/UniMatch-V2) 是基于强视觉骨干的半监督分割参考；本实现采用自己的冻结 DINOv3 + EMA 解码器流程，并未声称复刻其完整训练配方。
- `SHM_2026.pdf`、`Dataset/README.md` 是本地竞赛依据；`project_progress_20260918` 是同学提供的已有实验资料。
