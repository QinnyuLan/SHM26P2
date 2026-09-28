# DINOv3 语义教师

2026-09-26 新增独立渲染域对照 `configs/teacher_render_adapt_v1.json`：从已选 head-only 教师 EMA 开始，继续训练 2000 步；H+ 保持冻结，259 张原始 train 标注监督，无未标注一致性。用独立种子预先打乱 **1000 次真实 RGB + 1000 次 strong_rgb/last.pt 渲染 RGB** 的图像来源序列，局部/全景训练概率保持 50/50。该对照是相机渲染 RGB→二维分割的域适配基线，不作为新的三维结构创新。

`scripts/render_teacher_training_views.py` 从原始相机参数渲染全部 350 个 train 视角；`scripts/prepare_teacher_render_domain.py` 另外从纯相机 JSON 渲染全部 50 个 val 视角。二者渲染时均不读取原图、GT 或 valid 像素。独立 `runs/teacher_render_adapt_v1/manifest.json` 只替换图像来源，原 manifest、原教师和 `artifacts/pseudo_strong_v1` 均保留。训练图用 `image_path_sources` 显式列出 real/rendered 路径和 SHA；val 只保留 rendered 来源。衍生清单、两份渲染 receipt 和 renderer checkpoint 均绑定 SHA，所有标签、valid、K、原始相机和 split 字段逐项与原清单核验，生成的 renderer mask 禁止作为教师标签。缓存以 RGB/mask/valid 路径为键，防止同一相机的真实和渲染图互相覆盖。

域适配每 500 步在全部 50 个 **渲染** RGB 输入上评估，只有其中 41 张标签参与分数计算；固定使用 tile768/stride512、水平翻转、0.75 局部 + 0.25 全景概率。验证照片不作为模型输入。结果目录含输入逐图 SHA、checkpoint/source snapshot、混淆矩阵、边界指标和 50 张预测。原教师的 strongRGB 渲染域分数保存在 `baseline_strongrgb.json`，适配后同协议结果单独保存；不可把真实照片上的教师分数当作相机重建结果。

该 2000 步对照已经完成（训练连同周期评估 376 秒），best 与 last 均为第 2000 步，实际来源精确为 1000 real / 1000 rendered。独立 reload 后混淆矩阵与训练最终评测完全一致。50 个新生成 camera-only RGB 的 SHA 与 `strong_rgb/evaluation_native` 原评测 RGB 逐图一致；比较脚本进一步核验了全部输入 SHA、相机集合、标签有效像素数和推理协议一致。

| strongRGB 相机渲染 → DINOv3 | all-5 mIoU | 前景 mIoU | 缆索 IoU | 基础 IoU | 缆索 2px F1 |
|---|---:|---:|---:|---:|---:|
| 原选教师 | 94.1741% | 92.8895% | 93.9859% | 88.1460% | 0.884330 |
| 2000 步混合域延长训练 | 94.3866% | 93.1552% | 93.9099% | 88.6298% | 0.884676 |

mIoU 只增加 **0.2125 个百分点**，缆索 IoU 下降 **0.0760 个百分点**，没有达到 95%。因此保留为小增益/负对照，未覆盖已有教师或伪标签。对照没有匹配额外 2000 步纯真实图延长训练，不能把全部差异归因于域适配本身。相机 RGB 重建保持 strongRGB 的 PSNR 30.2735 / SSIM 0.902818 / LPIPS 0.224494；本项二维后处理不改善三维重建。这些都是同一插值开发集上的结果，没有独立测试集结论。完整数据见 `runs/teacher_render_adapt_v1/comparison.json`，检查点和输入来源见 `experiment_protocol.json`、`provenance.json`。

默认教师采用冻结的 **DINOv3 ViT-H+/16（840,592,640 参数）**，从 ModelScope 的 `facebook/dinov3-vith16plus-pretrain-lvd1689m` 下载。它是高容量版本，不能仅凭参数量认定桥梁语义精度高于 ViT-L 或已有 Swin-L 基线。官方通用评测也没有显示 H+ 在所有分割指标上领先 L。桥梁性能需要独立验证集和消融证明。

## 下载与环境

所有 Python 命令通过项目的 uv 环境运行：

```bash
uv sync
uv run python scripts/download_dinov3.py --output models/dinov3-vith16plus
```

下载固定 ModelScope 提交 `98b7096b2938406ade58801a0bf75eca3198b5bd`，核对 3,362,432,800 字节 `model.safetensors` 的 SHA256：

```text
3e1d4d18b9bfa9f28fad8e9de6a783f1313532d3460efa4cd0b12521d81d1a4d
```

下载器保存原始 LICENSE、配置、处理器配置以及 `download_provenance.json`。仅使用 ModelScope 正常公开的下载接口；模型加载只读取本地权重，不执行仓库自定义代码。编码器默认 CUDA BF16，解码器参数及损失为 FP32；冻结编码器不进入优化器，始终保持 eval 模式。

## 教师结构和历史经验

`project_progress_20260918/sources/SEM-381-DINOV3-VITB-SFP-ALL300.html` 记录了简单替换冻结 ViT-B + SFP 后，训练视图 mIoU 和缆索边界 F1 均下降。SEM-382 将新适配器学习率从 1e-5 提高到 1e-4 后追回部分差距；SEM-383 加入 RGB 空间先验只得到小幅改善，仍未超过其 Swin-L 对照。这些数值来自训练集拟合，不能当成泛化性能。

本实现吸收三点经验：

1. 第 8、16、24、32 层 patch tokens 形成四级 DPT 式特征融合，显式去掉 CLS/register tokens。新适配器与解码器均使用 1e-4 学习率。
2. 原生分辨率裁剪、stride-four RGB 细节分支、门控融合及标注边界辅助损失保留局部信息。缆索类按真实标注的区域训练；不假设标注表示每根物理钢索，也不把 RGB 梯度直接当作语义边界。
3. 未标注 **train** 视图采用共享几何裁剪和翻转的弱/强光度增强：EMA 解码器输出软分布，学生通过颜色、亮度、对比度和噪声扰动学习相机外观鲁棒性。每类阈值仅由训练标签上 EMA 的正确预测估计，拒绝不可靠伪标签。

EMA 只复制解码器，编码器由两个分支共享，避免重复加载 0.84B 权重。这里的 DPT、EMA、边界损失本身均为已有思想；研究贡献应由整个几何一致的 RGB—语义联合方案和对照实验支撑，不能把组合直接宣称为已验证创新。

## 训练

输入为准备后的 `manifest.json`，其中图片、类别索引 mask、有效像素 mask 均在同一去畸变网格。`split=train` 且有 mask 的视图进入监督训练；无 mask 的 train 视图进入一致性训练；val 仅用于固定频率评估和选择 checkpoint。

```bash
uv run python -m bridge_rgs.teacher train \
  --manifest artifacts/prepared/manifest.json \
  --model-dir models/dinov3-vith16plus \
  --output runs/teacher \
  --steps 6000 --crop-size 768 --lr 0.0001 \
  --consistency-start 1000 --eval-every 500
```

输出包含 `last.pt`、有验证标签时的 `best.pt`、每次验证的混淆矩阵和 IoU、`metrics.jsonl`、`provenance.json` 和 `result.json`。检查点只保存解码器/EMA、优化器、阈值、随机状态及来源信息，不重复存储冻结权重。`best.pt` 按验证 mIoU 选择；验证分数用于开发，另设未调参测试集才可报告最终泛化结果。若最终提交阶段使用全量标签，必须重新准备全 train manifest、重新训练，并明确其数值是训练拟合。

```bash
uv run python -m bridge_rgs.teacher train \
  --manifest artifacts/prepared/manifest.json \
  --model-dir models/dinov3-vith16plus --output runs/teacher \
  --steps 6000 --resume runs/teacher/last.pt
```

续训必须使用相同 manifest（按文件 SHA256 检查）。保留原始训练参数和总步数可恢复原来的随机流与学习率计划；改变总步数将改变后续学习率计划，应作为新的训练设置记录。图像内容如被外部修改，需重新准备 manifest 和运行目录，不应复用原有伪标签。

## 软标签导出

```bash
uv run python -m bridge_rgs.teacher predict \
  --manifest artifacts/prepared/manifest.json \
  --checkpoint runs/teacher/best.pt \
  --output artifacts/pseudo \
  --tile-size 768 --stride 512
```

只导出 train 视图；可加 `--unlabeled-only`。采用原生图像网格滑窗、重叠高斯加权与水平翻转 TTA；`--no-flip` 可关闭翻转。每个文件为 `<视图文件名去掉扩展名>.npz`（例如 `001.jpg` 对应 `001.npz`），不允许文件名主体重复。字段：

| 字段 | 形状/内容 |
|---|---|
| `probs` | `C,H,W`，float16 软类别分布；使用时可重新归一化 |
| `confidence` | `H,W`，float16，最大概率 × (1−归一化熵) × (1−归一化翻转 JS 散度) |
| `valid` | `H,W`，有效去畸变像素布尔 mask |
| `view_name`, `split` | 精确视图名以及 train |
| `manifest_sha256`, `checkpoint_sha256` | 来源绑定，供 3D 学生核验 |

目录 `provenance.json` 列出每个视图路径、SHA256、教师训练/验证划分和推理参数。只有全部输出成功，`complete` 才为 true；遇到中断应换新目录重新导出，不能把部分文件当作完整数据。无效像素置信度置零，概率保留但必须通过 valid 排除。

3D 学生在读取前调用 `verify_pseudo_provenance(manifest_path, pseudo_dir)`，核验划分、类别顺序、每个文件及其内嵌来源信息；这也拒绝未登记的额外 NPZ 文件。

导出 confidence 是保守的不确定性权重，**不是校准后的正确率**。3D 一致性再检查遮挡和几何后才能接受教师信息；不要用教师预测反过来污染 validation 标签或训练划分。

新版导出同时生成 `train_confidence_calibration.json`，只读取导出视图中 **train** 标签，针对 confidence 阈值 0.65、0.8、0.9，按预测类报告覆盖率与错误率、按真实类别报告覆盖率与正确覆盖率；无标签 train 图仅参与覆盖率。统计使用最终保存的 FP16 概率与置信度，避免序列化量化影响门槛。文件绑定同一 manifest/checkpoint SHA，并由消费者校验标签视图均属于 train。它是训练拟合可靠性统计，不能作为未见视角错误率保证，也没有使用验证标签选择阈值。

## 验证范围和消融

`uv run pytest tests/test_teacher.py` 在 CPU 小型模拟编码器上验证：编码器冻结、标签与 RGB 几何对齐、忽略区域无梯度、不可靠软标签拒绝、滑窗概率归一、训练/验证分离、保存/续训/导出来源检查。真实 H+ 的 CUDA 预检应额外确认输入形状、梯度有限、显存和吞吐；冒烟测试不是精度证据。

建议固定相机空间分组验证集、训练预算和至少 3 个随机种子，分别比较：DINOv3 纯解码器、加细节分支、加 EMA 光度一致性、加几何过滤后的 3D 学生。至少报告各类 IoU、mIoU、缆索标注区域边界 F1、RGB PSNR/SSIM 和验证相机距离分层指标。完整结果未产生前不作胜过历史方案的承诺。

对固定检查点评估全部带标签的 val 视图，并保存 RGB / GT / 预测预览：

```bash
uv run python -m bridge_rgs.teacher evaluate \
  --manifest artifacts/prepared/manifest.json \
  --checkpoint runs/teacher_pilot/best.pt \
  --output runs/teacher_pilot/validation_full \
  --tile-size 768 --stride 512 --previews 6
```

`evaluate` 不导出训练伪标签。边界 F1 的定义为类别内部一像素边界、原生网格 Chebyshev 距离 2 像素匹配，排除无效邻域；它不自动等同于历史报告的边界指标实现。

本地已执行真实 H+ CUDA 前后向预检，报告位于 `runs/teacher_smoke/cuda_report.json`：768×768 输入、冻结编码器无梯度、监督与一致性梯度有限，RTX 5090 上峰值分配显存约 2.68 GiB。另执行了 `runs/teacher_pilot` 的 **600 步短试跑**，使用 259 张带标签训练图与 91 张无标签训练图，512 裁剪，约 79.1 秒。固定前 6 张验证图上的 EMA mIoU 从第 300 步 70.29% 升至第 600 步 82.52%。选定第 600 步后，用 768 滑窗评估全部 **41 张带标签验证图**，mIoU 为 **86.80%**，缆索区域 IoU 为 **85.29%**，本文定义的缆索 2px 边界 F1 为 **0.5669**。这仍是开发验证，不能与同学记录中的 all300 训练拟合数值直接比较。细缆索连通性和基础/塔柱混淆仍明显，需要继续训练和对照实验。详细结果、6 张 RGB/GT/预测预览位于 `validation_full`，汇总为 `runs/teacher_pilot/pilot_report.json`。

本次数据划分是相机中心 PCA 排序后每 8 张交错留出，属于 **插值验证**；它不能证明未见桥段或新桥梁的外推能力。

## 更充分训练与全景上下文

`configs/teacher_strong_v1.json` 固定 6000 步、768 原生裁剪、全部 41 张带标签验证视图每 1000 步评估。第 3000 步起，有 50% 的监督样本改为完整画面：短边缩放到 768，保持纵横比，再把边缘补齐到 16 像素网格。RGB、标签与 valid 共享同一个变换；补齐部分全部忽略。类别引导局部裁剪仍保留，未标注一致性分支继续使用局部裁剪。

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 uv run python scripts/train_teacher_config.py \
  configs/teacher_strong_v1.json
```

这是对同学 SEM386 全景上下文经验的工程吸收，不单独主张学术创新。训练来源及完整配置记录在独立 `runs/teacher_strong_v1`，不会覆盖 600 步 pilot。继续训练时的上游 checkpoint、SHA256 与原始配置也会写入 provenance。

评估和伪标签导出可显式设置 `--context-weight 0.5 --context-short-side 768`，将原生滑窗预测与整幅上下文预测按固定权重融合。上下文输入只读取 RGB，不读取标签；像素网格先除去 patch padding，再恢复到原生尺寸。概率分歧会降低伪标签置信度。默认权重为零，保留原有推理协议，便于比较相同 checkpoint 下上下文推理是否有效。

强训练与上下文推理的最终数值应读取运行目录中的验证文件；教师读取真实验证 RGB 所得的分割精度并不是仅从目标相机渲染得到的联合重建精度，两者必须分开报告。

6000 步 v1 已完成：最佳检查点为第 5000 步，固定 41 张验证图上，纯滑窗 mIoU **95.0013%**，缆索 **94.6736%**，基础 **90.1931%**。同检查点用 0.5 上下文融合，mIoU 为 **95.1667%**、基础 **91.144%**，但缆索略降到 **94.484%**。完整学习曲线、混淆矩阵和来源记录保留在 `runs/teacher_strong_v1`。

同一教师改为只接收 `refined_long_v2` 渲染的验证 RGB，纯滑窗 mIoU 降为 **88.2884%**，缆索 **68.130%**；0.5 上下文分别为 **88.9165%**、**70.770%**。该诊断在读取前替换图像路径，完全不打开真实验证照片，GT 仅用于评分。它说明当前渲染域，尤其缆索，仍需几何语义条件或专门的域适配；不能把真实 RGB 教师的 95% 分割精度写成重建系统的分数。

## 可选低秩适配与匹配续训对照

`teacher_adapters.py` 在保持 ModelScope 基座权重固定的前提下，可给最后四个 Transformer block 的 Q/V 加 rank-16 低秩矩阵，共 **327,680** 个参数。适配器及其优化器状态为 FP32，基座 BF16；零初始化 B 保留初始函数。只保存适配器、解码器与对应 EMA，不在检查点中重复存储 0.84B 基座。弱预测和验证临时切换 EMA 适配器，退出后恢复在线参数对象，避免破坏优化器引用和未完成的反向图。

```bash
uv run python scripts/check_teacher_adapter_cuda.py
uv run python scripts/train_teacher_config.py configs/teacher_lora_v2.json
uv run python scripts/train_teacher_config.py configs/teacher_head_continuation_control.json
```

两个适配实验从同一 v1 EMA 检查点出发，各追加 2000 步纯监督，使用完全相同的视图/几何增强采样与重新设置的光度增强随机种子。对照仅训练解码器，用于分离增加训练步数的收益。LoRA 配置、初始化检查点 SHA256、EMA 适配器和 RNG 都进入检查点；不能把其工程收益直接归因于本文研究机制。[LoRA 是已有的低秩适配方法](https://arxiv.org/abs/2106.09685)。

匹配对照已完成：LoRA 最佳 mIoU **95.225234%**，仅续训解码器为 **95.226053%**，差 **−0.00082 个百分点**，本次配置未测到 LoRA 增益。两个实验最终 NumPy、CPU Torch、CUDA RNG 状态完全相同；LoRA 检查点重载后也逐元素复现原验证混淆矩阵。因此当前选择更简单的冻结 H+ 解码器续训结果，额外续训的收益不归于 LoRA。

最终选定 `runs/teacher_head_continuation_control/best.pt`，使用 768/512 滑窗、水平翻转 TTA、0.25 全景融合，在同一 41 张带标签开发验证图上 mIoU **95.3176196%**、前景 mIoU **94.2914160%**，较 600 步 pilot 提升 **8.5159 个百分点**。完整选择记录为 `teacher_selection.json`；伪标签导出目录为 `artifacts/pseudo_strong_v1`，只有 `provenance.json` 的 `complete=true` 且全部校验成功才可消费。这个精度仍是读取真实 RGB 的教师指标。

350 张 train 导出已经完成，259 张训练标注的实际 FP16 伪标签校准如下。覆盖率表示“预测为该类的训练标注像素中，达到门槛的比例”；错误率只在保留的预测中计算。

| confidence 门槛 | 缆索覆盖率 | 缆索错误率 | 基础覆盖率 | 基础错误率 |
|---|---:|---:|---:|---:|
| 0.65 | 93.271% | 0.4101% | 87.790% | 1.1387% |
| 0.80 | 90.372% | 0.1846% | 82.655% | 0.6763% |
| 0.90 | 86.933% | 0.0641% | 76.619% | 0.4387% |

首轮单项蒸馏实验可先固定全局 0.8 门槛：训练标注上所有类别的保留预测错误率均低于 0.7%，再以独立消融评价最终渲染收益。训练标注集中在编号 001–300，91 张无标注 train 图均为 301–400；这些训练拟合错误率不能直接当作后 100 张的准确率。统计文件同时保留全 train 的按类覆盖率以审计差异，GT 优先与跨视图可靠性检查仍应保留。

参考：[DINOv3 官方仓库](https://github.com/facebookresearch/dinov3)、[Transformers DINOv3 接口](https://huggingface.co/docs/transformers/en/model_doc/dinov3)、[ModelScope H+/16 权重](https://modelscope.cn/models/facebook/dinov3-vith16plus-pretrain-lvd1689m)。

## 坐标协议与伪标签来源绑定

新训练、续跑、warmstart、教师评价和伪标签导出都显式区分 `legacy_mixed_v1` 与 `colmap_corner_v2`。`checkpoint_pixel_protocol(checkpoint)` 只解析 checkpoint 根或其 provenance 中保存的声明；两处同时存在时必须一致，两处都缺失时固定返回 legacy。它不会重新读取记录路径中的 manifest 来推断或自动升级旧权重。新 checkpoint 在根及 provenance 保存完整协议，warmstart/resume 回执也记录上游解析结果。

`load_teacher(..., pixel_profile=...)` 绑定模型预期协议，默认仍是 legacy；`load_checkpoint_adapters` 在仅解码器和 LoRA 两条路径上都会检查 checkpoint 与模型协议。`require_teacher_manifest_protocol` 同时验证 manifest 内各 view 的显式声明。训练和推理入口将协议检查与既有 manifest SHA 检查同时执行：SHA 相同但协议不同仍被拒绝，协议相同但 manifest 字节不同也不会绕过 split 检查。当前没有跨协议 warmstart 的默认豁免；如研究迁移，须建立单独、可审计的实验。

新伪标签导出在目录 `provenance.json` 和 `train_confidence_calibration.json` 保存完整协议，教师来源也保存解析后的协议。为保留历史 NPZ 格式，各文件不强制增加 profile tag；目录协议通过逐文件 SHA256，以及 NPZ 中已有的 manifest/checkpoint SHA256 绑定到具体概率数组。`pixel_protocol_binding` 字段明确说明此规则。检查器比较目录、教师来源、manifest、校准记录的协议，并在 `verify_files=True` 时验证每份数组的文件 SHA、网格、视图及嵌入来源；如果某份 NPZ 自带 profile，则额外拒绝冲突。`verify_files=False` 仅用于元数据审计，不能代替首次消费时的文件校验。缺失目录或 checkpoint 协议按 legacy 处理；没有逐 NPZ tag 是明确的目录绑定格式，并不表示该数组自动采用 legacy。

渲染域适配还检查 original/derived manifest、train/val render receipt、图像来源协议及实际 renderer checkpoint，全部必须同 profile。v2 renderer 必须保存匹配的原始 manifest SHA；旧 renderer 缺少 profile/SHA 时仅保留原有 legacy 路径，已有 SHA 则仍须相等。`render_teacher_training_views.py` 和 `prepare_teacher_render_domain.py` 在 camera-only JSON、来源 receipt 和 image-source protocol 中透传完整协议，且在渲染前验证 renderer。上下文评价与域适配比较脚本也验证协议，不会静默混用旧新图像网格。

CPU 回归覆盖新旧两种协议的训练→续跑→warmstart→导出→校准→评价链、缺字段旧 checkpoint、同 SHA 异协议的四入口拒绝、渲染域不同协议和实际 renderer SHA、目录与可选 NPZ tag 冲突、350/50 相机清单及生成脚本。脚本测试仅 mock GPU rasterization，实际运行其元数据、图片文件与 receipt 校验，并保持真实 VAL 照片不存在。只读历史审计 `runs/teacher_pixel_protocol_audit.json` 检查了 12 个既有教师 checkpoint，均解析为 legacy，其中 4 个渲染适配 checkpoint 的完整域来源链通过；350 张既有 pseudo 的元数据兼容性通过，未重写原目录或权重，也未将旧模型宣称为 v2。
