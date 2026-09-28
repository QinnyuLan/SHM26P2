# 同冻结 RGB 场的 SEM386 风格参照：可实施性核查

后续状态：官方三文件现已按固定 revision 下载并核验，严格加载与四更新预检的 CPU 实现已完成；原文下方保留最初可行性核查的时间口径。当前进展与 BF16/有效支持差异见[预检说明](mask2former_reference_preflight.md)，尚未启动正式 6k 参考训练。

2026-09-27，CPU 和官方资料核查完成。**可以复用当前 uv 环境实现，但完整 baseline 尚未实现、下载权重或启动 GPU。** 此参考必须等待修复后的完整 v2 实验与当前科学诊断；它用于本项目共同划分下的效果／成本比较，不是同学 SEM386 的精确复现，也不证明创新。已披露配方及缺口仍以[近似复现范围](peer_approximation_scope.md)为准，没有寻找旧 Dev30 清单。

## 固定来源与本机可复用部分

- 官方发布账户下的模型：`facebook/mask2former-swin-large-ade-semantic`，本次查得 revision **`aa25c92404a40599614215e76514c79b427c7527`**。这是 ADE20K 语义预训练模型，HF 模型卡由 HF 团队编写；与作者 Detectron2 代码的执行等价性未验证。[模型页](https://huggingface.co/facebook/mask2former-swin-large-ade-semantic/tree/aa25c92404a40599614215e76514c79b427c7527)
- `model.safetensors`：**866,052,064 字节**，服务器声明 SHA-256 **`b143c144341c15b4f20165cc6d2c9305fb1b66792f68a6e0e06d2b20dc063b14`**。本轮只读取元数据，尚未下载／核验本地权重。[文件与 SHA](https://huggingface.co/facebook/mask2former-swin-large-ade-semantic/blob/aa25c92404a40599614215e76514c79b427c7527/model.safetensors)
- 将来如获执行许可，建议独立存到 `/mnt/data/SHM2026/models/mask2former-swin-large-ade-semantic-aa25c924`，只取同 revision 的 config、preprocessor、safetensors、模型卡／许可来源，逐项记录实际 SHA；不同时下载等价 `.bin`，不改现有 DINO 目录。结果与可选渲染 PNG 缓存也放 `/mnt/data/SHM2026`，不在根盘复制大缓存。
- 本机 uv 锁定环境已具备 `torch 2.8.0+cu128`、`torchvision 0.23.0+cu128`、`transformers 4.57.6`、`scipy 1.17.1`、`safetensors 0.8.0`、`accelerate 1.15.0`。`Mask2FormerForUniversalSegmentation` 和处理器导入成功；HF 多尺度可变形注意力使用 PyTorch `grid_sample`，匹配使用 SciPy。无需为此先安装 Detectron2、mmcv、mmseg 或 timm；这些包当前均不存在。[HF 对应版本实现](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/mask2former/modeling_mask2former.py)
- 检查了实际 HF 缓存 `/home/sky/.cache/huggingface/hub`、ModelScope 缓存 `/home/sky/.cache/modelscope/hub`、项目 `models` 和 `/mnt/data/SHM2026/models`：未发现匹配的 Swin-L／Mask2Former 权重；同学展示包也未附其权重。这里只限定上述目录，不声称扫描了整台机器。

## 最小训练与部署合同（建议，尚未冻结）

1. **输入和参照**：固定已选 legacy H3 场 `runs/h3_moments/02_cross/last.pt`，绑定完整 SHA、legacy manifest 和原 350/50 划分，只用同 259 个有标注 TRAIN 视角的渲染 RGB 与官方五类 mask。geometry／SH／相机全部冻结；不更换 RGB150、不把真实 VAL 图送入模型、不读取旧 Dev30。只生成这些 TRAIN 渲染或逐次相机渲染，不能误用旧 strong RGB 场的 350 图缓存。对比已选 DINO/场组合与同场学生时说明：标签集合可比，但预训练、真实 RGB 训练使用、无标签一致性、可训练参数与推理计算均不相同。
2. **初始化**：使用上述 ADE checkpoint 的 Swin 和通用 decoder，五类加 no-object 的 6 行分类头重新初始化；加载差异只允许预先枚举的分类头和类别权重 buffer，不用宽泛 `ignore_mismatched_sizes` 掩盖其它不匹配。类别保持 `background/deck/stay_cable/tower/foundation=0..4`，255 为 ignore，禁用 `do_reduce_labels`。
3. **形状／采样**：建议固定 6,000 次成功更新、batch 1、seed 20260805；奇数原生 768 裁块，偶数等比完整 1025×768 上下文，各 3,000 次，训练翻转概率 .5。视图、裁块、翻转的独立 RNG 与名单事先锁定。legacy 图与 mask 使用明示的 legacy resize，而非把 v2 重采样混入；图像与 mask／valid 联动。处理器默认配置包含 384 缩放，必须关闭隐式 resize；RGB 只进行一次缩放及 ImageNet 归一化。上下文若为网络要求补到 32 倍数，补边只影响网络画布，不成为监督像素。[处理器配置](https://huggingface.co/facebook/mask2former-swin-large-ade-semantic/blob/aa25c92404a40599614215e76514c79b427c7527/preprocessor_config.json)
4. **优化／损失**：四个 Swin 阶段与 decoder 全训练；参考作者 AdamW LR `1e-4`、backbone × `.1`、weight decay `.05`、norm／embedding／位置参数免衰减、全模型 clip `.01`，poly `.9` 无 warmup。保留 100 query、class/mask/dice=2/5/5、no-object=.1、辅助监督、12,544 采样点、oversample 3／importance .75。它们是作者及展示包的可借用配方，不能冒充 SEM386 完整 resolved 配置。AMP 仅用于模型计算，匹配与完整 criterion 显式 FP32；每次更新记录有限梯度、Swin 四阶段更新及成功步数，溢出不得静默算作成功步。[作者损失配置](https://github.com/facebookresearch/Mask2Former/blob/main/configs/ade20k/semantic-segmentation/maskformer2_R50_bs16_160k.yaml)、[优化器与分组](https://github.com/facebookresearch/Mask2Former/blob/main/train_net.py)、[基础训练配置](https://github.com/facebookresearch/Mask2Former/blob/main/configs/ade20k/semantic-segmentation/Base-ADE20K-SemanticSegmentation.yaml)
5. **无效像素是实现前置条件**：HF 4.57.6 的 `pixel_mask` 不会传给 criterion，`ignore_index` 仅从类别列表移除 255；转换后的 binary masks 在这些位置全零，现成点采样仍可能把它们当所有实例的负例。不能把二者当成 GT ignore 已生效。必须在 Hungarian matching 的 mask／dice 采点、uncertainty 候选与随机点、最终及每层辅助 mask 损失中共同限制到有效支持，含补边与未知标注。双线性采点要保证整个插值足迹有效或明确采用有效性加权／重新归一，避免 ignore 邻域泄漏；无有效支持的样本应显式处理。需要“只改无效区域 logit 不影响 matching/loss/梯度”的 CPU 合同，以及 padding／边界回归。作者 criterion 也保留未使用 padding valid 的 TODO，这个适配是公平标签支持所需的实现差异，不能声称逐行原版复现。[作者 criterion](https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/criterion.py)
6. **推理与评价**：只锁一个固定 768 tile／512 stride、明确末端 tile／padding／重叠平均的推理协议，初版不加未披露的推理翻转。以 FP32 计算 `softmax(class_logits)[..., :5]` 与 `sigmoid(mask_logits)` 的 query 加权和，先恢复各 tile 尺寸再累积，得到完整五类 score 后按固定规则归一；随后沿同 legacy 覆盖画布／官方 warp／argmax 导出。不能直接调用返回 argmax 的默认处理器，丢失映射前的 soft scores。记录这一 FP32 实现与同学局部零质量修复的差异。固定末点一次原图 50/41 评价，源图只由 scorer 打开；核验全部 RGB PNG 与选定 H3 场一致，报告 paired CI 和独占推理成本，不用 VAL 选择 crop、权重或 checkpoint。

## 参数与计算预算

本轮用官方 JSON 在 `torch.device('meta')` 构造五类模型，未加载权重／执行 forward，CUDA 未初始化。其总参数 **215,451,514**，Swin **195,201,204**；按合同全部可训练。Swin 配置为 embed 192、depths `[2,2,18,2]`、heads `[6,12,24,48]`、window 12、drop path .3，与作者 Swin-L 配置对应。[HF 固定配置](https://huggingface.co/facebook/mask2former-swin-large-ade-semantic/blob/aa25c92404a40599614215e76514c79b427c7527/config.json)、[作者 Swin-L 配置](https://github.com/facebookresearch/Mask2Former/blob/main/configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml)

现有 H+ 预检记录为冻结骨干 **840,592,640**、可训练 DPT **6,376,326**；Swin-L 参考总模型较小，但可训练参数约为该 DPT 的 **33.79 倍**，且更新整个 backbone。ADE 语义预训练与 DINO 自监督预训练不同。因此不是同可训练参数／同训练数据域的架构消融。[本机 H+ 记录](/mnt/data/SHM2026/preflight/dinov3_capacity_v1/hplus/worker_receipt.json)

FP32 权重约 .803 GiB，权重＋梯度＋Adam 两个状态约 **3.21 GiB**，这是不含激活、工作区、临时更新和冻结场的静态下限。HF 顶层模型不支持直接 `gradient_checkpointing_enable`；内部 Swin backbone 支持，四个 `SwinStage` 是 checkpoint layer，需对 encoder 单独启用并验证 RNG/四阶段梯度，不能只设置一个无效 flag。batch1、AMP、阶段重算有望在32GB卡上运行，但本轮无 CUDA 内存／速度实测，不能承诺峰值。

若以后批准，先用固定 TRAIN crop/context 各至少两次真实更新核验数值、重算和峰值，再按交替步实测 `6000×(t_crop+t_context)/2` 定时限；6k 步预留 **1–3 GPU 小时**仅为粗略排程范围（若每步 .6–1.8 秒），不是基准测量。纯 PyTorch deformable attention／SciPy 匹配可能改变吞吐，不能套用 DINO 耗时。下载约 .81 GiB，单份 FP32 推理权重约 .80 GiB，带 Adam 续训文件约 2.41 GiB；保存 last 及其原子临时副本、权重和可选 TRAIN RGB 时宜预留 **至少 8 GiB**。不保存逐步大概率缓存，不为本次准备更改 uv.lock。

随后已获准完成独立的[valid-support criterion CPU 适配](mask2former_valid_support.md)，采用有效 GT 像素中心抽样与整数 GT gather，明确区别于作者连续坐标采样。其余仍只到来源、缺口、合同和预算：没有新 baseline runner／配置／实验目录，没有权重下载和 GPU 使用；是否执行待完整 SSIM 修复结果之后另定。
