# Mask2Former 有效支持适配器（仅 CPU 实现与合同）

实现：[mask2former_valid.py](../src/bridge_rgs/mask2former_valid.py)。测试：[test_mask2former_valid.py](../tests/test_mask2former_valid.py)。当前接口核对 `transformers==4.57.6`；没有修改 HF 模型或 site-packages，没有下载权重、创建训练计划或使用 GPU。它是[SEM386 风格参照可行性](peer_style_mask2former_feasibility.md)中的标签支持适配，不是新方法或已完成的 baseline。

## 唯一采样约定

协议 ID 为 `valid_gt_pixel_centers_v1`。对一张已经补齐到**模型输入画布**的 GT，有效集合是 `valid & (label != 255)`。`valid` 必须显式排除无效区与补边。背景 0 是正常语义类；有效域内出现的每个类别各构成一个 binary mask，no-object 不属于 GT 类。每张图都必须有有效支持，否则抛错，不能把全 ignore 图变成 no-object 训练样本。

Hungarian matching 从有效 GT 像素中心均匀、有放回抽点，同一张图的所有 query 和 target 共用这些点；uncertainty 采样先从同一集合为每个匹配 query 抽 oversample 候选，按 `-|logit|` 取重要点，其余随机点仍从该集合抽取。每层辅助 head 独立重做 matching 与采样，但使用完全相同的有效域规则。生成器可以显式传入，主头再按 auxiliary 顺序消费其随机数。

GT 用整数索引读取，因此不会把 ignored target 混入双线性标签。预测仍调用 HF 的 `sample_point`，采用 **FP32、bilinear、align_corners=False、padding_mode=zeros**；中心坐标是 `((x+.5)/W, (y+.5)/H)`。这保留了预测网格的标准四点插值及图像边缘的零扩展，没有 erosion，所以单个有效像素与细结构不会被支持收缩删除。

必须区分两类位置：

- **没有参与任何有效采点的实际双线性 tap**：改变其有限 logit，不应改变匹配、采点排序、监督损失或梯度。
- **参与有效点插值的低分辨率 logit**：即使把其中心投影到 GT 后落在 ignore 区，也可能是合法的邻近 tap，不能强行置零其梯度。非整数比例时还存在 FP32 坐标舍入，应以实际插值足迹判断，不能只做 nearest-valid 判定。

这些合同只约束 criterion 对固定 logits 的直接监督依赖，**不意味着改变输入图像的 padding 后模型输出不变**；卷积、Swin attention 或 decoder 可以让该输入影响有效输出。

## 保留与改变的作者配方

复用 HF 的分类 CE、mask BCE、Dice 及 pairwise cost 函数；matching 的 class cost 仍是负类别概率，mask/dice 使用配置权重。未匹配 query 的类别是 `C`，其 CE 权重为 `no_object_weight`；背景类仍为 0。mask 损失按 batch 内有效 target masks 的总数归一，辅助层独立 matching，所有配置权重只在总 loss 加一次。只支持单进程；多进程会显式拒绝，未冒充已完成的 DDP 实现。[HF 4.57.6 criterion](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/mask2former/modeling_mask2former.py)

相对作者连续 `[0,1]²` 随机点和双线性 GT 采样，**本适配器改为有效 GT 像素中心分布与整数 GT 值**，因此类别边界不会产生连续点采样的软 target。matching、uncertainty 与 aux 都遵守该同一改变，不能声称逐点等价于作者或同学 SEM386。作者 criterion 自身也有未将 padding valid 用于 mask loss 的 TODO；这里明确解决有效支持，而不是把 `pixel_mask` 或 `ignore_index` 误当完整损失屏蔽。[作者 criterion](https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/criterion.py)

不改变类别、query 数、no-object 语义或 BCE/Dice 公式。criterion 在禁用 autocast 的 FP32 区域运行，半精度模型 logits 的 `.float()` 保留反传链；不在 nonfinite Hungarian cost 上静默补零。传入 targets 即使是浮点 binary masks，也会作为常量读取。

## 接入方式

```python
from bridge_rgs.mask2former_valid import (
    ValidSupportMask2FormerCriterion, semantic_targets,
)

# labels/valid 已共同变换并补齐到 pixel_values 的 HW，背景仍为 0。
targets = semantic_targets(labels, valid, num_labels=5)
criterion = ValidSupportMask2FormerCriterion(model.config).to(device)

# 不向 HF forward 传 mask_labels/class_labels，避免先执行不带 valid 的内置损失。
outputs = model(pixel_values, output_auxiliary_logits=True)
result = criterion(outputs, targets, generator=point_generator)
result.loss.backward()
```

这里 `point_generator` 与 logits 位于兼容设备；创建模型、AMP 和优化器仍由未来 runner 负责。`result.loss_dict` **未加权**，便于日志；`result.loss` 已按 config 对主头和每层 aux 加权。`indices` 与 `auxiliary_indices` 供匹配审计。若内置 HF loss 已经执行、辅助头缺失、canvas／类别／query 数不合法、有效目标无支持或 logits 非有限，直接拒绝，不回退为默认 HF criterion。

## 已验证的范围

`CUDA_VISIBLE_DEVICES='' uv run --no-sync pytest -q tests/test_mask2former_valid.py`：**14 passed**；对应模块和测试 Ruff 通过。包括：

- 固定有效域外 target 改为 NaN、真正无 tap 支持的 logit 改为 ±10,000，主头／所有辅助头的匹配、损失与梯度逐位不变，外域梯度为零。
- 显式四 tap 权重与 HF 对齐／零 padding 一致；合法的低分辨率 tap 可以影响有效监督，未将这种影响误判为泄漏。
- 单个有效像素不被 erosion 删除；255、padding、背景／no-object、不同图有效域、batch mask 归一与权重合同成立；全无有效域、遗漏 aux 等显式失败。
- CPU autocast 下 BF16 logits 可经 FP32 criterion 反传；不带权重下载的小型真实 HF Mask2Former 输出可接入，四个 Swin stage 和分类头均得到有限、非零梯度。

尚未验证真实 Swin-L 权重、768/context 的 CUDA 数值、速度／显存或 6k 训练收益；这些不能由本次 CPU 合同推定。未建立新的性能结论。
