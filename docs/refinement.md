# 多尺度、全景上下文语义精修

## 问题与设计范围

当前 `runs/refined/evaluation_native/metrics.json` 的五类 mIoU 为 89.952%，stay_cable IoU 为 74.003%；同一模型未经精修的原生三维 stay_cable IoU 仅为 19.217%。这说明语义特征投影后的解码能力明显影响最终结果。
41 张有标签验证图像的进一步混淆诊断中，stay_cable precision 为 80.58%、recall 为 90.06%；背景误报为 cable 的像素数约 809,334，cable 漏成背景约 355,118。
因此问题同时包括**索区外扩张误报和索区内部孔洞**，不能简单通过膨胀索区提高召回。

训练照片与官方 polygon 标注显示，stay_cable 的语义区域可覆盖缆索围成的区域；它不等同于逐根真实缆索的可见表面。
精修器要理解区域、桥塔和桥面之间的上下文关系，但不能为覆盖标签而向三维场景增加不存在的支撑平面。

原模块仅含全分辨率局部卷积和 4 倍池化上下文，而且输出为 `tanh(2 * logits)`，实际残差范围是 **±1**。
对任意两个类别，其最多只能改变 `exp(2)` 倍的概率比。当原生三维先验过度自信地预测背景时，这一上限可能阻止正确翻转。

新增 `MultiScaleRefinementHead` 使用同一共享几何的渲染证据，扩大上下文范围和纠错能力。
FPN、金字塔池化、方向池化都是已有通用组件；这里不把这些组件本身宣称为新算法。
这个模块是整套相机可靠性与结构分配方案中的可消融解码增强，是否改善真实新视角表现须由训练实验验证。

## 模块结构

输入全部由当前 Gaussian 场景渲染：

| 输入 | 含义 | 对几何反向传播 |
| --- | --- | --- |
| 语义特征 | 显式三维 Gaussian 特征的投影 | 继续更新语义特征；主渲染分支的几何已按原逻辑 detach |
| RGB | 同一场景渲染颜色 | detach |
| 深度 | `log1p` 后按当前渲染归一化 | detach |
| alpha | 渲染覆盖率 | detach |
| `p3d` | 原生三维五类概率 | 保留语义特征路径，遵循渲染器的 classifier 梯度开关 |
| 熵 | `p3d` 的归一化熵 | 同 `p3d` |
| RGB/深度梯度 | 渲染图像的局部边缘幅值 | detach |

默认特征维数 16 时，总输入宽度为 29 channels。不存在原始测试照片、GT、相机 ID、训练图像索引或固定图像坐标模板输入。

1. 原分辨率细节支路保留边界和局部类别信息。
2. 1/2、1/4、1/8、1/16 编码器用 GroupNorm 和 separable residual blocks 提取多尺度特征，适合单视图训练。
3. 在 1/16 特征上使用池化尺寸 `[1,2,4,8]` 的金字塔分支；全局分支能看到整个渲染。
4. 水平方向、垂直方向 strip pooling 保留一个空间轴，补充跨区域联系。它不包含桥梁专用形状模板。
5. FPN 自顶向下融合各尺度，并与原分辨率细节支路合并，输出五类 logits。

最终仍是原生三维概率加残差：

```text
z_centered = z - mean_classes(z)
r = B * tanh(z_centered / B)
p_final = softmax(log(p3d) + r)
```

默认 `B=6`，可以纠正更强的错误三维先验。较大纠错范围也意味着更需要背景监督和完整验证，不自动等于更高准确率。
输出卷积全零初始化，使插入新头时 `r=0`。原生 `p3d` 始终保留、单独评价，并用于三维语义 PLY 导出；二维精修不被伪装成三维标签改善。

## 配置与消融接口

完整默认结构：

```yaml
refiner:
  type: multiscale
  channels: 64
  residual_bound: 6.0
  context: pyramid_strip
```

建议按相同划分、同一个初始化 checkpoint、相同学习步数和分辨率比较：

| 消融 | `refiner` 配置 | 检验因素 |
| --- | --- | --- |
| 原始模块 | `{type: legacy}` | 完全保留旧模块、±1 范围 |
| 仅增加纠错幅度 | `{type: legacy, residual_bound: 6.0}` | 局部结构不变；扩大残差范围 |
| 多尺度但无额外全景分支 | `{type: multiscale, channels: 64, residual_bound: 6.0, context: none}` | FPN 与细节分支 |
| 多尺度 + 金字塔 | `{type: multiscale, channels: 64, residual_bound: 6.0, context: pyramid}` | 整体及区域池化 |
| 完整结构 | `{type: multiscale, channels: 64, residual_bound: 6.0, context: pyramid_strip}` | 再加入方向池化 |
| 新结构、小纠错幅度 | `{type: multiscale, channels: 64, residual_bound: 1.0, context: pyramid_strip}` | 区分上下文与幅度收益 |
| 轻量结构 | `{type: multiscale, channels: 32, residual_bound: 6.0, context: pyramid_strip}` | 容量、内存与质量权衡 |

`context: none` 表示移除显式金字塔/strip 分支；GroupNorm 仍统计整幅特征，不能把它解释为完全无全局信息的理论模型。
背景权重对照、数据采样对照属于训练损失实验，不在此模块内偷偷修改。

## 旧模型、warmstart 与恢复

`GaussianScene(..., refiner_config=None)` 默认生成原始 `RefinementHead`，保留旧参数名 `refiner.local/context/out.*` 和原来数学表达式。
规范化接口为 `bridge_rgs.refinement.normalize_refiner_config`；缺省结果是 `{"type": "legacy"}`，未知选项会报错。
模型的有效配置保存在 `scene.refiner_config`，checkpoint 保存这一字段，`load_scene` 用它重建模块；缺失字段按 legacy 处理。

严格 `resume` 不允许改变精修架构。更换架构应作为 **warmstart 新阶段**：保留所有非 `refiner.*` 的场景状态，重新初始化精修器和新阶段优化器，并保存新配置。
新头零初始化保留的是 `p3d`，并不会保留旧精修器带来的准确率增益。因此新头刚替换后的输出回到原生三维语义；应给足训练步数，不能用替换后第 0 步来比较最终质量。

若做仅语义阶段，须设置 `freeze_geometry: true`、`freeze_rgb: true`、`densification: none`，并从 `semantic_start: 1`、`refine_start: 1` 开始；否则冻结所有 RGB 参数后的 RGB 预热阶段没有可训练梯度。
这是一项训练协议要求，不能用新增标签专用几何代替。

## 已运行验证

`uv run python -m pytest tests/test_refinement.py -q` 已通过，包括：

- 随机非零权重下 legacy 前向与旧公式逐位一致，旧参数 strict load。
- 不规则尺寸图像及三种 context 模式零初始化时精确保留 `p3d`。
- 构造一个原生背景概率 0.98、cable 概率 0.005 的例子，证明 ±6 能翻转而 ±1 即使最优也无法翻转；这仅验证表达能力，不是数据集性能。
- 语义 features 和 `p3d` 获得有限梯度，RGB/depth/alpha 无梯度路径。
- 全景分支读取完整渲染，不局限于局部窗口。
- warmstart 仅遗漏 `refiner.*` 参数，所有三维参数、颜色和语义分类器保持一致；新模块可按保存配置 strict reload。
- 错误架构名、通道数、context 名、非有限 bound 等配置不会被静默忽略。

实际参数数量（feature_dim=16）：

| 模块 | 参数数 |
| --- | ---: |
| legacy | 21,733 |
| multiscale 32，全上下文 | 146,165 |
| multiscale 64，无显式上下文 | 532,645 |
| multiscale 64，全上下文 | 557,413 |

RTX 5090 上已运行一次 **1320×989、float32 的前向和反向操作检查**，没有执行 optimizer 更新：输出 `[989,1320,5]` 且有限，测试进程峰值分配显存为 4.185 GiB，前后向约 0.546 秒。
当时另有 GPU 作业并行，且这是随机渲染张量的模块检查；该时间不能当作公平速度 benchmark，显存也不包含完整 Gaussian 训练的其他缓冲区。
此记录不声称新精修器已取得更高竞赛指标。
