# DINOv3 ViT-7B：有限只读可行性审查

2026-09-26。结论：**7B 的 HF 格式权重在 ModelScope 实际公开可达，现有代码结构可适配；冻结 BF16、batch=1、768 crop 的单卡 32 GiB 运行有合理余量，但尚未经真实 CUDA 验证。H+ 已很接近 7B 的官方分割指标，建议把 7B 保留为一次匹配的精度工程对照，不直接替换选中模型。** 本轮没有下载张量权重、安装依赖、使用 GPU、改模型或移动已有文件。

## 存储核查

`/mnt/data` 是 `sky:sky`、权限 `0755`、ext4 `rw`，当前用户的写入与遍历权限检查为真。审查时可用 **869,064,429,568 bytes，约 809.38 GiB**；`/mnt/storage` 也由 sky 所有且可写，约 13 TiB 可用。只读搜索 `/mnt/data` 的前三层未发现 SHM2026 目录；root 专属 `lost+found` 不可读，没有绕过权限或进行全盘扫描。因此此前根分区余量确实不能代表整机可用容量。

后续新下载可明确放到 `/mnt/data/SHM2026/models/dinov3-vit7b16`，新实验输出与临时下载缓存也应显式放到该挂载盘；**本轮没有创建这个目录**。不用迁移旧产物或更改旧 receipt 路径。为 FP32 原始分片、缓存/临时副本、可选 BF16 导出和实验预留 **至少 80–100 GiB** 就足够宽裕；这是规划额度，不是当前写入。宿主机约 62 GiB RAM、审查时约 51 GiB available，仍应采用分片/低峰值加载，避免同时实例化 FP32 和 BF16 全模型。

## 权重与架构：已验证的元数据

来源是 [ModelScope facebook/dinov3-vit7b16-pretrain-lvd1689m](https://modelscope.cn/models/facebook/dinov3-vit7b16-pretrain-lvd1689m)，固定 revision **`5251e00b307184bb247d076713375235d554fefb`**。匿名 API 返回文件列表、config 与索引；首分片 Range 请求仅读取 **8 字节 safetensors 头部长度字段**，响应 `206 bytes 0-7/4980241600`，头部长度 13,496。未读取任何张量载荷，也没有以 HEAD 200 单独充当下载可达证据。文件声明的 SHA 与尺寸完整保存在 [metadata audit](../artifacts/dinov3_7b_metadata_audit.json)。大文件 SHA 是服务端声明值，本轮没有全量下载后校验。

| 项目 | 现有 H+ | 候选 7B |
|---|---:|---:|
| 精确骨干参数量 | 840,592,640 | **6,716,035,072** |
| encoder 层数 / hidden | 32 / 1280 | **40 / 4096** |
| heads / head dimension | 20 / 64 | **32 / 128** |
| SwiGLU intermediate | 5120 | **8192** |
| patch / register tokens | 16 / 4 | **16 / 4** |
| HF 权重存储 dtype | FP32 | **FP32** |
| safetensors 文件总字节 | 3,362,432,800 | **26,864,210,088** |
| 张量载荷字节 | 3,362,370,560 | **26,864,140,288** |
| BF16 骨干显存下限 | 1.5657 GiB | **12.5096 GiB** |

7B 有六个分片，字节数依次为 **4,980,241,600 / 4,967,510,232 / 4,967,510,568 / 4,967,543,448 / 4,967,543,320 / 2,013,860,920**；总文件约 26.864 GB 或 25.019 GiB。RoPE、SwiGLU、patch16 与 register4 可从 config 直接核实。7B 的 query/key/value bias 均关闭，不能用 H+ 的 bias 假设替代实际配置。

[Meta 官方模型卡](https://huggingface.co/facebook/dinov3-vit7b16-pretrain-lvd1689m)公开列出该模型与 AutoModel 用法；其 Hugging Face 权重入口需要接受访问条款。ModelScope 的上述公开端点本次可达，并附 DINOv3 License；公开可达不改变模型许可。应下载 LVD-1689M 的通用图像版本，不因“桥梁”关键词换成卫星版本。

## 官方 dense 结果与收益边界

同一官方模型卡的 dense 表如下；ADE20K 是冻结特征线性探测，**不是本项目四层 DPT、桥梁五类或渲染 RGB 的成绩**。

| 指标 | H+ | 7B | 差异 |
|---|---:|---:|---:|
| ADE20K mIoU ↑ | 54.8 | 55.9 | +1.1 pp |
| NYUv2 RMSE ↓ | .352 | .309 | −.043 |
| DAVIS ↑ | 79.3 | 79.7 | +.4 |
| NAVI recall ↑ | 63.3 | 64.4 | +1.1 |
| SPair recall ↑ | 56.3 | 58.7 | +2.4 |

这些数字支持“7B 在若干 dense 任务略强、深度有较明显改善”，不支持“必然提升本桥缆索 IoU”。[DINOv3 论文 Fig.16 / §7.1](https://arxiv.org/html/2508.10104v1#S7.SS1)也明确强调 H+ 与 7B 表现接近。论文给出的 512 输入计算量是 **1903 vs 14515 GFLOPs，7.63 倍**。二者都是 patch16；更大 hidden/深度不等于更细的像素采样。

## 现有 AutoModel / 教师接口合同

已使用 `uv run --no-sync python`、**CPU meta device** 构造当前 Transformers **4.57.6** 的 `AutoModel.from_config(..., attn_implementation="sdpa")`，没有分配 7B 实体张量：

- 构造类型是 `DINOv3ViTModel`，参数数目恰为 6,716,035,072，`state_dict` 的 **687 个键与 ModelScope 分片索引完全相同**。
- `DINOv3Teacher` 的四层选择会自动成为 **10/20/30/40**，四个投影输入自动成为 4096，解码器共 **8,539,014 参数**；H+ 头是 6,376,326 参数。
- `load_teacher` 已使用 BF16、SDPA、本地文件加载；骨干 `eval().requires_grad_(False)`，无 adapter 时 `extract_features` 用 `no_grad`，EMA 只复制 decoder，AdamW 也只持有 decoder。当前代码不会复制第二个 7B EMA 骨干。
- `output_hidden_states=True` 会暂存各层输出；四层提取后其他层释放。仅保存四层的 hooks 是可选内存优化，首轮不必增加这种代码变化。
- H+ decoder 第一层投影 shape 不兼容 7B；**不能 strict resume/warmstart 旧 H+ decoder**。首轮应创建新头；若以后做部分拷贝，那是另一个迁移对照。
- 现有 `scripts/download_dinov3.py` 写死 H+ ID、revision、单文件 SHA 与 1280/32 检查，**不能原样用于 7B**。需另建通用或 7B 下载 receipt，校验六个分片及 config、index、preprocessor、license。教师目前会哈希所有 `.safetensors`，但未单独绑定 index 文件 SHA，7B 下载来源应补齐这个绑定。

结构检查与本地代码 SHA 见 [local contract](../artifacts/dinov3_7b_local_contract.json)。它证明配置/键名/调用接口兼容，**不证明 `from_pretrained` 全量加载、768 forward、SDPA 具体 CUDA kernel 或梯度实测已经通过**。

## 32 GiB 的显存和时间估算

以下为 batch=1、冻结 BF16 encoder、FP32 decoder 参数与 AMP、adapter_rank=0 的解析规划值，**不是实测峰值或保证上界**。

| 7B 组件 | 768×768 crop | 768×1040 全景 |
|---|---:|---:|
| tokens（含 CLS/register） | 2309 | 3125 |
| 骨干 BF16 参数 | 12.510 GiB | 12.510 GiB |
| 暂存 41 层 BF16 hidden | .722 GiB | .978 GiB |
| 返回四层 BF16 patch feature | .070 GiB | .095 GiB |
| 若四层同时转 FP32 的体积 | .141 GiB | .190 GiB |
| decoder 参数+梯度+Adam两矩+EMA | .159 GiB | .159 GiB |
| **包含临时层激活、DPT反传、工作区的规划区间** | **16–20 GiB** | **18–22 GiB** |

最后一行给运行余量，不是把上面各行机械相加；临时 QKV/SwiGLU、卷积工作区、allocator reserved、CUDA context 和同时活跃的无标注分支张量都可能改变峰值。原配置第 3000 步后有 50% 全景，1320×989 会 pad 到 **1040×768**，因此只测正方形 crop 不够。teacher 当前先做监督 backward，再做无标注 weak/strong 分支，没有同时保留两套 encoder 反传图，但每步多次 encoder 前向仍耗时。首个 7B 预检应独占 GPU，不能用这份估算排两个大任务并发。

用 `2NP + 4 L N²d` 粗略计密集矩阵乘与 attention，768 crop 为 H+ **4.76 TFLOPs**、7B **34.51 TFLOPs**，约 **7.26 倍**；全景约 **6.85 vs 48.38 TFLOPs**。这个公式忽略部分逐元素与头部操作，不是测速。

本地 H+ 6000 步旧完整配置含周期验证用时 **1838.88 秒**（`runs/teacher_strong_v1/result.json`）。简单按 encoder 成本放大，7B 相同流程可暂按 **约 3–5 小时**排期；实际 decoder、I/O、kernel 效率占比会改变比值，应先短测后更新 ETA。现有 1320×989 tile768/stride512 有 6 个 tile，每 tile flip 两次，加全景两次，**每帧共 14 次 encoder forward**。已有 H+ 同卡旧记录约 .77–.95 秒/帧教师完整协议，7B 粗略可能到数秒/帧，不能把一次模型前向当完整提交耗时。冻结头训练检查点不含骨干，其 decoder/EMA/Adam 状态合计约 **137 MB**（不保存梯度），没必要每次保存 26.9 GB 权重副本。

## 最小工程路线建议

1. **先保留 H+ 作为选定实现。** 7B 是容量对照候选，官方数据不足以证明在本数据上值得全面替换。按 root 对现有赛事 PDF 的核查，Project2 视觉/语义准确度各占 50%，全文评分另含论文创新；现有 PDF 未列 Project2 的 FPS/推理硬时限。这允许考虑慢一些的精度方案，但不能推出后续组织方没有运行限制。
2. 若决定投入，只下载固定 revision 到新挂载目录，逐分片校验 SHA、记录完整环境/许可/index/preprocessor 来源。先做一张固定 **TRAIN** 768 crop 与 1040×768 context 的 2 步预检：确认所有 backbone grad=None、decoder grad 有限、BF16、shape、实测 allocated/reserved/进程显存、耗时、完整14调用推理。预检结果通过后再给训练预算；本轮没有执行。
3. 性能对照只换 encoder 与不可避免的投影输入维度，两头都从固定 seed 新初始化，保持同一 v2 manifest、标签、采样序列、768 crop、DPT192、优化器、6000 步/预定末态选择规则。首轮不叠加 LoRA、量化、任意层选择或调新阈值。若比较目标是相机任务，两个教师都必须用**同一个固定 renderer 的 RGB**接受评分；真实照片教师成绩单列。
4. 固定同一官方原图网格报告五类 IoU、缆索/基础/塔柱、RGB不变性和推理显存/时间。即使 7B 真有增益，也只说明预训练容量/工程选择有效；**不能作为高斯场机制或论文创新的证据**。若没有可信增益，保留负结果并继续用 H+，不围绕验证集反复挑层/步数/融合权重。

量化、磁盘权限及公式输入汇总在 [feasibility JSON](../artifacts/dinov3_7b_feasibility.json)。本轮新增的只有小型审查 JSON 与本文；没有修改任何训练配置、旧产物或选中模型。
