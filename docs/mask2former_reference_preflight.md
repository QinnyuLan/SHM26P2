# Swin-L Mask2Former 四更新预检

这项预检只判断本机参考模型是否可训练及其实际成本。它不产生候选 checkpoint、不评分、不启动 6k 训练，也不证明超过同学方案或具有学术创新。

官方 `facebook/mask2former-swin-large-ade-semantic` 固定 revision `aa25c92404a40599614215e76514c79b427c7527` 的三个文件已下载到 `/mnt/data/SHM2026/models/mask2former-swin-large-ade-semantic-aa25c924`。权重实际 SHA 为 `b143c144341c15b4f20165cc6d2c9305fb1b66792f68a6e0e06d2b20dc063b14`，下载回执 SHA 为 `5942dd0b86acb3d01ab52b019c9a5ef1ff5a794338a91bfde20bc0cff0ae2fc3`。没有第二种权重格式，没有修改 uv 环境或 site-packages。[来源与作者配方差异](peer_style_mask2former_feasibility.md)

`mask2former_reference.py` 严格复用所有非分类张量，仅重新初始化完整六行分类头的 weight/bias 和五类 criterion 的 no-object 类别权重 buffer；不能借 `ignore_mismatched_sizes` 跳过其它差异。真实文件的 CPU 载入已验证 779 个复用张量逐位相等。模型共 215,451,514 个可训练参数，四个 Swin 阶段均训练；使用作者式 AdamW 分组（backbone LR 1e-5、其余 1e-4、weight decay .05、norm/embedding/位置参数免衰减）、全局梯度裁剪 .01。四步只用固定 LR，没有暗含正式训练调度。

输入固定为选定 legacy H3 共享场 `22bc8a2d…` 的 TRAIN002。相机来自 checkpoint `training_cameras`，准备阶段把其 350 项与已排序 TRAIN 名称及 manifest 相机逐位绑定。只新渲染一次 RGB，clamp 后四舍五入为 uint8，随后释放整个场；不读取真实 RGB 或其它场缓存。GT 与 valid 仅取同一 TRAIN002。四次更新固定为 crop/context/crop/context：中心原生 768×768；完整上下文 1025×768，按 HF 规则在归一化后以零补到 1056×768。mask/valid 对齐缩放，补边始终无监督。没有随机翻转、增强、VAL 输入或精度评分。

本机预检采用 **CUDA BF16 autocast，FP32 参数、Adam 状态及完整 criterion，无 GradScaler/loss scaling**。这与作者 FP16 AMP 配方不同，属于本机可行性选择，不称精确 SEM386 复现。出现非有限 loss/梯度就停止，不通过缩放回退或重试凑成功步数。保持已审查的 [valid-support criterion](mask2former_valid_support.md) 版本：有效 GT 整数中心采样、GT 整数 gather、预测标准双线性，主头/aux 同支持，不腐蚀细索。该离散采样也区别于作者连续坐标采样。

只对内部 Swin backbone 启用四阶段非重入 checkpointing，保留 RNG。CPU 合同验证了真实重算、带 drop-path 的前向及参数梯度一致。每个实际更新检查四阶段及其余参数的有限非零梯度和参数变化，并记录 loss、BF16 设置、非有限检测、计算耗时及 allocated/reserved 峰值。为了验证参数确实更新，峰值含一套约 .803 GiB 的更新前 FP32 克隆；计时分别列出训练计算与梯度检查，不能直接当纯训练吞吐。首步也包含 Adam 状态建立。只两个不同尺寸各两步，不能代表完整训练稳定性。

准备入口只做 CPU 来源绑定；执行入口必须使用新输出目录内的冻结脚本和 package snapshot。计划绑定 checkpoint、manifest、固定两张标签/valid 文件、uv.lock、三个官方模型文件、下载回执与本机相关 Transformers 源码的 SHA，执行前后复核并核验实际 import 路径。原 field/cameras、模型来源文件均不可改。内部 600 秒按完整更新边界检查；外层 660 秒硬时限提供失败回执余量，不能把内部检查说成可抢占的 600 秒上限。失败保留且不覆盖重跑。

CPU 验证：`CUDA_VISIBLE_DEVICES='' uv run --no-sync pytest -q tests/test_mask2former_reference.py tests/test_mask2former_valid.py`，26 passed；相关 Ruff 通过。GPU 必须等待根任务交接，然后只执行这四次更新。

## 唯一一次实际执行结果

根任务交接后按原冻结计划执行，自然 exit 0，四次尝试均为成功更新，没有重试、缩放回退或非有限 loss/梯度。四个 Swin 阶段和其余可训练参数每步均有有限非零梯度与实际参数变化，所有参数的缺失梯度数为零。GPU 已交还，没有开启 6k 训练。

| 更新 | 输入 | loss | 训练计算秒 | 含梯度审计秒 | peak allocated GiB |
|---|---|---:|---:|---:|---:|
| 1 | crop | 77.05346 | .46531 | .49488 | 4.73806 |
| 2 | context | 61.81089 | .17297 | .20181 | 7.08591 |
| 3 | crop | 45.31284 | .13778 | .16671 | 6.07449 |
| 4 | context | 43.57365 | .16609 | .19597 | 7.09269 |

reserved 峰值 8.04492 GiB，含上文审计克隆；总 worker 5.383 秒，不含进入 worker 时的首次输入哈希检查。稳态最后两步只能给 6k 约 15.2 分钟的乐观计算外推，完整数据读取、增强、记录、保存及后续评价仍需另计，四步不验证长期收敛。loss 来自相同一张 TRAIN 图的四次训练，不是分割精度。

产物位于 `/mnt/data/SHM2026/preflight/mask2former_reference_v1`：plan SHA `a36c55bd41e899a41d0cbeb01556f3e893c4e2202492ae6cec11d047eeeef291`，runner SHA `b3211914d978849f2fca849d32b5fbcc826789ca105063f52cd19639f6685ab0`，38 文件 source-tree SHA `e37e170c9b47d874d9cf161d62e6cdb0fa92470ca596a83dbd6b04ee71421a48`（以计划 source_hashes 排序紧凑 JSON 计算），[completed 回执](/mnt/data/SHM2026/preflight/mask2former_reference_v1/execution_receipt.json) SHA `c0a503cfd59a3e573738f5d8af6ea48cf25e0b738c5f54e8eafa0b4f4977032b`。source/input、RGB 场与相机全部不变。
