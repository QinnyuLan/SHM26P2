# 同场 Mask2Former 固定参考：训练与评价合同

2026-09-27：四更新预检、259 张同场 TRAIN 渲染缓存、固定6,000次成功更新及共同原图50/41评价均已自然完成，CPU终点与结果复核通过。Swin五类94.3184%，当前已选H3＋H+为95.1090%，同场差值+0.7906pp、95%配对区间[+0.1362,+1.6536]；RGB逐文件相同。完整成本、历史选择与对照边界见[结果报告](mask2former_reference_results.md)。这项工程参照用于共同划分与共同 RGB 场的性能比较，不是 SEM386 精确复现，不把容量、域差异或标准 Mask2Former 训练作为创新。

唯一配置为 `configs/mask2former_reference_v1.yaml`。RGB 场固定 selected legacy H3 checkpoint `22bc8a2ddb260f93cb01b17857c97b2bb0873038efdb9318545cb2bdbb045226`；使用同 259 个官方有标注 TRAIN mask，完全不读取真实 RGB 作模型输入。其 259 张原生渲染 PNG 来自精确的 checkpoint 350 TRAIN 相机名单和姿态，scene tensors/cameras 前后逐位不变。缓存回执 `/mnt/data/SHM2026/runs/mask2former_reference_v1/train_render/cache_receipt.json` SHA `57c31c878115bc7ee343e34f025a0413cbb1db1a339bd6709a908ad8d436d2af`；训练准备会重核每一 PNG/label/valid 与所有源依赖 SHA。

固定 6,000 次成功 AdamW 更新，batch1，seed20260805。每个 epoch 将 259 个视角独立打乱；奇数步均匀随机原生 768 裁块，偶数步 legacy Pillow RGB bilinear / label+valid nearest 完整 1025×768 上下文；翻转概率 .5，补到32倍数的像素无监督。视图 RNG、增强 RNG、point-sampling CUDA RNG、torch/drop-path RNG 分离，续训保存全部状态。裁块若无有效像素就显式停止，不重新采样选择有利类别。

官方 ADE Swin-L 权重严格复用，仅重置五类＋no-object 六行分类头与对应类别权重 buffer，四阶段及 decoder 全训练。AdamW/分组/clip .01 与[预检](mask2former_reference_preflight.md)相同；倍率 `(1-(step-1)/6000)^.9`，首步完整 LR、末步仍正。BF16 autocast 无 loss scaling，匹配及完整 valid criterion、参数和优化器保持 FP32。不存在 AMP 跳步补步。Swin drop-path 能合法地产生某 block 的全零梯度，因此每步固定 query 探针仅要求梯度存在且有限，nonzero 只记录；最终探针参数必须发生变化，四阶段整体更新已由独立预检验证。没有中途 VAL、早停选点、教师、伪标签或融合。

`scripts/train_mask2former_reference.py prepare` 在根任务复核后才冻结：继承已执行的 preflight package，覆盖本次新增训练模块及三个参考脚本，保持旧 Gaussian/model/坐标/criterion 实现，不纳入并行 RGB 参考改动。每100步记录 loss、LR、时间、显存与梯度探针；每步保存轻量视角/裁块/翻转 trace；每1000步原子替换 `last.pt`，含完整 model/Adam/采样器/四类 RNG/trace/配置与来源。可显式恢复到新的输出目录，协议、输入和源码不得改变；不自动重试。最终仅 step6000＋completed 回执可进入评价。训练外部硬预算2400秒，内部2280秒在下一步前停止，失败时保留前一份完整周期 checkpoint，不保存半步状态。

推理在 `mask2former_training.predict_tiled` 固定 768 tile、512 stride，最后 tile 锚定边缘，没有 TTA 或全图 context 混合。输入来自本场 pinhole/overscan RGB 的 clamp→round uint8；使用一次 ImageNet 归一化及归一化后的零 padding。FP32 将 mask logits 双线性恢复到 padded tile 尺寸，再 sigmoid，与 query class softmax（去掉 no-object）求和。重叠处先均匀平均五类原始 score，最后按五类和归一化；这不同于先对每个 tile 强行归一。恢复有效 tile 区后再累积，不输出 argmax 中间量。与 HF 默认先固定到384的图像处理器路径不同，此处显式使用模型输入网格；不能称 HF 默认推理的逐位复现。

`scripts/evaluate_mask2former_reference.py` 使用已有 official 原始网格投影/覆盖画布/畸变 warp、GT rasterizer、PSNR/SSIM/LPIPS/CM scorer 与 fingerprint，不重写评分公式。模型只读取原生相机渲染 RGB，不读取场语义头、teacher 输出或 VAL 照片。全部50个预测完成后 scorer 才打开源 VAL RGB/标注；每张输出 RGB PNG 的实际 SHA 必须与 selected H3 历史官方评价相等，最终50/41 fingerprint必须完全相同。只有此固定终点参与后续 paired comparison，没有训练采用门槛或多权重搜索。

预算依据是已测稳态 crop/context .1378/.1661 秒，约15.2分钟仅为6k计算外推，加入PNG读取/增强/状态保存后预留40分钟硬限；不把四步小样本当正式吞吐保证。预检峰值7.093GiB含约.803GiB审计克隆，正式训练不逐步克隆整个模型。下载 .807GiB，完整 model/Adam checkpoint 约2.41GiB，周期原子替换短时约4.82GiB，缓存另计；产物均在空间充足的数据盘。ADE语义预训练且全模型可训练215.45M，与冻结 DINO H+＋DPT6.38M并非等容量或等预训练任务对照。

CPU 冻结计划为 `/mnt/data/SHM2026/runs/mask2former_reference_v1/preparation/plan.json`，SHA `de7fc9b821e3700066a9db18107155d1cd6a46ad38f4873085f473384f0e7600`；41文件 source-tree SHA `0317930c4afbbd976dcd621876b39465e8396e230329ec5f43c42b67397f7bd2`（source_hashes 排序紧凑 JSON），train/eval runner SHA 分别 `c3eab41c04cb8beecffee41e9ad740b88250197229ed981240cda7492e13565e`／`8c038d42c3a53fa00a89123e377b560ad6d8acdf04e35170a63ab4bf14e40d05`。533项实际来源哈希与259视角绑定通过，CPU准备阶段实际 import 路径通过且 CUDA 未初始化；48项 CPU 合同及 Ruff 通过。正式训练和评价随后由根任务按该冻结计划执行，均自然退出，无中途方案更改。

已执行的训练命令（留作来源记录，不自动重跑）：

```bash
PYTHONPATH=/mnt/data/SHM2026/runs/mask2former_reference_v1/preparation/source_snapshot OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 timeout --signal=TERM --kill-after=10s 2400s uv run --no-sync python /mnt/data/SHM2026/runs/mask2former_reference_v1/preparation/source_snapshot/train_mask2former_reference.py train --plan /mnt/data/SHM2026/runs/mask2former_reference_v1/preparation/plan.json
```

已执行的唯一固定终点评价命令：

```bash
PYTHONPATH=/mnt/data/SHM2026/runs/mask2former_reference_v1/preparation/source_snapshot OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 uv run --no-sync python /mnt/data/SHM2026/runs/mask2former_reference_v1/preparation/source_snapshot/evaluate_mask2former_reference.py --plan /mnt/data/SHM2026/runs/mask2former_reference_v1/preparation/plan.json --checkpoint /mnt/data/SHM2026/runs/mask2former_reference_v1/training/last.pt --output /mnt/data/SHM2026/runs/mask2former_reference_v1/official_evaluation
```
