# F 渲染域 DINOv3 教师迁移结果

## 目的与固定条件

旧域教师候选在 F 上退化后，本实验先用 F 的两个 RGB 成员（`top4_normalized` 与 MCMC）为 350 个 TRAIN 相机建立独立渲染缓存，再按原 `legacy_mixed_v1` 适配器生成教师训练域。缓存同时保存 350 张 `selected` 图、350 张等权 `composite` 图和 50 张留出评价域的 composite 图；缓存回执记录 700 次场渲染，`gt_payload_reads=0`。适配使用 float32 均值与 uint8 ties-to-even 舍入，未读取目标 RGB 或既有预测 PNG。

在同一 DINOv3 ViT-H+/16（ModelScope revision `98b7096b2938406ade58801a0bf75eca3198b5bd`，权重 SHA-256 `3e1d4d18…d1a4d`）上，从同一 warm-start 端点各训练 2000 步：

* `selected`：1000 个真实 RGB 域样本 + 1000 个 F top4 渲染域样本；
* `composite`：1000 个真实 RGB 域样本 + 1000 个 F 等权 composite 渲染域样本。

两臂使用相同 seed、相同相机/裁剪和预声明的域调度，关闭 VAL 读取与选择，固定使用 2000 步 `last.pt`。阶段回执显示两臂均自然完成，`validation_pixel_reads=0`、`domain_counts={real:1000, rendered:1000}`，训练 trace 的顺序、域计数和终端 RNG 一致。该实验是渲染域匹配的工程验证，不是跨桥或未见场景泛化训练；冻结 source snapshot 中增加的 `f_domain` 许可只放宽输入合同，没有改变教师网络或损失数学。

## 50/41 固定评价

两候选保持 F 的 RGB、H3 软读出和 0.5/0.5 融合，只替换教师。50 张 mask 全部写入并通过 hash 检查后才读取 41 张官方 JSON 标注。结果如下：

| 端点 | 五类 mIoU | 相对 F | 配对 95% 区间 |
|---|---:|---:|---:|
| F | 95.109016% | — | — |
| F RGB + selected 域教师 | 94.415452% | −0.693564 pp | [−1.393827, −0.002985] pp |
| F RGB + composite 域教师 | 94.414633% | −0.694383 pp | [−1.394497, −0.014049] pp |

两候选的 mIoU 门（至少 +0.20 pp、区间下界为正）均失败，缆索门也失败，因此不替换 F。该结果与旧域 `selected_2k` 候选（94.476038%，相对 F −0.632978 pp）方向一致：简单教师继续训练或把训练 RGB 换成 F 渲染域，都不足以提升当前 H3 语义读出。

运行目录为 `/mnt/data/SHM2026/runs/f_domain_teacher_adaptation_v1`。两个评价回执均标记 `natural_completion=true`、`prediction_count=50`、`annotation_reads=41` 和 `gt_reads_after_prediction_barrier=true`；独立评价脚本的 gate 结果保存在各自 `candidate_gate.json`。后续若继续做语义，应优先改变三维语义监督或融合读出并预先锁定消融，而不是继续无条件迁移教师。
