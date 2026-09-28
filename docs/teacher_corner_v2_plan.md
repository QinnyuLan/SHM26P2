# V2 教师候选配置与容量计划（未执行）

`configs/teacher_corner_v2.json` 仅将 `teacher_strong_v1.json` 的 manifest 和输出目录改为 `artifacts/prepared_corner_v2/manifest.json`、`runs/teacher_corner_v2`，训练配置逐项完全相同：6000 步、768 crop、192 通道、原学习率/EMA/一致性分支、3000 步后的 50% 全景课程、seed20260926、每 1000 步评价完整 41 张标注验证图。现有历史 checkpoint 中重叠的有效配置也逐项一致；后加的 adapter/domain/推理配置均为关闭默认值。

复用已下载的 ModelScope DINOv3 H+/16 冻结骨干，从新 decoder 初始化，不跨协议 warmstart 旧头，不恢复旧 optimizer、不使用 LoRA。新 manifest 为 `colmap_corner_v2`，SHA256 `91b41aeecedc352e4882eb80a1bb635d479ae432251a4c96c85a9dc110cbb327`。本配置训练期间只保存验证 JSON 与 best/last，不导出另一个 350 张概率缓存，也不消费 legacy `pseudo_strong_v1`。

配置 SHA256 为 `5ebd8bfc70bde7ed9a870351615f8f21c5758aa105b9ba37326730de27da3644`，完整 CPU 计划为 `configs/teacher_corner_v2_plan.json`，状态为 **prepared_not_started**。须先完成已授权的 v2 RGB 30000 步、语义 8000 步及共同 official-grid 评价，再决定是否运行；当前未开启教师训练，没有为它占用 GPU。

六个已有 head-only best/last 实测单文件最大 **97.416 MiB**。其中 decoder tensor 为 25,505,304 bytes，EMA 为同样大小，Adam tensor 为 51,010,880 bytes；骨干不重复写入 checkpoint。

| 教师 checkpoint 场景 | 占用 |
| --- | ---: |
| 单文件实测上限 | 97.416 MiB |
| best + last 常驻 | 194.833 MiB |
| 两份常驻 + 原子替换临时文件 | 292.249 MiB |
| 单文件增加 10% 裕度，三份并存，再留 16 MiB 日志/来源文件 | 337.474 MiB |

预算沿用已批准 v2 的保守槽位：RGB final 447 MiB、semantic final 241 MiB、每次 native 评价 75 MiB、源及日志 5 MiB，留存底线 256 MiB。考虑已生成的 v2 文件与当时约 1730 MiB 空闲空间，合计可用于这些未来槽位的容量约 1847 MiB；以下不需要删除任何旧结果。

| 顺序阶段峰值 | 预计剩余空间 | 超出 256 MiB 保底的余量 |
| --- | ---: | ---: |
| RGB 原子替换：两份 RGB checkpoint | 948 MiB | 692 MiB |
| RGB 保留后，semantic 原子替换 | 838 MiB | 582 MiB |
| 两个 v2 阶段及 native 评价完成后留存 | 1004 MiB | 748 MiB |
| 上述全部留存，再进行教师原子替换 | 667 MiB | 411 MiB |
| 再暂留 225 MiB 给额外 official-grid 评价 | 442 MiB | 186 MiB |

若直接将教师峰值叠到当前 v2 计划的全部保守预留峰上，预计仅余 **220 MiB**，低于 256 MiB 保底。因此不能用这份计划为并发训练提供容量保证。顺序等待 v2 两阶段完成后可能足够，但余量不大：开始前必须重新读取实际 checkpoint 大小、official-grid 输出占用和可用空间。这只是运行期间的一次估算，没有取得磁盘预留，也没有删减或压缩旧权重。

原 v1 6000 步历史总耗时约 **1838.9 秒（30.6 分钟）**，包含验证和当时其他 GPU 作业；这不是独占基准或 v2 的时长保证。该候选属于坐标协议修正后的教师工程对照，不因使用更大骨干或重训而获得学术创新性，也不能将真实 RGB 教师分割分数冒充 camera-only 联合渲染指标。
