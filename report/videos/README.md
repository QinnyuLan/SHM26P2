# 渲染视频索引

这些视频均为本地渲染结果，编码为 MP4/H.264-compatible MPEG-4，12 fps，帧顺序按相机编号排列。

| 文件 | 内容 | 来源 |
|---|---|---|
| `full400_rgb.mp4` | 全量 400 视角 RGB 拟合渲染 | 全量 400 视角模型 |
| `full400_semantic_mask.mp4` | 全量 400 视角彩色语义 mask | 全量 400 视角模型 |
| `previous_final_rgb.mp4` | 上一版最终多场 RGB 渲染 | 350/50 版本 |
| `previous_final_semantic_mask.mp4` | 上一版最终多场语义 mask | 350/50 版本 |
| `unlabeled_301_400_rgb.mp4` | 301–400 无标注图像 RGB 评估渲染 | 上一版部署模型 |
| `unlabeled_301_400_semantic_mask.mp4` | 301–400 无标注图像语义 mask | 上一版部署模型 |

全量 400 视角模型的 RGB/语义指标见 `runs/full400_evaluation/`；原始帧和完整 JSON 指标保留在该目录中。由于全量模型使用了全部 400 个视角，相关结果是训练拟合诊断，不是独立泛化测试。
