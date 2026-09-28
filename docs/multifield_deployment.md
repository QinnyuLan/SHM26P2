# 新相机 RGB 与语义导出

当前可用的高成本工程版本 E 使用三套已冻结高斯场：1M/MCMC 输出 RGB 原图均值，原 H3 场与 ModelScope DINOv3 H+ 固定各半输出语义。它在共同原图开发集达到 **30.035661 dB / .874420736 SSIM / .264741845 LPIPS / 95.109016% mIoU**。三场共 1,994,145 个高斯，不是单场共享几何，不构成已验证学术创新。

入口支持任意给定的原图相机，不依据文件名查询已有预测。模型及主干、像素协议、实际加载源码、依赖版本均固定并校验。

```bash
uv run --project /home/sky/workspace/SHM2026 --no-sync python \
  /mnt/data/SHM2026/runs/multifield_h3_teacher_v1/deployment/render_multifield_bundle.py \
  --bundle /mnt/data/SHM2026/runs/multifield_h3_teacher_v1/deployment/bundle.json \
  --cameras /absolute/path/to/cameras.json \
  --output /absolute/path/to/new_output_directory
```

`--output` 必须是尚不存在的目录。`cameras.json` 可以是一个相机对象或对象列表。每个对象必需字段为：

| 字段 | 内容 |
|---|---|
| `name` | 不含目录的唯一 `.png` 文件名 |
| `K` | 原始畸变图像的 3×3 内参 |
| `w2c` | 官方世界坐标系到相机坐标系的 4×4 外参，平移与训练场同单位 |
| `width`, `height` | 原图整数尺寸 |
| `distortion` | OpenCV 顺序的畸变系数，4/5/8/12/14 项；无畸变时显式填写四个零 |

可选 `image_id`、`camera_id`、`split` 仅作为元数据保留。照片、标签、valid map 或其他额外输入字段被拒绝。需要使用数据集官方相机坐标系；该入口不会自动配准另一个坐标系。

输出 `rgb/*.png` 为三通道 uint8 RGB 文件，`mask/*.png` 为单通道类别 ID：0 background、1 deck、2 stay_cable、3 tower、4 foundation。另保存两个 RGB 成员文件和包含相机、模型来源、输出哈希、实际场渲染数及成本的 `execution_receipt.json`。临时失败保留 failed 回执，不覆盖已有结果。

## 已执行验证

[两相机输入](/mnt/data/SHM2026/runs/multifield_h3_teacher_v1/deployment_test_cameras.json)的第一项采用验证相机001的参数，但改名 `reproduction.png`；RGB 与 mask 均逐字节复现本轮 E。第二项改名 `novel_shift.png`，同时把 `w2c[0,3]` 增加0.1世界单位；实际生成新的1320×989输出、合法0–4类别，RGB变化1,282,952像素、mask变化115,023像素。没有这个新位姿的 GT，因此该检查证明导出功能，不衡量新位姿质量。

从 `/tmp` 启动的真实 GPU 运行自然退出0，完成6次 scene 调用及2次完整教师预测，耗时14.776898秒、峰值 allocated 3,544,005,632字节。包含模型加载、输出和来源检查，保留桌面 GPU 应用，不是独占 GPU FPS。GPU推理没有读取GT照片或标注。

- [部署模型清单](/mnt/data/SHM2026/runs/multifield_h3_teacher_v1/deployment/bundle.json)，SHA `323b9620e8d596021c643fbe51638a54e55f93a51deb2d5f10a5335d75ad8ae4`。
- [实际导出审查](/mnt/data/SHM2026/runs/multifield_h3_teacher_v1/deployment_smoke/review.json)及[运行回执](/mnt/data/SHM2026/runs/multifield_h3_teacher_v1/deployment_smoke/execution_receipt.json)。
- [50/41正式对照的独立审计](/mnt/data/SHM2026/runs/multifield_h3_teacher_v1/independent_cpu_review.json)核实全部软概率融合、PNG、混淆矩阵和比较区间，见[结果](multifield_h3_teacher_results.md)。

清单引用当前数据盘上的模型和冻结包，不复制数GB权重；搬到另一机器时需要同时迁移依赖并重建路径清单。研究用源入口位于 `scripts/render_multifield_bundle.py`，实际验证使用上面锁定的副本。环境继续由项目 `uv.lock` 与 `.venv` 管理。
