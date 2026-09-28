# 工程版本 F 的相机导出

使用本机uv环境和经过52个相机实际验证的冻结入口：

```bash
uv run --project /home/sky/workspace/SHM2026 --no-sync python -B \
  /mnt/data/SHM2026/runs/ibgs_joint_replacement_v2/deployment/render_ibgs_joint_bundle.py \
  --bundle /mnt/data/SHM2026/runs/ibgs_joint_replacement_v2/deployment/bundle.json \
  --cameras /absolute/path/to/cameras.json \
  --output /absolute/path/to/new_output_directory
```

输出目录必须尚不存在。入口自动顺序运行原IBGS独立uv环境及主gsplat/DINOv3 uv环境，不用pip或conda。完整运行成功才有status=completed的execution_receipt.json；子进程错误保留日志和失败回执，不能把部分结果当成功交付。

JSON可为单个相机或相机数组。每个相机必需name、K、w2c、width、height、distortion；可选image_id、camera_id、split。禁止照片、标注和图像路径字段。name必须为不含目录的.png文件名，数组中名字不能重复，名字和ID不参与来源选择。w2c为官方坐标系下camera-from-world，旋转必须合法。输出rgb/<name>为RGB PNG；mask/<name>为单通道uint8，0背景、1桥面、2拉索、3桥塔、4基础，禁止提交可视化颜色替代ID。

当前来源缓存和头只验证固定官方传感器：width=1320、height=989、K=[[925.7016189245708,0,660],[0,925.7016189245708,494.5],[0,0,1]]、distortion=[0.008987863345268233,0,0,0,0]。允许该传感器的不同位姿。来源按相机几何选择至多4个TRAIN邻居，距离严格介于.01与1.5、方向夹角小于30度；没有邻居明确报错，不承诺任意场外位姿或其他内参质量。改名和位姿偏移已实际核验。

清单绑定三套场、RGB头、DINOv3 H+/16骨干和语义解码器、350个TRAIN来源图、21.93GB来源层缓存、冻结Python/CUDA代码及两个uv运行时。来源照片是训练银行，不是目标/测试照片。语义教师始终读取H3自身渲染RGB，新组合RGB不输入教师。每张图三个场前向、一个选择器、一个教师调用；不做测试相机优化。

推理文件仍引用本机的绝对路径。迁移机器必须同时迁移这些资产、保持原模型SHA，并重新建立两个uv环境、验证CUDA扩展及运行时版本；只复制bundle.json或一个.pt不足以运行。完整成本和同协议指标见[结果](ibgs_joint_bundle_results.md)。
