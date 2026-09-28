# IBGS 官方实现接入可行性（只读）

2026-09-27。结论：**可作为完整强工程参考的移植候选，但当前环境不能直接运行，尚未证明能在5090编译/训练。**本次仅从[作者项目 Code 链接](https://hoangchuongnguyen.github.io/ibgs)克隆官方仓库到 `/mnt/data/SHM2026/third_party/ibgs`，固定 commit `977e96c6574f20b456c760942e615031154b4cd5`，工作树干净；无安装、编译、模型下载、GPU或项目依赖修改。固定ED两次阴性不排除IBGS。

**实际方法与接口。**[作者论文 §3.2–3.6](https://arxiv.org/html/2511.14357v1)及当前代码使用围绕透射率中位贡献位置的多个Gaussian射线/平面交点，而非整个射线的单ED；默认 `buffer_length=4`，每Gaussian另有可学习normal/offset。交点处源RGB按贡献权重聚合，再由每源7维输入MLP、均值聚合及9个卷积层的hourglass预测RGB残差；有源深度过滤、几何/photometric损失与联合训练。入口是 `gaussian_renderer.render(camera, GaussianModel, Scene, pipe, args, …)`，输出 `render/warped_image/cam_feat/median_intersected_depth/min_depth_diff/camera_ray`，接 `color_aggregation_network.fuse_color`；不是把本次λ换成网络即可复现。关键源码：[CUDA交点/过滤](https://github.com/HoangChuongNguyen/ibgs/blob/977e96c6574f20b456c760942e615031154b4cd5/submodules/diff-plane-rasterization/cuda_rasterizer/forward.cu#L437)、[融合网络](https://github.com/HoangChuongNguyen/ibgs/blob/977e96c6574f20b456c760942e615031154b4cd5/color_aggregation_network.py#L154)、[训练目标](https://github.com/HoangChuongNguyen/ibgs/blob/977e96c6574f20b456c760942e615031154b4cd5/train.py#L303)。

| 接入项 | 已核事实及最小工作 |
|---|---|
| 构建/依赖 | [README](https://github.com/HoangChuongNguyen/ibgs/blob/977e96c6574f20b456c760942e615031154b4cd5/README.md#L28)推荐Python3.8、Torch2.1.2/cu121；本机3.11、2.8/cu128不是作者已验证组合。两个扩展为 `diff_plane_rasterization`、`simple_knn`，不能替换为现有gsplat二进制。setup.py走Torch CUDAExtension且未硬编码架构，本机Torch构建工具支持12.0；独立CMake默认却仅70/75/86。旧Tensor.data<T>()本机仍有兼容包装，未找到已证实移除API阻断，但未编译不能保证。requirements未pin且含Git PyTorch3D；本机缺PyTorch3D/plyfile/open3d/trimesh/tensorboard，原入口import即需它们。应另建固定依赖的隔离uv环境，不动现有uv.lock。 |
| 相机/数据 | reader只支持已去畸变PINHOLE/SIMPLE_PINHOLE，支持显式 `split.json`；写入既有350/50名单而非默认每8图切分。可用现成corner-v2 preparedRGB、TRAIN点云/原pose，但必须核像素中心：`Camera`忽略输入主点并取W/2,H/2，CUDA射线用整数pix减该值，投影 `ndc2Pix` 又是 `(W−1)/2`中心；不能假定与当前corner `j+.5` 数值相同。该差异须在数据/相机适配与二维投影测试中显式处理，不把旧PNG当identity。所有源使用目标同尺寸/同内参的核假设，本数据同K可满足；任意K部署需另扩接口。 |
| checkpoint/目标 | 当前1M `.pt`可导出means/SH/logscale/quat/logit-opacity，但没有其normal/offset或融合网络；普通PLY也不满足其含 `nx,ny,nz,nd` 的加载格式。要么完整作者流程从同TRAIN初始化重训，要么明说1M暖启变体，不能称精确复现。作者默认SH2、30k、最高5M点、7k启几何损失、10k启融合；与当前SH3/1M容量、AA、区域RGB权重不同。原RGB loss未接当前valid_mask，不能悄悄把黑边当有效监督。 |
| 存储/成本 | `Scene._initialize_train_buffers`驻留TRAIN RGB与深度；350×1320×989的FP32 RGB约5.11GiB、单通道深度1.70GiB，stack瞬时复制与CUDA raster工作区另算。可用CPU源bank按4邻相机上传，但这是需验证的内存工程改动。原train每步主渲染并更新深度缓存，test入口可额外渲染源深度；不能按本轮78次ED计时推断完整IBGS速度。 |
| 许可 | 仓库根目录未提供LICENSE，部分Python及simple-knn头部引用缺失的根 `LICENSE.md`；唯一明确主算法许可证在[diff-plane子目录](https://github.com/HoangChuongNguyen/ibgs/blob/977e96c6574f20b456c760942e615031154b4cd5/submodules/diff-plane-rasterization/LICENSE.md)，限定非商业研究/评估并要求保留许可与归属。未见IBGS新增网络/修改的独立明确授权，不能把整仓称MIT/任意商业或比赛重分发可用；正式打包分发前需要澄清授权范围。这里不推断公开仓库自动授予这些权利。 |

**实施建议与工期估计。**先按上述commit做独立参考端口：预计0.5–1个工程日固定依赖与验证两个扩展；再1–2日完成350/50数据、主点/像素中心、valid域、checkpoint/任意相机入口及少量前向反向合同。此为人力估计，不是执行承诺；30k训练/GPU预算必须用新扩展实测再定，未执行前不给虚假分钟数。原 `Scene`会加载相机照片，交付推理应构造不读取目标RGB的camera对象，仅源TRAIN资产可读。

**公平性边界。**把它列为已有IBGS工程强参考；保留普通ED阴性。完整重训应固定同TRAIN/初始化/官方50图RGB评分、最终checkpoint，披露不同容量、额外源资产、网络/法向参数和实际总成本；暖启版需有同底座同预算无IBR控制，不能把整个作者配方的提升归为单一深度机制。普通RGB残差网络、交点采样和多视图photometric目标已有直接先例，不宣称本项目创新。语义可保留既有camera-only分支而不声称新语义收益；若声称共享几何/语义改善，必须另做同场真实语义评价。
