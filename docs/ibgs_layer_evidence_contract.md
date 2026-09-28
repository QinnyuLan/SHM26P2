# 固定四层×四源的同 ID 质量支持

`bridge_rgs.ibgs_layer_evidence.build_layer_evidence` 是独立、无梯度的 PyTorch 前向模块，不渲染、不编译，也不在 import 时初始化 CUDA。当前只完成 CPU 合同；实际训练和质量收益尚未验证。

目标 `ids/depth/weights` 为 `[H,W,4]`，分别是选择器的 ID、有限正平面深度及原 `alpha*T`；`base_rgb` 为 `[H,W,3]`。四个 source 字典固定原槽位，包含 `ids/depth/weights[H,W,4]`、`rgb[H,W,3]` 和已经 erode 的 `valid[H,W]` bool。所有浮点输入为实际 FP32，ID 为 int32/int64，同 device、连续布局、共同网格和内参。空槽 ID=-1；零质量、非法深度或非有限量不能产生证据。

显式关键字包括：`ref_to_src[4,4,4]`、`focal[2]`、**数组中心** `principal[2]`、行主序 `target_world_to_camera[4,4]`、`target_campos[3]`、`source_campos[4,3]`。相机矩阵必须是 caller 实际使用的值，不在这里重算/正交化。原图像素中心为整数：目标相机点 `((u-cx)*z/fx, (v-cy)*z/fy,z)`，运算采用先乘深度再乘 reciprocal；源投影使用原 `ref_to_src` 与 `1/(z_source+1e-8)`。只有 finite、`z_source>0` 且闭区间 `0≤u≤W-1, 0≤v≤H-1` 可采样。此接口仅覆盖原 IBGS 共同内参与尺寸范围，不假装支持任意 source K。

采样明确采用整数数组坐标上的**理想双线性**，边界 tap clamp。它不是 CUDA texture 的 8-bit 小数舍入复现；所有未来对照必须共用此采样器。source RGB 始终普通双线性，不按 ID、valid 或质量重加权。连续支持为：

`q = Σ_four_taps β_t · valid_t · Σ_four_source_slots [ID_s=ID_target, finite(z_s)>0] · w_s`。

没有 median 深度门、层深度误差门、质量阈值或保留质量归一化。source invalid tap 只把该 tap 的 q 贡献置零，不重归一 RGB。有限正平面支持与 eroded-valid 分开核；同一 tap 重复 live ID 会报错，防止重复算质量。目标质量为零、投影无效或实际采样特征非有限时 q=0、特征为零。保留极小正质量，不将其变成可靠性二值标签。

输出 `features[N,4,4,7]`、`support[N,4,4]`、`target_weights[N,4]`，N=H×W。七维为源 RGB−base RGB、**target world center−source world center**、该层点两视角单位射线 cosine。世界点沿原 shader `R_targetᵀ(p_camera−t)` 构造，射线归一化沿原 `norm+1e-8`。源数不压缩、层不重排。所有输入 detach；原正质量不变，非法目标槽返回零质量。caller 继续向融合器传同次原 render 的 `camera_ray`，不以本模块逐层射线替换 CNN 的共同 ray 输入。

默认 `chunk_pixels=65536`，只限制中间计算；CPU 已覆盖 chunk=1 与大于总像素数的逐项相同输出。完整输出本身不会变小：1320×989 下三项 FP32 输出合计约 689 MB（features 585 MB、support 84 MB、weights 21 MB），另需四源缓存与单 chunk 临时量。按源逐一处理，每步不建立全图×4tap×4cache-slot 的所有源联合临时张量。输入检查会做 device 同步，尚无吞吐性能声明。

融合器必须继续在 MLP 前清零无效特征、在 MLP 后乘 support/原目标质量，最后无源像素精确返回 base。source q 是模型同 ID 的贡献质量，不是颜色正确概率、遮挡真值或物理索身份。本模块不控制 normalized 对照；该因素应只在独立融合器中显式改变，三个 selector/control 共用同一源支持策略。
