# TRAIN 残差重投影的坐标合同

2026-09-27，CPU 源码与元数据核查。没有读取 RGB/标签/valid 像素、启动 GPU、加载 checkpoint 或改动训练产物。本文规定坐标接口，不预设残差运输的性能。

## 1M 场实际采用 corner 网格

此处使用的 [1M 冻结计划](/mnt/data/SHM2026/runs/rgb_capacity_1m_reference_v1/plan.json) SHA 为 `40191687627d54da91b78c8b819e83bc04b04b986ae6568efe042c6dbcd447f3`。它的配置是 **`colmap_corner_v2`，不是 H3 的 `legacy_mixed_v1`**。对应 [manifest](/home/sky/workspace/SHM2026/artifacts/prepared_corner_v2/manifest.json) SHA 为 `91b41aeecedc352e4882eb80a1bb635d479ae432251a4c96c85a9dc110cbb327`。[既有 CPU 终点审计](/mnt/data/SHM2026/runs/rgb_capacity_1m_reference_v1/cpu_endpoint_audit.json) 绑定 `training/last.pt` SHA `35b489fe45ad34ad279018609b6d5eaf6493e4c0d82f9c86ec16925268078092`，确认原始 350 TRAIN 相机；本次不重新反序列化该模型。

350 张 TRAIN 均为 1320×989，`w2c == w2c_original`；未优化相机。唯一源相机为 SIMPLE_RADIAL：`fx=fy=925.7016189245708, cx=660, cy=494.5, k1=0.008987863345268233`，其余 OpenCV 畸变参数为零。manifest 的 K 是 corner K，不能写回减半像素的 K。

实际训练入口是冻结的 [io.load_view](/mnt/data/SHM2026/runs/rgb_capacity_1m_reference_v1/source_snapshot/bridge_rgs/io.py:34)，而非仅凭 Dataset 类推断：prepared RGB 经 `INTER_AREA` 缩放；K 先 FP32，再分别把整行乘实际宽、高比。`progressive_resolution=True` 在训练前 15%、随后至 60%、最后阶段采用约 .5/.75/1：尺寸分别为 660×494、990×742、1320×989。高度比例是 `494/989`、`742/989`，不能替换为精确 .5/.75。最终原生缓存使用 scale=1，不能拿早期训练分辨率的 K 或深度直接混用。上述冻结 `io/data/coordinates/model/prepare/train` 六文件已逐项与计划 SHA 匹配。

## 深度与可见性

冻结 [model.render](/mnt/data/SHM2026/runs/rgb_capacity_1m_reference_v1/source_snapshot/bridge_rgs/model.py:115) 使用 gsplat `RGB+ED`、`antialiased`、near=.01、far=1e6。安装的 [gsplat rendering.py](/home/sky/workspace/SHM2026/.venv/lib/python3.11/site-packages/gsplat/rendering.py:614) 把每 Gaussian 的相机中心深度 `z_i=(R mu_i+t)_z` 当附加颜色通道，背景该通道为 0；末端在 [第 760 行](/home/sky/workspace/SHM2026/.venv/lib/python3.11/site-packages/gsplat/rendering.py:760) 返回

`ED(p) = Σ_i W_i(p) z_i / max(alpha(p), 1e-10)`。

它是合成权重下的 **相机 z 期望值**，不是欧氏射线距离、最近表面交点、深度中位数或真实表面。无覆盖时不能把零值当有效三维位置。多层混合的 ED 可能落在空处；alpha 和跨视图 ED 一致性只能作已声明的支持代理，不能证明遮挡正确或排除共同浮层。不要再次除 alpha，也不要把归一化的单位射线直接乘 ED。

## 推荐的同 prepared 网格运输链

使用 `prepared_corner_v2/images` 的源 RGB 及其 valid，与同相机原生渲染 RGB/ED/alpha 配对。具体允许源名单、alpha/深度门、残差权重由独立实验协议固定；这里不增加或选择阈值。

1. 缓存绑定 **实际送入渲染的 FP32 K 和 w2c**，以及实际宽高；CPU 几何可将这些数值转 FP64 运算，但不宣称与 FP32 CUDA 逐位等价。gsplat [像素位置](/home/sky/workspace/SHM2026/.venv/lib/python3.11/site-packages/gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu:62) 为 `p_t=(j+.5,i+.5,1)`。
2. 令 `r_t=K_t^{-1} p_t`，取 `r_t/r_t.z` 保持 z=1，则 `x_t=ED_t*r_t`。世界位置用 `X=inv(w2c_t) [x_t,1]`；理想刚体下等价 `R_t^T(x_t-t_t)`。对 FP32 保存矩阵求 FP64 逆可避免在纯坐标 roundtrip 中额外假定其严格正交。
3. 源相机点 `x_s=R_s X+t_s`，要求有限且 `z_s>0`；投影得到 corner 坐标 `u_s=(K_s x_s)[:2]/z_s`。比较深度时用这个 **源相机 z_s** 对比源 ED，不比较目标 z 或射线长度。
4. prepared/render 数组的整数元素中心是 `(0,0)`。OpenCV `remap` 消费 **`u_s-(.5,.5)`**；PyTorch `grid_sample(align_corners=False)` 则消费 **`2*u_s/[W_s,H_s]-1`**。两者不得再叠加一次 .5。若 align_corners=True，使用 `2*(u_s-.5)/([W_s,H_s]-1)-1`，不能直接使用上一公式。
5. 源残差须在同一 native prepared 数组上先形成，例如 `I_s_prepared - R_s_native`，再按同一个源位置采样。RGB 的 clamp/uint8/浮点选择是独立光度合同，必须预先统一；不能直接把原畸变 RGB 与 pinhole 渲染逐元素相减。样本支持应显式处理有限值、边界及所有正权双线性 tap 的 valid；`padding/clip` 不能把缺失样本伪装为有效源颜色。连续数组内完整双线性中心域对应 corner `u∈[.5,W-.5], v∈[.5,H-.5]`。

冻结 [prepare 的 map](/mnt/data/SHM2026/runs/rgb_capacity_1m_reference_v1/source_snapshot/bridge_rgs/data.py:182) 已用 `K_array = K_corner` 仅把主点减 .5，分别传入原图与目标图 OpenCV map；两端的 K 转换只在 raster API 临时使用。无需再对已有 prepared RGB 做一次去畸变或 official 导出 warp。

## 若必须采样原畸变源图

这与 prepared 源图是不同的重采样路径，不能声称像素逐位相同。源相机归一化坐标 `(x,y)=x_s[:2]/z_s`，本数据用 `a=1+k1*(x²+y²)`，再得 `u_d=(fx*a*x+cx, fy*a*y+cy)`；OpenCV 源数组索引仍是 **`u_d-.5`**。等价地 `cv2.projectPoints` 使用临时 `K_array`，畸变参数不变。

从原图目标数组索引反投影则先 `+.5`，用原 corner K/distortion 去畸变到归一化射线；不要将两端转换各做两次。若最终导出到官方原图，复用 **CORNER** 的 `distortion_render_grid`，并保留返回的 overscan K/尺寸：该函数执行 `array+.5 → undistortPoints → corner-.5` 后的 canvas 平移。H3 legacy adapter 的无 .5 分支不适用于此 1M 场。

最低限度 CPU 合同是同相机 identity、非中心主点、非整数缩放、畸变前后接口、source z 与 range 的区别、边界/invalid tap 拒绝。OpenCV 的实际双线性插值有相位量化；坐标自洽不等于与另一采样后端逐位相同。

## 数据与解释边界

16 个 target 即使从残差 source 名单剔除，也已被 1M 场训练看过。这只能叫 **TRAIN source-leave-out 残差运输诊断**，不是模型 OOF、盲测或泛化证据。target 的 RGB/标签不得参与源选择或预测；若用于本轮评分，须遵守预测完成后的读取屏障。本次源图只允许固定 TRAIN 子集，不读取 VAL 资产。

本次重读本地 [SHM_2026.pdf](/home/sky/workspace/SHM2026/SHM_2026.pdf) 项目二（印刷页 9–10）和 [Dataset/README.md](/home/sky/workspace/SHM2026/Dataset/README.md)：未发现明确禁止提交方法保留 TRAIN 照片资产的条款；这不是主办方对该具体形式的额外确认。交付仍须从提供的测试视点生成 RGB/官方类别语义图，并如实列出源照片、相机和缓存的依赖、存储、推理成本。已有 [可实施性记录](/home/sky/workspace/SHM2026/docs/train_photo_transport_feasibility.md) 的 IBR/非创新边界保持；这里没有重做文献搜索，也不把坐标正确当作性能保证。
