# IBGS revision 1：已完成的后端验证

2026-09-27。修订后的官方 IBGS 后端在 RTX 5090 / Torch 2.8.0+cu128 上完成三项固定合成检查及两张真实 TRAIN 的接线预检，根代理均观察到自然 exit 0。**这是端口可运行性、局部梯度和采样支持验证；尚不是本桥真实重建质量、完整梯度正确性或训练收益的证据。** 没有训练出的 IBGS 候选或新 VAL 成绩。

测试使用 [revision 1](ibgs_port_revision1.md) 的二进制 SHA `436b2b55df2b37cddda59e606ec0855ad125eb211058b502197674bd2b93adaf`，simple-knn SHA `2bfc7f1b145fe8a91fe5f63327a49bb662687d0749014677f1d01142b48d8f2f`。没有调整原烟测阈值；这是明确的工程端口，不是原作者实现的逐位复现，也不构成方法创新。

| 固定检查 | 实测结果 | 调用与内部 wall |
|---|---|---|
| Median-depth plane-distance 导数 | 0/1/4 个有效源 × 两方向 × 三个固定 h，共 **18/18 通过**；最大绝对 FD 误差 `4.63724e-5` | 39 forward、3 backward；1.370269 s |
| 后端＋融合网络烟测 | identity 源 RGB 最大差 `1.01328e-6`；内部支持率 1；plane-depth 最大误差 `4.05312e-6`；depth-only 与普通深度差 0 | 2 forward、1 backward；1.562423 s |
| 无效 source RGB 污染检查 | 无效 RGB 从 `+10000` 改为 `−10000` 后，全部保存输出与 fused RGB **逐值相同**；2218 支持像素、200 个混合有效性交点见证像素 | 2 forward、0 backward；1.395564 s |

第一项固定 h 为 `.002/.004/.008`，绝对容差 `1e-4`、相对容差 `.005`。全部 plane-distance 同向扰动的解析导数均为 `1.000000238`；交替方向在三种源数下为 `0.68256837–0.68256843`。因此当前检查覆盖了“无源时直接深度梯度不消失、有多个源时不重复累加”的局部修复，不能推广成 RGB texture、opacity/transmittance、buffer 选择或整个后端的梯度认证。

烟测使用 25 个高斯、64×48 合成输入和 66095 参数融合网络；simple-knn 最大误差 `7.45058e-9`。修正后的 camera ray 与 corner-v2 约定最大差 `1.19209e-7`，与旧 `integer−W/2` 约定差 `.00833276`。所检查梯度均有限，旋转梯度在该对称 fixture 中为零；有限不代表精确。主 forward/backward 段为 `.181487 s`，整个检查 peak allocated 为 23652352 bytes，不是实际场景吞吐量。

污染检查用 z=2/4 两个平面与非 identity 的 source x 平移 `.4`，固定 source RGB-valid 孔洞和边界。source depth 乘半径 1 腐蚀后的 valid；中心两个交点投到 x=44、38，median 投影约 39.764。检查中 `depth_error_threshold=10` 专用于隔离支持问题，**不是训练深度阈值**。它证明该合成构造下无效颜色没有进入输出，不证明一般遮挡正确性；构造中的交点见证由 CPU 几何推导，并非保存的 CUDA buffer ledger。peak allocated 为 14756352 bytes；深度导数检查为 1649152 bytes。

三项均无数据集读取、无 checkpoint 加载或优化器更新。这里的耗时仅为报告内部 wall；根代理未测外层 elapsed，外限为 120 秒，不能补估或把这些小图时间当本桥 FPS。GPU 检查由根代理执行，本报告只读整理结果与 SHA，没有重复运行。

随后真实接线预检从已完成 1M 场导入 **996009 点、SH3**，测试固定 `002.png / 041.png`，各用 4 个既有几何邻居。新 normals / offsets 分别为 2988027 / 996009 个标量，融合网络 66095 参数。执行 **8 次 source-depth forward、2 次 target forward、2 次 backward、0 次 optimizer step**；原 field 参数前后不变。8 组 field 参数梯度均有限且范数非零，融合网络梯度有限。002/041 的 forward/backward 记录用时分别 `.401813 / .198691 s`，融合支持率 `.716727 / .457537`。这些只是初始化接线和覆盖量，不是训练后的质量分数；导入到不同 rasterizer 的原 RGB 不要求与 gsplat 数值相等。

真实预检内部 wall **3.431237 s**，peak allocated **6785746432 bytes**、reserved **8151629824 bytes**；外限 180 秒，外部 elapsed 未测。实际解码 10 张 TRAIN RGB 和 1 张共同 valid，0 VAL、0 semantic-label 解码。实际执行绑定 `plan_v2.json`，SHA `a67805a19045a49b83355777a8f58ba8fea07866a0f941aefb82f5bef3d73cee`；先前 `plan.json` 只准备未执行，v2 修改标签解码字段命名及收尾核验，未替换数值配方。[真实预检报告](/mnt/data/SHM2026/runs/ibgs_port_preflight_v1/result.json) SHA `36a6a8c570c963422cb336dacf33b1c49883a934fe3316e3878229ab54d06545`。

原失败完整保留：[原烟测](/mnt/data/SHM2026/runs/ibgs_backend_smoke_v1/result.json) 为 identity transport 失败；[追加前向记录](/mnt/data/SHM2026/runs/ibgs_backend_smoke_diagnostic_v1/result.json) 最大 RGB 差 `.756250024`；[原深度差分](/mnt/data/SHM2026/runs/ibgs_original_depth_gradient_v1/result.json) 无源时解析导数 0 而 FD 约 1 / `.68259`，6 项失败，一源时解析梯度非有限，未执行四源。修订前后使用同一深度检查源码，旧失败不追认通过。

结果与绑定：

- [深度导数结果](/mnt/data/SHM2026/runs/ibgs_port_depth_gradient_v1/result.json)，SHA `79f612861e000f770a585ffafeba69dc724d0909b0012b908bcf9ec2f8f4a741`。
- [后端烟测结果](/mnt/data/SHM2026/runs/ibgs_port_backend_smoke_v1/result.json)，SHA `e4513cd8822eba320e47e3e674c39b18e93b35a27876d17c898732af6c4decbb`。
- [source-valid 结果](/mnt/data/SHM2026/runs/ibgs_port_source_validity_v1/report.json)，SHA `b89259504c2c60e93d0174de37131b679139f677128d0f7b27841f03a35965b2`。
- [根代理观察的启动回执汇总](/mnt/data/SHM2026/third_party/ibgs_port_revision1/verification_launch_receipts.json)，SHA `851fb54c5059265d03c28cd997c92ab1cc75adf7a7bdce93e2f48235b698b2f5`。汇总记录各 session / exit、entry 与报告 SHA，含真实预检 session 14912；外部耗时为 null，不伪造执行记录。
