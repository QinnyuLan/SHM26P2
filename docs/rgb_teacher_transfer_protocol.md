# 固定组合 RGB 的纯 H+ 显式像素迁移

本轮只评价已交付 selected 官方 RGB 与 1M/MCMC 固定各半的实际 RGB。两臂都使用历史 `00f5b84a…` H+ EMA、tile768/stride512/flip/context0.25，纯教师权重 1，无 H3 语义融合或第三场。renderer/checkpoint 的 legacy/corner_v2 声明保持原样；这是明确的图像坐标转换，不绕过单场加载保护。

以原 K、畸变和尺寸调用冻结 `distortion_render_grid(...,legacy_mixed_v1)`，再用 `initUndistortRectifyMap(K,d,None,Kcanvas)` 及 INTER_LINEAR 把原图 RGB 映到教师画布。不作半像素共轭；overscan 的 K 平移与画布尺寸由原函数给出。边界固定黑填充，不借旧场补像素。完整教师 soft probability 经同一 legacy grid 双线性映回原图后 argmax。相机外参仅核对共同来源，不参与二维 warp。原 RGB PNG 字节保持不变。

现有 50 相机均为 1320×989 同内参：预计约 1.124% 教师画布无法完整取样，约 0.3603% 原图往返插值足迹受影响；逐图报告实际值，不裁掉评分边缘。重采样、量化及黑边使此控制不能复现历史 H3+教师分数；它只隔离同一新适配器内的 RGB 输入变化。

先保存、hash 全部 2×50 mask，随后才读取 41 份官方 annotation，使用原冻结 rasterizer/255 规则重新评分。RGB 三指标只在逐 PNG 字节核对后复用既有真实评分，不重算 LPIPS、不读真实 RGB。完整 fingerprint 的 RGB GT 身份继承已完成回执，annotation 与 raster SHA 在本轮重新核验。新回执为 multi-component/pureteacher，不能伪装成单 checkpoint 结果。

固定两比较：组合−同 adapter 的旧 RGB 纯教师；组合−当前 selected 完整系统。复用 50 RGB/41 semantic 视图 bootstrap（5000，seed20260926）。提出联合候选需旧 RGB 四门已通过，同时 all5 比 selected 至少 +0.20pp、paired95 下界 >0，cable 点差 ≥−0.10pp。无 TTA/权重/填充/适配强度搜索，失败不重试。开发集已反复使用，区间不是选择校正后的盲测证据。

旧 H+ 50-view 预测约 53.944s、整评约 68s，据此固定外限 300s、内限 270s。新耗时只包括已有 PNG 迁移、教师推理与评分/I/O；新相机仍需两场约 1,496,009 高斯的两次渲染，不能将缓存评分时间称端到端 FPS。

即便过门，这也是多场 RGB + 纯二维 H+ 读出的工程候选；不证明共享三维语义几何或学术创新，不自动采用、训练或改变当前 selected。仅 CPU 准备与冻结，GPU 由 root 单次交接。
