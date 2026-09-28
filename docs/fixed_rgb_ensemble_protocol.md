# 固定 1M＋MCMC PNG 等权 RGB 工程对照

本轮只评价已完成的 `rgb_capacity_1m_reference_v1` 与 `rgb_mcmc_reference_500k`，默认权重各 0.5。没有 TRAIN 筛选、成员搜索或 VAL 权重扫描；这批开发视角已多次使用，不能称新盲测或选择校正后的泛化结论。

唯一候选逐像素计算 `np.rint((A.astype(float32)+B.astype(float32))*.5).astype(uint8)`。输入是两成员共同官方原图网格的各 50 张已交付 PNG，使用 round-to-even。这是量化后编码 RGB 的平均，不是线性光辐射平均，也不等同未量化渲染结果的平均。两个原模型均为 corner-v2；相机、原图来源和已有共同评价 fingerprint 必须一致。

CPU 准备绑定两次完成回执、各 50 张预测 PNG 的 SHA、原 checkpoint／训练 manifest 来源、固定 500k strong 参考与其成功使用的冻结评分包。GT 的预期 SHA 仅取已有回执 `source_records`；准备和生成阶段不读或 hash GT 像素，也不读标签。运行先生成、保存并 hash 全部 50 张组合 PNG，写完预测完成回执后，才读取原图 GT 并核 SHA。

复用原冻结 `official.score_official_arrays` 与 AlexNet LPIPS。旧 API 要求的全零 predicted-mask 仅为内存参数占位；target-mask 必须是 None，返回键必须恰为 PSNR、SSIM、LPIPS、RGB像素数。不输出 mask、混淆矩阵或任何语义成绩。继承的完整官方 fingerprint 仅标识已有共同参考；本次只重新验证 RGB GT，不重新读取标注或建立新语义证据。

独立 RGB 配对：50 名称排序视图、NumPy seed20260926、5000 次同相机有放回抽样、两侧分位数 .025/.975，差为 candidate−reference。主参考固定为 `ssim_fixed_corner_v2_rgb_full` 的官方500k结果。四门固定：PSNR增幅≥.15 dB、其配对区间下界>0、SSIM点估计不降、LPIPS点估计不增。另报告对两成员的描述性差异，不据此另选赢家或改权重。无论门是否通过，都不自动启动语义训练或采用联合系统。

内部200秒／外层240秒，无自动重试、无训练或新场渲染。只计缓存 PNG 组合及评分耗时，不能当作双场端到端 FPS。实际新视角生成仍需保留约1,496,009个高斯的两个场及两次相机渲染。本轮不能声称单一共享几何创新；如未来考虑联合采用，语义必须从实际组合 RGB 重新预测并另行核定匹配的来源和像素协议，不能拼旧 mask，也不继承已停止的1M语义训练链。

冻结测试必须用 `pytest -p no:cacheprovider`；不修改旧模型、旧评分包或任何已有结果目录。
