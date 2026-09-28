# 现有 IBGS buffer 的固定射线 CPU 重建

2026-09-27。仅实现 CPU library 与合成测试；尚未 prepare、GPU 导出或实际射线分析。是否执行、选用哪个固定已完成端点、实际 renderer profile／binary 和连续量容差，都由后续根进程在导出前绑定，不从结果挑选。

`src/bridge_rgs/ibgs_ray_replay.py` 不加载模型／图像，不导入 Torch，也不改 CUDA。它包含固定采样、128 字节对齐前缀解析、按 tile 复用候选列表的 NumPy 重建及生产摘要比较。仅支持实际 `render_geo=True, render_depth_only=False, buffer_length=4`；depth-only 的额外 early stop 不在合同内。

## 最小导出接口

根进程可在 `_C.rasterize_gaussians` 返回处暂存输入 `all_map`、实际 `colors_precomp`（若为空则用 geometry RGB）、background、焦距／主点、模式开关，以及返回的 `num_rendered`、原始RGB、median depth、geom/binning/img三个buffer。每个buffer须记录原GPU `data_ptr()%128` 与字节长度；CPU副本地址不能作布局基准。需要的geom属性都在CUB scan workspace前，binning point_list／keys都在sort workspace前，image的finalT／n_contrib／ranges／median sum／low／high可直接解析，无需计算CUB临时区长度。`num_rendered` 是Gaussian×tile实例数，不是N。只核实际tile数量的ranges，不能读其余未初始化分配槽。

绑定实际binary与对应`rasterizer_impl.{h,cu}`、`forward.cu`、config常量；ABI为bool1／int4／float4、float2 8／float4 16、uint2 8字节及128对齐。`parse_buffers`只返回只读view，对长度、ID范围、tile范围进行检查；`replay_rays`再核选中tile keys与tile身份。只对被实际列表引用的Gaussian读取属性，不全量要求culled槽有限。提供colors_precomp时，不能从未初始化的geom.rgb替代。

## 采样与重建

沿[原设计](ibgs_thin_evidence_diagnostic_design.md)的16个TRAIN名称，每图32×16格固定512中心，SHA确定坐标；每个中心及其8邻点全部保留。合计8192中心、73728射线，边界／低alpha／空buffer／失配不重采样。库中的`sample_rays`仅使用名称、格编号与尺寸。

同tile的Gaussian属性只gather一次，每chunk至多32条射线矢量化计算power、alpha、前缀透射率；保持贡献ordinal包含被skip的候选，严格复现alpha<1/255跳过、testT<1e−4先终止而不累计、正plane depth才可进入median、pre-T>0.5的两槽环形及其后首两槽。RGB的所有有效贡献不因plane无效而删除。输出每条射线的ID、ordinal、alpha、incoming T、w、交点z、Gaussian中心z、median4/top4标记及buffer槽顺序。top4独立要求finite positive plane交点后按w降序、ordinal破平票；原median仍保留CUDA的z>0判断（若出现+inf则整射线标nonfinite），不为比较器修改生产ring；它是已知等槽数质量控制，不是新方法。

生产核对包括finalT、最后有效ordinal `n_contrib`、raw RGB（含背景）、median总重、low/high ordinal及median depth。离散项必须完全相同；连续项用后续执行前固定的显式atol/rtol，库没有默认宽松门。保留所有误差、阈值距离及质量闭合误差。

**NumPy exp、非融合乘加和CUDA `__expf`／FMA并不逐位等价。** 不匹配全部标记为`production_summary_mismatch`或`nonfinite_cpu_replay`，纳入固定总体的失配统计，不能换样本／改阈值后称实际账本。即使全部摘要一致，也只叫`summary_consistent_reconstruction`，始终`exact_cuda_contribution_ledger=False`；这些摘要不是逐Gaussian权重的完整校验和。连续量容差匹配无法独立证明所有中间离散选择相同。

首轮仅判断此已训场中是否存在可重复的、具有实际贡献的窄前置群被median4遗漏；正深度、窄足迹和同ID重复只是模型代理，不能直接命名真实cable。top4质量增加是定义性质，不能作创新或画质收益。失配严重／没有可重复前置遗漏时，报告覆盖或机制不支持，不扩采样、K或训练。
