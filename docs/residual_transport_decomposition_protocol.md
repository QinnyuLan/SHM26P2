# 已缓存残差传输的固定 CPU 分解

2026-09-27。协议 `residual_transport_decomposition_v1`。仅准备实现，真实执行由 root 冻结后启动；不改父运行或 helper，不新增渲染、模型加载、原 RGB／GT 解码、语义或 VAL 读取。内部90秒、外部120秒，一次执行，失败保留不重试。没有性能采用门。

父运行固定 `residual_transport_probe_v1`：plan SHA `061586d6309693a41c66b720159076e9bb52d6c1d7d605ec5d939a51402b297b`，独立审计 SHA `c2476de5dac9951b3f4318c1f55711f903fbf9dde5eea1c866d7cec877ce2b33`。prepare 必须核自然退出0、completed、审计passed及其自然退出回执；绑定78个render、16个transport、16个target NPZ与原分析、来源链。允许提前hash缓存，不提前加载target数组。helper从父冻结包逐字节复制，不从live包导入，不导入Torch。新目录 `/mnt/data/SHM2026/runs/residual_transport_decomposition_v1`。

16目标、62源、每目标4源、相机矩阵、common支持、source_count、alpha／ED阈值、OpenCV插值以及A偶数／B奇数划分全部继承。令 `M` 为有共同支持的目标像素，`T` 为每像素相同有效源的等权重投影平均，`Rt` 为目标场RGB：

- `source_residual s`：直接复用父缓存的 `T(Is−Rs)`，不重算。
- `render_only d = M * (T(Rs)−Rt)`：以原helper将源residual字段换为源render RGB，其余输入不变。必须逐位复现父 valid 和 source_count。没有支持处 d 为0，不能全图减Rt。
- `full_photo = s+d`，未clip，支持外0。
- `zero=0`。父wrong只链接原系数和汇总，不重新拟合，不加新对应控制。

由于父源残差先经过FP32减法，`s+d` 是本次缓存定义下的恒等分解，不能声称重建了原源照片的逐位数值。任何中间RGB裁剪都会破坏该线性分解。它是普通IBR工程控制，既不是 [IBGS](https://arxiv.org/html/2511.14357v1) 的完整多交点／网络实现，也不构成新方法。

必须先保存全部16张的 d、full、support/count 并记录SHA，随后才加载target NPZ。首遍target缓存生成三臂充分统计，次遍用于固定系数评分，故32次target缓存加载、0次原图解码。每次只保留一个目标及4源，避免驻留全部源bank。source_residual的两λ与zero/source-residual汇总必须精确复现父结果。

每个非zero臂独立拟合A、B两个全局λ∈[0,1]；每目标只用另一折λ。拟合分子／分母先按每图**完整目标RGB-valid**像素与3通道平均，再等图平均；unsupported residual是0，不能改用覆盖区分母。拟合为未clip MSE，另报clipped结果。固定报告全部16图、A和B两折的 clipped／unclipped MSE、等图mean log10 MSE及等价mean PSNR；若有恰好零MSE，log均值明确NA并计数，不添加数值floor。零臂λ固定0；无二维λ、权重扫描或根据结果改源／门。

在每张图的heldout full_photo λ下，固定保存未clip误差增量的五项：令 `e=Rt−It`，则

`ΔMSE = 2λ<E,s> + 2λ<E,d> + λ²<s,s> + λ²<d,d> + 2λ²<s,d>`，

其中 `E=e`，所有内积仍为同完整valid均像素／通道平均。与直接计算增量校验，容差为 `64*eps64*max(1,|Δ|,sum|terms|)`。此恒等式不能用于clipped指标；交叉项不能被忽略或归为独立因果贡献。

独立λ的full-vs-render比较检验固定工程管线中的增量，而非严格照片信息因果效应；render-only改善也可能来自平滑、SH视角差、遮挡误投影或场自身误差。若full没有实用增益，不扫描参数救援本固定ED方案。基础场见过全部TRAIN目标、源照片由旧开发选择得到，OOF仅指λ，不能称模型泛化或盲测。
