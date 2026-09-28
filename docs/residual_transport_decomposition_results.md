# 源照片与场自重投影的缓存分解

2026-09-27。固定四路CPU分解及独立复核完成，完整源照片修正的16图平均MSE微降，但平均逐图PSNR变化为 **−.004552518dB**，分折不一致，没有实用收益。停止本固定ED传输配方，不扫描源相机、深度门或λ范围。此结果不否定有学习法向、多交点和神经解码的完整IBGS；当前工程E不变。

## 具体检验

继承[原诊断](residual_transport_probe_results.md)的16个TRAIN目标、62源、相机、支持与权重。令目标原渲染为Rt、投影平均算子为T、有支持区域为M，保留三项修正：

- `source_residual s = T(Is−Rs)`：原缓存逐位复用。
- `render_only d = M*(T(Rs)−Rt)`：不需要源真实照片的场自重投影。
- `full_photo = s+d`：普通源照片辅助；支持外全部为0。

另保留zero，三项分别在A/B拟合两个λ∈[0,1]，只应用另一折λ。每图使用完整prepared RGB-valid区域作等图MSE拟合。完整[协议](residual_transport_decomposition_protocol.md)未因结果更改。父源残差经过FP32减法，所以full=s+d是缓存定义的恒等式，不称重新还原了原照片的逐位值。

## 固定结果

| 修正 | A拟合λ（用于B） | B拟合λ（用于A） | 16图clipped MSE | native有效域平均PSNR | 对zero变化 |
|---|---:|---:|---:|---:|---:|
| zero | 0 | 0 | .000885358904 | 32.300961 | — |
| source_residual | .246583438 | .092834604 | .000882931250 | 32.301105 | +.000144dB |
| render_only | .013820140 | 0 | .000885388498 | 32.300637 | −.000324dB |
| full_photo | .094902366 | .026602284 | .000884993560 | 32.296409 | −.004553dB |

这是条件于既有场的TRAIN缓存评分，**不是新视角或官方原图PSNR**；只有λ交叉拟合，基础场见过全部目标。source_residual与zero的原系数和MSE逐位复现父结果，原wrong只引用，不重新拟合。

full_photo的A折平均PSNR +.004534dB、B折−.013639dB。全16图平均MSE虽下降.041265%，平均逐图对数误差仍恶化；不能只报有利的MSE汇总。render-only在B拟合λ为0，因此A的输出与zero相同，其B输出反而略退。当前没有支持将新照片信息或场自重投影作为稳定提分模块的结果。

在每图heldout full_photo系数下，以`e=Rt−It`作未clip MSE精确展开，五项等图均值为：

| 项 | MSE增量 |
|---|---:|
| `2λ<e,s>` | −2.104870880e−6 |
| `2λ<e,d>` | −9.435787329e−8 |
| `λ²<s,s>` | +5.521017258e−7 |
| `λ²<d,d>` | +1.303359116e−6 |
| `2λ²<s,d>` | −2.157597360e−8 |

该表说明加入d同时引入不可忽略的二次误差项，但不是独立因果归因。不同臂采用独立拟合的λ，full-vs-render也不是严格的照片因果效应。五项恒等式只适用于未clip误差，不能拿来解释裁剪后的非线性变化。

## 执行与复核

v1在CPU prepare阶段因查询未安装的`opencv-python`分发包而失败：0数组加载、0渲染、尚未写plan。保留失败目录。只把元数据名称改成实际安装的`opencv-python-headless==5.0.0.93`，另用v2目录；数学、数据和阈值不变。6项新合成测试与Ruff通过，独立只读代码审查通过；没有重跑未变的旧19项测试。

v2内部23.499875s／外层23.607310s自然退出0。CPU读取已保存缓存，生成16组分解数组共1,107,063,456 bytes；全部分解生成后才加载目标缓存，共32次target-cache加载。0Torch／GPU／新渲染／原图解码／标签／VAL／模型更新。父支持与source_count逐位一致，全部5源文件与117输入SHA检查通过。

| 记录 | SHA256 |
|---|---|
| plan | `832b6d6e936a1efd6c0c69cbd95efa2740436af1fa08a269e8f1765767cf3d71` |
| worker | `eab81cc4a4a6ce58e6f01730d02ef27f2fd97ca9b918850747395a3cf06730e4` |
| execution | `b37ac62e8ccf0e3c34ade124f0074fe770106c13efbc0e1033a0ffc775792dcc` |
| analysis | `bb0c9243a62141873fcec5ef698675c8df6c72994bf917b88397469b413a1f7c` |
| independent CPU review | `102b2c014486796d0542de3e2417112389a446028be5115570dc8c167fbba742` |
| independent audit launch | `06ee7c1384389e24c1b0797f4340cf4fa1055b122651cad2f77597ea19bfc1e4` |

运行：`/mnt/data/SHM2026/runs/residual_transport_decomposition_v2`。[独立CPU报告](/mnt/data/SHM2026/runs/residual_transport_decomposition_v2/independent_cpu_review.json)及[启动回执](/mnt/data/SHM2026/runs/residual_transport_decomposition_v2/independent_audit_launch_receipt.json)均自然退出0：内部20.657916s／外层 **20.748514s**。复核5份冻结源、134个绑定输入/输出文件，以独立NumPy/OpenCV公式从保存源RGB重算16组传输、d与full、共同支持与source_count，再核六个λ、全部16图/A/B指标及五项展开。208,877,606个数组/标量比较的最大绝对差为 **7.105427358e−15**（预定复算容差1e−11）；支持与计数逐位一致。

复核没有调用producer数学函数、GPU、模型反序列化、原图/标签解码或重新渲染。评分只使用已保存的decoded目标RGB；原渲染值、原图解码、运行读取先后及32次target-cache加载仍是SHA绑定回执证据，不声称由独立复算重新观察。v1准备失败保留为失败，不与v2成功混写。

普通ED传输与这组代数分解均不构成新方法。[IBGS](https://arxiv.org/html/2511.14357v1)的真实实现另有交点、法向与残差网络。完整IBGS路线目前仅进入构建与相机/数据接口合同核查，尚未验证训练或效益；应先建立该强工程基线，再提出和验证区别于它的具体机制，不能把本轮改名为创新成功。
