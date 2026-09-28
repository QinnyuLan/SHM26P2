# 固定simplex端点的FW方向诊断：一致且单步下降

2026-09-27，v2自然退出0，独立CPU审计通过。结果支持**当前固定端点、唯一FW方向的数值链一致，且一次候选前向使目标下降**；不证明全方向VJP正确、属性已收敛、IoU提高或几何容量不足。没有VAL、head或teacher评价，不采用新模型。

沿用[全量PG诊断](raw_simplex_fullbatch_results.md)留下的q、H3 `22bc8a2d…5226`/legacy、完整259 TRAIN、类别权重及正仿射噪声δ=5e−7。模型与相机冻结；用完整梯度的逐行argmin生成唯一FW顶点v。A为实际FP32位移 `v32−q32` 与各view VJP的FP64点积均值；B64由两次独立前向的target通道差和FP64 CE导数计算，B32upstream另模拟反传上游的FP32转换。同一背景在端点差中抵消，仍保留在CE分母。

| 核验量 | 实测 |
|---|---:|
| A | -0.086292239227 |
| B32upstream | -0.086292246941 |
| B64 | -0.086292248063 |
| A−B64 | 8.83658986e-09 |
| 固定64次导数二分的γ | 0.009662661771 |
| F：旧q → 真实候选重渲染 | 0.103402089840 → 0.103390119366 |
| 实际下降 | 1.19704741e-05 |
| 缓存预测下降与实际下降差 | 7.52474472e-11 |

两项方向误差、非零负信号及候选确认均通过预定门。完整顶点F=0.136725875718，因此负初始导数不代表整步移动有益；γ由冻结规则在这一个方向上确定，没有换方向或按结果选步长。

预定分箱中，仅4个 `p̃_y<1e−6` 像素贡献B64负量−0.0747984881，而总B64为−0.0862922481。它显示**这条新FW方向**的CE导数高度受稀少小概率像素影响；分箱只分解前向B，不是逐箱VJP验证，不能据此认定这些像素是旧PG回溯失败的唯一原因，也不能推断删除/降权它们会提高性能。

实际成本：[执行回执](/mnt/data/SHM2026/runs/simplex_direction_diagnostic_v2/execution_receipt.json)内部208.594321秒，[外层回执](/mnt/data/SHM2026/runs/simplex_direction_diagnostic_v2/launch_receipt.json)209.018355秒自然0。三遍259 TRAIN，共777 scene、1554 gsplat加777直接q shader（2331 raster）、259 VJP、518 mask/valid解码；peak CUDA allocated 1,054,332,928字节。包含载模、缓存写入/校验、CPU线搜索、真实候选确认及恢复，不能当渲染FPS；没有主机RAM峰值记录。0模型更新/0head/0teacher/0VAL，未计算mIoU；仅保存诊断q候选，原场/flags/梯度/数值设置恢复且来源输入不变。

[独立CPU审计](/mnt/data/SHM2026/runs/simplex_direction_diagnostic_v2/independent_cpu_review.json)耗时21.292207秒，核64源、295输入、1037缓存，2116项标量最大绝对差6.48370246e-14。它独立归约缓存直线目标/导数、分箱及门，核候选和来源；梯度仍来自保存g32，没有重新执行Wᵀ，64次搜索未重跑，**actual F仅由保存逐view行均值核验，未独立重渲染**。审计SHA `eefcb75c7ad19e2e24397b97d0e83c04e656e4bdf449add1361c691ee8557e07`。

v1由root运行冻结测试时误启pytest cacheprovider，源快照新增缓存文件，worker在GPU/渲染/标签解码前拒绝来源清单，0GPU工作且失败目录保留。v2仅修改协议/失败绑定与禁缓存测试流程，数学、阈值、相机、64次二分和预算不变；12项冻结CPU测试通过。详见[固定协议](simplex_direction_protocol.md)。plan SHA `0d684781d76582c554f7803f4a42bbca989e0ee742243e8c99e4c2c060108984`；[analysis](/mnt/data/SHM2026/runs/simplex_direction_diagnostic_v2/analysis.json) SHA `71c6d5b2b277a757eab13400da75b4ef757c30cc82698a6c7322a24fad54e346`；execution SHA `a7a33d154337c75ff341526adcdc7db26ade1130d340da2b61c14a071e7a952d`。旧PG未收敛结论不变，标准FW线搜索不是学术创新，工程E保持。
