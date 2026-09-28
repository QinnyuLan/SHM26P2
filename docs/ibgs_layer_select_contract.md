# IBGS 四层选择器：仅前向草稿

独立新 `ibgs_layer_select.{cpp,cu}`，不改旧IBGS ABI、源码或二进制。Python导入只定义接口；本轮仅CPU合同测试，**未编译、未执行CUDA、未证明真实场逐射线一致或画质收益**。遍历适配自IBGS/Inria光栅器，保留原[研究/评估许可](/mnt/data/SHM2026/third_party/ibgs/submodules/diff-plane-rasterization/LICENSE.md)归属。

`buffer_views(geometry, binning, image, *, point_count, num_rendered, width, height)` 从实际tensor地址按128字节对齐取typed views：FP32 means2d `[N,2]`、conic+opacity `[N,4]`、int32 ranges `[tile_count,2]`、point_list `[instances]`。旧uint32索引按int32位视图使用，负值拒绝；只暴露实际tile数，不读取image中未初始化ranges尾部。`all_map[N,5]` 是原调用的显式输入，另传。不能把迁移到不同地址的保存缓冲区当作原ABI内存；CPU副本应使用已有带原地址的NumPy parser。

显式 `load_extension(build_directory='/mnt/data/...')` 才调用Torch JIT；建议caller设置 `CUDA_HOME=/usr/local/cuda, TORCH_CUDA_ARCH_LIST=12.0, MAX_JOBS=4`。生成文件限定数据盘，不安装覆盖原包。`select_layers(..., width, height, focal, principal, extension)` 不自动编译；principal必须原backend数组坐标主点，不能传未减半像素的corner K。仅接收连续、同CUDA设备、无requires_grad的FP32/整数输入；不注册旧backward，不静默detach可训练场。

同一次16×16 tile共享加载扫描同时生成median4和top4，每模式固定四槽，输出ID/depth/原始 `alpha*T`/遍历ordinal；缺失ID=-1，其余0。另返回final T、最后RGB接受ordinal、非有限plane计数和错误位。保留原power、`min(.99,opacity*__expf(power))`、`alpha>=1/255`和严格`next_T<1e−4`提前停止；触发停止的贡献不纳入。负/零plane仍消耗RGB透射率，不进入选择。

median使用**入射**T：T>.5时交替更新前两槽，否则只写后两槽首次交点；不重排环形槽。top4按原w降序，同w保留较早遍历ordinal，与Gaussian ID大小无关。两模式仅选择有限正plane。旧CUDA median只测z>0、可能存入+inf；新选择器对此记录错误位并默认fail-loud，不能把删去非有限交点后的输出冒充原median等价。`strict=False`仅暴露错误诊断数组，禁止进入训练。无来源相机/深度门、无窄前策略、无邻域采样。

射线算术沿绑定 `forward.cu:352` 的 `(pixel-c)/f` 除法，用于plane交点；该源码后续聚合median的来源投影才使用预计算 `1/f` 乘法，两条链不能混用。本选择器没有实现后者。

CPU测试覆盖环形槽、跨.5归属、相同权重tie、alpha/提前停止、无效plane仍保质量、非有限状态、原地址对齐/零拷贝、梯度拒绝，并用现有NumPy replay交叉核一个有限fixture。CPU exp与CUDA `__expf`/FMA仍有差别；编译后必须以root固定真实ledger和合成边界独立核对，不能仅凭此测试声称CUDA逐位等价。所有原始权重不再归一化，槽位数也不随候选不足而补救。
