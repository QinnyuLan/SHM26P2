# 固定 mild 两臂 means pilot：隔离入口

状态：**prepared_not_started**。独立校准实际结果为 `direction_calibration_not_met`；冻结入口的 gate 已在 CPU 核实拒绝启动，不更改门槛。CPU 实现：`scripts/run_pose_profile_pilot.py`。没有修改主 `train.py` 或任何旧 snapshot；本 pilot 未执行 GPU/训练。启动有两个独立前提：不可变校准 plan `d14423c9…892f69` 的 completed receipt/audit SHA 链成立，结果为 `independent_direction_calibration_supported`（5 视角方向可靠且≥4 保留），以及 root 明确 GPU 交接。旧 v1 的 inconclusive 不能满足这个条件。

沿用 [决策备忘](pose_profile_pilot_decision.md)：相同 clean 6k/177,378 点 H1 场、固定 mild TRAIN 位姿、350 张 TRAIN RGB/valid、宽 320、SH3。每臂独立重载共同起点，只有 `splats.means` 有梯度和优化器。三轮 `np.default_rng(42)` successive shuffle 共 1,050 步，在 CPU plan 中完整保存，既不共享全局随机流也不根据 loss 改抽样。raw 使用有效 RGB 标量 L2 / 2ΣW；profile 每步先以当前 means/当前固定相机作原 13-render J，沿原 Λ、FP64 求和和 stop-J/stop-δ envelope 求梯度。每步 δ 用后丢弃；无位姿更新、增密、重置、语义、区域加权、L1/SSIM 或额外先验损失。

两臂均新建空 Adam，固定共同起点末态 means lr=2.6018353271484376e−5、betas=(.9,.999)、eps=1e−15；不继承旧动量、不继续 scheduler。TRAIN 输入调用冻结的 `io.load_view`，只给 RGB/valid 与相机白名单，强制 `mask_path=None`，保留原 AREA RGB / nearest valid 缩放。读取 guard 禁止训练期间解码 mask 或 VAL；CPU prepare 只哈希不解码。clean/stress manifest 分别显式绑定，并逐个对齐旧 H1 已绑定 SHA；除 TRAIN w2c 外，全部图像/标签路径、尺寸/K/split/original pose 和 VAL w2c 必须一致。

固定预算：raw 1,050 render；profile 14,700 render；合计 15,750 train render / 2,100 backward。每步记录 raw L2、当前目标、means 梯度范数、实际 Adam 位移、g·实际位移、相对共同起点漂移，以及完整 view/index。两臂完成固定 last 后，才调用同一旧 native evaluator 对起点/raw/profile 做全部 50 视角、native scale1、LPIPS，共额外 150 render。LPIPS 权重须已存在本地缓存且 SHA 绑定，不下载。固定比较器只输出 RGB 的 5,000 次视图 bootstrap；无中间 VAL 选择、语义收益结论或等算力表述。

整个 execute 固定 **600 秒**，子进程均使用剩余 deadline，包含载模/输入核验/训练/末点评分。最终来源核验也必须在 600 秒内才可记 completed；超时记 `inconclusive_timeout`、非零退出，保留已有日志和磁盘产物，不重跑/续训/改预算。硬 kill 不保证子进程内 finally；源模型/数据始终只读。正常或 Python 异常路径通过已冻结的恢复 context 还原全部内存模型张量/梯度权限/原 grad。训练完成前逐位检查所有非 means 模型张量以及基底保存的 camera array。

**新 checkpoint 的含义。** 每臂只输出固定终点 `last.pt`，完整 CPU 推理模型+架构字段、新阶段 1,050 步配置、实际固定 mild TRAIN camera array、base 路径/SHA/6000 步来源及 namespaced pilot 记录；不复制旧 optimizer、RNG、density 或 stats。kind=`pose_profile_pilot_inference_or_warmstart`，并明确 `ordinary_resume_allowed=False`。该 flag 本身不是主训练器通用拒绝接口；实际保护是独立 `config.parameter_scope='pose_profile_means_only_pilot'`，当前主训练器的 scope 验证会拒此配置；改成合法 scope 做 strict resume 时又会触发 saved/current scope 不同的现有检查。旧 snapshot 训练器没有此友好 guard，但也没有可续训的 optimizer/RNG，不能称支持其 resume。推理 `load_scene` 可正常读取；新的 warmstart 要给合法新配置并使用一致的 mild manifest，不以输出为原清洁训练场。

CPU 合同 9 项通过、Ruff 通过：350×3 顺序及全局 RNG 独立、真实 GaussianScene 仅 means 更新、空 Adam 与精确超参、非默认源 Adam 拒绝、完整 checkpoint 实际 CPU 装载、旧状态剥离、实际主 train 对独立 scope 的提前拒绝（仅 mock availability，不初始化 CUDA）、合法新 scope 模型加载、TRAIN 读取白名单、固定 calibration 条件、deadline 不因 phase 重置。未做 GPU warmstart/训练/效果测试。

局限：mature-field 低分辨率 mild-only 两臂只能问“在这次固定真实 RGB 适应中 profile 是否比 raw 少产生有害 means 改动”。不能替代 clean×stress 交互、从头重建或 joint BA 对照，也不能把标准线性消元声称新数学。若独立校准不过，入口即拒绝；不根据训练结果追加候选或重新调步长。

CPU 锁定计划：`/mnt/data/SHM2026/runs/pose_profile_means_pilot_v1/plan.json`，SHA256 `5c8228103579ae94964e8f859df119093a630f913a1dd857724bc9351b06f26f`；冻结 runner SHA256 `6d2564212f64dade42a422932741eef9baaf9ef0444b7c46d92ac952b0f5846b`。准备记录 CUDA 未初始化、0 像素解码。未执行任何 GPU 命令；仍须独立校准实测通过和 root 交接。
