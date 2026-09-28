# 第四角对照的预检与运行状态

原三臂结果之后新增median4_normalized，补齐选择×归一化对照，见[协议](ibgs_factorial_completion_protocol.md)。这不是原三臂预注册的一部分，也不是主假设已成功。

训练体由原冻结worker `5dd48b57…4987c43`复用，只增加首次更新前的跨实验初始化哈希检查，并将总更新计数从固定三臂改成实际臂数；其余逐步数学源码未改。单臂profile只改变协议名、臂列表、选择说明及scope，全部数值训练设置保持一致。三项CPU检查通过，Ruff通过。

002/041各一步的真实GPU预检自然exit0：2 raster／2 selector／2 head updates，几何版本和梯度不变，初始逐参数哈希与原三臂全等；0VAL、语义、teacher或场更新。第二步MLP梯度L1=.0004758043，CNN=.1215800494，所有参数/梯度/Adam状态有限。峰值12,656,191,488B；内24.625572秒／外25.10秒（含全量来源校验），单臂更新阶段2.412444秒。该预检验证可执行性，不验证画质。

预检目录`/mnt/data/SHM2026/runs/ibgs_factorial_preflight_v1`，plan `f785bd3c70e08834f5892a2a5bcfedd82c9c9c7e197e561695c3ce4e32c4614e`，执行回执 `53a8920f1f74eb5974fa681e048bdab830ddaefcd64b6a556c11221ea3d9c583`。wrapper SHA `3d27a597368b98fa775566b5db87ed9c18a57443a4ffcd7638875efbcd020681`；两处检查适配后的训练体SHA `5bce9d7dc8f874babf4435fd5e8d326ca4d44efb5ae1d789c8576bc732db53a8`。

正式6000步已在fresh目录`/mnt/data/SHM2026/runs/ibgs_factorial_completion_v1`启动；plan `8455dc63e6680f9c7828bda255c0f0a464cddf74c29d61789db0a501b5af7d19`。原三臂快照、旧场及source缓存不修改，预检更新不继承。正式训练全部源码哈希与本预检一致，使用原seed42初始头、原6000相机顺序、相同学习率/损失及单GPU，内部4800秒／外部4860秒预算。此处尚不表示完整训练或评价完成，最终以自然退出和运行回执为准。
