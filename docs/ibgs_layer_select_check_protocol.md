# IBGS 四槽选择器：固定前向检查

这是新独立 CUDA **forward-only** selector 的一次编译与接线检查，不渲染场、不读取 RGB/标签、不训练、不做 backward。旧 IBGS 后端及全部冻结实验不变。不以通过本检查证明完整后端梯度、有效源纹理或新方法收益。

`check_ibgs_layer_select.py --prepare --output …` 仅冻结已审定的 selector Python/C++/CUDA 和其 CPU tests、旧 `ibgs_ray_replay.py`、固定父计划 `133ca861…13e6055` 下的 16 capture/replay 引用与本 worker/tests。Torch 固定 `2.8.0+cu128`，CUDA 12.8，sm120。`--run --plan … --expected-plan-sha256 …` 才在数据盘 run/build 显式 JIT 编译；不安装包或覆盖旧 `.so`，生成二进制 SHA、实际编译源 SHA 与工具链身份写回执。总预算编译加检查内600/外660秒，输出上限4 GiB。失败保留，不改容差重跑。

先执行六个预设小合成条件：alpha cap/终止候选不纳入、负 plane 消耗 RGB 透射但不入槽、正 power/alpha-cut 跳过、incoming T 恰 .5、相等 w 按较早 traversal ordinal、median 环槽。GPU 结果与既有独立 NumPy replay 合同比较，而不是用结果挑 fixture。

随后逐个处理 16 已捕获视图。历史 byte buffers 必须先用**历史真实绝对地址**在 CPU 按 ABI 解码；再复制连续 typed means2d/conic/all_map/ranges/point-list 到 GPU。禁止把迁移后的历史 byte buffer 传给按当前地址解析的 `buffer_views`。每图仅一次 selector，输出 median4 与 top4 后立即落盘/释放；共22次新 selector 调用、0 raster、0 GT、0 backward/optimizer。

全 H×W 对照生产 `T`、last contributor、原 median 槽按顺序累积的 weight/depth/low/high。离散量必须精确，所有 status 和 nonfinite-plane count 必须0；浮点沿原容差 `2e-5 + 2e-4*abs(reference)`。固定73728条旧采样射线再对照 CPU replay 的 median 槽 ID/ordinal、top4 按 w 降序及较早 ordinal tie 排序；ID/ordinal 精确，depth/w 用同一浮点门。包括有效域外射线，不删失配、不只验抽样。

保存每图全图重建摘要 NPZ、抽样 actual/reference 槽 NPZ、各检查全量失配数/非有限数/最大差及最多16个位置示例；生产参考仍绑定原 captures。全部检查完成后才作 numerical_status，任何离散不符或超容差均失败并保留文件。CPU与CUDA的 exp/FMA 差异不能用于事后豁免门；本次通过也不是所有未来场/相机的数学逐位证明。未实现或批准 selector 的后向与新训练。
