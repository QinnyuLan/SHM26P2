# 因子式 FP64 AA：固定 warm-IBGS 双臂 v2

2026-09-27，执行前合同。协议 `ibgs_aa_stable_matched_v2`，渲染身份 `ibgs_centered_corner_v2_aa_factored64_near001_v2`。旧 AA v1 在第一臂完成 61 步后，第 62 步相机 004 的 FP32 Gram 行列式消减触发保护而失败；失败目录、检查点和回执保持原样。本次不是续训，不追认旧失败通过，也不称已知 AA／IBGS 为新方法。

新入口 `scripts/train_ibgs_aa_stable_matched.py` 直接复用已完成旧 warm 的冻结 `train_arm` 与旧 AA 封装的作用域／端点保存实现。从原 `35b489fe…078092`、996009 点、SH3 的 1M 检查点重新加载两臂，fresh Adam、seed42及原预存 6000 相机顺序不变。任何 `failure.pt` 或其它初始 checkpoint 路径／哈希都被拒绝。旧失败场仅用于验证中的 004 单相机回归。

两臂 `full`／`no_source` 均固定 6000 步：1000 几何暖启、1000 融合梯度阻断、4000 联合优化。原损失、LR、Adam epsilon、源选择、支持标志修正、空源回退与固定末态规则全部继承；仅网络入口七维源特征是否清零是两臂差别。350 次初始源深度＋6000 次目标调用均通过新 AA hook，每臂 6350 次；场更新 6000，网络更新最多 5000，实际计数保留。没有 VAL、语义目标、增密或相机优化。

新 provider 为 `bridge_rgs.ibgs_antialias_stable`。原实际激活 FP32 值先转 FP64，不重新 exp 或归一化 quaternion；构造 `B=J Rcam Rquat diag(scale)`，以三个 2×2 minor 的平方和计算 `D`，`H=D+.3||B||²+.09`，`rho=sqrt(D/H)`。正 D 走该前向的真实 autograd，计算为零的秩退化使用明确零导数约定，不加 floor／clip；有效 opacity 再转回输入 dtype。近面仍 .01，沿输入 dtype 相机深度决定剔除；FoV／modifier先按输入 dtype 量化，blur 为 nominal FP64 .3，不能声称逐位等于 CUDA .3f。`PRECISION_POLICY`、provider、模块 SHA 和现有 near-.01 binary 均进入 plan／端点。内部 CUDA 协方差、alpha-cap及其它近似 VJP 没有被此修补认证。

prepare 必须绑定新 stable provider 的四项独立自然完成证据：16 TRAIN 兼容前向、原固定 33 前向／3 VJP 合成检查、002／041 的 8 深度＋2 目标＋2 backward 预检、旧失败 004／61 步状态的 1 raw 前向＋1 backward 回归。各 receipt 必须 `completed` 且 `numerical_status=passed`，外层自然 exit0、plan／receipt SHA 一致，实际 provider／precision／binary匹配、恢复通过。cap-active 的已知反例保留，不替换非饱和主门。四项检查的旧 FP32-AA passed 记录不能替代新证据。

FP64 增加成本，因此执行前将时间上限固定为**每臂 3600 秒、外层全流程 7500 秒**，只改变超时预算，不改变迭代数；不声称相同墙时或 FLOPs。实际耗时／显存照实报告。失败保留，不延时补训、改参数或选择 best。最终评价须用相同 stable provider 和渲染身份刷新全部源深度及目标输出。

仅 CPU 准备入口（四项真实验证完成后）：

```bash
uv run --no-sync python scripts/train_ibgs_aa_stable_matched.py \
  --prepare /mnt/data/SHM2026/runs/ibgs_aa_stable_matched_v2 \
  --compatibility-run <new-stable-16-run> --gradient-run <new-stable-FD-run> \
  --preflight-run <new-stable-two-TRAIN-run> --failure-run <new-stable-004-run>
```

GPU入口由 root 使用隔离 IBGS interpreter 执行新 snapshot 的同名脚本，传 `--run <plan.json> --expected-plan-sha256 <sha>`，外部固定 7500 秒。prepare 不加载模型／图像，不自动启动训练。两臂输出继续是 `<arm>/last.pt`、`training_receipt.json`、`aa_scope_receipt.json`及顶层自然执行记录；源码、样本顺序、初始参数身份和失败产物均保留。
