# Swin-L 6k 参考终点 CPU 审计

`scripts/audit_mask2former_reference_endpoint.py` 已准备，4项必要 CPU 合同及 Ruff 通过。正式训练仍由根任务调度；本准备没有读取活动 `last.pt`，没有使用GPU或重复训练。入口只读取不可变plan及小型执行回执，只有 `completed_step=successful_updates=6000` 且 status completed 后才允许对最终checkpoint进行 hash/load。当前固定审计仅接受这次从头训练；显式恢复的变体需要另核其来源，不能假定初始化差值仍代表整个阶段。

审计绑定正式plan SHA `de7fc9b821e3700066a9db18107155d1cd6a46ad38f4873085f473384f0e7600`、冻结41文件source、533输入及同场cache。确认 checkpoint/config/provenance 的来源一致、最终步为6000、五类加no-object的六行分类头、全部 model/Adam 张量有限，四组Adam每个参数计数均为6000，moment shape/dtype与最终poly LR匹配。这些检查与源程序中 Adam.step/synchronize 后才增加completed计数的顺序、6000条连续observed trace共同支持无跳步；合法drop-path零梯度不会被误判为失败。

从实际manifest锁定 **259** 个有标注TRAIN名称（不是256），以seed20260805的独立NumPy视图生成器重演完整shuffle，以seed+1重演每步crop坐标与flip：奇数两次整数抽样再flip，偶数context只抽flip。独立重演须与磁盘trace及checkpoint内6000条trace逐条相同，最终sampler permutation/cursor/RNG及augmentation RNG也须一致；全23轮259视图＋最后43步。训练日志必须为step1及每100步，共61条，并逐项对齐trace和最终checkpoint。

四stage更新证据是每stage固定第一个block的query weight。审计只从官方safetensors读取这四个预训练tensor，比较最终参数的CPU L1变化，要求有限、非零，且与runtime回执的变化量一致（容许CPU/GPU求和的微小舍入）。它不冒充全阶段每个tensor每步都发生变化；四阶段整体的有限梯度/实际更新已由独立四更新预检验证。

完整checkpoint保留torch CPU/CUDA及point-generator RNG状态。CPU审计检查其payload，但只重演 **视图与增强**，不声称独立重现GPU point sampling或drop-path逐位结果。不做精度评分，不把不同抽样视图的首尾loss当作准确率改善，不读取VAL输入。

固定完成后执行：

```bash
CUDA_VISIBLE_DEVICES='' uv run --no-sync python scripts/audit_mask2former_reference_endpoint.py
```

默认新输出 `/mnt/data/SHM2026/runs/mask2former_reference_v1/endpoint_audit.json`，拒绝覆盖，执行结束前后复核所有输入与最终checkpoint SHA。CPU合同覆盖活动endpoint早拒、独立采样重演与真实训练采样函数一致、Adam任一参数漏步拒绝、官方四stage探针实际变化检查；没有扩大成另一套训练框架。
