# AA／near兼容后的固定warm-IBGS双臂

2026-09-27，训练前合同。新协议`ibgs_aa_warm_matched_v1`。旧6000步双臂及其评价结果保留；本轮无新VAL指标，不作创新声明。原始1M场AA到IBGS的移植损失已由固定48次前向分解定位，加入补偿与near=.01后16TRAIN恢复到32.2999814 dB，接近原AA的32.3009613 dB；该前向结果不认证完整后向或训练收益。

只共同修改两件事：IBGS已有`C+.3I`足迹上的有效opacity乘`rho=sqrt(max(det(C)/det(C+.3I),0))`；隔离backend近裁面由.2改为.01。前者通过`ibgs_antialias.aa_opacity_rasterizer`覆盖每个相机的原RGB与源深度入口，`capture_last=False`；不把补偿固化进原始opacity参数，不重复补偿，不更改源深度阈值.01。正ratio沿Torch自身前向求导，非正ratio安全常零，detblur非正／非有限直接失败，无隐含floor。原IBGS的alpha cap、clamp和median权重梯度近似明确保留。

## 严格继承

从已完成`ibgs_warm_matched_v1/source_snapshot`逐文件SHA复制原27份绑定源码，包括任何合法路径中的`.py`；不从live官方包重拷。新worker直接调用旧`train_arm`主体和原helper函数，旧损失／优化器／抽样代码字节不改。运行期SPEC仅替换协议名并加入renderer_profile；端点保存增加明确渲染身份。

仍从同一996009点1M原checkpoint开始：350 TRAIN、SH3、seed42、相同6000相机顺序与初始8场参数／22网络张量；每臂1000几何暖启、1000融合stop-gradient、4000联合优化。固定skip1、残差末层零初值；损失、全部LR／Adam eps、源选择／支持和空源回退均沿旧协议。full与no_source仍只差网络入口七维源残差／相机特征置零，双方照常接受source-photo监督。每臂6350次AA调用=350初始源深度+6000目标调用，6000场更新，网络最多5000次真实有梯度更新；不增删高斯，不更新相机，无VAL／语义目标／教师。

保持**每臂1800秒、外层3900秒**，不缩步、不延时或自动重试。旧full约750秒；额外Torch协方差及其图会增加实际成本，当前不保证耗时相同。逐臂报告实际墙时、峰值显存、AA与光栅调用数；这些不作为FPS。相同seed／顺序不保证非确定性CUDA训练逐位一致。

## 执行前证据与输出合同

CPU prepare必须先绑定：旧训练自然0与合同复核；AA16前向自然0；新局部FD自然0且`numerical_status=passed`、四项主非饱和方向与零控制通过；两个真实TRAIN backward预检自然0且数值／恢复检查通过。三项新检查须绑定同一AA模块SHA与同一隔离binary。FD的cap反例不属于主通过门，也不被抹去。只成功退出而数值失败不能放行。

prepare仅读metadata／哈希，不加载checkpoint或图像；GPU只由root从新冻结目录单次启动。现有文件／失败不覆盖。两臂外部AA上下文异常时恢复原类方法，另存`aa_scope_receipt.json`；完成端点`last.pt`不能省略profile，失败只保留独立failure产物。旧helper／旧eval因协议不同应拒绝新端点，不得以无AA渲染默默评价。

新plan及每个端点标明：`renderer_profile.id=ibgs_centered_corner_v2_aa_near001_v1`、near=.01、eps2d=.3、`aa_module_sha256`与`renderer_binary`身份。plan另存`backend_python`、`binary`、完整runtime来源；隔离新binary与原安装均绑定，旧环境文件不改。未来评价必须先核profile，再载入新SPEC并在所有源深度／目标渲染周围启用同一AA上下文；不能只对输出RGB事后补偿。

终点固定full/no_source各6000步，不按TRAIN或VAL选分支。本轮与旧结果的区别是共同渲染兼容性修订；如果后续有提升，仍是已有IBGS工程参考，不能将已知AA补偿、near、来源融合或普通梯度公式包装成新的桥梁机制。

入口（待真实前置全部通过再执行prepare）：`uv run --no-sync python scripts/train_ibgs_aa_warm_matched.py --prepare <fresh-data-dir> --fd-run <completed-FD-dir> --backward-run <completed-2TRAIN-dir>`。GPU入口为隔离IBGS解释器运行新snapshot的同名worker：`--run <plan.json> --expected-plan-sha256 <sha>`，root负责外层timeout与自然退出回执。
