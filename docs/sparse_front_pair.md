# RGB + ED 底座上的前侧质量下界控制

本实验是经典稀疏深度/射线支持的工程对照，不作新颖性声明。固定两组各2000步、350个TRAIN、native grid、独立shuffle seed42，从同一 `runs/strong_semantic_coupled/last.pt` 开始。共同底座为原 RGB loss 与 `sparse_depth_weight=.05`；唯一科学变量是 `sparse_front_weight=0` 对 `.05`。region RGB权重沿用此前opacity-only控制的0。代码、warmstart、数据与配置SHA由各自 `experiment_receipt.json` 绑定，两个正式运行都复用 `runs/sparse_front_preflight/source_snapshot`。

只训练opacity，保留全部几何位置/shape、SH、background、语义特征/heads、人工prior、相机。SH3从step1，densification/opacity reset/teacher/fusion/semanticgeometry关闭。`sparse_front_weight` 默认0，不构建额外front support、不额外渲染；正权重只允许 `parameter_scope: opacity_only`。孤立front正则鼓励去除密度，故保留RGB+ED共同底座，并报告同一组固定射线上的总alpha及近表面代理，不用正则项本身下降替代外部RGB/语义评价。

正则复用 `docs/ray_termination_diagnostic.md` 的固定32边、保守floor和完整遮挡机制。对每个SfM观测，目标/位置std、confidence、门控/edge均detached，`L_front=Σ confidence*M(edge_floor)/Σ confidence`，**不除alpha**。低alpha仍保留。16/32 bins不加密、不根据VAL或训练结果选边；本阶段使用32。所有新损失仅经opacity求导。

近表面区间 `[D−margin,D+margin]` 仍用同一32边CDF夹界：若前后CDF分别在 `[L_b,U_b]`、`[L_c,U_c]`，则近表面质量代理夹在 `[max(0,L_c−U_b),max(0,U_c−L_b)]`。这是粗离散中心CDF的代理区间，不是精确表面命中率。训练日志会记录lower/upper/区间宽/alpha/near-band，但每条随机视图不同，不能直接凭日志趋势判定是否塌缩。

两臂真实2步预检均通过：off额外front render0次、on2次；front单项opacity梯度有限非零；实际更新只改变opacity；其他全部模型张量/相机逐位相同；冻结参数无Adam state；sampler状态完全相同。输出 `runs/sparse_front_preflight/preflight_report.json`。无支持目标的所有front诊断显式写None/0并附view ID，防止继承上一个视图的值。front/scope/ED回归45项CPU通过、2项CUDA跳过；真实预检补足大场景CUDA权限检查。

运行配置：

```bash
uv run python scripts/preflight_sparse_front.py
uv run python scripts/run_experiment.py configs/generated_sparse_front_pair/00_rgb_ed.yaml \
  --source-snapshot runs/sparse_front_preflight/source_snapshot
uv run python scripts/run_experiment.py configs/generated_sparse_front_pair/01_rgb_ed_front.yaml \
  --source-snapshot runs/sparse_front_preflight/source_snapshot
```

起始诊断已使用新的固定 `runs/sparse_front_pair/diagnostic_plan`，其全部SfM目标/16和32边与此前 `runs/ray_termination_diagnostic/targets.npz` **逐位一致**；新报表仅增加near-band夹界字段。before/after共享这个计划，不按训练结果重选目标。所有测量都使用正式训练的同一源码快照，无新增VAL像素；最终RGB/语义评价单独保留完整50/41视图与LPIPS。

```bash
PYTHONPATH=runs/sparse_front_preflight/source_snapshot uv run python scripts/audit_ray_termination.py \
  --output runs/sparse_front_pair/diagnostic_plan --measure \
  --checkpoint runs/strong_semantic_coupled/last.pt \
  --measurement-output runs/sparse_front_pair/diagnostics_before
# 两个after只换checkpoint和measurement-output；目标/相机/edges全固定。
```

保留已知局限：新CDF使用corner UV→array UV−.5双线性采样；已有ED sampler/整个CV2预处理约定未被统一修复。条件SfM协方差不等于可校准3D真值，高斯中心深度不等于真实表面首次交点；32-bin区间仍宽。正结果也只能支持本桥梁数据上这次开发控制，不能外推跨场景或声称创新机制有效。

## 完成结果：有损取舍，暂不提升为默认方案

两组均完成2000步及全50 RGB / 41有标签VAL的native评测（包含LPIPS）。所有模型张量中只有 `splats.opacity_logits` 改变；其他参数、人工prior、相机、architecture metadata逐位不变。两个run的输入SHA、源码SHA、sampler顺序/cursor/RNG完全一致；唯一训练配置变量是front权重（另有输出目录）。冻结参数没有Adam状态。严格审计见 `runs/sparse_front_pair/pair_invariance_audit.json`，CPU聚合见 `runs/sparse_front_pair/summary.json`。

| 指标 | 同一warmstart | RGB+ED控制 | RGB+ED+front | front相对控制 |
|---|---:|---:|---:|---:|
| PSNR↑ | 30.27351 | 30.42886 | 29.88247 | −.54640 dB |
| SSIM↑ | .902818 | .903052 | .901051 | −.002001 |
| LPIPS↓ | .224494 | .223992 | .226328 | +.002336 |
| final all5 mIoU↑ | 93.9082% | 93.8143% | 94.0344% | +.2201 pp |
| final cable IoU↑ | 93.6436% | 93.5190% | 93.1092% | −.4098 pp |
| final tower IoU↑ | 91.3164% | 91.3793% | 92.1259% | +.7466 pp |
| final foundation IoU↑ | 89.0627% | 88.6864% | 89.4062% | +.7198 pp |
| raw3D all5 mIoU↑ | — | 78.9341% | 78.8515% | −.0827 pp |
| raw3D cable IoU↑ | — | 31.2817% | 29.3821% | −1.8996 pp |

“raw3D”仍是原生场投影后的二维语义评价，不是3D点真值分数。虽然语义features/classifier完全冻结，opacity变化仍会改变投影混合，所以raw和final可随之变化。不能将final提升归因于更好的原生3D语义参数。

补充5,000次成对视图bootstrap（seed20260926）：五类差+.2201pp的95%区间为[−.2155,+.9485]pp；索差−.4098pp区间[−.7095,−.1064]pp；PSNR差−.5464dB区间[−.7481,−.3587]dB。平均语义没有稳定正增益，RGB与索的损害更明确。完整数据为`runs/sparse_front_pair/paired_comparison.json`，该区间不包含跨训练种子或跨场景方差。

235保持在全部评价内，没有从主指标扣除。该视图final all5从55.4914%到61.7898%，LPIPS从.377653到.363364，但PSNR从20.37421降至20.10582。逐图查看，两组仍有明显半透明大片浮层；front后部分下部结构更清楚，远未解决视觉缺陷。235只是已知诊断案例，不用于阈值选择或替代全视图结果。

## 同一16 TRAIN目标的前后复测

19,956个目标、原观测UV、confidence、覆盖mask与边界均固定。结果使用Float64汇总全部目标的confidence加权均值，未按每视图等权平均，低alpha不从集合剔除。每次诊断仅计算一次梯度检查，零optimizer step，结束后checkpoint及scene不变；没有读取任何RGB/语义/VAL像素。

| 固定射线量 | before | RGB+ED控制 | RGB+ED+front |
|---|---:|---:|---:|
| 32-bin前侧下界均值 | .177276 | .172991 | .045788 |
| 32-bin前侧上界均值 | .724132 | .721108 | .644747 |
| 上下区间宽均值 | .546856 | .548117 | .598959 |
| 上下区间宽中位数 | .623075 | .627483 | .718164 |
| 总alpha加权均值 | .995314 | .995216 | .993766 |
| alpha<.5目标数（不剔除） | 14 | 15 | 50 |
| 近表面质量下界均值 | .001245 | .001252 | .001399 |
| 近表面质量上界均值 | .727911 | .730802 | .815477 |

front的目标下界比控制下降约73.5%，其上界也下降，但区间**更宽**。没有证据可将这73.5%解释为真实错误质量或浮层数量的减少。总alpha平均没有大规模归零，但低alpha目标增多；这些稀疏位置也不能排除其他像素出现孔洞。近表面夹界过宽，不能用下界略升、上界上升证明近表面质量保全。

本次实验显示：经典前侧质量下界正则能强烈改变目标量，但在当前固定32边、opacity-only和权重.05设置下，以明显RGB和索类损失换来少量final平均语义改善。保留该混合/负结果，不改默认配置，不通过加密bins、选择权重或去除235追求更好报告。本结果不验证新颖性，也不支持跨场景推广。

完整复测和汇总产物：`runs/sparse_front_pair/diagnostics_before`、`diagnostics_00`、`diagnostics_01`、`summary.json`。可重复CPU汇总并验证计划/结果SHA与固定目标：

```bash
uv run python scripts/summarize_sparse_front_pair.py
```

两个训练checkpoint及共同warmstart均保留；该控制不替换既有模型。两长训练与所有短诊断已经自然结束，无后台GPU任务留存。

最终回归：`test_ray_termination.py`、`test_sparse_front_support.py`、`test_ray_support.py`、`test_training_scope.py`共47项通过；相关新增模块/诊断脚本/tests的Ruff检查通过。真实场景2步及完整2000步权限不变性另由上述独立审计覆盖。
