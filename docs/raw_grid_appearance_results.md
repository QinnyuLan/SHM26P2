# Native/raw appearance-only：固定双臂结果与独立审计

**本次精修为负结果，预先锁定的 engineering gate 未通过，应停止本配置，不扫描学习率或训练步数。** 两臂都在共同官方网格上明显损失 RGB 质量，raw 臂没有超过 native 臂。训练与保存合同审计通过不等于已经解释退化原因；目前不能将现象简单归为已知过拟合，也没有证据将其归为 warp、相机错配或 delta 保存错误。

## 固定协议及完整终点

计划 SHA 为 `a8f889c3c9d7678cb0a1376dfd6bb1f06423e1e470f3f3853991259832869a1d`，根任务外层 session 64576 自然 exit0，`execution_receipt.status=completed`。没有失败后重启或中途选模。

共同 base 是已完成的 v2 semantic checkpoint，350 TRAIN、同一 seed42 独立 shuffle 顺序、各 3,000 步、fresh Adam。只训练 SH0/SH-rest/background，学习率分别为 2.5e-4/1.25e-5/1e-4。两臂使用同一 antialiased pinhole overscan renderer、full SH3、先 clamp(0,1)；native 裁原 prepared canvas，raw 用已验证的连续 float-map 四邻可导 warp。共同 loss 为 .8 L1 + .2(1−7-box SSIM)，SSIM 仅用完整有效 7×7 窗口，无标签/teacher/增密/相机更新。

每步 native 监督 RGB **1,290,806** 像素、SSIM **1,277,048** 个中心；raw 全图 RGB **1,305,480** 像素、SSIM **1,291,662** 个中心。两者有效支持不同，因此是目标网格及支持的工程对照，不能把差值解释为仅插值的因果效应。

以下全部是相同 `official_original_grid_v1`、50 RGB / 41 semantic 的 plain 推理结果；没有 native 指标混入：

| 模型 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 五类 mIoU ↑ | 索区 IoU ↑ |
|---|---:|---:|---:|---:|---:|
| v2 base | 29.496634 | 0.870771586 | 0.271327749 | 94.262986% | 93.219355% |
| `00_native` 3k | 27.524011 | 0.854932100 | 0.297081873 | 94.260011% | 93.199660% |
| `01_original` 3k | 27.490377 | 0.854903135 | 0.296743804 | 94.260014% | 93.203734% |
| 旧 support，仅事后背景比较 | 29.400950 | 0.870649924 | 0.271236874 | 94.552931% | 93.565369% |

语义参数冻结，但 RGB 输入变化仍会影响 refiner，所以必须报告微小的语义变化，不能宣称最终 mask 不变。

## 三个预定配对与 gate

独立 CPU 审计重新计算全部三个已保存配对，数值及输入 receipt 绑定逐项相同。bootstrap 固定 5,000 次、seed20260926，单位为对应相机；区间只描述这批开发视图，不代表独立训练种子或跨桥泛化。

| 候选−参考 | ΔPSNR [95% CI] | ΔSSIM | ΔLPIPS | Δ五类 mIoU（pp） | Δ索区 IoU（pp） |
|---|---|---:|---:|---:|---:|
| native−base | −1.972623 [−2.380027, −1.557568] | −0.01583949 | +0.02575412 | −0.002975 | −0.019695 |
| raw−base | −2.006258 [−2.407147, −1.598267] | −0.01586845 | +0.02541605 | −0.002971 | −0.015621 |
| raw−native | −0.033634 [−0.078177, +0.011612] | −0.00002896 | −0.00033807 | +0.000003 | +0.004074 |

raw−native 的 LPIPS 小幅改善区间为 [−0.00050168, −0.00017293]，但不能覆盖其余 gate 失败。预定 gate 要求 raw 同时较 native/base 提高至少 .15 dB，两个 PSNR 区间下界都大于零，SSIM 点值不降且 LPIPS 点值不升，并且相对 base 的五类/索区 IoU 下降不超过 .2 pp。**只有 raw 对 native 的 LPIPS 方向、两项语义保护条件通过，其余条件失败。** 门槛按计划解释，未按结果修改。

## 来源、冻结与保存核验

[result_audit.json](../runs/raw_grid_appearance_v1/result_audit.json) 为 `status=passed`，这表示执行合同通过，而非效果 gate 通过。审计在外层 completed 后才读取终点：

- 重算 **708 个输入文件 SHA、35 个冻结 source 文件 SHA**，plan 与全部 bound input 未变。
- 两臂均恰好 3,000 步，完整实际视图序列 digest 与预定顺序相同，SHA `5c9370ca5ddf7349cd35291f070cb0011b735c48d8e7beb7d863689564c872ad`。另直接核对每臂 step1/every100 的 31 条日志；不把稀疏日志冒称全 3,000 步逐步记录。
- 从同一完整 base 出发，**128 个非 appearance 模型张量逐位相同**；training cameras、scene_scale、SH degree、feature_dim、refiner 架构与 pixel/manifest metadata 均一致。
- 三个允许的颜色张量都发生有限值更新。每份 compact delta 只含这三个 model key；重新 CPU 加载和叠加得到的完整 state 与训练保存前合同一致，没有旧 optimizer/RNG/density 状态，普通 resume 明确禁止。
- 两臂各 100 个官方 PNG SHA、50/41 指标、实际加载 source、checkpoint SHA、预测完成后才开始 GT scoring 的时间顺序均核验。三个 locked pair 均精确重算。
- 共同 fingerprint：`21a2f19c5d4d703403a0107402d5dd5e006dcd98e5f8e023820c388c3e69a85d`。

实际 002 TRAIN CUDA smoke 在两臂初始化时一致：overscan native crop 对直接 native render 最大差 **0**，四邻 warp 对官方 float cv2 最大差 **1.788×10⁻⁷**，三项允许参数有非零有限梯度。该 smoke 没有执行 optimizer step。非零梯度本身不能替代整个 rasterizer backward 的数值正确性证明；这里的 warp CPU double gradcheck 是更窄的验证。

只读检查冻结 runner 的 target/predict 路径也未发现错配：按同一 sorted TRAIN view 选择原始 pose；source/prepared K 与尺寸逐项相同；`read_rgb` 把 OpenCV BGR 转 RGB 后除 255；native/raw 分别取 manifest 的 image_path/source_image_path，native valid/raw 全一；两者先对同一 overscan RGB clamp，再 crop/warp。未修改这些锁定源或输入来解释结果。

两臂训练时间约 **82.64 / 71.34 s**，不含完整终点评价。每份 delta 为 **95,646,625 bytes**，无需保存完整 scene 副本。

| 终点 | SHA-256 |
|---|---|
| 共同 base | `a77d304f32de4356c1a4608ed5192258b6a9ced4c67a4297809f439481408d89` |
| native delta | `ce6162e2859b9d85396fa78fdb4c7882e26c1c3baeb444cef1fb1b9c4f3ca343` |
| raw delta | `fea32ae75ddd633e76fa1357c491141ebcfcd3751098809080f7d72909db5a8c` |

## 退化是否仅少量视图

以下为明确的事后描述，保留全部 50 个视图，没有排除 worst view 后重报成绩。

| 相对 v2 base | native | raw |
|---|---:|---:|
| PSNR 下降视图 | 46/50 | 50/50 |
| 中位 ΔPSNR | −1.575367 dB | −1.579011 dB |
| 下降超过 1 / 2 / 3 dB | 35 / 21 / 12 | 36 / 22 / 12 |
| 最差 5 视图占总 PSNR 下降量 | 23.953% | 23.763% |

raw 最差 5 个为 127、177、110、183、187，分别 −5.098723、−5.002458、−4.792844、−4.472683、−4.470435 dB。退化广泛存在，不能用少数异常相机解释全部均值变化。根任务人工检查 127 的官方 RGB 发现远景/地平线彩色椭圆层更显著；本 CPU 分析没有将这一观察量化为已知物理成因。

## SH 与背景变化：仅描述关联

[posthoc_delta_diagnostics.json](../runs/raw_grid_appearance_v1/posthoc_delta_diagnostics.json)绑定完成审计，记录逐视图与系数统计。RMS 是每个张量内全部系数差的 RMS，不是 RGB 误差。

| 参数差 | native RMS / max abs | raw RMS / max abs |
|---|---|---|
| SH0 | .101420 / .642395 | .108221 / .639891 |
| SH-rest | .0044244 / .0325587 | .0046435 / .0319866 |
| background logits | .0139678 / .0171856 | .0763306 / .0797115 |

背景 sigmoid 从 `[.47186,.46281,.38735]` 变为 native `[.46758,.45963,.38469]`、raw `[.49142,.48268,.40423]`，均未接近 .01/.99 饱和阈值。SH 系数本身无 0–1 饱和定义；只能另算 DC-only 代理 `.28209479*SH0+.5`。任一 DC 颜色通道超出 0–1 的高斯比例从 **1.11235%** 变为 native **1.28961%**、raw **1.34943%**。这不是带方向 SH 的实际渲染 clamp 饱和率，不能据此断言大椭圆由饱和造成。

按 base 的 `max(exp(log_scales))` 固定 50/90/99 分位分桶，世界尺度阈值为 **.0116925 / .1407124 / 4.0707209**（scene_scale 16.26147）。分桶只用于事后描述，不使用 VAL 来选滤除阈值。

| 尺度分位 | 高斯数 | native 总系数平方变化占比 | raw 总系数平方变化占比 | raw 每高斯系数 ΔRMS 均值 |
|---|---:|---:|---:|---:|
| 0–50% | 249,068 | 35.292% | 38.701% | .0187743 |
| 50–90% | 199,254 | 46.353% | 44.815% | .0226100 |
| 90–99% | 44,832 | 17.877% | 16.073% | .0258394 |
| 99–100% | 4,982 | 0.478% | 0.411% | .0127315 |

最大 1% 高斯并不是系数变化能量集中的组；这**不能排除**少量大高斯凭大覆盖面积产生显著图像影响。系数能量没有 alpha、可见性、遮挡或投影面积加权。

背景票主导的 144,163 个高斯（约28.94%）占 raw 系数平方变化 **43.68%**；无人工票的 50,882 个占 **7.43%**。这些是继承的二维标签票分组，既不等于物理三维真值，也不能把无票组直接称为人工自由背景高斯。现有统计只能提出相关性，不能定位责任点或因果。

## 对旧 support 的独立事后比较

[posthoc_paired_01_original_minus_legacy_support.json](../runs/raw_grid_appearance_v1/posthoc_paired_01_original_minus_legacy_support.json)仅使用双方 completed official receipt 绑定的指标，逐 source SHA 核验；没有 native 混用，不参与原 gate。

raw−旧 support：PSNR **−1.910573 dB**，95% CI **[−2.333171, −1.492411]**；SSIM **−.01574679**；LPIPS **+.02550693**；五类 mIoU **−.292916 pp**（CI [−.979558,+.147684]）；索区 **−.361635 pp**（CI [−1.136922,+.545032]）。这是已开发过的两个模型之间的上下文比较，不能倒推坐标修正或网格训练的独立效应。

## 当前结论边界

保留全部负结果与三个锁定配对，停止这次精修。来源/冻结/重载核验排除了已检查的实现错配，但不证明 renderer 梯度、SH 优化行为或场景欠约束是已知原因；不能仅凭训练顺利与 nonzero gradient 把明显退化归为普通随机泛化波动。任何后续检查应独立锁定为诊断，不能在本报告中追加调参后改写原 gate。

审计代码：[audit_raw_grid_appearance_result.py](../scripts/audit_raw_grid_appearance_result.py)、[describe_raw_grid_appearance_deltas.py](../scripts/describe_raw_grid_appearance_deltas.py)。gate 单测 **3 passed**，覆盖双参考 CI、点值方向与 pp 单位边界；Ruff 通过。审计阶段无 GPU 调用、训练、删除或对锁定计划/source/input 的修改。

后续独立的[TRAIN002 CUDA 梯度数值审计](appearance_cuda_gradient_audit.md)已完成：固定未更新的v2 base，三种RGB路径×SH0/SH-rest/background×三个预定步长，共27组中心有限差分全部通过，最大相对误差1.67124e-5；57次渲染耗时2.476秒，模型恢复exact、输入SHA未变。这仅验证一个相机、平滑线性RGB探针和每参数组一个方向，未发现这一范围的CUDA外观导数错误；不验证完整训练损失/优化轨迹/泛化，也不解释此次退化原因。原实验gate与停止结论不变，没有重训或改动锁定prereg。
