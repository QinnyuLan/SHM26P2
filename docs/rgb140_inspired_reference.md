# RGB140-inspired mixed-gradient RGB reference

状态：CPU合同、610步CUDA smoke及固定30k/native终点评价现已完成，CPU审计通过；
原生三项RGB差异区间均跨零，本项目实际训练更慢，详见[结果与成本](rgb140_reference_results.md)。配置为
`configs/rgb140_inspired_mixed_500k.yaml`。这是本项目共同 350/50、corner_v2
协议下的 RGB140 启发参考，**不是 RGB150 exact，也不是增密单因素消融**。
RGB140 明确披露 signed clone .0002、absolute split .0004；RGB150 的完整
resolved 配置没有在已有进展包展开。来源为
[RGB140](../project_progress_20260918/sources/RGB-140-STANDARD-ABSGRAD-ALL400.html) 与
[RGB130](../project_progress_20260918/sources/RGB-130-STANDARD-ALL400.html)。后者明确
Standard opacity LR=.025，故本配置取 .025，不沿用本项目 .05。

## 共同条件及非等价边界

- 与修复 SSIM 后的本项目 30k RGB 配方共用 v2 manifest/初始化点、seed42、固定相机、
  2000 初始背景 shell、SH3/每1000步升阶、shuffle、渐进分辨率与其余 Adam/均值LR日程。
  沿用当前 gsplat AA、可学习背景、有效像素损失、7×7 box SSIM（显式 NCHW 修复）、
  现有微小尺度正则和参数约束。它们不冒充 Graphdeco/RGB140 的完整上游默认值。
- 初始 RGB/几何构造共用 `GaussianScene`。新模式跳过语义计数初始化、类别权重读取和
  mask 像素加载，所有语义/教师/相机归因/区域RGB加权/结构配额关闭。这不改变初始
  RGB/几何张量；同时意味着与旧强场的 `.15` 标签区域RGB损失、语义配额等也有差异。
- 世界尺度沿用当前模型点距中位中心的90%半径（含 shell），并非宣称与对方相机extent
  相同。小点阈值 `.01*scene_scale`、大点剪枝 `.1*scene_scale` 是明确选择的 gsplat
  Standard-style 默认。后者会移除部分初始 shell，是该参考的固定行为，不能看结果后关闭。
- 同500k上限、同30k并不保证最终实际点数/累计算力相同。日志记录
  `reference_gaussian_steps`（每次训练实际渲染前的 N 求和）、
  `reference_render_pixels`（每次训练 H×W 求和）、`reference_peak_gaussians`。
  本模式没有诊断额外渲染；另报真实墙时、最终N与显存。两者都不是栅格化精确 FLOP。

## 明确的增密合同

只新增 `densification: rgb140_mixed`。gsplat 1.5.3 `DefaultStrategy(absgrad=True)`
本身对 clone/split 使用同一个 absolute accumulator，不满足双梯度配方；当前旧
`absgrad` 模式也仍走本项目结构算法。因此新模块独立实现，旧模式不换算法。

同一次 RGB backward 前保留 `means2d.grad`，同时读取 renderer `.absgrad`；后者是
每像素梯度绝对贡献的聚合，不能由最终 signed 向量的绝对值替代。每视图先乘
`(W/2,H/2)`、取二维 L2 范数，再按该窗口可见次数平均；可见要求两轴 radius>0。
仅支持当前 unpacked、单相机训练。没有 signed/abs 任一梯度即报错，不静默退化。

1. 从 step1 累积。`500 < step < 15000` 且整100步增密：600,…,14900。
2. 最大轴尺度≤`.01*scene_scale` 的点仅按 signed均值>.0002 clone；更大点仅按
   absolute均值>.0004 split。阈值严格大于、无最少独立视图门/自适应阈值。
3. 可用净增容量为 `500000 - 当前N`，每 clone 或两子 split 均净增1。
   合格候选按各自梯度/各自阈值降序，同分原点索引升序。没有前景配额、recycle
   或提前借用本轮预计剪掉的容量。初始超过cap直接拒绝，不隐式删除输入点。
4. clone 保留父点和复制点。split 删除父点，两子偏移为 `R diag(scale) ε`，
   两个 ε 独立标准正态，所有轴尺度除1.6，opacity/SH原样继承。
   保留点与 clone 父点沿用 Adam 矩，新 clone 行和两 split 子行矩归零、Adam step保留。
5. grow 后 prune：opacity<.005；从 step3100 开始同时剪最大轴>.1scene_scale。
   不用屏幕尺寸剪枝。先缩小子点再判断其大小。全部剪空为显式失败，无自动救场。
6. 在3000/6000/9000/12000，grow→prune之后 cap opacity=.01，并清空所有opacity行的
   Adam矩，保留step；15000及之后无增密/reset，无reset后暂停。

所有动作位于**本项目 optimizer.step 之后**，因此只是 Standard/gsplat 风格，
不是 Graphdeco train.py 逐行复刻。本机 gsplat1.5.3 的
`step % reset_every == 0 & step > 0` 条件被 Python 解析成恒不成立的链式比较；
新模块用明确布尔条件，不调用或改动 site-packages 的该分支。

## 恢复与执行边界

独立双统计保存于 `density_state.mixed_gradient`，包括完整policy/单位版本/三数组。
同模式 resume 严格核对policy、shape、dtype及非负有限值；缺失窗口拒绝。
新/旧模式之间不允许伪装 resume。普通模型、Adam和torch/CUDA/NumPy RNG仍走原checkpoint链。
CPU合同验证保存窗口+torch RNG可重放下一随机split；不承诺CUDA训练逐位可复现。
旧模式 checkpoint 不添加该键，旧统计/动作保持原逻辑。

CPU测试覆盖 signed消除但abs保留、双阈值分派、归一化cap排序/同分、实际500k cap、
随机旋转偏移与三轴收缩、grow/prune/reset先后及边界、Adam/AMSGrad迁移、窗口/RNG恢复，
以及禁用mask读取。16项新合同加相邻训练/恢复/SSIM合同合计79 passed、6 CUDA skipped，
Ruff及编译检查通过。使用真实v2 NPZ在CPU分别初始化两个配方，62000点的means/quats/
scales/opacity/SH/background逐位相同，scene_scale均为16.261470794677734。

root随后授权一次610步、180s的无VAL CUDA smoke，以确认实际 `grad+absgrad`、
首次600步增密和之后10步优化，30k尚未授权。独立准备目录
`/mnt/data/SHM2026/runs/rgb140_inspired_mixed_smoke` 继承已审核的
`mask2former_reference_v1/source_snapshot`，仅替换train.py、增加本helper；
来源/config/input绑定在plan中。smoke的610步会压缩LR/分辨率日程，
它不是30k训练前缀，也不输出质量结论。运行状态以execution_receipt与smoke_audit为准。

该smoke于2026-09-27自然exit0，用时21.505秒（含导入、训练、终点CPU检查，未跑VAL）。
610次均实际收集双梯度；step600有1238 clone、2914 split、53 prune，
62000→66099点，无cap拒绝。新行12组Adam矩检查全部为零、旧引用已移除；
随后601–610继续优化，终点模型/优化器张量全部有限、形状匹配，新窗口263884次
可见观测。只在指定几个步记录原始双梯度快照，所有步由helper检查可见梯度有限性。
父源、输入与worker前后SHA均相同，GPU已释放并交接。此结果仅证明实际执行合同，
不能代替30k质量/容量/计算预算比较。证据：

- `execution_receipt.json`：状态completed，worker returncode=0。
- `smoke_audit.json` SHA：`ecd3671346a581f7fab5be5273a62e98c4b04e0b5b35a1069a8bb3a42ca19282`。
- `last.pt` SHA：`ffd9712c431bc720211df818e6c9cf73355c768c545cff936273bcbcd3254808`。

建议正式执行使用 `scripts/run_experiment.py` 冻结源码，输出独立run；最终仅解读RGB
PSNR/SSIM/LPIPS与成本，语义字段未训练不作为结果。共同v2 native和独立official评分
仍各自保留fingerprint，不与同学全400 TRAIN拟合或未知Dev30数字直接相减。
