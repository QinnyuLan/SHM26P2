# MCMC RGB参考：值得补齐的工程路线，不是新方法

2026-09-27。仅CPU／一手资料审查；未打开活动checkpoint、未实现策略、未运行GPU。核查限于3DGS-MCMC论文、gsplat官方实现这一来源组，以及已有SalientGS先例，不扩展候选名称。

**建议：MCMC应比继续调整已无支持的局部opacity／分裂启发式更优先，作为一次完整、明确标注实现差异的RGB工程参考。** 它检验的是持续位置探索和容量重定位这条尚未覆盖的训练路线，不能预设会提升桥梁精度，也不能用其成功替代本项目的学术贡献。

## 本项目尚未做过的部分

对`src/bridge_rgs`、`configs`、`docs`以及本地两处runs/preflight的JSON计划／receipt检索，未发现MCMC训练入口、启用配置或完成记录；命中主要是文献审查中的SalientGS。当前mixed策略按signed／abs图像梯度clone/split，并定期reset/prune；hybrid里的有界回收也不等于持续SGLD探索。历史若有未登记的外部试验不在此核查范围内。

Mip固定3D平滑和G3004固定opacity弱化的负结果，不直接否定“把失效点搬到其他位置再持续优化”。反过来，它们也没有证明剩余误差来自初始化或探索不足。这是补强参考的理由，不是新机制启动证据。

## 三个最相关一手来源及边界

1. **[3DGS-MCMC，§3.3–3.6、附录C](https://arxiv.org/html/2404.09591v3)**：位置噪声、低opacity点重定位、opacity／scale正则组成完整训练方案。式9同时修正复制后的opacity与covariance，作者明确其渲染保持只是近似；论文报告同点数及受限预算实验，同时承认仍有3DGS的混叠、反射等表示局限。它不是只在终点删除一组点，也没有证明几何真实或语义准确。
2. **[gsplat v1.5.3 MCMCStrategy](https://raw.githubusercontent.com/nerfstudio-project/gsplat/v1.5.3/gsplat/strategy/mcmc.py)**及同版本[官方simple_trainer](https://raw.githubusercontent.com/nerfstudio-project/gsplat/v1.5.3/examples/simple_trainer.py)：本机已有可用策略实现，无须修改site-packages。需区分策略类和完整配方：官方mcmc入口另设初始opacity=.5、scale multiplier=.1、opacity与scale平均值正则各.01；示例默认世界坐标归一化、AA关闭、opacity LR=.05，均不同于本项目。
3. **[SalientGS，§3.3式2–9、§3.4](https://arxiv.org/html/2607.11285)**：已经用多视图高／低误差估计欠拟合与冗余，修改MCMC birth/relocation采样，并结合相机光度与轨迹重投影优化。作者明确重要性分配是启发式，不是新的MH接受步骤。它直接覆盖“给MCMC加多视图残差权重／冗余回收，再联合姿态优化”这一泛化表述；不能将这些组件换名当创新。这里核查了方法正文，未复现其结果或代码。

## 本机实现的实际接入风险

已读安装包`gsplat/strategy/mcmc.py`与`ops.py`，版本1.5.3；SHA分别为`0c2fc042833f39fae0b785de07d0472fe0d1f22437108b0d04a45c0e4fe67c16`、`b30fa9e4d60a97ec619eb54d43d158d3e9dc8102c3d971acbb46da8385b53074`。以下是代码事实，不能用文档概括替代：

- 默认`noise_lr=5e5`、`min_opacity=.005`，500<step<25000且每100步relocate/add；新增目标为`min(cap, floor(1.05*N))`。达到cap后仍可relocate；位置噪声在refine_stop之后仍执行，不能顺手与增密一同关闭。策略没有本项目的opacity reset或大尺度剪枝，不能不经声明继续叠加它们。
- 噪声是`means_lr * noise_lr * sigmoid(100*(.005-opacity)) * Sigma * Normal(0,I)`；乘的是covariance，不是其平方根。当前世界尺度与官方归一化坐标不同，必须先说明坐标／噪声／scale正则的单位映射；直接照搬5e5并不能称相同物理扰动。
- relocate时，存活目标的Adam非step状态清零，被搬迁的原dead行保留自己的矩；birth时新增行矩为零。它和本项目“所有新行置零”的通用迁移语义不同。API还要求`scales`是log-scale、`opacities`是logit，并会替换Parameter对象：仅建别名字典、不回写scene会导致渲染和optimizer指向不同参数。
- 所有逐点字段必须一起迁移；`semantic_prior_counts`是注册buffer，不会被gsplat参数字典自动处理。RGB阶段语义关闭也不能留下错误行数／来源，破坏后续共享场语义训练和checkpoint加载。全dead无合法采样目标、重复父点、cap、状态恢复及随机数流也须有合同。500k在`torch.multinomial`的2^24元素上限内；但`relocation.py`会将重复次数ratio夹到51，若实际同父副本更多，参数校正并不按真实副本数，需披露／计数而不能悄悄改kernel。
- 我们现有`train.py`固定加`1e-4*mean(scale)/scene_scale`，并非MCMC的完整正则。`MCMCStrategy`本身不替用户加这两项loss。官方simple_trainer调用策略在optimizer／scheduler之后；接入时应固定清楚步数约定，不能误用回调名理解成optimizer之前。

这些问题可通过小范围适配和数值合同处理，未发现使该路线原则上不适合桥梁任务的证据；本轮没有验证CUDA relocation kernel或桥梁噪声尺度。

## 建议的比较口径与学术边界

若后续实施，优先一个**配方明确的gsplat-MCMC桥梁RGB参考**：固定350/50、corner_v2数据／相机、相同SfM点和彩色输入、30k和500k上限，沿用修复SSIM及共同原图评分；关闭语义／教师／姿态更新／Mip和本项目结构分配。初始化opacity与scale、背景shell、AA、正则归一及世界尺度若采用MCMC配方，应逐项列出差异；不要同时要求“完全相同初始张量”又声称完整官方复现。它是整套训练配方参考，**不是单独relocation消融，更不是RGB150精确复现**。

同500k cap及30k步不等于同算力：5%增长可能更早占满预算，每步另有全点噪声操作。必须报告实际点数轨迹、Gaussian-step总量、渲染像素和墙时；采用独立视图采样RNG，使额外随机噪声／采样不悄悄改变相机序列。先验证接口与更新合同，再决定是否投入完整训练，不在本轮直接生成训练计划或根据VAL调噪声。

即使RGB提高，仍须在这个新场上用相同冻结几何语义配方评价，才能说共享RGB+语义系统变好；不拼接旧场mask，不保证raw语义随RGB上升。后续若要提出学术区别，必须以vanilla MCMC为对照，证明新增机制实际区分了相机扰动、赋值欠优化与几何容量，并在控制预算／适配后具有可测作用。现有profile校准、assignment和局部干预结果尚未提供这组证据。**本轮只推荐补齐一个未覆盖的标准强参考，不推荐在其上立即追加“可靠性／语义感知MCMC”。**
