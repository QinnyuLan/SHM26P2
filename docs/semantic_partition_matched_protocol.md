# 固定密度 Gaussian 类别分区：四臂匹配实验草稿

**状态：仅 CPU 实现/合同测试，尚未准备或冻结正式计划，未训练。** 必须先由 root 的两个真实 TRAIN 相机集成预检证明初值、RGB/alpha、梯度和运行成本可接受，再确定一次完整运行的时间上限。失败预检不能由本 runner 绕过；协议、预算和源都在正式 prepare 时绑定。这不是已有 Gaussian CDF 积分公式的创新声明，也不预设提高准确率。

## 固定来源与解释对象

四臂共同从旧 H3 `runs/h3_moments/02_cross/last.pt` 开始，SHA `22bc8a2ddb260f93cb01b17857c97b2bb0873038efdb9318545cb2bdbb045226`。legacy manifest SHA 为 `551546979a583d46e840bd485559721f361bc28fa4dca60826374ceb74b315fa`。几何、opacity、SH/background、相机、Gaussian sem_features、原 semantic_decoder 及 cross-moment 特征全部冻结；不新增 Gaussian，不优化 RGB，不注入真实照片，不训练或蒸馏教师。

三分区臂都用 `SemanticPartitionField`：每个原 Gaussian 的 inside/outside 五类 logits 初始等于原 decoder logits，另有本地法向三维参数、中心 raw 参数和宽度 raw 参数。中心为 `2*tanh(raw)`，宽度为 `.05+softplus(raw)`、初始 `.75`，softness 固定 `.3`；法向初始本地 x 轴。初始两个类别端点完全相同，因此初始 shape 梯度为零是预期，不应作为失败条件。

| 臂 | 原场 context | 类别先验及可训练字段 |
|---|---|---|
| `refiner_only` | 原 features/RGB/depth/alpha/cross moments | 原 p3d，只有原 refiner 可训练 |
| `marginal` | 同原场冻结 context | 两端类别+无条件边缘 CDF，refiner 可训练 |
| `point` | 同原场冻结 context | 条件均值处点读出，refiner 可训练 |
| `integrated` | 同原场冻结 context | 条件 Gaussian 分布积分读出，refiner 可训练 |

三分区臂参数形状、初始化、optimizer 分组都一致。marginal 法向不影响输出，可能始终没有梯度；它并不具有与其他臂完全相同的活跃自由度。它提供静态类别重赋值/端点容量控制，point 提供同类空间函数但忽略条件方差的控制；refiner-only 控制继续训练本身。**不能仅凭胜过 refiner-only 把额外字段容量的收益归因于积分。** integrated 是预定候选，不从四臂择优。

`render_partition` 从同相机一次原场渲染取得冻结 context，在相同原几何/可见性上执行新的类别 shader。新的 p3d 同时进入 refiner 的先验/entropy 通道和最终 `log(p3d)+residual`，两条反向均保留；不能使用会把它一起 detach 的旧 field 梯度开关。原 features/cross moments 仍来自固定旧场，不因新 q 而重定义。初始分区与原渲染可能有 shader 浮点误差，精确容差继承真实预检，不能伪称逐位初值一致。

## 固定训练

使用350个 TRAIN 保存相机，按 base `training_cameras` 与 manifest 顺序逐项校验，再以名称映射259个有标签视图。视图名称排序，fresh `numpy.default_rng(42)` 每完整 epoch shuffle，提前保存完整2,000步索引；所有臂复用该序列，Python/Torch seed42。7个完整epoch加第8个epoch的187图。全native1320×989，SH阶数3，无progressive/crop/flip、teacher、camera优化或密度调整。只读取 TRAIN mask/valid，不读取真实 RGB 字节。

每臂固定2,000次真实 Adam 更新，唯一最终端点，不中途VAL、不best、不按损失扩步。fresh Adam，betas=(.9,.999)、eps=1e−8、weight_decay=0，无调度或梯度裁剪。head lr=`3e−4`；分区两端 logits lr=`.01`；方向/中心/宽度 lr=`.001`。三分区臂直接复用 `field.optimizer_groups()`，不改变数值。梯度为 None 或合法零仅记录，非有限 loss/梯度则停止；末尾 refiner 和类别端点须实际更新且 finite。

目标固定为：

`weighted_CE(final) + .2*Lovasz_present(final) + .001*mean(residual²) + .25*weighted_CE(raw)`。

raw 不加 Lovasz。refiner-only 的 raw 项对训练参数为常数，仍保留以使日志目标定义相同。使用冻结源 `semantic_loss` 的有效像素分母与 ignore=255 语义，不重写损失；不额外加边界目标或参数正则。五类权重固定为 `[.45604488253593445,.9227690100669861,.8530434370040894,1.1935913562774658,1.5745513439178467]`，绑定已完成 source audit，不重估权重。

全部2,000视图、分项 loss 与 elapsed 写日志，每100步及首末步记录有限性/梯度组。末态检查所有非 refiner 原模型 tensor 摘要与训练前相同、相机相同。独立 `semantic_partition_matched_delta_v1` 只保存原 refiner 和新增 partition 状态、base/manifest/plan/profile/schema 与命名隔离的新阶段 Adam/RNG；不伪装完整 scene，不支持普通 resume，不覆写旧 checkpoint。加载先读相同22bc完整场，再strict载两类状态。

## 源、预检和运行预算

新包必须**完整继承实际通过预检的 package**，逐文件 SHA 相同，不从当前主树混拣 model/core，也不把需要新接口的 adapter 塞进旧36文件包。旧已完成 projective-deck 的评价工具仅作为 PNG/soft保存、五通道软warp和配对统计 helper；工具源码单独复制/绑定。official rasterizer/coordinates/data 必须与旧完成评价源字节相同，实际 imports 全部核对新冻结路径。

prepare 需要真实预检的 passed receipt+analysis、外层 launch receipt 的 natural-exit0 与 execution/plan SHA、6个固定 case 的全部门通过和来源链；不硬编码预检目录版本，可使用修复后新的独立版本，旧失败原样保留。此前纯合成 CDF 和 shader 结果作为预检依赖继承。源码/输入/安装的关键 gsplat 文件及 uv.lock 均绑定，训练后复核。

时间上限在 root 得到实测后用 `--internal-seconds` 明确指定，外层加60秒用于退出/回执；现在**没有冻结时间数值，也未生成实验目录**。预计候选臂比原 head 续训多一个 categorical shader 前反向，不能声称同算力。共8,000个训练 scene context 调用，其中6,000次新增分区 shader；只计实际执行，不以函数调用数替代 CUDA kernel 数。记录每臂耗时、allocated峰值、参数量。三分区臂约增加15N标量；refiner-only参数更少。桌面 RustDesk 允许但必须按 `/proc/PID/exe` 证明真实可执行文件，拒未知compute进程，成本不冒充独占GPU。

## 唯一共同终点评价

所有四臂使用旧 E 相同50原图相机/41标注，在 legacy canvas 上将新 H3 S 与旧 `00f5b84a…524bff` H+ 概率固定等权，然后一次官方五通道软warp后argmax。H+缓存仍只对原 H3 RGB有效，因此每臂每相机须重新渲染H3并逐字节复现原canvas量化RGB和原图PNG。E交付RGB另用已完成1M+MCMC固定mean原图PNG及原评分行，逐SHA核对；本轮不重渲染这两个RGB场、不重新读GT RGB评分，不把本次缓存评价耗时当完整三场系统FPS。

先额外用未续训22bc场及同教师重现50张旧E联合mask，要求PNG字节exact，作为加载/转换/来源控制。随后四臂共200张joint、200张scene-final、200张raw新mask和200份FP32新scene概率全部落盘，之后才打开41份原标注。基线50次加四臂200次共250次原场context推断，其中150次分区shader；教师调用0。评分不会在第一臂完成后提前解码GT。RGB改善不能记作本次语义模块贡献；输出仍属于E的多场工程系统。

固定5,000次seed20260926的相机配对bootstrap，语义重采样逐图CM再pool。候选 integrated 对以下四个参照都必须通过既有单deck实验对应的保守门：refiner-only all5 ≥+.15pp，marginal与point各≥+.10pp，原始E ≥+.20pp；四项paired95%下界均>0。对每个参照分别要求索IoU不降超过.10pp、其余各类不降超过.20pp。scene-only与raw完整报告但不能替代联合采用门，任一失败不换候选/阈值/步数或融合比例救援。即使全过也仅为人工复核资格，不自动替换E或证明跨桥泛化/学术原创。

## 准备入口（尚未执行）

`CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python scripts/train_semantic_partition_matched.py --prepare <new_data_run> --preflight <passed_real_preflight_dir> --internal-seconds <root_fixed_budget>`。

冻结后仅由 root 用同一 plan SHA、uv环境与外层有限deadline启动 `--execute <plan.json> --expected-plan-sha256 <sha>`。合成测试禁GPU、使用 `-p no:cacheprovider`；不污染冻结源码。
