# 像素坐标约定的兼容迁移计划

日期：2026-09-26。依据 [只读像素坐标审计](pixel_coordinate_audit.md)，逐项阅读了当前 `prepare.py`、`data.py`、`io.py`、`evaluate.py`、`fusion.py`、`reliability.py`、`ray_support.py`，以及相关 teacher/pseudo、checkpoint 和 front 调用。首稿为只读实施计划，之后获准分阶段实现兼容分支；实际完成范围见文末。真实 prepared 数据、旧 snapshot、checkpoint 和历史评价协议没有改动，没有启动训练。

**建议只引入两个明确的整体协议：`legacy_mixed_v1` 与 `colmap_corner_v2`。缺少版本的历史产物一律解释为前者；只有新 prepare 生成后者。** 修复必须在来源版本明确的接口边界发生，不能全局给 K/UV 减去 0.5，也不能给旧文件补上新版本标签。

## 1. 不变的坐标合同与必须保留的历史行为

两个协议的相机 K、COLMAP 连续观测和三维投影都仍为 corner-origin：左上像素中心是 `(0.5,0.5)`。数组索引原点 `(0,0)` 指向左上数组元素。两者不同的是历史上如何连接这些接口，旧版不是一个自洽的“全部 integer-center”体系。

令 `h=(.5,.5)`，corner 坐标的 warp 为 W，则传给数组重采样器的映射为 `W_array(p)=W(p+h)-h`。`grid_sample(align_corners=False)` 接受 corner UV 时使用 `2*uv/[W,H]-1`。[OpenCV remap 文档](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html)、[PyTorch grid_sample 文档](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.grid_sample.html)

必须保留的例外是旧 front：其实际执行的是“旧 ED 的 target/valid 筛选 → 额外正确 corner valid 筛选 → 正确 corner CDF/alpha 采样”。v1 不能为了名字统一而退回错误 CDF 采样；v2 则只因 ED 前置筛选修正而获得全链一致，front 的正确 sampler 本身不改。

## 2. 最小版本与传递方案

### Manifest 与运行时解析

新 prepare 在全新的输出目录写 `schema_version: 2`，增加一个整体 `pixel_protocol` 对象，例如：

```json
{
  "id": "colmap_corner_v2",
  "version": 2,
  "intrinsics_and_projected_uv": "corner_origin; first_center=(0.5,0.5)",
  "array_coordinates": "integer_element_centers; first_center=(0,0)",
  "prepare_warp": "corner_conjugate",
  "categorical_resize": "center_nearest_half_up_v1",
  "appearance_prior_nearest": "floor_corner_uv",
  "sparse_sampling": "corner_bilinear",
  "official_export": "corner_conjugate"
}
```

这些字段是可验证的整体配置，**不是供用户任意混搭的七个开关**。解析器只接受两个已知 profile；显式未知版本、v2 缺字段或字段组合不一致时直接报错。旧 schema=1 或历史无 schema 的小型 manifest 没有 `pixel_protocol` 时，运行时解析为 `legacy_mixed_v1`，不写回原 JSON。v1 的解析说明应明确列出 ED/fusion 的旧采样和 front 的历史混合行为。

当前有两个 manifest loader：`io.load_manifest` 和 `data.load_manifest`，还有 `SparseDepthSupport.from_manifest` 内部直接 `json.loads`。最小做法是共用一个小型协议解析/坐标工具模块，并让这些入口都调用它；不顺手合并其他路径解析、类别检查或 I/O 行为。`load_view` 和 teacher reader 只收到 view，故 loader 需在 **内存副本** 给每个 view 附上已解析的协议 ID，避免丢失顶层来源；不得在旧 manifest 文件上补字段改变 SHA。直接传 dict 的入口也要解析，不能绕过版本验证。

### Checkpoint、teacher/pseudo 与评价

| 产物/接口 | 版本来源与最小行为 |
|---|---|
| 新训练 checkpoint | 顶层保存已解析 `pixel_protocol`、原 manifest SHA；config/receipt 同记。张量 schema 无需因坐标元数据改名。 |
| 旧 checkpoint | 顶层无协议即 v1；不得根据当前磁盘上的 manifest 或相机文件猜成 v2。 |
| resume / 普通 warmstart | 校验 manifest SHA 与协议；跨 v1/v2 默认拒绝。当前仅比较 manifest 路径不足以防原路径内容被替换。对缺 SHA 的旧 checkpoint，可继续原有路径约束并如实注明无法从 checkpoint 单独证明原字节；若 receipt 有历史 SHA 则核验，不把今天读到的 SHA冒充历史值。 |
| teacher / pseudo | 新 reader/augmentation 根据来源协议缩放 mask/valid；新 teacher checkpoint 和 pseudo provenance/NPZ 保存协议。现有 manifest SHA 校验继续保留，v1 pseudo 不能仅改 metadata 用于 v2。 |
| 语义平均/transfer 等派生 checkpoint | 继承目标场的协议并记录输入协议；不能掉元数据后被误判为 v1。不同协议的迁移不纳入本次最小实现，默认拒绝或另行显式实验设计。 |
| prepared-grid evaluation | 根据 manifest 选择 load/resize；默认要求 checkpoint 协议匹配。v1 fingerprint 算法原样保留；v2 fingerprint 增加协议 ID、manifest SHA 和有效尺寸。不同协议不冒用同一评价名/指纹。 |
| 官方 PNG export | **auto 根据 checkpoint 的 export 协议选择 warp**，不由 `cameras_path` 的新旧决定。旧模型即使收到 v2 camera manifest，默认仍走旧 export。receipt 写明 checkpoint/相机 SHA、实际协议与 canvas。 |

严格 resume 不能悄悄升级协议。若将来要研究旧几何在 v2 网格继续优化，应是单独批准的新 stage、独立输出、明确迁移记录，而非这次兼容补丁自动放行。已有 source snapshot 不改；新版元数据无法防止人为用不识别它的旧 snapshot 处理 v2，因此新 pipeline 入口必须校验其支持的协议，并在 receipt 固定实际源码。

## 3. 逐接口最小修改

### A. 新 prepare：只在 OpenCV 栅格边界做共轭

`data.undistortion_maps` 新增显式 protocol 参数；历史默认调用保留 v1，新 `prepare_dataset` 明确传 v2。按最终整数尺寸分别计算 `sx=Wout/Win`、`sy=Hout/Hin`，先得到 `Kout_corner=diag(sx,sy,1)*Ksource_corner`；随后仅在临时 OpenCV K 上把 cx/cy 各减 .5：

```
Ksource_cv = corner_to_array_K(Ksource_corner)
Kout_cv = corner_to_array_K(Kout_corner)
maps = initUndistortRectifyMap(Ksource_cv, d, None, Kout_cv, out_size, ...)
```

返回并写 manifest 的仍是 `Kout_corner`。不能先减 .5 再缩放，否则下采样会再次错相。连续 `undistortPoints(xy_colmap, K_colmap, d)` 保持原样；xy、K 本来就在同一体系，三角化、重投影及位置协方差不需要修正。

RGB、mask、valid 共用这一组 map。RGB 保持现有 LINEAR；mask 的 `cv2.remap(..., INTER_NEAREST)` 本来就在数组坐标上选最近元素，不是 `resize(..., INTER_NEAREST)` 的旧缩放相位问题。**不要全局替换为 INTER_NEAREST_EXACT**：OpenCV remap 并不支持所有 resize 插值模式。[OpenCV remap/resize 文档](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html)

LabelMe 原生 polygon 栅格化/覆盖顺序保持原样；不根据 COLMAP 约定去移动标注顶点。新 grid、valid、init appearance/prior 和 manifest 必须写新目录，拒绝覆盖已有 prepared manifest。新旧 split 仍用同一相机决策和 seed，不能趁修复更换验证视图。

valid 的最小语义仍是源数组双线性可用范围 `[0,Win-1]×[0,Hin-1]`，不同时引入更宽松边界或自动 crop。已知零畸变 identity 的负浮点尾差要在 CPU 合同中捕获：优先为零畸变 identity 返回解析整数 map，避免为通过测试全局放宽 valid；如采用边界 snap，应预先固定数值容差并记录 v2 政策，不能按保留面积或评分调它。

### B. 下采样 mask/valid：按来源版本切换，RGB/K 不再额外移位

`io.load_view` 的 K 按实际尺寸整行缩放已经适合 corner 坐标，保持不变；RGB 的 AREA/LINEAR 策略也保持原实现。v1 mask/valid 继续 OpenCV legacy NEAREST，v1 pseudo/fusion valid 继续当前 PyTorch nearest。v2 的离散图改用像素中心最近邻：

```
source_index(j; Nin,Nout) = floor((j+0.5)*Nin/Nout)
                         = ((2*j+1)*Nin) // (2*Nout)
```

第二式用 int64 可明确最近邻 tie 向较大索引取值，避免不同库的浮点边界实现漂移。建议小型 NumPy/Torch 共用定义各自执行 gather；也可使用经合同验证的 OpenCV NEAREST_EXACT / Torch nearest-exact，不能在不支持时静默回退旧 nearest。[PyTorch interpolate 文档](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.interpolate.html)

需要接入的离散 resize 不止 `io`：`train.pseudo_for_view`、`fusion.fuse_scene_evidence` 的 valid，teacher `aligned_crop` 和 scale-jitter/crop 的 mask/valid 都必须从其数据来源协议选择。概率/置信度的 bilinear `align_corners=False` 原来已使用中心相位，不需要再平移；resize 后概率归一化不变。teacher 模型内部 feature/logit 多尺度插值、refiner 的同 buffer 下采样、纯整数 crop/flip 不属于 COLMAP UV 边界，不做全局半像素补偿。

最近邻 valid 是目标中心的有效性判定，**不保证 AREA RGB 的整个源像素面积都有效**。本次只修采样相位；若想升级到完整面积支持，需另一个明确协议，不能顺手改变评分 support。

只读 CPU 小探针已在本机 OpenCV 5.0.0、Torch 2.8.0+cu128 上检查 `4→2、5→2、989→494、989→360、7→11`：上述整数索引与两库 exact 模式全部一致。这不是所有尺寸/版本的等价证明，正式实现仍应以明确的索引合同为准。

### C. 初始化 appearance / semantic prior：只在新 prepare 改

`prepare._cloud_appearance` 从归一化射线得到的 `pixels` 是 corner UV。v1 保留 `np.rint(uv)`；v2 先检查有限值及 `0<=u<W, 0<=v<H`，再用 `floor(uv)` 选最近像素中心，对恰在两个中心中间的 tie 明确选较大数组索引。不要用 `rint(uv-.5)` 后假定它与 floor 完全等价：NumPy 的 ties-to-even 会在整数 corner UV 处产生不同选择。

这会改变颜色和 `semantic_counts`，但不移动三角化点。只能随新 prepare 重新计算，不能覆写旧 `init_points.npz`、旧模型 prior buffer 或已训练特征。

### D. Sparse ED / fusion：转换采样器，同时转换边界

二者都继续接收 corner UV；projection Jacobian、相机/点位置协方差和连续 SfM 射线不减 .5。v2 sampler 使用 `grid=2*uv/[W,H]-1`，`align_corners=False`；若整个双线性 footprint 必须在图像内，中心域边界为 `.5<=u<=W-.5` 和 `.5<=v<=H-.5`。有 b 像素边界时为 `b+.5<=u<=W-.5-b`，不能只改 grid 却留下 `[0,W-1]` 的旧 bounds。

- `ray_support`：targets 仍存 corner pixels；valid erode 后的 footprint 采样、depth、alpha 全部按解析协议路由。把 `border_px` 定义固定为距离可用像素中心的整数层数。给 target/support provenance 带上协议，确保独立 `sparse_depth_loss` 不因丢失上下文选错 sampler。v1 保留原 align_corners=True 公式与旧边界，重现旧历史。
- `reliability.ellipse_sample_probabilities`：v2 同时修正全部 ellipse 点的 bounds 以及概率、teacher confidence、depth、alpha 的同坐标采样；调用处显式传协议，不能凭张量形状猜。v1 保留当前 `_sample_map` 的 `+.5` 和旧 bounds。
- `fusion`：先缩放 corner K、UV/协方差到目标网格，再由 sampler 做 corner→array；不能先 `uv_native-.5` 再乘 scale。480 网格的半像素不是 native 半像素。当前 opacity×valid 门控及容差参数保持数值不变；若额外要求 valid footprint 全真，是独立可见性策略变化，不夹带在修坐标时。
- `ray_termination`：正确的 corner sampler 保持不变。v1 继续历史两次 gate；v2 继承修正后的 ED gate 和既有 corner gate。缓存 key/provenance 加协议，避免相同相机 ID/尺寸误复用旧筛选。

### E. 官方导出：旧 checkpoint 默认保持旧 warp

`render_cameras` 必须保留 `load_scene` 返回的 checkpoint 元数据，解析后显式传给 `distortion_render_grid`。v1 逐行保留当前 `arange → undistortPoints → canvas shift`。v2 则：

```
target_corner = stack(arange(W), arange(H)) + .5
source_corner = undistortPoints(target_corner, Kcorner, d, P=Kcorner)
source_array = source_corner - .5
# 由 source_array 的 floor(min)/ceil(max) 选择整数 overscan 边界
render_Kcorner[:2,2] -= minimum_array_index
remap_source = source_array - minimum_array_index
```

保持现有 overscan 上限、RGB/probability 线性 remap、argmax 和 PNG 类别 ID。零畸变的直接 pinhole 输出路径本来正确，不平移整幅图。相机目录及 manifest 的 K 都仍按 COLMAP corner 解读；是否采用新 warp 由 checkpoint profile 决定。最小版本不增加能静默修改旧模型输出的默认 override；如果将来显式比较旧模型新 warp，另存实验输出并注明不等于历史导出。

## 4. 实际必要的测试，不靠长训练验坐标

| 最小测试组 | 必须捕获的错误 |
|---|---|
| 协议解析与来源传递 | 缺字段旧产物→v1；完整v2→v2；未知/不完整v2报错；两个 loader、dict入口、teacher/pseudo、target对象不丢版本；原 manifest bytes/SHA不变。 |
| prepare 解析 oracle | 使用独立 SIMPLE_RADIAL 解析式检查 `D(p+.5)-.5`；零畸变 native identity、半尺寸、奇数高989→494、正负畸变、异主点。验证先scale后shift，K输出仍corner，mask/RGB/valid共map。identity有效边界不因浮点尾差丢列。 |
| 离散 resize 与 prior | 1D索引坡度/交替label，包含tie与奇数比例；NumPy/Torch索引逐位相同、标签集合不变、scale1 identity；floor corner首末像素/负值/整数tie；v1 rint与旧nearest保留golden。 |
| Sparse / ellipse / front | 在真实像素中心定义线性场 `F(u,v)=1+u+2v`，验证(.5,.5)、亚像素及最后中心；v1保留审计中的+1.5误差，v2精确采样。测试带b边界、valid洞、完整footprint、alpha/depth/confidence共同采样；zero-cov ellipse中心合同；v2 ED与front同射线，v1 front仍历史混合链。 |
| Export 双协议 | 保留旧整数RayScene的legacy golden，但不能再把它当gsplat正确性证明。新增 `j+.5/i+.5` 的corner RayScene，以独立目标corner射线检查v2正/负畸变、overscan、identity和常量平移；旧checkpoint+新camera manifest仍走legacy。禁止RGB/GT读文件。 |
| Checkpoint/评价兼容 | 新元数据往返、派生checkpoint保留协议；默认拒绝跨协议resume/eval；旧checkpoint默认旧export和旧fingerprint；v2 fingerprint明确不同，输出目录不覆盖历史结果。 |

上述核心可全部在 CPU 跑。接线完成后再做一次很小的真实 gsplat CUDA 渲染合同：验证 j+.5 的投影/采样约定及所选 K，无优化步骤、无长训练；用于排除替身再次与真实 rasterizer 不一致。对 legacy 回归要求相同锁定依赖下的 maps、索引、fingerprint 与导出 PNG 相同；GPU浮点重渲染本身若不逐位确定，应另报容差，不把坐标兼容性夸大为全训练逐位复现。

## 5. 建议实施顺序与交付边界

先实现协议解析和小坐标/离散缩放 helper，并锁定 legacy golden；再接新 prepare/appearance、新数据读取与teacher/pseudo、ED/fusion、最后checkpoint与export。全部测试通过后才生成单独的 `artifacts/prepared_corner_v2` 及审计 SHA。这个新目录不能被现有运行配置自动选中，旧 prepared 和所有既有 snapshot 始终保持原样。

不需要修改三维相机约定、gsplat renderer K、COLMAP 连续观测、点位置协方差、PLY 导出、H3 矩公式或 refiner 内部普通张量插值。坐标修复是数值合同修正；需要未来单独受控实验测实际影响，不预言它会消除浮层、推翻已有负结果或构成学术创新。

## 6. 获准后的实施状态

兼容实现已落在 `coordinates.py`、`data.py`、`prepare.py`、`io.py`、`evaluate.py`、`reliability.py`、`fusion.py`、`ray_support.py`、`ray_termination.py`。新增 CPU 合同分别在 `tests/test_coordinates.py`、`tests/test_coordinate_protocol.py`、`tests/test_coordinate_sampling.py`。默认 API 仍为 legacy；新 profile 名称是 `colmap_corner_v2`。本实施者没有编辑训练主循环、teacher 或 CLI；这些入口的 metadata/调用传播由 root 同步集成，必须一起交付，不能仅将 prepare helper 拷贝到旧 snapshot 使用。

以下内容已经落实：

- 两个固定协议的解析、view 内存继承、未知/冲突协议拒绝；两个 loader 对所有协议仅在内存记录源 manifest SHA，不修改源文件。旧 checkpoint 一旦已保存 SHA，同样不能跳过当前文件匹配；真正无 SHA 的历史 checkpoint 保留原兼容路径。
- v2 正确共轭 prepare、零畸变 identity 的精确 map、corner 最近邻 prior、中心相位离散 resize。新 v2 NPZ 加 `pixel_protocol` 标量字符串；旧 NPZ 不重写。独立 sparse/fusion 入口同样检查 NPZ 来源，缺字段视为 legacy。
- prepare 在任何写入前拒绝 **两个方向** 的跨协议覆盖；v2 对同协议非空目录也拒绝，旧 v1→v1 行为保留。只在 pytest 临时目录成功生成合成三相机 v2 数据，没有生成真实 `prepared_corner_v2`。
- ED 与 ellipse 的所有相关采样及边界一同版本化；front 保留正确 sampler 和历史旧 gate 组合；fusion 拒绝跨协议 scene/views。
- 官方 export 根据 checkpoint 顶层 metadata 决策，旧 checkpoint 即使收到新 camera manifest 也走 legacy；zero-distortion 仍直接输出。v1 fingerprint payload/hash 不变；v2 加 profile 与 manifest SHA，评价拒跨协议。

实现没有添加新的 visibility 阈值、改变 RGB resize 插值或重写现有结果。CPU 合同包含真实 OpenCV 操作、独立径向解析式、临时文件 I/O、旧导出 golden 和正确 corner 射线替身。没有做 CUDA 渲染、优化步骤或新模型评价，因此不能称“新协议已验证有性能收益”，也不能用 CPU 替身测试冒充真实 rasterizer 的新 GPU 验证。

2026-09-26 补充验证：loader/SHA/evaluation 相关 33 项 CPU 测试通过。只读复核现有 `artifacts/prepared/manifest.json` 的 SHA 为 `551546979a583d46e840bd485559721f361bc28fa4dca60826374ceb74b315fa`，全 50 个 VAL、原生尺度的历史 fingerprint 仍为 `750b9f53046bc104093715c6c26c090837746c467445364584570feb21f10b7b`；给 legacy checkpoint 声明错误的已知 SHA 会在评价输出前报错。

后续 root 完成整仓373项测试（含当时已有CUDA测试）及Ruff，并新增实际gsplat合同 `tests/test_coordinates_cuda.py`：单高斯投影到corner坐标(6.5,5.5)时，alpha峰值落在数组[5,6]；相邻中心对称，v2在中心/半像素处的采样与真实alpha网格一致，而legacy保留历史偏移。该新增CUDA测试已通过。随后TTA实验用新loader完整复现三个旧模型的150张原始mask，以及对应RGB/raw混淆矩阵，确认兼容性；这些均不是v2模型效果提升的证据。真实v2数据准备与独立重训另行登记，不覆盖旧实验。
