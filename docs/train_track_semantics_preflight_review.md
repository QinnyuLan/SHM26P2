# TRAIN 轨迹标签一致性诊断：独立执行前审查

结论：未发现阻断执行的坐标、统计或来源接线问题。此结论仅为代码与合成合同审查；未运行真实人口诊断，未解码真实标签、valid 或 RGB，未使用 GPU，未修改被审查代码与协议。

## 审查范围与版本

| 文件 | SHA256 |
|---|---|
| `scripts/collect_train_track_semantics.py` | `74c37b3e56704f2a2e6b5860e8de8eba09137f1568dbd2690c1335a9ded914d8` |
| `src/bridge_rgs/track_semantic_audit.py` | `60a5f1fb473f395420d863018d7d2192fcb28c78d9de8941881a0bd1b00c69b3` |
| `scripts/diagnose_train_track_semantics.py` | `cdcd72a3132bb97b04197c362707f2df5e672a5ee5d7cae53148437a8ea0f7e0` |
| `docs/train_track_semantics_protocol.md` | `db0ab554bf04ee2153884ea59ce74cacdfd99b3371d25d442f2091b64c4098e8` |

## 核验要点

- 人口固定为保存的 60,000 个 TRAIN track、457,102 个观察对。恢复原 COLMAP 实际关键点；同 track/image 取最后观察，与原三角化去重规则一致。保存 image_id 必须严格递增；统计入口再次拒绝重复观察，包括随后被排除的标签。
- 原格使用 raw COLMAP corner 坐标的 `floor`；敏感性分支使用实际观察经去畸变、prepared K 后的 `np.rint`，两者不混票。官方已知类 rasterizer 加 unknown overlay 与 collector 按同一绘制顺序直接画 255 等价，后画已知类会覆盖先画的 ignore。
- 重投影误差比较保存 XYZ 与实际观察的去畸变原生像素坐标；世界 ray 来自实际观察，不冒充 camera-to-saved-point 方向。strict 固定为误差 ≤1、边界距离 >10；边界 EDT 包括其他类、ignore、valid 洞与画布外。
- 各层先筛观察再计票。观察加权少数票比例与 track 等权少数票比例分开；bootstrap 重采样 track，不重采样单个观察。四组按全部 350 TRAIN 名称划分，删除对应观察后重新计算支持。
- LOO 先扣除 target，再用至少三个其他不同 image_id 和 ≥0.8 共识接受；`3A1B` 的接受偏差有合成回归。LOO 不用于必要条件门。覆盖条件与信号条件独立，覆盖不足不会被较大冲突率替代。
- driver 绑定三模块 SPEC、全部来源与运行库版本；准备阶段仅哈希标签文件字节，不解码。collector 执行时只解码有标注 TRAIN 的 JSON/mask/valid；无标注保持 NA。源码与输入终检、禁止覆盖、失败回执及 CPU-only 入口与协议一致。

## 合成验证

执行：`CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 uv run --no-sync pytest -q -p no:cacheprovider tests/test_train_track_semantics.py tests/test_track_semantic_audit.py tests/test_train_track_diagnostic.py`。

结果：35 passed，0.23 秒。覆盖坐标舍入、重复观察、pose/camera 来源拒绝、绘制顺序、边界和无效域、NA、LOO 接受偏差、两种加权、独立重采样复算、四组删除、gate 独立条件、JSON 有限序列化及禁止覆盖。测试只用合成数组与临时文件。

## 解释边界

误差单位是去畸变 native 像素；边界距离是原始畸变栅格或 legacy 栅格中的像素中心 EDT，两者不是统一物理距离。低误差与深内区仍不能排除误匹配、遮挡或标注约定差异。区间条件于既有相机和筛选后的保存轨迹，删相机组未重建 SfM，也不是 OOF。

即使必要条件门通过，也只得到“每条轨迹一个固定硬标签”的条件性冲突证据；track 不等于 Gaussian，视角相关的 alpha 合成可以改变像素类别。不得将结果解释为 3DGS 表达能力上限、当前 raw/final 差距的原因，或视角语义模型有效性的证明。
