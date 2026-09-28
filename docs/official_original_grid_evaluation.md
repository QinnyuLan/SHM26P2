# 独立的官方原始网格评价协议

协议 ID：`official_original_grid_v1`。这是我们为官方原始像素网格定义的共同开发评分协议，不改变已有 prepared/native 评价，不重算或替换历史 native 分数。它固定本项目旧 manifest 的 50 个 VAL 相机，其中 41 个有原始 LabelMe 标注；不是同学尚未完整公开的 Dev30，也不是主办方已经给出的评分实现。

## 预先固定的规则

- 相机参考只允许原 `artifacts/prepared/manifest.json`（文件 SHA `551546979a583d46e840bd485559721f361bc28fa4dca60826374ceb74b315fa`）。恢复 `source_cameras` 中的原始 K、尺寸、畸变及 `w2c_original`，不读取 prepared RGB/mask/valid。
- checkpoint 的训练 manifest 必须声明这些视角仍是 VAL，原始相机与 source 数据路径匹配。保存过的训练 manifest SHA 必须匹配。旧 checkpoint 缺 profile 就是 legacy；新的相机或数据 metadata 不能把旧模型升级为 corner。
- 预测函数只接相机白名单和模型；按模型 profile 调用已有 `distortion_render_grid`，渲染覆盖所需 pinhole canvas，再把 RGB 与 soft semantic probabilities 双线性变换回原始输出网格，最后做语义 argmax。默认没有 teacher ensemble 或 test-time augmentation。
- 先写完全部 50 张预测 RGB PNG 与 ID mask，之后才开始读取原始 `source_image_path` / `source_annotation_path` 的内容、哈希与解码。RGB 在 warp 前 clamp 到[0,1]，双线性 warp 后乘 255、round 后转 uint8；评分再读交付 PNG，因此分数与所存预测文件一致。
- RGB MSE/PSNR 与 LPIPS 使用完整官方图像，不使用 undistort valid 区域、不用 GT 填充边缘。PSNR 使用 `[0,1]`、三通道全图 MSE、最小 MSE `1e-12`。SSIM 固定 Gaussian σ=1.5、11×11 窗口、population covariance，仅对完整窗口中心（四边各去 5 px）求均值。LPIPS 固定 AlexNet/version 0.1、完整图像输入映射到 `[-1,1]`；不缩放图像。
- 原始标注尺寸必须与原图/相机完全相同。复用既有 LabelMe polygon rasterizer 和原 draw order；未知类仅在新评价器内标为 ignore=255，后画的已知类可覆盖它。未画区域保持 background=0。已知类别的栅格化必须与现有函数逐像素一致。
- 原始网格没有 undistort 无效区；只有未知语义标签忽略语义分母。按全部 41 个有标注视角汇总 5×5 confusion matrix，再计算 all/foreground mIoU；不是逐图 mIoU 简单平均。

## 指纹和可比性

公共 fingerprint 包含独立协议 ID、公共评分/栅格化版本、固定名称与 split、原相机、源图/annotation 文件 SHA、实际 rasterized GT mask 原始 uint8 字节 SHA、类别与 ignore ID。它不包含模型 profile、checkpoint、训练 manifest SHA 或模型预测；后者单独记录在 execution receipt。

结果使用 `evaluation_family=official_original_pixel_grid` 和 `official_evaluation_fingerprint`，不冒用 native 的 `evaluation_fingerprint`。只有同一 official fingerprint 的结果才允许比较。legacy 与 corner 模型可以拥有同一公共 fingerprint，但各自的导出 warp 保持自己的训练历史协议；这样比较的是同一原始 GT 网格上的实际交付行为。两种模型的原 prepared/native 指纹仍不同，不允许直接作 native 分数差分。

## 当前验证范围与风险

实现阶段先完成 CPU 合同测试，覆盖 analytic ray warp、soft-warp-before-argmax、模型 profile 主导、未知标签覆盖顺序、原始尺寸/路径校验、预测与 GT 读取隔离、uint8 评分、CM/SSIM/LPIPS 输入合同、指纹的 profile 独立与 GT 敏感性。随后两个固定模型已完成真实GPU评价，见[共同网格实测](corner_v2_official_comparison.md)；下文保留实现阶段的具体检查范围与接口说明。

原始像素协议会改变评分分母、畸变插值和 RGB 量化，与既有 native float 结果的差值不代表模型变好或变坏。首先应在一个固定 legacy 候选上建立新的 official 基线，之后才比较 v2。原始 LabelMe 栅格器和 LPIPS 选择也是本项目明确声明的实现；若组织方后来给出不同规则，应另建版本，不能静默替换此 fingerprint。

## 实现与调用

独立模块是 `src/bridge_rgs/official_evaluate.py`，脚本为 `scripts/evaluate_official.py`。未修改 `evaluate.py`、训练循环、CLI、原栅格器或旧结果。以下只是未来执行命令，本次没有运行真实评分：

```bash
uv run python scripts/evaluate_official.py runs/support_split_semantic_coupled/last.pt \
  --output runs/support_split_semantic_coupled/evaluation_official_v1 \
  --workspace-root /home/sky/workspace/SHM2026
```

`--workspace-root` 默认当前工作目录；checkpoint config 中的相对 manifest、reference 和 source 路径统一按该根解析，不能按模块 `__file__` 推断根目录，因此从 immutable `source_snapshot/bridge_rgs` 导入也可使用。新输出目录必须为空；没有 `max_views`、临时分辨率或关闭 LPIPS 的正式评分开关。

输出为 `rgb/*.png`、`mask/*.png`、`official_metrics.json` 与 `execution_receipt.json`。receipt 记录模型 profile、checkpoint SHA、实际与已声明训练 manifest SHA、固定 reference SHA、实际导入核心源码路径/SHA、脚本入口 SHA、依赖版本、所有交付预测 SHA、源文件/实际 GT mask SHA、先预测后评分的阶段时间及 metrics 文件 SHA。Python API 没有提供脚本入口时该字段为 null；不能凭空推测外部 runner。

当前新增 13 项 CPU 合同通过，与旧导出/协议相关测试合计 29 项通过，Ruff 通过；其中增加了从真实 snapshot 目录动态加载模块并使用相对 manifest/source 路径的回归。另只读检查真实旧 reference：50 个 RGB / 41 个 annotation 路径存在、原相机恢复完整；当前 support checkpoint 的训练 metadata 可通过来源检查。没有为此解码真实源 RGB/GT，也没有 CUDA render、真实 LPIPS 调用或产生模型分数。

当前实现仅支持 plain inference，暂无 teacher/TTA 分支。在第一轮真实评分前，已将`inference_protocol`独立放入每份metrics/receipt，与公共评分协议分离；模型推理策略不进入共同GT fingerprint。未来如扩展推理，必须分别记录实际模型输入及成本，不能把方法配置变化伪装成相同预测；若改变评分或GT定义则必须另建评分协议版本。

## 独立配对比较

`scripts/compare_official_evaluations.py`只接受本协议的完整50/41结果、相同公共fingerprint及绑定指标SHA的completed回执。它核验逐图尺寸、完整像素分母、annotation集合、汇总RGB均值和pooled confusion，不修改native比较器或伪装成native输入。RGB按50个相机成对重采样；语义按41个有标注相机重采样并重新汇总混淆矩阵，固定5000次、seed20260926。差值是candidate−reference，LPIPS负值才是改善。

```bash
uv run python scripts/compare_official_evaluations.py \
  runs/support_split_semantic_coupled/evaluation_official_v1/official_metrics.json \
  runs/corner_v2_semantic_coupled/evaluation_official_v1/official_metrics.json \
  --output runs/corner_v2_semantic_coupled/comparison_official_to_legacy.json
```

以上命令仍待两组正式评分后执行，不代表已有新结果。模型训练profile分别写入比较回执的来源段，不参与公共GT网格的fingerprint。此区间仅描述已多次使用的开发视角集合，不覆盖种子、桥梁、同学Dev30或盲测不确定性。
