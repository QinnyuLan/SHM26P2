# Corner v2：独立数据与训练准备

本轮是修正像素坐标约定后的工程正确性/性能实验。数据为全新 `artifacts/prepared_corner_v2`，不覆盖 `artifacts/prepared`，不把协议修正称为新方法。准备用2026-09-26冻结的main package，显式 `colmap_corner_v2`、原生1320×989、max_points60000、seed42、val_every8、no BA。

来源及准备核验已通过：

- 400个view的名字、image/camera IDs、官方K/source_cameras、原始及使用的poses、pose covariance、原始image/annotation路径逐项同旧；350/50 split和259/41 labeled split完全相同。
- NPZ中60000个点、位置covariance、track IDs、重投影误差、457102个观测view IDs/offsets及scene_radius逐数组dtype/值精确一致；所有观测属于TRAIN。新增NPZ profile tag为`colmap_corner_v2`。
- 旧prepared全部705文件的SHA在准备前后完全一致。新数据518,789,148 bytes，约494.76 MiB。
- 协议变更作用于raster warp和离散取样；新RGB400个、mask300个、valid1个文件SHA变化。valid数量仍1290806，但位置改变40个（各增减20），因此不能只凭valid像素总数认定评测网格相同。
- 初始化colors有56325行变化，semantic_counts有736行变化，源于修正后的图像和corner UV取样，不能说新模型只有训练时投影采样变化。

原始输入703文件（400RGB、300annotation、2COLMAP text、uv.lock）及旧数据逐文件SHA在`runs/corner_v2_preparation/preparation_receipt.json`。全部新数据SHA另存`prepared_hashes.json`；结构对应检查在`comparison_audit.json`，网格差异和fingerprint在`grid_audit.json`。

训练/准备共同source snapshot：`runs/corner_v2_preparation/source_snapshot`；30个Python文件的tree SHA为`107bfe24f70ba35a4b359c1520e0d97abfd8a59b8209409a720994b880a4f224`。运行不得改用后续main。对应源的34项data/coordinates/coordinate_protocol CPU测试通过；准备runner Ruff通过。

磁盘锁定预算在`budget.json`：开始free 2,449,481,728 bytes；reserved为新数据497MiB、RGB last加atomic临时2×447MiB、未来语义last241MiB、两次终点评价150MiB、source/log5MiB及reserve256MiB，总2043MiB。准备后free约1.928GB。RGB完成后必须重新评估语义atomic峰；不删除既有文件以满足预算。

RGB配置`configs/corner_v2_rgb_full.yaml`只相对`support_split_rgb_full.yaml`改变manifest/output/profile及eval_every=0；保持30k、support_constrained split及原始所有训练参数，save_every=5000滚动last。语义配置`configs/corner_v2_semantic_coupled.yaml`相对原support coupled配置仅改变manifest/output/profile、新RGB warmstart及eval_every=0；保持8000步及原本coupled语义参数、save_every=2000。step checkpoint仅在eval分支产生，所以两配置均无中途step文件或验证选模，最终runner仍评价完整native50/41与LPIPS。

```bash
# 两阶段均需使用这一个source snapshot；新RGB从新NPZ初始化，无旧模型warmstart。
uv run python scripts/run_experiment.py configs/corner_v2_rgb_full.yaml \
  --source-snapshot runs/corner_v2_preparation/source_snapshot
# 仅在RGB completed、profile/inputSHA及磁盘核验通过后执行：
uv run python scripts/run_experiment.py configs/corner_v2_semantic_coupled.yaml \
  --source-snapshot runs/corner_v2_preparation/source_snapshot
```

Legacy native fingerprint是`750b9f53046bc104093715c6c26c090837746c467445364584570feb21f10b7b`；v2是`f5c4419d3d81183310ae322b2c6343406a3e00a3e903a5f9a6e3d549bb63fca9`。不能把两个native分数并入同协议汇总、直接跨fingerprint配对bootstrap或宣称坐标修正的因果增益。独立公共官方raw-grid评价若另行实现，应对两个模型采用各自checkpoint的导出warp，并在同一原始GT网格评分，明确其新评价协议。

相机仍使用竞赛上游根据全体view产生的标定，因此是该已给定标定条件下的held-out image supervision，不能称独立SfM泛化。


## RGB 30k固定终点（2026-09-26）

`runs/corner_v2_rgb_full` 已自然完成训练与native50/41、LPIPS终点评价。训练1118.76秒、498136个高斯、训练峰值allocated GPU1.437GiB；无中途验证/step checkpoints。v2 native PSNR=30.20533188、SSIM=.9030958486、LPIPS=.2239661789。RGB阶段没有训练语义头，其语义输出不用于候选选择。

checkpoint SHA=`7fac0df69f34cbb8ecefdd32c34d1ffd4ae8b7f41f54293c75056effaa36951f`，v2 manifest SHA=`91b41aeecedc352e4882eb80a1bb635d479ae432251a4c96c85a9dc110cbb327`。`stage_audit.json`证实source/config/输入与checkpoint profile/manifest/fingerprint绑定正确。

语义阶段启动前重新锁定`runs/corner_v2_semantic_gate.json`：free2,018,426,880 bytes；语义两份241MiB atomic峰+75MiB最终评价+5MiB logs/source+256MiB reserve共857,735,168 bytes，预算满足。语义8k已按原计划使用此completed新RGB和同一source snapshot完成，未依据RGB终点评价改变超参。两阶段独立成绩和冻结审计见 [corner_v2_results.md](corner_v2_results.md)，不能将本节RGB值对旧native作因果差分。
