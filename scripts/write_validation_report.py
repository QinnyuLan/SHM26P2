"""Build a readable report exclusively from completed local evaluation receipts."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    experiments = [
        ("监督基线，1500步", "pilot_supervised"),
        ("初版研究模块，1500步", "pilot"),
        ("局部蒸馏修正版，1500步", "pilot_revised"),
        ("修正版＋原生精修，总3000步", "refined"),
    ]
    rows = []
    for label, directory in experiments:
        path = ROOT / f"runs/{directory}/evaluation_native/metrics.json"
        if not path.exists():
            continue
        value = json.loads(path.read_text())
        if value.get("scale") != 1:
            raise ValueError(f"Expected native evaluation in {path}")
        lpips = f"{value['lpips']:.4f}" if value.get("lpips") is not None else "未计算"
        rows.append(
            f"| [{label}](../runs/{directory}/evaluation_native/metrics.json) | "
            f"{value['gaussians']:,} | {value['psnr']:.3f} | {value['ssim']:.4f} | "
            f"{lpips} | {value['miou_foreground'] * 100:.2f}% | {value['miou_all'] * 100:.2f}% | "
            f"{value['semantic_3d']['miou_all'] * 100:.2f}% |"
        )
    lines = [
        "# 实现验证与实际预跑结果",
        "",
        "补充：同学SEM-386已有Dev30留出五类mIoU93.74%，当前没有证明超越；详见[完整比较](peer_comparison.md)。",
        "",
        "所有下表数据均来自保存的 metrics.json，不使用同学的全量拟合分数作为本项目成绩。",
        "训练与选择模型使用的是提供的相机标定条件；50个验证视角没有参与教师/高斯监督或新三角化，其中41个有语义标签。",
        "相机按位置排序留出，属于轨迹内插开发验证；同一验证集经过多次开发使用，不能代替最终盲测。",
        "",
        "## 原生分辨率三维模型结果",
        "",
        "下表全部在1320×989去畸变PINHOLE网格、原始未优化验证相机上评估。PSNR/SSIM/LPIPS覆盖50图，IoU由41图的累计混淆矩阵计算。",
        "前景mIoU只含4种桥梁构件；五类mIoU另含背景。raw3D列来自未经过二维精修的显式三维语义。",
        "",
        "| 实验 | Gaussian数 | PSNR↑ | SSIM↑ | LPIPS↓ | 前景mIoU↑ | 五类mIoU↑ | raw3D五类mIoU↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "前三个1500步实验使用相同初始点数、图像集合、半分辨率训练与相同容量上限，但实际点数、随机采样顺序和耗时并不完全相同。后两个研究版共享相同600步教师。",
        "3000步精修模型额外训练了1500步原生分辨率，不能将其相对1500步模型的变化单独归因于研究机制。",
        "PSNR、SSIM、LPIPS没有被任意合并成竞赛分数；不同模块的预算匹配、多种子显著性和官方盲测仍需完成。",
        "",
        "## DINOv3教师",
        "",
        "实际从ModelScope下载并校验ViT-H+/16，冻结840,592,640个骨干参数，解码器6,376,326参数。",
        "600步512裁剪预跑约79秒，最初6张验证图用于选取600步EMA检查点。随后固定此检查点评价全部41张有标注验证图：",
        "五类mIoU86.80%，缆索IoU85.29%；仅采用前6图时为82.52%，两个统计不能混用。",
        "教师训练使用259个带标注train视角与91个无标注train视角；完整导出350份训练软概率，约2.9GiB，每份包含来源/网格/完整性校验。",
        "详见[教师完整指标](../runs/teacher_pilot/validation_full/metrics.json)和[困难视角对比](../runs/teacher_pilot/validation_full/001_comparison.jpg)。",
        "",
        "## 机制检查及修正",
        "",
        "初版研究模块并未在短训练内全面胜出，因此保存其原始结果。对一个融合快照的检查显示，295个可靠目标中没有缆索目标；9个具有稳定人工缆索标注的点全部遭遇教师冲突。这是局部样例，不能推广为整个数据集的错误比例。",
        "修正版将教师蒸馏限制为局部语义特征更新，停止辅助损失到共享分类器的梯度，按类别证据量平衡，并让至少2次、纯度≥0.8的人工轨迹证据优先。轨迹频数随高斯分裂继承，避免最近邻重新匹配到邻近构件。",
        "这些约束是实际代码与单元测试覆盖的机制；是否提高泛化看上表和后续消融，不作预先保证。",
        "",
        "[受控残差诊断](../runs/residual_diagnostic/diagnostic.json)使用6个训练相机及模型自身合成图像：微小相机扰动的误差平均99.18%可被局部相机补偿解释；随机去掉25%不透明度造成的误差仅约0.89%。",
        "目标图由同一检查点合成，这是一项实现/机制自检，不是真实桥梁误差的因果分解证明，也没有几何真值。",
        "",
        "![合成场景残差诊断](../runs/residual_diagnostic/residual_attribution.png)",
        "",
        "## 运行与测试凭据",
        "",
        "- 环境：[依赖与代码SHA](../artifacts/environment.json)，uv.lock固定依赖，CUDA12.8/RTX5090。",
        "- 数据：60,000个训练专用点；训练原生重投影误差中位0.327px。此指标不等同于渲染质量。",
        "- 120步真实smoke验证RGB/语义/精修/相机诊断/相机更新/结构分裂，含163次实际分裂；没有将smoke低分写作性能结论。",
        "- 单元测试覆盖：验证观测投毒不改变点云、COLMAP姿态/畸变、SE(3)零点梯度、协方差MonteCarlo、遮挡/边界门控、官方标签优先、局部KD梯度权限、固定预算/Adam迁移、断点恢复、纯相机导出/PLY。",
        "- 真实768×768 H+ CUDA前后向已验证：骨干无梯度、监督与EMA一致性梯度有限，峰值分配约2.7GiB。",
        "- 导出的[官方PNG样例](../runs/pilot_supervised/submission_preview/render_receipt.json)为1320×989，语义ID0..4；输入只含相机信息。",
        "",
        "## 继续复现",
        "",
        "```bash",
        "uv run pytest -q",
        "uv run bridge-rgs train --config configs/pilot_revised.yaml",
        "uv run bridge-rgs train --config configs/refine_native.yaml --resume runs/pilot_revised/last.pt",
        "uv run bridge-rgs evaluate --checkpoint runs/refined/last.pt --manifest artifacts/prepared/manifest.json --output runs/refined/evaluation_native --lpips",
        "# 完整6000步教师 + 30000步3D：",
        "uv run python scripts/run_pipeline.py --resume",
        "# 开发选型后，全400最终拟合（独立目录，无验证分数）：",
        "uv run python scripts/run_pipeline.py --all-data --resume",
        "```",
        "",
        "本次没有完成完整30000步、多随机种子消融、相机扰动曲线或官方盲测。因此不声称已达到最终竞赛最优，也不声称理论创新已由这些预跑证明。",
    ]
    (ROOT / "docs/validation.md").write_text("\n".join(lines) + "\n")
    print(ROOT / "docs/validation.md")


if __name__ == "__main__":
    main()
