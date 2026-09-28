"""Summarize completed native evaluations; keep RGB-only stages separate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = [
    ("原联合模型，3,000步", "refined", True),
    ("旧架构延长至9,000步", "refined_long_v2", True),
    ("局部精修±6，相同6,000步", "refiner_bound6_control", True),
    ("多尺度精修，相同6,000步", "refiner_multiscale_v1", True),
    ("强RGB，6,000步", "strong_rgb_sanity", False),
    ("强RGB，30,000步", "strong_rgb", False),
    ("强几何＋语义，允许精修回传", "strong_semantic_coupled", True),
    ("强几何＋语义，隔离精修梯度", "strong_semantic_detached", True),
    ("强几何，后期语义参数平均", "strong_semantic_averaged", True),
    ("旧多尺度，后期语义参数平均", "multiscale_semantic_averaged", True),
    ("仅opacity续训RGB对照", "opacity_rgb_control", True),
    ("仅opacity＋SfM深度", "opacity_sfm_depth", True),
    ("仅opacity熵实验，匹配RGB对照", "opacity_entropy_control_v2", True),
    ("仅opacity，熵正则0.05", "opacity_entropy_005_v2", True),
    ("稀疏射线前方质量实验，匹配RGB＋ED对照", "sparse_front_pair/00_rgb_ed", True),
    ("仅opacity，稀疏射线前方质量约束", "sparse_front_pair/01_rgb_ed_front", True),
    ("H3，相同容量零矩输入", "h3_moments/00_zero", True),
    ("H3，仅深度方差输入", "h3_moments/01_variance", True),
    ("H3，深度与语义特征交叉矩", "h3_moments/02_cross", True),
    ("仅精修器续训3k，翻转匹配对照", "refiner_flip/00_control", True),
    ("仅精修器续训3k，水平翻转0.5", "refiner_flip/01_flip", True),
    ("修正结构分裂，6,000步RGB", "support_split_rgb_sanity", False),
    ("修正结构分裂，30,000步RGB", "support_split_rgb_full", False),
    ("修正结构分裂，恢复完成30,000步RGB", "support_split_rgb_full_resumed", False),
    ("修正结构分裂，再次恢复完成30,000步RGB", "support_split_rgb_full_resumed2", False),
    ("修正几何＋相同8,000步语义", "support_split_semantic_coupled", True),
    ("修正15k几何，3k仅颜色优化", "support15k_appearance_polish", False),
    ("修正20k几何，3k仅颜色优化", "support20k_appearance_polish", False),
    ("修正20k几何，邻近阶段1NN语义迁移", "support20k_semantic_transfer", True),
    ("旧多尺度，3k全图续训对照", "refiner_crop_pair/00_fullframe", True),
    ("旧多尺度，3k裁剪/全图混合", "refiner_crop_pair/01_mixed_crop", True),
    ("强语义平均，3k常数LR续训", "semantic_lr_pair/00_constant", True),
    ("强语义平均，3k余弦LR续训", "semantic_lr_pair/01_cosine", True),
    ("旧语义1NN迁移，零训练控制", "semantic_transfer_1nn", True),
    ("H2：仅GT续训", "h2_multiscale/00_gt_continuation", True),
    ("H2：教师，无融合", "h2_multiscale/01_teacher_no_fusion", True),
    ("H2：教师，固定范围融合", "h2_multiscale/02_teacher_fixed_sampling", True),
    ("H2：教师，投影范围融合", "h2_multiscale/03_teacher_projection_sampling", True),
    ("强几何＋教师联合语义", "strong_joint_semantic", True),
]


def renderer_transfer_records(expected_fingerprint):
    """Include every fixed mixture, with RGB from its verified common renderer."""
    run = ROOT / "runs/teacher_renderer_transfer_support"
    if not (run / "execution_receipt.json").is_file():
        return []
    receipt = json.loads((run / "execution_receipt.json").read_text())
    if receipt["status"] != "completed":
        return []
    serialized = (run / "metrics.json").read_bytes()
    if hashlib.sha256(serialized).hexdigest() != receipt["metrics_sha256"]:
        raise ValueError("Renderer-transfer metrics changed after completion")
    m = json.loads(serialized)
    if not (m["all_50_student_rgb_pngs_and_fresh_renders_exact"]
            and m["all_student_global_and_per_view_confusions_reproduced"]
            and m["all_student_nonrefiner_tensors_and_cameras_exact"]):
        raise ValueError("Renderer-transfer identity audit missing")
    if (m["validation_views"], m["semantic_validation_views"]) != (50, 41):
        raise ValueError("Renderer-transfer view counts differ")
    records = []
    labels = {"base": "原support", "00_zero": "H3零输入", "01_variance": "H3方差", "02_cross": "H3交叉项"}
    for name, item in m["plan"]["students"].items():
        raw = Path(item["evaluation"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["evaluation_sha256"]:
            raise ValueError("Renderer-transfer student evaluation changed")
        student = json.loads(raw)
        if student["evaluation_fingerprint"] != expected_fingerprint:
            raise ValueError("Renderer-transfer comparison protocol differs")
        values = m["metrics"][f"ensemble_{name}_0.5"]
        records.append({"label": f"{labels[name]}＋DINOv3固定0.5",
            "run": f"{run.relative_to(ROOT / 'runs')}/ensemble_{name}_0.5",
            "model_kind": "fixed_teacher_student_combination", "trained_semantics": True,
            "metrics_source": str((run / "metrics.json").relative_to(ROOT)),
            "metrics_sha256": receipt["metrics_sha256"],
            "completion_receipt": str((run / "execution_receipt.json").relative_to(ROOT)),
            "student_checkpoint": item["checkpoint"],
            "teacher_checkpoint": m["plan"]["teacher_checkpoint"],
            "parameters": m["parameters"]["each_student_and_teacher"][name],
            "gaussians": student["gaussians"], **m["common_renderer_rgb_metrics"],
            "miou_all": values["miou"], "miou_foreground": values["miou_foreground"],
            "iou": values["per_class_iou"], "semantic_3d": student["semantic_3d"],
            "boundary_f1_2px": values["boundary_f1_2px"]})
    return records


def refiner_tta_records(expected_fingerprint):
    run = ROOT / "runs/refiner_flip_tta"
    receipt_path = run / "execution_receipt.json"
    if not receipt_path.is_file():
        return []
    receipt = json.loads(receipt_path.read_text())
    if receipt["status"] != "completed":
        return []
    report_raw = (run / "report.json").read_bytes()
    if hashlib.sha256(report_raw).hexdigest() != receipt["report_sha256"]:
        raise ValueError("TTA report changed after completion")
    report = json.loads(report_raw)
    plan = json.loads((run / "plan.json").read_text())
    labels = {"base": "原support＋固定翻转TTA", "00_control": "续训对照＋固定翻转TTA",
              "01_flip": "翻转训练＋固定翻转TTA"}
    records = []
    for name, label in labels.items():
        audit_raw = (run / name / "audit.json").read_bytes()
        if hashlib.sha256(audit_raw).hexdigest() != report["all_three_predeclared_arms"][name]["audit_sha256"]:
            raise ValueError("TTA audit changed")
        audit = json.loads(audit_raw)
        metrics_path = run / name / "evaluation_native/metrics.json"
        metrics_raw = metrics_path.read_bytes()
        if hashlib.sha256(metrics_raw).hexdigest() != audit["metrics_sha256"]:
            raise ValueError("TTA metrics changed")
        m = json.loads(metrics_raw)
        if ((m["validation_views"], m["semantic_validation_views"], m["scale"]) != (50, 41, 1.0)
                or m["evaluation_fingerprint"] != expected_fingerprint):
            raise ValueError("TTA comparison grid differs")
        records.append({"label": label, "run": f"refiner_flip_tta/{name}",
            "model_kind": "fixed_horizontal_refiner_tta", "trained_semantics": True,
            "student_checkpoint": plan["arms"][name]["checkpoint"],
            "metrics_source": str(metrics_path.relative_to(ROOT)),
            "metrics_sha256": audit["metrics_sha256"],
            "completion_receipt": str(receipt_path.relative_to(ROOT)),
            **{key: m[key] for key in ("gaussians", "psnr", "ssim", "lpips", "miou_all",
                                      "miou_foreground", "iou", "semantic_3d", "boundary_f1_2px")}})
    return records


def main():
    records = []
    fingerprint = None
    for label, directory, trained_semantics in EXPERIMENTS:
        run = ROOT / "runs" / directory
        metrics_path = run / "evaluation_native/metrics.json"
        if not metrics_path.is_file():
            continue
        receipt_path = run / "experiment_receipt.json"
        receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else None
        if receipt and receipt["status"] != "completed":
            continue
        serialized = metrics_path.read_bytes()
        sha = hashlib.sha256(serialized).hexdigest()
        if receipt and receipt["evaluation_sha256"] != sha:
            raise ValueError(f"Evaluation changed after completion: {metrics_path}")
        m = json.loads(serialized)
        if m["scale"] != 1 or m["validation_views"] != 50 or m["semantic_validation_views"] != 41:
            raise ValueError(f"Unexpected validation protocol: {metrics_path}")
        if m.get("evaluation_fingerprint"):
            fingerprint = fingerprint or m["evaluation_fingerprint"]
            if m["evaluation_fingerprint"] != fingerprint:
                raise ValueError(f"Different camera/grid fingerprints: {metrics_path}")
        record = {"label": label, "run": directory, "trained_semantics": trained_semantics,
                  "metrics_source": str(metrics_path.relative_to(ROOT)), "metrics_sha256": sha,
                  "completion_receipt": str(receipt_path.relative_to(ROOT)) if receipt else None,
                  **{key: m[key] for key in ("gaussians", "psnr", "ssim", "lpips")}}
        derivation_path = run / "last.provenance.json"
        if derivation_path.is_file():
            record["derivation_provenance"] = json.loads(derivation_path.read_text())
        if trained_semantics:
            record.update({key: m[key] for key in ("miou_all", "miou_foreground", "iou", "semantic_3d")})
            cm = m.get("confusion_matrix")
            if cm:
                record["cable_precision"] = cm[2][2] / max(1, sum(row[2] for row in cm))
                record["cable_recall"] = cm[2][2] / max(1, sum(cm[2]))
            record["boundary_f1_2px"] = m.get("boundary_f1_2px")
        records.append(record)
    records.extend(refiner_tta_records(fingerprint))
    records.extend(renderer_transfer_records(fingerprint))
    payload = {"scope": "fixed development split; not blind-test or matched peer Dev30 evidence",
               "evaluation_fingerprint": fingerprint, "records": records}
    (ROOT / "runs/optimization_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    lines = ["# 原生分辨率实测汇总", "",
             "由 `uv run python scripts/write_optimization_report.py` 从已完成的原始评价生成。",
             "固定50个RGB/41个语义开发验证视图，1320×989去畸变网格，原始验证相机。",
             "每行是一个实际检查点或明确标出的固定模型组合；不拼接不同场的最优指标。",
             "DINOv3组合额外需要约8.47亿参数教师；raw3D栏属于组合中的同一学生场，未被教师更新。", "",
             "| 实验 | 高斯数 | PSNR↑ | SSIM↑ | LPIPS↓ | 五类mIoU↑ | 前景mIoU↑ | 拉索IoU↑ | raw3D五类/拉索↑ |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        semantic = (f"{100*r['miou_all']:.3f}% | {100*r['miou_foreground']:.3f}% | "
                    f"{100*r['iou'][2]:.3f}% | {100*r['semantic_3d']['miou_all']:.3f}% / "
                    f"{100*r['semantic_3d']['iou'][2]:.3f}%") if r["trained_semantics"] else "— | — | — | —"
        lpips = f"{r['lpips']:.5f}" if r["lpips"] is not None else "未计算"
        lines.append(f"| [{r['label']}](../{r['metrics_source']}) | {r['gaussians']:,} | "
                     f"{r['psnr']:.3f} | {r['ssim']:.5f} | {lpips} | {semantic} |")
    lines.extend(["", "同学SEM386的Dev30数字为五类93.7418%、前景92.3300%、拉索91.4347%。",
                  "本地缺少其具体Dev30视图清单及RGB150留出指标，不能将不同划分的数值差称为同协议提升。",
                  "研究机制配对实验与局限见[研究定位](research_positioning.md)、[优化记录](optimization_log.md)。",
                  "机器可读来源与SHA见[汇总JSON](../runs/optimization_results.json)。", ""])
    output = ROOT / "docs/measured_results.md"
    output.write_text("\n".join(lines))
    print(output)


if __name__ == "__main__":
    main()
