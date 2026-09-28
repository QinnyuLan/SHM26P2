"""Render completed, SHA-bound capacity results as Chinese Markdown; no scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

DEFAULT_RECEIPT = Path('/mnt/data/SHM2026/runs/teacher_capacity_v1/evaluation_preparation/execution/execution_receipt.json')
PLAN_SHA = 'be5a47bd4ace31f796c6d69cc9ed0f4c769a52278d3aa103a74e668da6f53f69'
LABELS = {'selected_reference': '当前已选 H+ 组合', 'hplus': '本轮匹配 H+ 组合', 'vit7b': '本轮 7B 组合'}
PAIRS = {'vit7b_minus_hplus': ('hplus', 'vit7b'),
         'vit7b_minus_selected_reference': ('selected_reference', 'vit7b'),
         'hplus_minus_selected_reference': ('selected_reference', 'hplus')}
STAGES = ('hplus_real', 'hplus_render_adapt', 'vit7b_real', 'vit7b_render_adapt')
METRICS = (('psnr', 'PSNR (dB)', 1), ('ssim', 'SSIM', 1), ('lpips', 'LPIPS', 1),
           ('miou_all', 'all5 (%)', 100), ('miou_foreground', '前景 (%)', 100),
           ('stay_cable_iou', '索 IoU (%)', 100), ('foundation_iou', '基础 IoU (%)', 100))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def bound_json(path, expected):
    require(sha(path) == expected, f'SHA mismatch: {path}')
    return read(path)


def metric_value(metrics, name):
    if name in ('stay_cable_iou', 'foundation_iou'):
        return metrics['iou'][2 if name == 'stay_cable_iou' else 4]
    return metrics[name]


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_results(receipt_path):
    """Check existing values and provenance, without recomputing any score or CI."""
    receipt_path = Path(receipt_path).resolve()
    receipt = read(receipt_path)
    require(receipt.get('status') == 'completed', 'Evaluation execution must be completed')
    require(receipt['plan_sha256'] == PLAN_SHA, 'Wrong fixed capacity evaluation plan')
    plan_path = Path(receipt['plan']).resolve()
    require(receipt_path == plan_path.parent/'execution/execution_receipt.json', 'Wrong execution receipt location')
    plan = bound_json(plan_path, receipt['plan_sha256'])
    launch_item = plan['training_launch_manifest']
    launch = bound_json(launch_item['path'], launch_item['sha256'])
    require([item['key'] for item in launch['stages']] == list(STAGES), 'Wrong four-stage order')
    stages = {}
    for item in launch['stages']:
        path = item['stage_receipt']
        value = bound_json(path, receipt['training_inputs_sha256'][path])
        capacity, stage = item['key'].split('_', 1)
        expected_steps = 6000 if stage == 'real' else 2000
        require(value.get('status') == 'completed' and value['capacity'] == capacity
                and value['stage'] == stage and value['steps'] == expected_steps, 'Incomplete/wrong training endpoint')
        require(finite(value['elapsed_seconds']) and value['elapsed_seconds'] > 0, 'Missing stage elapsed time')
        endpoint = receipt['endpoints'][item['key']]
        require(endpoint == {'path': value['last_checkpoint'], 'sha256': value['last_checkpoint_sha256'],
                             'steps': expected_steps}, 'Stage/endpoint source mismatch')
        stages[item['key']] = value
    results = {}
    require(set(receipt['result_sources']) == set(LABELS), 'Missing one of the three fixed evaluations')
    for key, source in receipt['result_sources'].items():
        metrics = bound_json(source['path'], source['sha256'])
        scored = bound_json(source['receipt'], source['receipt_sha256'])
        require(scored.get('status') == 'completed' and scored['official_metrics_sha256'] == source['sha256'],
                'Metrics are not bound by a completed scorer receipt')
        require(metrics['official_evaluation_fingerprint'] == scored['official_evaluation_fingerprint']
                == plan['official_evaluation_fingerprint'], 'Original-grid fingerprint differs')
        require(metrics['validation_views'] == 50 and metrics['semantic_validation_views'] == 41
                and len(metrics['views']) == 50 and len({v['name'] for v in metrics['views']}) == 50
                and sum('confusion_matrix' in v for v in metrics['views']) == 41, 'Require full 50/41 views')
        require(metrics['scoring_protocol']['class_names'] == ['background', 'deck', 'stay_cable', 'tower', 'foundation'],
                'Class order differs')
        require(metrics['scoring_protocol'] == scored['scoring_protocol']
                and metrics['inference_protocol'] == scored['inference_protocol'] == source['inference_protocol'],
                'Scoring/inference source differs')
        require(scored['checkpoint_sha256'] == source['checkpoint_sha256'] == plan['inference_scene']['sha256'],
                'Shared RGB field differs')
        require(scored['teacher_ensemble'] == source['teacher_ensemble'] == metrics['teacher_ensemble'],
                'Teacher source differs')
        if key != 'selected_reference':
            endpoint = receipt['endpoints'][key+'_render_adapt']
            teacher = source['teacher_ensemble']
            require(teacher['teacher_checkpoint'] == endpoint['path']
                    and teacher['teacher_checkpoint_sha256'] == endpoint['sha256'], 'Teacher is not fixed last endpoint')
        require(all(finite(metric_value(metrics, name)) for name, _, _ in METRICS), 'Nonfinite/missing report score')
        results[key] = metrics
    hashes = receipt['rgb_files_sha256']
    require(set(hashes) == set(LABELS) and len(hashes['selected_reference']) == 50
            and hashes['selected_reference'] == hashes['hplus'] == hashes['vit7b'], 'Recorded RGB SHA sets differ')
    pairs = {}
    for name, (reference, candidate) in PAIRS.items():
        filename = name+'.json'
        pair = bound_json(receipt_path.parent/filename, receipt['report_files_sha256'][filename])
        require(pair['reference_source'] == receipt['result_sources'][reference]
                and pair['candidate_source'] == receipt['result_sources'][candidate], 'Pair source/direction differs')
        require(pair['difference_direction'] == 'candidate minus reference; LPIPS improves when negative'
                and pair['official_evaluation_fingerprint'] == plan['official_evaluation_fingerprint'],
                'Pair direction/fingerprint differs')
        require(pair['bootstrap_repeats'] == plan['bootstrap']['repeats']
                and pair['seed'] == plan['bootstrap']['seed'], 'Bootstrap protocol differs')
        for metric, _, _ in METRICS:
            row = pair['metrics'][metric]
            require(row['reference'] == metric_value(results[reference], metric)
                    and row['candidate'] == metric_value(results[candidate], metric), 'Pair endpoint score differs')
            interval = row['paired_view_bootstrap_95_interval']
            require(finite(row['difference']) and isinstance(interval, list) and len(interval) == 2
                    and all(finite(v) for v in interval) and interval[0] <= interval[1], 'Missing/nonfinite pair interval')
        pairs[name] = pair
    gate = bound_json(receipt_path.parent/'adoption_gate.json', receipt['report_files_sha256']['adoption_gate.json'])
    require(gate == receipt['adoption_gate'] and type(gate['passed']) is bool
            and bool(gate['checks']) and all(type(v) is bool for v in gate['checks'].values())
            and gate['passed'] == all(gate['checks'].values()), 'Gate source/decision differs')
    return receipt, plan, results, pairs, stages, gate


def markdown(receipt_path, values):
    receipt, plan, results, pairs, stages, gate = values
    decision = '采用本轮 7B 固定组合。' if gate['passed'] else '保持当前已选 H+ 固定组合；不采用本轮 7B，不据此改选匹配 H+。'
    lines = ['# DINOv3 容量对照：正式原图结果', '',
             f"预定采用门槛：**{'通过' if gate['passed'] else '未通过'}**。{decision}", '',
             ('三组共用同一 H3 共享场，只向教师输入该场的相机渲染 RGB；固定 0.5 概率融合、tile768/stride512、flip、context0.25。'
             '评价为同一原图网格 50 个 RGB 视角、41 个有语义标注视角，执行回执确认全部 50 张 RGB PNG 的 SHA 集合相同。'), '',
             '| 模型 | '+' | '.join(label for _, label, _ in METRICS)+' |',
             '|---|'+'---:|'*len(METRICS)]
    for key, label in LABELS.items():
        lines.append('| '+label+' | '+' | '.join(f'{metric_value(results[key], name)*scale:.5f}'
                                                     for name, _, scale in METRICS)+' |')
    lines += ['', '差异方向全部为“候选 − 参照”，括号为已有的视图配对 bootstrap 95% 区间；语义差为百分点，LPIPS 负差更好。', '',
              '| 对比 | '+' | '.join(label.replace('(%)', '(pp)') for _, label, _ in METRICS)+' |',
              '|---|'+'---:|'*len(METRICS)]
    for name, (reference, candidate) in PAIRS.items():
        cells = []
        for metric, _, scale in METRICS:
            row = pairs[name]['metrics'][metric]
            lo, hi = row['paired_view_bootstrap_95_interval']
            cells.append(f"{row['difference']*scale:+.5f} [{lo*scale:+.5f}, {hi*scale:+.5f}]")
        lines.append('| '+LABELS[candidate]+' − '+LABELS[reference]+' | '+' | '.join(cells)+' |')
    lines += ['', '采用门槛要求 7B 相对匹配 H+ 和当前已选组合均 all5 至少 +0.30 pp、配对区间下界大于 0、前景与索点估计不降，且全部 RGB PNG 相同。', '',
              '| 已保存门槛检查 | 结果 |', '|---|---|']
    lines += [f"| `{key}` | {'通过' if value else '未通过'} |" for key, value in gate['checks'].items()]
    lines += ['', '| 实际训练阶段 | 固定步数 | 已记录耗时（秒） |', '|---|---:|---:|']
    for key in STAGES:
        lines.append(f"| {key} | {stages[key]['steps']} | {stages[key]['elapsed_seconds']:.2f} |")
    ratios = {stage: stages['vit7b_'+stage]['elapsed_seconds']/stages['hplus_'+stage]['elapsed_seconds']
              for stage in ('real', 'render_adapt')}
    total = sum(stages['vit7b_'+s]['elapsed_seconds'] for s in ratios)/sum(stages['hplus_'+s]['elapsed_seconds'] for s in ratios)
    lines += ['', f"7B/H+ 阶段耗时比：real **{ratios['real']:.3f}×**，render_adapt **{ratios['render_adapt']:.3f}×**，两阶段合计 **{total:.3f}×**。",
              ('耗时取各 `stage_receipt.elapsed_seconds`，包含训练函数调用（其内部加载、验证、保存）和阶段末端来源核验，不含 wrapper 起始预检；不是纯训练步吞吐或独占推理 FPS。'
              '此表只使用阶段时间，不混入真实照片阶段的精度诊断。'), '',
              '这是固定开发视角的工程容量对照，不证明学术创新、跨随机种子/桥梁泛化，也不构成对同学 peer Dev30 的胜利。区间直接读取既有结果，本工具没有重算评分、bootstrap 或采用门槛。', '',
              f"共享场：[checkpoint]({plan['inference_scene']['path']})，SHA `{plan['inference_scene']['sha256']}`。", '']
    for key, label in LABELS.items():
        source = receipt['result_sources'][key]
        teacher = source['teacher_ensemble']
        lines += [(f"- {label}：[教师检查点]({teacher['teacher_checkpoint']})，SHA `{teacher['teacher_checkpoint_sha256']}`；"
                  f"[正式指标]({source['path']})，SHA `{source['sha256']}`。")]
    lines += ['', (f"来源：[完整执行回执]({Path(receipt_path).resolve()})，SHA `{sha(receipt_path)}`；"
              f"[锁定计划]({receipt['plan']})，SHA `{receipt['plan_sha256']}`。"),
              f"报告工具：`{Path(__file__).resolve()}`，SHA `{sha(__file__)}`。"]
    return '\n'.join(lines)+'\n'


def report(receipt_path, output):
    output = Path(output)
    require(not output.exists(), 'Refuse to overwrite report')
    payload = markdown(receipt_path, load_results(receipt_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        handle.write(payload)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(report(args.receipt, args.output))


if __name__ == '__main__':
    main()
