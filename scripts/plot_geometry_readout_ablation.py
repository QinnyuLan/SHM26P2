"""Plot stored paired comparisons; no model inference or metric recomputation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RUN = Path('/mnt/data/SHM2026/runs/continuous_geometry_rgb_only_evaluation_v1')


def main():
    rows = []
    arms = ('joint', 'pcgrad', 'trust')
    titles = ('Native 3D classes', 'Frozen scene head', 'Scene + DINOv3 H+')
    labels = ('Joint RGB + raw CE', 'PCGrad', 'RGB acceptance')
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.8), sharey=True)
    for ax, kind, title in zip(axes, ('raw', 'scene', 'joint'), titles, strict=True):
        for y, arm in enumerate(arms):
            path = RUN/'evaluation'/f'paired_{arm}_minus_rgb_{kind}.json'
            record = json.loads(path.read_text())['metrics']['miou_all']
            value = 100*record['difference']
            low, high = np.asarray(record['paired_view_bootstrap_95_interval'])*100
            ax.errorbar(value, y, xerr=[[value-low], [high-value]], fmt='o',
                        color='#167a70' if value > 0 else '#ad4c47', capsize=4, markersize=6)
            rows.append({'candidate': arm, 'reference': 'rgb', 'readout': kind,
                         'difference_pp': value, 'lower95_pp': low, 'upper95_pp': high,
                         'source': str(path)})
        ax.axvline(0, color='#777777', linewidth=.8, linestyle='--')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('mIoU change vs RGB-only (pp)')
        ax.grid(axis='x', alpha=.16)
        ax.set_yticks(range(3), labels)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylim(2.5, -.5)
    fig.suptitle('Improved native semantics did not improve the deployed readout', fontsize=13)
    fig.text(.5, .015, 'Same 41 labeled development views; 5,000 paired-view bootstrap samples; fixed endpoints.',
             ha='center', fontsize=9, color='#555555')
    fig.tight_layout(rect=(0, .05, 1, .92))
    output = RUN/'figures'; output.mkdir(exist_ok=True)
    for suffix in ('png', 'svg'):
        fig.savefig(output/f'readout_ablation.{suffix}', dpi=180, facecolor='white')
    with (output/'readout_ablation.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(output/'readout_ablation.png')


if __name__ == '__main__':
    main()
