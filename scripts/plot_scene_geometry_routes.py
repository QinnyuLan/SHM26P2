"""Plot the two fixed route endpoints from saved audited statistics."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

RUN = Path('/mnt/data/SHM2026/runs/scene_geometry_routes_evaluation_v1')


def main():
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.2), sharey=True)
    for ax, kind, title in zip(axes, ('raw', 'scene', 'joint'),
                               ('Native 3D classes', 'Frozen scene head', 'Scene + DINOv3 H+'), strict=True):
        for y, arm in enumerate(('full', 'prior_only')):
            source = RUN/'evaluation'/f'paired_{arm}_minus_baseline_{kind}.json'
            metric = json.loads(source.read_text())['metrics']['miou_all']
            value = metric['difference']*100
            low, high = [v*100 for v in metric['paired_view_bootstrap_95_interval']]
            ax.errorbar(value, y, xerr=[[value-low], [high-value]], fmt='o', capsize=4,
                        color='#167a70' if low > 0 else '#ad4c47' if high < 0 else '#5a6b83')
            rows.append({'arm': arm, 'readout': kind, 'difference_pp': value,
                         'lower95_pp': low, 'upper95_pp': high, 'source': str(source)})
        ax.set_yticks([0, 1], ['Full gradient', 'Prior path only'])
        ax.set_ylim(1.5, -.5)
        ax.axvline(0, color='#777', linestyle='--', linewidth=.8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('mIoU change vs unchanged geometry (pp)')
        ax.grid(axis='x', alpha=.15)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Scene-head improvement remains uncertain after teacher fusion', fontsize=13)
    fig.text(.5, .02, '41 labeled development views; 5,000 paired-view bootstrap samples; both fixed endpoints.',
             ha='center', fontsize=9, color='#555')
    fig.tight_layout(rect=(0, .06, 1, .93))
    out = RUN/'figures'; out.mkdir(exist_ok=True)
    for suffix in ('png', 'svg'):
        fig.savefig(out/f'geometry_routes.{suffix}', dpi=180, facecolor='white')
    with (out/'geometry_routes.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(out/'geometry_routes.png')


if __name__ == '__main__':
    main()
