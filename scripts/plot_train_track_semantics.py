"""Display frozen descriptive counts; no model/data selection or new statistics."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('run', type=Path)
    parser.add_argument('--folder', default='figures')
    args = parser.parse_args(); run = args.run
    path = run/'analysis.json'; data = json.loads(path.read_text())
    receipt = json.loads((run/'execution_receipt.json').read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if receipt['status'] != 'completed' or receipt['output_files']['analysis.json'] != digest:
        raise ValueError('Incomplete or changed analysis')
    dest = run/args.folder; dest.mkdir(exist_ok=False)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, (ax, heat) = plt.subplots(1, 2, figsize=(11.2, 4.7), gridspec_kw={'width_ratios': [1, 1.1]})
    colors = {'primary': '#176b8e', 'legacy': '#cc7c29'}
    for j, kind in enumerate(('primary', 'legacy')):
        means = np.array([data[kind][s]['point_equal_conflict'] for s in ('all', 'strict')])*100
        bounds = np.array([data[kind][s]['bootstrap']['point_equal_conflict_95_interval'] for s in ('all', 'strict')])*100
        x = np.arange(2)+(j-.5)*.3
        ax.errorbar(x, means, yerr=np.stack([means-bounds[:, 0], bounds[:, 1]-means]), fmt='o',
                    color=colors[kind], capsize=5, label='Original grid' if j == 0 else 'Legacy sensitivity')
        for xx, yy, stat in zip(x, means, ('all', 'strict')):
            ax.annotate(f'{yy:.2f}%\nn={data[kind][stat]["tracks_ge2"]:,}', (xx, yy),
                        xytext=(0, 10 if j == 0 else -32), textcoords='offset points', ha='center', fontsize=9)
    ax.axhline(1., ls=':', color='.5', label='Strict lower-CI threshold')
    ax.set(xticks=[0, 1], xticklabels=['All usable', 'Error ≤1 px\nBoundary >10 px'],
           ylabel='Track-equal hard-label conflict (%)', title='Same-track label conflict')
    ax.set_xlim(-.5, 1.5)
    ax.set_ylim(0, .8+100*max(data[k][s]['point_equal_conflict'] for k in colors for s in ('all', 'strict')))
    ax.legend(fontsize=8, loc='lower left')
    rows = ('le1', 'gt1_le2', 'gt2'); cols = ('le3', 'gt3_le10', 'gt10')
    stats = [[data['primary']['strata'][f'error_{r}__boundary_{c}'] for c in cols] for r in rows]
    values = np.array([[s['point_equal_conflict']*100 if s['point_equal_conflict'] is not None else np.nan
                        for s in row] for row in stats])
    chart = heat.imshow(values, cmap='YlOrBr', vmin=0, vmax=max(1., np.nanmax(values)))
    for i in range(3):
        for j in range(3):
            heat.text(j, i, f'{values[i,j]:.2f}%\nn={stats[i][j]["tracks_ge2"]:,}',
                      ha='center', va='center', color='white' if values[i,j] > .65*np.nanmax(values) else '#222')
    heat.set(xticks=range(3), xticklabels=['≤3', '(3, 10]', '>10'], yticks=range(3),
             yticklabels=['≤1', '(1, 2]', '>2'], xlabel='Own-class boundary distance (px)',
             ylabel='Native reprojection error (px)', title='Original-grid strata (descriptive)')
    fig.colorbar(chart, ax=heat, fraction=.046, pad=.04, label='Conflict (%)')
    fig.suptitle('TRAIN SfM tracks: diagnostic only, no rendering accuracy claim', fontsize=13, y=.98)
    fig.text(.5, .02, 'n = tracks with ≥2 eligible observations; 95% track bootstrap only for left panel.\n'
             'Saved TRAIN correspondences and shared calibration; neither a 3DGS bound nor held-out performance.',
             ha='center', fontsize=9, color='.3')
    fig.tight_layout(rect=(0, .1, 1, .95))
    for ext in ('png', 'svg'):
        fig.savefig(dest/f'track_semantic_conflict.{ext}', dpi=160)
    record = {'analysis_sha256': digest, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(dest.iterdir())}}
    (dest/'receipt.json').write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()
