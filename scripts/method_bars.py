"""Method comparison bar figure in the style of the original WISER-Moderna
plot (plots/plot3_structural_accuracy.png): sensitivity, PPV and MCC per method,
value printed on every bar.

All methods are evaluated on the same 25 Q-set chains (every 4th chain; the set
on which VQE was run), with the same pruned 24-stem M3 Hamiltonian, against the
experimental FR3D structures.  Readout per method, as each method is defined:
  Exact (CP-SAT)   the exact optimum of the model (enumeration = CP-SAT optimum)
  SA               lowest-energy of the SA samples
  SKQD, SQD        sample, then diagonalise H(G) in the sampled subspace
                   (qiskit-addon-sqd style); readout = largest-weight bitstring
  VQE              baseline readout (most probable bitstring after COBYLA) and,
                   separately, the lowest-energy of its samples
  Random           lowest-energy of uniformly random bitstrings (floor)
Per chain, values are averaged over seeds; bars are means over chains.

    python scripts/method_bars.py
"""

import json
import os
from collections import defaultdict
from statistics import mean

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results', 'method_comparison.jsonl')
FIG = os.path.join(ROOT, 'report', 'figures')
OUT_MD = os.path.join(ROOT, 'report', 'method_comparison.md')

C = ['#2a78d6', '#eb6834', '#1baf7a']           # validated palette, slots 1-3
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'

# (label, method, tag, readout prefix)
METHODS = [
    ('Exact\n(CP-SAT)', 'skqd', 'g0.5_dt0.3_r10', 'opt'),
    ('SA\n5 sweeps', 'sa', 'sweeps5', 'best'),
    ('SKQD\nconstrained\ndrive', 'skqd', 'cx_g0.5_dt0.3_r10', 'sqd'),
    ('SKQD\ntransverse\ndrive', 'skqd', 'g0.5_dt0.3_r10', 'sqd'),
    ('SA\n1 sweep', 'sa', 'sweeps1', 'best'),
    ('VQE\nbest\nsample', 'vqe', 'vqe_reps1', 'best'),
    ('SQD\nLR-QAOA\nsampler', 'lrqaoa', 'p4_dg2.0_db0.4', 'sqd'),
    ('VQE\nbaseline\nreadout', 'vqe', 'vqe_reps1', 'vqe_argmax'),
    ('Random\nbitstrings', 'random', 'uniform', 'best'),
]


def collect(rows, shots):
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r['shots'] == shots:
            by[(r['method'], r['tag'])][r['id']].append(r)
    ids = sorted(set.intersection(*(set(by[(m, t)]) for _, m, t, _ in METHODS)))
    table = []
    for lab, m, t, pre in METHODS:
        g = by[(m, t)]
        vals = {}
        for metric in ('sens', 'ppv', 'mcc'):
            vals[metric] = mean(mean(x[f'{pre}_{metric}'] for x in g[i]) for i in ids)
        if pre not in ('opt', 'vqe_argmax'):
            vals['hit'] = mean(mean(x[f'{pre}_hit'] for x in g[i]) for i in ids)
        elif pre == 'vqe_argmax':
            vals['hit'] = mean(mean(x['vqe_argmax_hit'] for x in g[i]) for i in ids)
        else:
            vals['hit'] = 1.0
        table.append((lab, vals))
    return ids, table


def draw(table, n_chains, shots, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.facecolor': SURF,
        'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2,
        'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linestyle': '-',
        'grid.linewidth': 0.6, 'legend.frameon': False, 'savefig.dpi': 200,
        'savefig.bbox': 'tight'})
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    w = 0.26
    xs = range(len(table))
    for k, (metric, name) in enumerate((('sens', 'Sensitivity'), ('ppv', 'PPV'),
                                        ('mcc', 'MCC'))):
        pos = [x + (k - 1) * w for x in xs]
        vals = [v[metric] for _, v in table]
        ax.bar(pos, vals, width=w - 0.02, color=C[k], label=name, zorder=2,
               edgecolor=SURF, linewidth=1)
        for p, v in zip(pos, vals):
            ax.text(p, v + 0.012, f'{v:.2f}', ha='center', va='bottom', fontsize=7,
                    color=INK)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([lab for lab, _ in table], fontsize=8.5)
    for tl, (lab, _) in zip(ax.get_xticklabels(), table):
        if lab.startswith(('SQD', 'SKQD')):      # the methods this study adds
            tl.set_fontweight('bold'); tl.set_color(INK)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Score')
    ax.grid(axis='x', visible=False)
    ax.legend(loc='upper right', ncol=3)
    ax.set_title('Structural accuracy per method', loc='left', color=INK,
                 fontsize=12, pad=22)
    ax.text(0, 1.02, f'Mean over {n_chains} held-out PDB chains, 24-qubit stem QUBO, '
            f'{shots} shots per sampling method. Bold: quantum methods added here',
            transform=ax.transAxes, fontsize=9, color=INK2)
    fig.savefig(path)
    plt.close(fig)


def main():
    rows = [json.loads(l) for l in open(RES) if l.strip()]
    os.makedirs(FIG, exist_ok=True)
    md = ['# Method comparison (generated by scripts/method_bars.py)', '',
          'Same chains, same 24-stem M3 Hamiltonian, scored against the FR3D structures. '
          'Hit = fraction of chains where the readout is the exact optimum of the model.', '']
    for shots in (100, 1000, 10000):
        ids, table = collect(rows, shots)
        table.sort(key=lambda t: (not t[0].startswith('Exact'), -t[1]['mcc']))
        name = 'fig6_method_comparison.png' if shots == 1000 else \
            f'fig6_method_comparison_{shots}shots.png'
        draw(table, len(ids), shots, os.path.join(FIG, name))
        md += [f'## {shots} shots ({len(ids)} chains)', '',
               '| method | sensitivity | PPV | MCC | hit |', '|---|---|---|---|---|']
        md += [f"| {lab.replace(chr(10), ' ')} | {v['sens']:.3f} | {v['ppv']:.3f} | {v['mcc']:.3f} | {v['hit']:.2f} |"
               for lab, v in table]
        md.append('')
    with open(OUT_MD, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
