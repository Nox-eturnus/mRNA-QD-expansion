"""Figures for the short project report (report/project_report.docx).

    fig_workflow.png    methodology workflow
    fig_noise_mcc.png   MCC of SKQD under gate noise (Pauli trajectories, 10000 shots)

    python scripts/report_figures.py
"""

import json
import os
from collections import defaultdict
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
FIG = os.path.join(ROOT, 'report', 'figures')

C = ['#2a78d6', '#eb6834', '#1baf7a']
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'savefig.dpi': 220, 'savefig.bbox': 'tight'})


def box(ax, x, y, w, h, num, title, body, edge=INK2, lw=1.0, fill='white'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12',
                                fc=fill, ec=edge, lw=lw))
    head = f'{num}  {title}' if num else title
    ax.text(x + 0.18, y + h - 0.2, head, ha='left', va='top', fontsize=9.5, weight='bold', color=INK)
    ax.text(x + 0.18, y + h - 0.58, body, ha='left', va='top', fontsize=8, color=INK2,
            linespacing=1.35)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color=INK2, lw=1.1, shrinkA=0, shrinkB=0))


def workflow():
    fig, ax = plt.subplots(figsize=(7.2, 9.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.95, 13.2)
    ax.axis('off')
    W, X = 9.4, 0.3
    rows = [
        (11.75, 1.3, '1', 'Experimental structures',
         '792 RNA chains from the PDB (BGSU non-redundant list), base pairs annotated\n'
         'by FR3D. Chains are split by RNA family: 317 for calibration, 475 for testing.'),
        (9.95, 1.3, '2', 'Stem candidates (mapping)',
         'Every helix the sequence can form is listed, together with its shortened\n'
         'versions. Each candidate stem becomes one binary variable, later one qubit.'),
        (8.15, 1.3, '3', 'Energy model and Hamiltonian',
         'Stacking energy for each stem, a loop correction for stems nested directly\n'
         'inside each other, and a penalty for stems that clash. The 24 best stems are kept.'),
    ]
    for y, h, n, t, b in rows:
        box(ax, X, y, W, h, n, t, b)
    for (y0, h0, *_), (y1, h1, *_) in zip(rows, rows[1:]):
        arrow(ax, 5, y0, 5, y1 + h1)

    # step 4: three samplers side by side
    y4, h4 = 5.55, 1.95
    w3, gap = 2.95, 0.275
    xs = [X, X + w3 + gap, X + 2 * (w3 + gap)]
    box(ax, xs[0], y4, w3, h4, '4a', 'SKQD',
        'Evolve a simple start\nstate under the\nHamiltonian; measure\nthe Krylov states.',
        edge=C[0], lw=1.8)
    box(ax, xs[1], y4, w3, h4, '4b', 'SQD',
        'Measure a linear-ramp\nQAOA circuit built\nfrom the same\nHamiltonian.',
        edge=C[0], lw=1.8)
    box(ax, xs[2], y4, w3, h4, '4c', 'Baselines',
        'Exact optimum (CP-SAT),\nVQE as in the original\nproject, simulated\nannealing, random.')
    for x in xs:
        arrow(ax, 5, 8.15, x + w3 / 2, y4 + h4)

    y5, h5 = 3.2, 1.75
    box(ax, X, y5, xs[1] + w3 - X, h5, '5', 'Diagonalise in the sampled subspace',
        'The driven Hamiltonian is restricted to the sampled\nbitstrings. Its ground state is found, and its\nlargest component is the predicted structure.',
        edge=C[0], lw=1.8)
    for x in xs[:2]:
        arrow(ax, x + w3 / 2, y4, x + w3 / 2, y5 + h5)

    y6, h6 = 1.1, 1.5
    box(ax, X, y6, W, h6, '6', 'Score against experiment',
        'Predicted base pairs are compared with the FR3D pairs: sensitivity, PPV and MCC.\n'
        'The SKQD circuits are also run with gate noise, and their gate counts are found\n'
        'after compiling for an all-to-all and an IBM heavy-hex device.')
    arrow(ax, 5 - 1.6, y5, 5 - 1.6, y6 + h6)
    arrow(ax, xs[2] + w3 / 2, y4, xs[2] + w3 / 2, y6 + h6)
    fig.savefig(os.path.join(FIG, 'fig_workflow.png'), facecolor='white')
    plt.close(fig)


def noise():
    rows = [json.loads(l) for l in open(os.path.join(RES, 'traj.jsonl')) if l.strip()]
    g = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r['shots'] == 10000:
            g[r['tag']][r['id']].append(r)
    ids = sorted(g['traj_x'])
    settings = [('', 'Noiseless'), ('t_a2a_1e-3', 'All-to-all\n0.1 % error'),
                ('t_a2a_3e-3', 'All-to-all\n0.3 % error'), ('t_hh_1e-3', 'Heavy-hex\n0.1 % error'),
                ('t_hh_3e-3', 'Heavy-hex\n0.3 % error')]
    series = [('SKQD, transverse drive', 'traj_x'), ('SKQD, constrained drive', 'traj_cx')]
    vals, hits = {}, {}
    for name, pre in series:
        for key, lab in settings:
            gg = g.get(pre + (f'|{key}' if key else ''))
            if gg and all(i in gg for i in ids):
                vals[(name, lab)] = mean(mean(x['best_mcc'] for x in gg[i]) for i in ids)
                hits[(name, lab)] = mean(mean(x['best_hit'] for x in gg[i]) for i in ids)

    plt.rcParams.update({'axes.facecolor': SURF, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                         'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
                         'grid.linewidth': 0.6, 'legend.frameon': False})
    fig, ax = plt.subplots(figsize=(8, 4.2))
    w = 0.36
    for k, (name, _) in enumerate(series):
        for j, (_, lab) in enumerate(settings):
            x = j + (k - 0.5) * w
            v = vals.get((name, lab))
            if v is None:
                ax.text(x, 0.015, 'not run', ha='center', va='bottom', fontsize=7, color=INK2,
                        rotation=90)
                continue
            ax.bar(x, v, width=w - 0.03, color=C[k], zorder=2, edgecolor=SURF, lw=1,
                   label=name if j == 0 else None)
            ax.text(x, v + 0.012, f'{v:.2f}', ha='center', va='bottom', fontsize=7.5, color=INK)
    ax.set_xticks(range(len(settings)))
    ax.set_xticklabels([s[1] for s in settings], fontsize=8.5)
    ax.set_ylim(0, 0.88)
    ax.set_ylabel('MCC')
    ax.grid(axis='x', visible=False)
    ax.legend(loc='upper right', fontsize=8.5)
    ax.set_title(f'SKQD under two-qubit gate noise ({len(ids)} chains, 24 qubits, 10000 shots)',
                 loc='left', color=INK, fontsize=10.5)
    fig.savefig(os.path.join(FIG, 'fig_noise_mcc.png'))
    plt.close(fig)
    for k in vals:
        print(k[0], '|', k[1].replace(chr(10), ' '), f'MCC {vals[k]:.3f}  hit {hits[k]:.2f}')


if __name__ == '__main__':
    os.makedirs(FIG, exist_ok=True)
    workflow()
    noise()
