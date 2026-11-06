"""Follow-up tables: warm start, sensitivity, noise, VQE, scaling.

    python scripts/followup_report.py > results/followup_summary.txt
Reads results/qset.jsonl, vqe.jsonl, traj.jsonl and scale.jsonl.
Every number is a mean over chains of the seed-averaged value; chain sets
are matched within each table (only chains present for every row).
"""
import json
import os
import random
import sys
from collections import defaultdict
from statistics import mean

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name):
    p = os.path.join(ROOT, 'results', name)
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def boot(vals, n=2000, seed=0):
    rng = random.Random(seed)
    k = len(vals)
    m = sorted(sum(rng.choice(vals) for _ in range(k)) / k for _ in range(n))
    return m[int(0.025 * n)], m[int(0.975 * n)]


def index(rows):
    """(method, tag, shots) -> id -> [rows].  Scaling runs carry their
    budget in the tag (s26_..., s28_...), so the qubit count is not part of
    the key (Q-set chains have 20-24 qubits at the 24-stem budget)."""
    d = defaultdict(lambda: defaultdict(list))
    for r in rows:
        d[(r['method'], r['tag'], r['shots'])][r['id']].append(r)
    return d


def cm(rs, f):
    v = [x[f] for x in rs if x.get(f) is not None]
    return mean(float(t) for t in v) if v else float('nan')


def opt(rs):
    r = rs[0]
    return r.get('opt_mcc') if r.get('opt_mcc') is not None else r.get('ref_cpsat_mcc')


def table(title, idx, keys, shots, fields=('best_mcc', 'best_hit', 'feasible_frac'),
          ids=None, paired=None):
    groups = []
    for (m, t) in keys:
        if (m, t, shots) in idx:
            groups.append(((m, t), idx[(m, t, shots)]))
    if not groups:
        return
    common = set.intersection(*(set(g) for _, g in groups))
    if ids is not None:
        common &= set(ids)
    common = sorted(common)
    if not common:
        return
    print(f'\n=== {title}  (shots = {shots}, n = {len(common)} chains)')
    print(f'  exact optimum of the model: MCC {mean(opt(groups[0][1][i]) for i in common):.3f}')
    head = f'  {"method / tag":42s}' + ''.join(f'{f:>14s}' for f in fields)
    if paired:
        head += f'   d(best_mcc) vs {paired[1][:24]}'
    print(head)
    ref = dict(groups).get(paired) if paired else None
    for key, g in groups:
        line = f'  {key[0] + " " + key[1]:42s}'
        for f in fields:
            line += f'{mean(cm(g[i], f) for i in common):14.3f}'
        if ref is not None and key != paired:
            d = [cm(g[i], 'best_mcc') - cm(ref[i], 'best_mcc') for i in common]
            lo, hi = boot(d)
            line += f'   {mean(d):+.3f} [{lo:+.3f},{hi:+.3f}]'
        print(line)


def main():
    q = index(load('qset.jsonl'))
    X = 'g0.5_dt0.3_r10'
    for s in (100, 1000, 10000):
        table('Drives and warm start (24 qubits)', q,
              [('skqd', X), ('skqd', 'cx_' + X), ('skqd', 'warm_' + X),
               ('skqd', 'warm_cx_' + X), ('sa', 'sweeps1'), ('sa', 'warm_sweeps1'),
               ('sa', 'sweeps5'), ('sa', 'warm_sweeps5')], s,
              fields=('best_mcc', 'sqd_mcc', 'best_hit', 'feasible_frac', 'warm_mcc'),
              paired=('skqd', X))
    table('Sensitivity of transverse SKQD (24 qubits)', q,
          [('skqd', X), ('skqd', 'sens_r5'), ('skqd', 'sens_r20'),
           ('skqd', 'sens_gd0.01'), ('skqd', 'sens_gd0.5'), ('skqd', 'sens_gd1.0')], 1000,
          fields=('best_mcc', 'sqd_mcc', 'best_hit', 'sqd_w'), paired=('skqd', X))
    for s in (1000, 10000):
        keys = [('skqd', X + '_wn')] + [('skqd', f'{X}_wn|{lab}') for lab in
                                        ('w_a2a_1e-3', 'w_a2a_3e-3', 'w_hh_1e-3', 'w_hh_3e-3')]
        keys += [('skqd', 'cx_' + X)] + [('skqd', f'cx_{X}|{lab}') for lab in
                                         ('w_a2a_1e-3', 'w_a2a_3e-3', 'w_hh_1e-3', 'w_hh_3e-3')]
        keys += [('lrqaoa', 'p4_dg2.0_db0.4_wn'), ('lrqaoa', 'p4_dg2.0_db0.4_wn|w_a2a_1e-3'),
                 ('lrqaoa', 'p4_dg2.0_db0.4_wn|w_hh_1e-3'), ('sa', 'sweeps1'), ('sa', 'sweeps5')]
        table('White-noise model (pessimistic bound), 24 qubits', q, keys, s,
              fields=('best_mcc', 'rep_best_mcc', 'rec_mcc', 'best_hit', 'feasible_frac'))
    t = index(load('traj.jsonl'))
    for s in (1000, 10000):
        keys = [('skqd', 'traj_x')] + [('skqd', f'traj_x|{lab}') for lab in
                                       ('t_a2a_1e-3', 't_a2a_3e-3', 't_hh_1e-3', 't_hh_3e-3')]
        keys += [('skqd', 'traj_cx')] + [('skqd', f'traj_cx|{lab}') for lab in
                                         ('t_a2a_1e-3', 't_hh_1e-3')]
        table('Pauli-trajectory noise model, 24 qubits', t, keys, s,
              fields=('best_mcc', 'rep_best_mcc', 'rec_mcc', 'best_hit', 'feasible_frac'))
    v = load('vqe.jsonl')
    if v:
        vi = index(v)
        ids = {r['id'] for r in v}
        merged = defaultdict(lambda: defaultdict(list))
        for k, g in list(vi.items()) + list(q.items()):
            for i, rs in g.items():
                if i in ids:
                    merged[k][i] = rs
        for s in (100, 1000, 10000):
            table('Baseline VQE vs SKQD and SA (same chains)', merged,
                  [('vqe', 'vqe_reps1'), ('vqe', 'vqe_reps1|w_a2a_1e-3'),
                   ('vqe', 'vqe_reps1|w_hh_3e-3'), ('skqd', X), ('skqd', 'cx_' + X),
                   ('sa', 'sweeps1'), ('sa', 'sweeps5')], s,
                  fields=('best_mcc', 'best_hit', 'feasible_frac', 'vqe_argmax_mcc', 'p_opt'),
                  paired=('skqd', X))
        vv = [r for r in v if r['tag'] == 'vqe_reps1' and r['shots'] == 1000]
        if vv:
            print(f'\n  VQE: mean final energy {mean(r["vqe_energy"] for r in vv):.2f} kcal/mol, '
                  f'mean evaluations {mean(r["vqe_evals"] for r in vv):.0f}, '
                  f'argmax readout feasible {mean(r["vqe_argmax_feasible"] for r in vv):.2f}, '
                  f'hits optimum {mean(r["vqe_argmax_hit"] for r in vv):.2f}')
    sc = load('scale.jsonl')
    if sc:
        si = index(sc)
        for b in (26, 28):
            ids = {r['id'] for r in sc if r['n_qubits'] == b}
            for s in (1000, 10000):
                rows = []
                for m, tg in (('skqd', X), ('skqd', 'cx_' + X), ('lrqaoa', 'p4_dg2.0_db0.4'),
                              ('random', 'uniform'), ('sa', 'sweeps1'), ('sa', 'sweeps5'),
                              ('sa', 'sweeps50')):
                    rows.append((m, f's{b}_{tg}'))
                table(f'Scaling: {b} qubits', si, rows, s,
                      fields=('best_mcc', 'rec_mcc', 'best_hit', 'feasible_frac', 'p_opt_mean'),
                      ids=ids)
                table(f'Same chains at 24 qubits (for comparison with {b})', q,
                      [('skqd', X), ('skqd', 'cx_' + X), ('lrqaoa', 'p4_dg2.0_db0.4'),
                       ('random', 'uniform'), ('sa', 'sweeps1'), ('sa', 'sweeps5'),
                       ('sa', 'sweeps50')], s,
                      fields=('best_mcc', 'rec_mcc', 'best_hit', 'feasible_frac', 'p_opt_mean'),
                      ids=ids)


if __name__ == '__main__':
    sys.exit(main())
