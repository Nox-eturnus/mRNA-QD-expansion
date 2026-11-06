"""Summarise run_models outputs: mean MCC per model with bootstrap 95 % CI,
split by pk_kind, restricted to chains every listed model solved.

    python scripts/summarize_models.py results/calib.jsonl [--models M0,M1,M2]
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mrnasqd import DATA_DIR  # noqa: E402


def boot_ci(vals, n=2000, seed=0):
    rng = random.Random(seed)
    k = len(vals)
    means = sorted(sum(rng.choice(vals) for _ in range(k)) / k for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--models', default='')
    ap.add_argument('--optimal-only', action='store_true')
    ap.add_argument('--extra', default='', help='comma list of extra numeric fields')
    ap.add_argument('--paired-ref', default='',
                    help='model (or ViennaRNA MFE / pKiss) for paired MCC differences')
    args = ap.parse_args()

    kinds = {r['id']: r['pk_kind'] for r in
             csv.DictReader(open(os.path.join(DATA_DIR, 'pdb_fr3d_folds.csv')))}
    census = {r['id']: r for r in
              csv.DictReader(open(os.path.join(DATA_DIR, 'census.csv')))}
    vienna = {i: float(r['vienna_mfe_mcc']) for i, r in census.items()}
    refs = {'ViennaRNA MFE': vienna,
            'ceiling (trunc)': {i: float(r['ceil_mcc_truncations_m2'])
                                for i, r in census.items()}}
    cb = os.path.join(os.path.dirname(DATA_DIR), 'results', 'classical_baselines.jsonl')
    if os.path.exists(cb):
        refs['pKiss'] = {d['id']: d['pkiss_mcc'] for d in map(json.loads, open(cb))
                         if 'pkiss_mcc' in d}
    rows = [json.loads(l) for p in args.paths for l in open(p) if l.strip()]
    by = defaultdict(dict)
    for d in rows:
        by[d['model']][d['id']] = d
    models = args.models.split(',') if args.models else sorted(by)
    common = set.intersection(*(set(by[m]) for m in models))
    if args.optimal_only:
        common = {i for i in common if all(by[m][i]['optimal'] for m in models)}
    ids = sorted(common)
    print(f'{len(ids)} chains common to {models}')
    extras = [e for e in args.extra.split(',') if e]
    groups = [('all', ids)] + [(k, [i for i in ids if kinds[i] == k])
                               for k in ('nested', 'helical_pk', 'tertiary_pk')]
    for gname, gids in groups:
        if not gids:
            continue
        print(f'\n[{gname}] n={len(gids)}')
        for rname, rmap in refs.items():
            v = [rmap[i] for i in gids if i in rmap]
            if len(v) == len(gids):
                lo, hi = boot_ci(v)
                print(f'  {rname:14s} MCC {sum(v)/len(v):.3f} [{lo:.3f},{hi:.3f}]')
        for m in models:
            v = [by[m][i]['mcc'] for i in gids]
            lo, hi = boot_ci(v)
            opt = sum(by[m][i]['optimal'] for i in gids) / len(gids)
            q = sorted(by[m][i]['n_qubits'] for i in gids)[len(gids) // 2]
            sens = sum(by[m][i]['sensitivity'] for i in gids) / len(gids)
            ppv = sum(by[m][i]['ppv'] for i in gids) / len(gids)
            line = (f'  {m:14s} MCC {sum(v)/len(v):.3f} [{lo:.3f},{hi:.3f}]  '
                    f'sens {sens:.3f} ppv {ppv:.3f}  qubits~{q}  optimal {opt:.0%}')
            if args.paired_ref and args.paired_ref != m:
                ref = (by[args.paired_ref] if args.paired_ref in by else None)
                rv = ([ref[i]['mcc'] for i in gids] if ref is not None
                      else [refs[args.paired_ref][i] for i in gids])
                d = [a - b for a, b in zip(v, rv)]
                dlo, dhi = boot_ci(d)
                line += f'  d(vs {args.paired_ref}) {sum(d)/len(d):+.3f} [{dlo:+.3f},{dhi:+.3f}]'
            for e in extras:
                ev = [by[m][i][e] for i in gids if e in by[m][i]]
                if ev:
                    line += f'  {e} {sum(ev)/len(ev):.3f}'
            print(line)


if __name__ == '__main__':
    main()
