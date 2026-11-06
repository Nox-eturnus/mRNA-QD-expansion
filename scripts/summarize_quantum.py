"""Summarise run_quantum outputs: per (method, tag, shots), chain means of the
seed-averaged readout metrics.  Usage:
    python scripts/summarize_quantum.py results/qdev.jsonl [--by-kind]
"""
import argparse
import json
from collections import defaultdict
from statistics import mean

READOUTS = ('best', 'sqd', 'rep_best', 'rec_best', 'rec')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--by-kind', action='store_true')
    ap.add_argument('--shots', default='')
    args = ap.parse_args()
    rows = [json.loads(l) for p in args.paths for l in open(p) if l.strip()]
    if args.shots:
        keep = {int(s) for s in args.shots.split(',')}
        rows = [r for r in rows if r['shots'] in keep]
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        kinds = ['all'] + ([r['pk_kind']] if args.by_kind else [])
        for k in kinds:
            groups[(r['method'], r['tag'], r['shots'], k)][r['id']].append(r)
    print(f'{"method":7s} {"tag":22s} {"shots":>6s} {"kind":11s} {"n":>3s} {"ref":>6s} '
          + ' '.join(f'{x+"_mcc":>12s}' for x in READOUTS) + f' {"best_gap":>8s}'
          + f' {"best_hit":>8s} {"rec_hit":>8s} {"feas":>6s} {"uniq":>7s} {"p_opt":>8s}')
    for key in sorted(groups):
        per_chain = groups[key]

        def cm(field):
            vals = [mean(float(x[field]) for x in rs) for rs in per_chain.values()
                    if all(field in x and x[field] is not None for x in rs)]
            return mean(vals) if vals else float('nan')
        popt = cm('p_opt') if any('p_opt' in rs[0] for rs in per_chain.values()) else cm('p_opt_mean')
        method, tag, shots, kind = key
        print(f'{method:7s} {tag:22s} {shots:6d} {kind:11s} {len(per_chain):3d} {cm("ref_cpsat_mcc"):6.3f} '
              + ' '.join(f'{cm(x+"_mcc"):12.3f}' for x in READOUTS)
              + f' {cm("best_gap"):8.2f}'
              + f' {cm("best_hit"):8.2f} {cm("rec_hit"):8.2f} {cm("feasible_frac"):6.3f}'
              f' {cm("n_unique"):7.0f} {popt:8.2e}')


if __name__ == '__main__':
    main()
