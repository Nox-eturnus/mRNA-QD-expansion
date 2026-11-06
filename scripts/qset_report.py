"""Report on the Q-set: per (method, shots), chain means of the
seed-averaged readouts with bootstrap 95 % CIs, paired differences against a
reference method, and per pseudoknot class.

    python scripts/qset_report.py results/qset.jsonl \
        --ref "skqd g0.5_dt0.3_r10" > results/qset_summary.txt
"""
import argparse
import json
import random
from collections import defaultdict
from statistics import mean, median


def boot(vals, n=2000, seed=0):
    rng = random.Random(seed)
    k = len(vals)
    m = sorted(sum(rng.choice(vals) for _ in range(k)) / k for _ in range(n))
    return m[int(0.025 * n)], m[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--ref', default='skqd g0.5_dt0.3_r10')
    ap.add_argument('--readout', default='best')
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.path) if l.strip()]
    per = defaultdict(lambda: defaultdict(list))        # (method tag, shots) -> id -> rows
    kind = {}
    ref_mcc = {}
    for r in rows:
        per[(f"{r['method']} {r['tag']}", r['shots'])][r['id']].append(r)
        kind[r['id']] = r['pk_kind']
        ref_mcc[r['id']] = r['ref_cpsat_mcc']

    def chain_mean(rs, f):
        v = [x[f] for x in rs if x.get(f) is not None]
        return mean(float(t) for t in v) if v else None

    shots_all = sorted({s for _, s in per})
    methods = sorted({m for m, _ in per})
    ids = sorted(ref_mcc)
    print(f'Q-set: {len(ids)} chains; CP-SAT optimum of the pruned model: '
          f'MCC {mean(ref_mcc.values()):.3f}')
    print(f'Readout for paired differences: {args.readout}; reference method: {args.ref}\n')
    for s in shots_all:
        print(f'=== shots = {s}')
        print(f'{"method":26s} {"best MCC [CI]":>22s} {"sqd":>6s} {"repair":>7s} {"rec":>6s}'
              f' {"hit":>5s} {"rec_hit":>7s} {"feas":>6s} {"p_opt(med)":>10s}  d({args.readout}) vs ref [CI]')
        refrows = per.get((args.ref, s), {})
        for m in methods:
            g = per.get((m, s))
            if not g:
                continue
            cid = [i for i in ids if i in g]
            v = [chain_mean(g[i], 'best_mcc') for i in cid]
            lo, hi = boot(v)
            line = (f'{m:26s} {mean(v):6.3f} [{lo:.3f},{hi:.3f}]'
                    f' {mean(chain_mean(g[i], "sqd_mcc") for i in cid):6.3f}'
                    f' {mean(chain_mean(g[i], "rep_best_mcc") for i in cid):7.3f}'
                    f' {mean(chain_mean(g[i], "rec_mcc") for i in cid):6.3f}'
                    f' {mean(chain_mean(g[i], "best_hit") for i in cid):5.2f}'
                    f' {mean(chain_mean(g[i], "rec_hit") for i in cid):7.2f}'
                    f' {mean(chain_mean(g[i], "feasible_frac") for i in cid):6.3f}')
            po = [chain_mean(g[i], 'p_opt') or chain_mean(g[i], 'p_opt_mean') for i in cid]
            po = [p for p in po if p is not None]
            line += f' {median(po):10.2e}' if po else f' {"-":>10s}'
            if m != args.ref and refrows:
                common = [i for i in cid if i in refrows]
                d = [chain_mean(refrows[i], f'{args.readout}_mcc') - chain_mean(g[i], f'{args.readout}_mcc')
                     for i in common]
                dl, dh = boot(d)
                line += f'  ref-this {mean(d):+.3f} [{dl:+.3f},{dh:+.3f}]'
            print(line)
        print()
    s = 1000
    print(f'=== by pseudoknot class, shots = {s}, best-sample MCC (CP-SAT optimum in brackets)')
    for k in ('nested', 'helical_pk', 'tertiary_pk'):
        cid = [i for i in ids if kind[i] == k]
        line = f'{k:12s} n={len(cid):3d} [{mean(ref_mcc[i] for i in cid):.3f}]'
        for m in methods:
            g = per.get((m, s))
            if g:
                line += f'  {m.split()[0]}:{m.split()[1][:10]}={mean(chain_mean(g[i], "best_mcc") for i in cid if i in g):.3f}'
        print(line)


if __name__ == '__main__':
    main()
