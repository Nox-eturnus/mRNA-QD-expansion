"""Re-rank stored CP-SAT top-K solutions by full-structure Turner energy.

Input: run_models output made with --topk K (solutions in field 'topk').
For each chain the QUBO is rebuilt from the stored spec (deterministic), and
each solution gets E(beta) = e_nested + stack_rest + beta * n_rest
(``qubo_v2.rerank_components``).  The lowest-E solution is the prediction.

    python scripts/rerank.py results/topk.jsonl --model M3 \
        --betas -2,-1,0,1,2,3,4,6 --out results/rerank.jsonl
"""

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mrnasqd import DATA_DIR  # noqa: E402


def work(d, rec, betas):
    from mrnasqd import baseline_path
    from mrnasqd.qubo_v2 import build_stem_qubo, rerank_components
    from mrnasqd.metrics import structure_metrics
    baseline_path()
    import RNA
    q = build_stem_qubo(rec['sequence'], **d['spec'])
    fc = RNA.fold_compound(q.sequence)
    truth = [tuple(p) for p in rec['pairs']]
    sols = []
    for s in d['topk']:
        x = [0] * q.n
        for i in s['sel']:
            x[i] = 1
        e0, srest, nrest = rerank_components(q, x, fc)
        mcc = structure_metrics(q.pairs(x), truth, rec['length'])['mcc']
        sols.append((e0, srest, nrest, mcc, s['energy']))
    out = {'id': d['id'], 'model': d['model'], 'k': len(sols),
           'qubo_mcc': sols[0][3], 'oracle_mcc': max(s[3] for s in sols),
           'all_optimal': all(s['optimal'] for s in d['topk'])}
    for b in betas:
        best = min(sols, key=lambda s: (s[0] + s[1] + b * s[2], s[4]))
        out[f'rr_{b}'] = best[3]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--model', required=True)
    ap.add_argument('--betas', default='-2,-1,0,1,2,3,4,6')
    ap.add_argument('--procs', type=int, default=16)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    betas = [float(b) for b in args.betas.split(',')]
    recs = {json.loads(l)['id']: json.loads(l)
            for l in open(os.path.join(DATA_DIR, 'pdb_fr3d_chains.jsonl'))}
    F = {r['id']: r for r in csv.DictReader(open(os.path.join(DATA_DIR, 'pdb_fr3d_folds.csv')))}
    rows = [json.loads(l) for l in open(args.path) if l.strip()]
    rows = [d for d in rows if d['model'] == args.model and d.get('topk')]
    with ProcessPoolExecutor(args.procs) as pool:
        res = list(pool.map(work, rows, [recs[d['id']] for d in rows],
                            [betas] * len(rows), chunksize=4))
    with open(args.out, 'w') as f:
        for r in res:
            r['fold'] = F[r['id']]['fold']
            r['pk_kind'] = F[r['id']]['pk_kind']
            f.write(json.dumps(r) + '\n')
    for fold in ('calibration', 'test'):
        for kind in ('all', 'nested', 'helical_pk', 'tertiary_pk'):
            S = [r for r in res if r['fold'] == fold and (kind == 'all' or r['pk_kind'] == kind)]
            if not S:
                continue
            line = (f'{fold:11s} {kind:11s} n={len(S):3d}  QUBO {sum(r["qubo_mcc"] for r in S)/len(S):.3f}'
                    f'  oracle@K {sum(r["oracle_mcc"] for r in S)/len(S):.3f}  rerank:')
            for b in betas:
                line += f' b{b:g}={sum(r[f"rr_{b}"] for r in S)/len(S):.3f}'
            print(line)


if __name__ == '__main__':
    main()
