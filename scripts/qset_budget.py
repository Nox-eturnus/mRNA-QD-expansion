"""Q-set selection: prune the calibrated model (M3 or M4, per --params) to simulator-sized qubit
budgets and record what survives.

For each chain (length <= --max-len) and budget K, stems are pruned to the
K most stable by stacking energy (``max_stems``).  Recorded per (chain, K):
qubits, expressible-ceiling MCC, CP-SAT optimum MCC under the pruned model
(the reference a quantum solver can at best reach), and the unpruned
CP-SAT optimum MCC.

    python scripts/qset_budget.py --params results/m3_params.json \
        --out results/qset_budget.jsonl
"""

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mrnasqd import DATA_DIR  # noqa: E402


def work(rec, fold, budgets, params, time_limit):
    from mrnasqd.qubo_v2 import build_stem_qubo, cpsat_solve
    from mrnasqd.encoding import expressible_coverage
    from mrnasqd.metrics import structure_metrics
    truth = [tuple(p) for p in rec['pairs']]
    n = rec['length']
    rows = []
    for K in budgets:
        q = build_stem_qubo(rec['sequence'], model='turner_direct',
                            mode='truncations', min_len=2, max_stems=K,
                            **params)
        cov = expressible_coverage(q.stems, truth)
        tset = set(truth)
        covered = sorted({p for s in q.stems if set(s.pairs) <= tset
                          for p in s.pairs})
        ceil = structure_metrics(covered, truth, n)['mcc']
        r = cpsat_solve(q, time_limit=time_limit, workers=2)
        m = structure_metrics(q.pairs(r.x), truth, n)
        rows.append({'id': rec['id'], 'fold': fold, 'length': n,
                     'genus': rec['genus'], 'budget': K, 'n_qubits': q.n,
                     'covered_true_pairs': cov, 'ceiling_mcc': ceil,
                     'cpsat_mcc': m['mcc'], 'cpsat_optimal': r.optimal,
                     'cpsat_energy': r.energy, 'cpsat_x': r.x})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--params', required=True,
                    help='JSON with c_far, pk_beta, ml_sibling [, pk_mode, pk_init, pk_unp]')
    ap.add_argument('--budgets', default='12,16,20,24,28,0')
    ap.add_argument('--max-len', type=int, default=60)
    ap.add_argument('--time-limit', type=float, default=30)
    ap.add_argument('--procs', type=int, default=16)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    params = {k: v for k, v in json.load(open(args.params)).items()
              if k in ('c_far', 'pk_beta', 'ml_sibling', 'pk_mode', 'pk_init',
                       'pk_unp')}
    budgets = [int(b) or None for b in args.budgets.split(',')]

    folds = {r['id']: r['fold'] for r in
             csv.DictReader(open(os.path.join(DATA_DIR, 'pdb_fr3d_folds.csv')))}
    recs = [json.loads(l) for l in open(os.path.join(DATA_DIR, 'pdb_fr3d_chains.jsonl'))]
    recs = [r for r in recs if folds[r['id']] in ('calibration', 'test')
            and r['length'] <= args.max_len]
    print(f'{len(recs)} chains x {len(budgets)} budgets', flush=True)
    with ProcessPoolExecutor(args.procs) as pool, open(args.out, 'w') as f:
        futs = [pool.submit(work, r, folds[r['id']], budgets, params,
                            args.time_limit) for r in recs]
        for k, fu in enumerate(futs, 1):
            for row in fu.result():
                row['budget'] = row['budget'] or 0
                f.write(json.dumps(row) + '\n')
            if k % 50 == 0:
                print(f'  {k}/{len(recs)}', flush=True)


if __name__ == '__main__':
    main()
