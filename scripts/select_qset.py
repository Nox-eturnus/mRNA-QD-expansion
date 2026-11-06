"""Select the quantum benchmark chain sets from results/qset_budget.jsonl.

Q-set (test fold, length <= 60): every helical and tertiary-only pseudoknot
plus a seeded, length-stratified sample of nested chains, to N_TEST total.
Q-dev (calibration fold, length <= 60): the same recipe, to N_DEV total,
for tuning quantum hyperparameters.  Both use the calibrated M3 model
pruned to BUDGET stems; the CP-SAT optimum of the pruned model is stored as
the reference a quantum solver is measured against.

    python scripts/select_qset.py [--budget 24]
"""

import argparse
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mrnasqd import DATA_DIR  # noqa: E402


def pick(rows, kinds, n_total, rng):
    pk = [r for r in rows if kinds[r['id']] != 'nested']
    nested = sorted((r for r in rows if kinds[r['id']] == 'nested'),
                    key=lambda r: (r['length'], r['id']))
    k = max(0, n_total - len(pk))
    if k >= len(nested):
        chosen = nested
    else:
        # one draw from each of k equal-width strata of the length-sorted list
        edges = [round(i * len(nested) / k) for i in range(k + 1)]
        chosen = [nested[rng.randrange(edges[i], edges[i + 1])] for i in range(k)]
    return sorted(pk + chosen, key=lambda r: r['id'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=int, default=24)
    ap.add_argument('--n-test', type=int, default=100)
    ap.add_argument('--n-dev', type=int, default=30)
    ap.add_argument('--seed', type=int, default=20261104)
    ap.add_argument('--model-desc', default='turner_direct, truncations, '
                    'min_len 2, c_far 0, pk_beta 0, ml_sibling 0, max_stems=budget')
    args = ap.parse_args()

    root = os.path.dirname(DATA_DIR)
    kinds = {r['id']: r['pk_kind'] for r in
             csv.DictReader(open(os.path.join(DATA_DIR, 'pdb_fr3d_folds.csv')))}
    rows = [json.loads(l) for l in open(os.path.join(root, 'results', 'qset_budget.jsonl'))]
    full = {r['id']: r for r in rows if r['budget'] == 0}
    rows = [r for r in rows if r['budget'] == args.budget]
    rng = random.Random(args.seed)
    out = {'budget': args.budget, 'seed': args.seed,
           'model': args.model_desc,
           'sets': {}}
    for name, fold, n in (('qset', 'test', args.n_test), ('qdev', 'calibration', args.n_dev)):
        sel = pick([r for r in rows if r['fold'] == fold], kinds, n, rng)
        out['sets'][name] = [{
            'id': r['id'], 'length': r['length'], 'pk_kind': kinds[r['id']],
            'n_qubits': r['n_qubits'], 'ceiling_mcc': r['ceiling_mcc'],
            'ref_cpsat_mcc': r['cpsat_mcc'], 'ref_cpsat_energy': r['cpsat_energy'],
            'ref_cpsat_x': r['cpsat_x'], 'ref_cpsat_optimal': r['cpsat_optimal'],
            'unpruned_cpsat_mcc': full[r['id']]['cpsat_mcc']} for r in sel]
        S = out['sets'][name]
        print(f"{name}: {len(S)} chains  "
              f"{sum(s['pk_kind'] == 'helical_pk' for s in S)} helical PK, "
              f"{sum(s['pk_kind'] == 'tertiary_pk' for s in S)} tertiary PK; "
              f"qubits {min(s['n_qubits'] for s in S)}-{max(s['n_qubits'] for s in S)}; "
              f"ref CP-SAT MCC {sum(s['ref_cpsat_mcc'] for s in S) / len(S):.3f}, "
              f"ceiling {sum(s['ceiling_mcc'] for s in S) / len(S):.3f}")
    with open(os.path.join(DATA_DIR, 'qset.json'), 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
