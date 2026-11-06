"""Exact (CP-SAT) optimum of each QUBO model vs the FR3D structure.

Run in the WSL env:
    python scripts/run_models.py --fold calibration --models M0,M1,M2 \
        --max-len 100 --out results/calib.jsonl

Resumable: (id, model) keys already in --out are skipped.  Model specs are
named in MODELS; ``--set name.key=value`` overrides a parameter, e.g.
``--set M2.pk_beta=0.5``, and ``--tag`` renames the stored model key.
"""

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrnasqd import DATA_DIR  # noqa: E402

MODELS = {
    # Baseline QUBO exactly as in WISER-Moderna (maximal stems, mu* = -0.25).
    'M0': dict(model='baseline', mode='maximal', min_len=2, pk_beta=-0.25),
    # Encoding fix only: same energy terms, truncated stems added.
    'M1': dict(model='baseline', mode='truncations', min_len=2, pk_beta=-0.25),
    # Encoding fix + exact pairwise Turner corrections.
    'M2': dict(model='turner2', mode='truncations', min_len=2, pk_beta=1.0),
    # Exact pairwise corrections only where provably direct (see qubo_v2).
    # pk_beta = 0.0 locked by the calibration sweep (results/m3_params.json).
    'M3': dict(model='turner_direct', mode='truncations', min_len=2,
               pk_beta=0.0, c_far=0.0),
    # M3 + Dirks-Pierce-style pseudoknot loop term for direct crossing pairs;
    # pk_init = 12, pk_unp = 0.1 locked by calibration (results/m4_params.json).
    'M4': dict(model='turner_direct', mode='truncations', min_len=2,
               pk_beta=0.0, c_far=0.0, pk_mode='loop', pk_init=12.0,
               pk_unp=0.1),
    # Literature comparison: FOX22 length-based Hamiltonian, all sub-stems.
    'FOX22': dict(model='fox22', mode='all', min_len=3),
}


def _parse_value(v):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return None if v == 'None' else v


def work(rec, key, spec, time_limit, workers, topk, rerank_beta,
         topk_time_limit=None):
    from mrnasqd.qubo_v2 import (build_stem_qubo, cpsat_solve, cpsat_topk,
                                 turner_rerank_energy)
    from mrnasqd.metrics import structure_metrics
    truth = [tuple(p) for p in rec['pairs']]
    t0 = time.time()
    q = build_stem_qubo(rec['sequence'], **spec)
    out = {'id': rec['id'], 'model': key, 'spec': spec, 'length': rec['length'],
           'genus': rec['genus'], 'n_qubits': q.n, 'n_J': len(q.J),
           'n_conflicts': len(q.conflicts), 'build_s': round(q.build_seconds, 2)}
    if q.n == 0:
        m = structure_metrics([], truth, rec['length'])
        out.update({'optimal': True, 'energy': 0.0, 'pairs': [], 'solve_s': 0.0,
                    'status': 'EMPTY', **{k: m[k] for k in ('tp', 'fp', 'fn', 'sensitivity', 'ppv', 'mcc')}})
        return out
    if topk > 1:
        sols = cpsat_topk(q, k=topk, time_limit=time_limit, workers=workers,
                          next_time_limit=topk_time_limit)
        best = sols[0]
    else:
        best = cpsat_solve(q, time_limit=time_limit, workers=workers)
        sols = [best]
    pairs = q.pairs(best.x)
    m = structure_metrics(pairs, truth, rec['length'])
    out.update({'optimal': best.optimal, 'status': best.status,
                'energy': best.energy, 'pairs': pairs,
                'solve_s': round(time.time() - t0, 2),
                **{k: m[k] for k in ('tp', 'fp', 'fn', 'sensitivity', 'ppv', 'mcc')}})
    if topk > 1:
        rr = sorted(sols, key=lambda s: turner_rerank_energy(q, s.x, rerank_beta))
        mr = structure_metrics(q.pairs(rr[0].x), truth, rec['length'])
        out['topk'] = [{'sel': [i for i, b in enumerate(s.x) if b],
                        'energy': s.energy, 'optimal': s.optimal}
                       for s in sols]
        out.update({'topk_found': len(sols),
                    'topk_all_optimal': all(s.optimal for s in sols),
                    'rerank_mcc': mr['mcc'],
                    'topk_best_mcc': max(structure_metrics(q.pairs(s.x), truth,
                                                           rec['length'])['mcc']
                                         for s in sols)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', default='calibration')
    ap.add_argument('--models', default='M0,M1,M2')
    ap.add_argument('--max-len', type=int, default=100)
    ap.add_argument('--ids', default='')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--time-limit', type=float, default=60.0)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--procs', type=int, default=8)
    ap.add_argument('--topk', type=int, default=1)
    ap.add_argument('--rerank-beta', type=float, default=1.0)
    ap.add_argument('--topk-time-limit', type=float, default=None,
                    help='time limit for the 2nd..k-th solves (default: --time-limit)')
    ap.add_argument('--set', action='append', default=[])
    ap.add_argument('--tag', default='')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    specs = {k: dict(MODELS[k]) for k in args.models.split(',')}
    for s in args.set:
        lhs, v = s.split('=', 1)
        name, key = lhs.split('.', 1)
        specs[name][key] = _parse_value(v)
    if args.tag:
        specs = {f'{k}{args.tag}': v for k, v in specs.items()}

    folds = {r['id']: r['fold'] for r in
             csv.DictReader(open(os.path.join(DATA_DIR, 'pdb_fr3d_folds.csv')))}
    recs = [json.loads(l) for l in open(os.path.join(DATA_DIR, 'pdb_fr3d_chains.jsonl'))]
    want = set(args.ids.split(',')) if args.ids else None
    recs = [r for r in recs if folds.get(r['id']) == args.fold
            and r['length'] <= args.max_len and (want is None or r['id'] in want)]
    recs.sort(key=lambda r: r['length'])
    if args.limit:
        step = max(1, len(recs) // args.limit)
        recs = recs[::step][:args.limit]

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            if line.strip():
                d = json.loads(line)
                done.add((d['id'], d['model']))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    jobs = [(r, k, s) for r in recs for k, s in specs.items() if (r['id'], k) not in done]
    print(f'{len(recs)} chains x {len(specs)} models: {len(jobs)} jobs to run', flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.procs) as pool, open(args.out, 'a') as f:
        futs = [pool.submit(work, r, k, s, args.time_limit, args.workers,
                            args.topk, args.rerank_beta, args.topk_time_limit)
                for r, k, s in jobs]
        for n, fu in enumerate(as_completed(futs), 1):
            try:
                d = fu.result()
            except Exception as err:
                print('ERROR', repr(err)[:300], flush=True)
                continue
            f.write(json.dumps(d) + '\n')
            f.flush()
            if n % 25 == 0 or n == len(futs):
                print(f'  {n}/{len(futs)}  {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
