"""Classical thermodynamic baselines on every kept chain: pKiss (pseudoknots,
MFE mode) and ViennaRNA RNAfold MFE (nested).  Run in WSL from the mrna-sqd
env; pKiss is called in the separate pk-tools env.

    python scripts/classical_baselines.py --out results/classical_baselines.jsonl
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mrnasqd import DATA_DIR, baseline_path  # noqa: E402
from mrnasqd.metrics import structure_metrics  # noqa: E402

baseline_path()
import RNA  # noqa: E402
from genus import parse_dotbracket  # noqa: E402

PKISS = [os.path.expanduser('~/.local/bin/micromamba'), 'run', '-n', 'pk-tools',
         'pKiss', '--mode', 'mfe']


def run_pkiss(seq: str, timeout: float):
    clean = ''.join(c if c in 'ACGU' else 'A' for c in seq)
    t0 = time.time()
    p = subprocess.run(PKISS, input=clean + '\n', capture_output=True,
                       text=True, timeout=timeout)
    lines = [l for l in p.stdout.splitlines() if l.strip()]
    db, energy = None, None
    for l in lines[::-1]:
        parts = l.split()
        if len(parts) == 2 and len(parts[1]) == len(seq):
            energy, db = float(parts[0]), parts[1]
            break
    if db is None:
        raise RuntimeError(p.stderr[-300:] or p.stdout[-300:])
    pairs = [(i, j) for i, j in parse_dotbracket(db)
             if seq[i] in 'ACGU' and seq[j] in 'ACGU']
    return pairs, energy, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/classical_baselines.jsonl')
    ap.add_argument('--threads', type=int, default=24)
    ap.add_argument('--timeout', type=float, default=1800)
    args = ap.parse_args()

    folds = {r['id']: r['fold'] for r in
             csv.DictReader(open(os.path.join(DATA_DIR, 'pdb_fr3d_folds.csv')))}
    recs = [json.loads(l) for l in open(os.path.join(DATA_DIR, 'pdb_fr3d_chains.jsonl'))]
    recs = [r for r in recs if folds[r['id']] in ('calibration', 'test')]
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)['id'] for l in open(args.out) if l.strip()}
    todo = [r for r in recs if r['id'] not in done]
    todo.sort(key=lambda r: -r['length'])          # long ones first
    print(f'{len(todo)} chains to run', flush=True)

    def work(r):
        truth = [tuple(p) for p in r['pairs']]
        out = {'id': r['id'], 'fold': folds[r['id']], 'length': r['length']}
        s, e = RNA.fold(r['sequence'])
        mv = structure_metrics(parse_dotbracket(s), truth, r['length'])
        out.update({'vienna_mcc': mv['mcc'], 'vienna_sens': mv['sensitivity'],
                    'vienna_ppv': mv['ppv'], 'vienna_db': s})
        try:
            pairs, ep, sec = run_pkiss(r['sequence'], args.timeout)
            mp = structure_metrics(pairs, truth, r['length'])
            out.update({'pkiss_mcc': mp['mcc'], 'pkiss_sens': mp['sensitivity'],
                        'pkiss_ppv': mp['ppv'], 'pkiss_energy': ep,
                        'pkiss_pairs': pairs, 'pkiss_s': round(sec, 1)})
        except Exception as err:
            out['pkiss_error'] = repr(err)[:300]
        return out

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0 = time.time()
    with ThreadPoolExecutor(args.threads) as pool, open(args.out, 'a') as f:
        for n, d in enumerate(pool.map(work, todo), 1):
            f.write(json.dumps(d) + '\n')
            f.flush()
            if n % 50 == 0:
                print(f'  {n}/{len(todo)} {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
