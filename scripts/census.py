"""Census: folds, encoding sizes, expressibility ceiling, ViennaRNA MFE.

Run in the WSL env (needs ViennaRNA + Biopython):
    python scripts/census.py
Writes data/pdb_fr3d_folds.csv, data/census.csv and prints a summary.
"""

import csv
import math
import os
import sys
import time
from statistics import mean, median

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrnasqd import DATA_DIR, baseline_path  # noqa: E402
from mrnasqd.datasets.splits import (load_jsonl, cluster_records,  # noqa: E402
                                     assign_folds, family_keyword,
                                     cap_clusters, pk_kind)
from mrnasqd.encoding import (generate_substem_candidates,  # noqa: E402
                              expressible_coverage)
from mrnasqd.metrics import structure_metrics  # noqa: E402

baseline_path()
import RNA  # noqa: E402
from genus import parse_dotbracket  # noqa: E402


def mcc_from_counts(tp, fp, fn, n):
    tn = n * (n - 1) // 2 - tp - fp - fn
    d = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return (tp * tn - fp * fn) / math.sqrt(d) if d > 0 else 0.0


def main():
    recs = load_jsonl(os.path.join(DATA_DIR, 'pdb_fr3d_chains.jsonl'))
    t0 = time.time()
    labels, edges = cluster_records(recs)
    keep = cap_clusters(recs, labels, cap=20)
    print(f'clustered {len(recs)} chains into {len(set(labels))} clusters '
          f'({len(edges)} links) in {time.time() - t0:.0f}s; '
          f'{sum(keep)} kept after capping families at 20')
    kept_idx = [i for i, k in enumerate(keep) if k]
    kept_folds = assign_folds([recs[i] for i in kept_idx],
                              [labels[i] for i in kept_idx])
    folds = ['dropped_family_cap'] * len(recs)
    for i, fo in zip(kept_idx, kept_folds):
        folds[i] = fo

    with open(os.path.join(DATA_DIR, 'pdb_fr3d_folds.csv'), 'w',
              newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'cluster', 'fold', 'family_keyword', 'pk_kind'])
        for r, lab, fo in zip(recs, labels, folds):
            w.writerow([r['id'], lab, fo, family_keyword(r['description']),
                        pk_kind(r)])
    with open(os.path.join(DATA_DIR, 'pdb_fr3d_cluster_links.csv'), 'w',
              newline='') as f:
        w = csv.DictWriter(f, fieldnames=['a', 'b', 'why'])
        w.writeheader()
        w.writerows(edges)

    rows = []
    for r, fo in zip(recs, folds):
        if fo == 'dropped_family_cap':
            continue
        seq, n = r['sequence'], r['length']
        truth = [tuple(p) for p in r['pairs']]
        row = {'id': r['id'], 'fold': fo, 'length': n, 'genus': r['genus'],
               'pk_kind': pk_kind(r),
               'n_true_pairs': len(truth), 'method': r['method']}
        for m in (2, 3):
            for mode in ('maximal', 'truncations', 'all'):
                st = generate_substem_candidates(seq, min_len=m, mode=mode)
                cov = expressible_coverage(st, truth)
                key = f'{mode}_m{m}'
                row[f'n_{key}'] = len(st)
                row[f'ceil_mcc_{key}'] = round(
                    mcc_from_counts(cov, 0, len(truth) - cov, n), 4)
        # Isolated (helix length 1) true pairs: not expressible by any stem.
        tset = set(truth)
        row['n_isolated_true'] = sum(
            1 for i, j in truth
            if (i + 1, j - 1) not in tset and (i - 1, j + 1) not in tset)
        s, e = RNA.fold(seq)
        met = structure_metrics(parse_dotbracket(s), truth, n)
        row['vienna_mfe_mcc'] = round(met['mcc'], 4)
        row['vienna_mfe'] = round(e, 2)
        rows.append(row)

    with open(os.path.join(DATA_DIR, 'census.csv'), 'w',
              newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def summ(sel, label):
        if not sel:
            return
        print(f'\n== {label}: {len(sel)} chains, length median '
              f'{median(x["length"] for x in sel)}')
        for key in ('maximal_m2', 'truncations_m2', 'all_m2',
                    'maximal_m3', 'truncations_m3', 'all_m3'):
            print(f'  {key:15s} qubits median {median(x["n_"+key] for x in sel):6.0f}'
                  f'  max {max(x["n_"+key] for x in sel):6d}'
                  f'  ceiling MCC mean {mean(x["ceil_mcc_"+key] for x in sel):.3f}')
        print(f'  ViennaRNA MFE MCC mean {mean(x["vienna_mfe_mcc"] for x in sel):.3f}')
        print(f'  isolated true pairs: {sum(x["n_isolated_true"] for x in sel)}'
              f' of {sum(x["n_true_pairs"] for x in sel)}')

    summ(rows, 'all')
    for fo in ('calibration', 'test'):
        summ([x for x in rows if x['fold'] == fo], fo)
    summ([x for x in rows if x['genus'] == 0], 'nested')
    summ([x for x in rows if x['genus'] >= 1], 'pseudoknotted')
    summ([x for x in rows if x['pk_kind'] == 'helical_pk'], 'helical pseudoknots')
    summ([x for x in rows if x['pk_kind'] == 'tertiary_pk'], 'tertiary-only pseudoknots')
    for fo in ('calibration', 'test'):
        sel = [x for x in rows if x['fold'] == fo]
        print(fo, {k: sum(x['pk_kind'] == k for x in sel)
                   for k in ('nested', 'helical_pk', 'tertiary_pk')})
    small = [x for x in rows if x['length'] <= 60]
    summ(small, 'length <= 60')


if __name__ == '__main__':
    main()
