"""Build data/pdb_fr3d_chains.jsonl from the BGSU NR list + FR3D annotations.

Usage:
    python scripts/build_pdb_dataset.py [--release current] [--cutoff 4.0A]
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrnasqd import DATA_DIR  # noqa: E402
from mrnasqd.datasets.pdb_fr3d import build_dataset, record_to_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', default='current')
    ap.add_argument('--cutoff', default='all')
    ap.add_argument('--min-len', type=int, default=20)
    ap.add_argument('--max-len', type=int, default=150)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    records, exclusions, manifest = build_dataset(
        release=args.release, cutoff=args.cutoff, workers=args.workers,
        min_len=args.min_len, max_len=args.max_len)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, 'pdb_fr3d_chains.jsonl'), 'w') as f:
        for r in records:
            f.write(record_to_json(r) + '\n')
    with open(os.path.join(DATA_DIR, 'pdb_fr3d_exclusions.csv'), 'w',
              newline='') as f:
        w = csv.DictWriter(f, fieldnames=['nr_class', 'id', 'reason', 'detail'])
        w.writeheader()
        w.writerows(sorted(exclusions, key=lambda e: (e['reason'], e['id'])))
    manifest['exclusion_counts'] = dict(Counter(e['reason'] for e in exclusions))
    with open(os.path.join(DATA_DIR, 'pdb_fr3d_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
