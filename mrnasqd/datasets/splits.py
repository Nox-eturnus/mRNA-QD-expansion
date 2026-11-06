"""Family-disjoint calibration/test split for the PDB/FR3D chains.

The BGSU NR list already removes near-identical chains (one representative
per equivalence class), but distinct classes can still belong to one RNA
family (e.g. many tRNAs).  Neither RCSB nor PDBe returned Rfam labels for
these chains, so families are approximated by single-linkage clustering on

  * normalised local-alignment similarity  s(a,b) / min(s(a,a), s(b,b))
    >= ``sim_threshold`` together with an absolute score s(a,b) >=
    ``min_abs_score`` (match +1, mismatch -1, gap open -2, extend -1), so a
    short chance match between short chains cannot chain families together;
  * a shared family keyword in the RCSB entity description
    (tRNA, 5S rRNA, a named riboswitch class, ...).

Whole clusters are then assigned to calibration or test, stratified so both
folds contain nested and pseudoknotted chains.
"""

from __future__ import annotations

import json
import random
import re
from typing import Dict, List, Tuple

from Bio import Align


FAMILY_PATTERNS: List[Tuple[str, str]] = [
    ('trna', r'\btrna\b|transfer rna|trna-'),
    ('5s_rrna', r'\b5s\b'),
    ('5.8s_rrna', r'5\.8s'),
    ('srp', r'\bsrp\b|signal recognition'),
    ('tmrna', r'tmrna'),
    ('rnase_p', r'rnase p|ribonuclease p'),
    ('hammerhead', r'hammerhead'),
    ('hdv', r'\bhdv\b|hepatitis delta'),
    ('twister', r'twister'),
    ('pistol', r'pistol'),
    ('hatchet', r'hatchet'),
    ('glms', r'glms'),
    ('group_i_intron', r'group i\b|group-i'),
    ('group_ii_intron', r'group ii\b|group-ii'),
    ('snrna_u', r'\bu[1-6]\b.*snrna|\bu[1-6] snrna|snrna'),
    ('telomerase', r'telomerase'),
    ('frameshift', r'frameshift'),
    ('ires', r'\bires\b'),
    ('tar', r'\btar\b'),
]
RIBOSWITCH_LIGANDS = [
    'preq1', 'sam', 'sah', 'tpp', 'fmn', 'lysine', 'glycine', 'purine',
    'guanine', 'adenine', 'cobalamin', 'adocbl', 'glutamine', 'fluoride',
    'mn', 'manganese', 'nico', 'c-di-gmp', 'c-di-amp', 'zmp', 'ppgpp',
    'thf', 'tetrahydrofolate', 'moco', 'ykok', 'mg', 'magnesium', 'glna',
    'guanidine', 'nad', 'xanthine', 'li', 'na',
]


def family_keyword(description: str) -> str:
    d = description.lower()
    if 'riboswitch' in d or 'aptamer' in d:
        for lig in RIBOSWITCH_LIGANDS:
            if re.search(rf'\b{re.escape(lig)}\b', d):
                return f'riboswitch_{lig}'
        return ''
    for name, pat in FAMILY_PATTERNS:
        if re.search(pat, d):
            return name
    return ''


class _DSU:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _aligner() -> Align.PairwiseAligner:
    al = Align.PairwiseAligner()
    al.mode = 'local'
    al.match_score = 1.0
    al.mismatch_score = -1.0
    al.open_gap_score = -2.0
    al.extend_gap_score = -1.0
    return al


def cluster_records(records: List[dict], sim_threshold: float = 0.7,
                    min_abs_score: float = 15.0, use_keywords: bool = True
                    ) -> Tuple[List[int], List[dict]]:
    """Return (cluster label per record, list of linking edges)."""
    n = len(records)
    al = _aligner()
    seqs = [r['sequence'] for r in records]
    self_score = [al.score(s, s) for s in seqs]
    dsu = _DSU(n)
    edges: List[dict] = []
    for a in range(n):
        for b in range(a + 1, n):
            raw = al.score(seqs[a], seqs[b])
            s = raw / min(self_score[a], self_score[b])
            if s >= sim_threshold and raw >= min_abs_score:
                dsu.union(a, b)
                edges.append({'a': records[a]['id'], 'b': records[b]['id'],
                              'why': f'similarity {s:.2f}'})
    by_kw: Dict[str, int] = {}
    for idx, r in enumerate(records if use_keywords else []):
        kw = family_keyword(r.get('description', ''))
        if kw:
            if kw in by_kw:
                dsu.union(by_kw[kw], idx)
                edges.append({'a': records[by_kw[kw]]['id'], 'b': r['id'],
                              'why': f'keyword {kw}'})
            else:
                by_kw[kw] = idx
    roots = [dsu.find(i) for i in range(n)]
    relabel = {r: k for k, r in enumerate(sorted(set(roots)))}
    return [relabel[r] for r in roots], edges


def assign_folds(records: List[dict], labels: List[int],
                 calib_fraction: float = 0.4, seed: int = 20261104
                 ) -> List[str]:
    """Greedy cluster-to-fold assignment, stratified by topology class.

    Clusters are visited in a seeded random order; each goes to the fold
    whose nested and pseudoknotted counts are furthest below target.
    """
    clusters: Dict[int, List[int]] = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(lab, []).append(i)
    tot_n = sum(r['genus'] == 0 for r in records)
    tot_p = sum(r['genus'] >= 1 for r in records)
    target = {'calibration': (calib_fraction * tot_n, calib_fraction * tot_p),
              'test': ((1 - calib_fraction) * tot_n,
                       (1 - calib_fraction) * tot_p)}
    have = {'calibration': [0, 0], 'test': [0, 0]}
    order = sorted(clusters)
    rng = random.Random(seed)
    rng.shuffle(order)
    # Large clusters first so they do not unbalance the end of the pass.
    order.sort(key=lambda c: -len(clusters[c]))
    fold = [''] * len(records)
    for c in order:
        members = clusters[c]
        n_nest = sum(records[i]['genus'] == 0 for i in members)
        n_pk = len(members) - n_nest

        def deficit(f: str) -> float:
            tn, tp = target[f]
            hn, hp = have[f]
            dn = (tn - hn) / max(tn, 1)
            dp = (tp - hp) / max(tp, 1)
            return dp if n_pk else dn

        f = max(('calibration', 'test'), key=deficit)
        for i in members:
            fold[i] = f
        have[f][0] += n_nest
        have[f][1] += n_pk
    return fold


def load_jsonl(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


_METHOD_RANK = {'X-RAY DIFFRACTION': 0, 'ELECTRON MICROSCOPY': 1}


def cap_clusters(records: List[dict], labels: List[int], cap: int = 20,
                 seed: int = 20261104) -> List[bool]:
    """Keep at most ``cap`` chains per family cluster.

    Large families (tRNA, 5S rRNA) would otherwise dominate any fold they
    land in.  Within a family, chains are ranked by method (X-ray, EM, NMR)
    and resolution, ties broken by a seeded shuffle.
    """
    rng = random.Random(seed)
    jitter = [rng.random() for _ in records]
    clusters: Dict[int, List[int]] = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(lab, []).append(i)
    keep = [False] * len(records)
    for members in clusters.values():
        members.sort(key=lambda i: (
            _METHOD_RANK.get(records[i]['method'], 2),
            records[i]['resolution'] if records[i]['resolution'] else 99.0,
            jitter[i]))
        for i in members[:cap]:
            keep[i] = True
    return keep


def helices(pairs) -> List[List[Tuple[int, int]]]:
    s = {tuple(p) for p in pairs}
    out = []
    for i, j in sorted(s):
        if (i - 1, j + 1) in s:
            continue
        h = [(i, j)]
        while (h[-1][0] + 1, h[-1][1] - 1) in s:
            h.append((h[-1][0] + 1, h[-1][1] - 1))
        out.append(h)
    return out


def pk_kind(record: dict) -> str:
    """'nested', 'helical_pk' (two crossing helices of >= 2 bp each), or
    'tertiary_pk' (crossing only through lone pairs, e.g. tRNA G19:C56)."""
    if record['genus'] == 0:
        return 'nested'
    hs = [h for h in helices(record['pairs']) if len(h) >= 2]
    for a in range(len(hs)):
        for b in range(a + 1, len(hs)):
            (i, j), (k, l) = hs[a][0], hs[b][0]
            if i < k < j < l or k < i < l < j:
                return 'helical_pk'
    return 'tertiary_pk'
