"""Stem candidates with sub-stems.

The baseline ``stems.generate_stem_candidates`` keeps only *maximal* stems:
a stem that is a strict subset of a longer one is discarded.  When the true
helix is shorter than the longest helix the sequence allows on that diagonal
(common: terminal pairs fray, or a loop nucleotide could extend the stem),
the true helix is not expressible and the QUBO must choose between a stem
that over-predicts and no stem at all.

``generate_substem_candidates`` keeps every contiguous sub-run of length
>= ``min_len`` of each maximal stem (mode ``'all'``), or only runs that share
the maximal stem's outer or inner end (mode ``'truncations'``, O(L) per stem
instead of O(L^2)).
"""

from __future__ import annotations

from typing import List, Tuple

from mrnasqd import baseline_path

baseline_path()
from candidates import generate_pair_candidates  # noqa: E402
from stems import Stem, generate_stem_candidates  # noqa: E402


def _make_stem(pairs: List[Tuple[int, int]]) -> Stem:
    ordered = tuple(sorted(pairs))
    return Stem(pairs=ordered, outer=ordered[0], inner=ordered[-1])


def generate_substem_candidates(sequence: str, min_len: int = 2,
                                mode: str = 'all',
                                max_len: int | None = None) -> List[Stem]:
    pair_cands = generate_pair_candidates(sequence)
    maximal = generate_stem_candidates(sequence, pair_cands,
                                       min_stem_len=min_len)
    seen = set()
    out: List[Stem] = []
    for st in maximal:
        L = len(st.pairs)
        runs: List[Tuple[int, int]] = []          # [start, end) over st.pairs
        if mode == 'all':
            runs = [(a, b) for a in range(L) for b in range(a + min_len, L + 1)]
        elif mode == 'truncations':
            runs = ([(0, b) for b in range(min_len, L + 1)]
                    + [(a, L) for a in range(1, L - min_len + 1)])
        elif mode == 'maximal':
            runs = [(0, L)]
        else:
            raise ValueError(mode)
        for a, b in runs:
            if max_len is not None and b - a > max_len:
                continue
            key = st.pairs[a:b]
            if key in seen:
                continue
            seen.add(key)
            out.append(_make_stem(list(key)))
    out.sort(key=lambda s: (s.outer, -len(s.pairs)))
    return out


def expressible_coverage(stems: List[Stem],
                         true_pairs: List[Tuple[int, int]]) -> int:
    """Number of true pairs reachable using only stems made of true pairs.

    True pairs use each nucleotide once, so stems that are subsets of the
    true structure never conflict except by sharing pairs on one diagonal;
    the union of their pairs is therefore jointly selectable when sub-stems
    are present, and an upper bound (exact for maximal-only) otherwise.
    """
    truth = set(true_pairs)
    covered = set()
    for st in stems:
        if set(st.pairs) <= truth:
            covered.update(st.pairs)
    return len(covered)
