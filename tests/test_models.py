import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrnasqd import baseline_path  # noqa: E402
from mrnasqd.qubo_v2 import (build_stem_qubo, cpsat_solve, cpsat_topk,  # noqa: E402
                             dotbracket, relation)

baseline_path()
import RNA  # noqa: E402

BASELINE_SEQS = [
    'GCAAUAUAUUGC',                       # synth_hairpin_003
    'GCAUAUGCAUAUGCAUAAUGC',              # pk_htype_001
    'GCAAUUGCUGCAUAAUGGCACAU',            # pk_kissing_001
    'GCAAUAUGCUAUGCAAUAAUGUAUCAUAUAUGC',  # fse_3_5
]


@pytest.mark.parametrize('seq', BASELINE_SEQS)
def test_M0_matches_baseline_qubo(seq):
    from genus_penalty import build_qubo_with_genus_penalty
    from classical_solvers import cpsat_solve as base_solve
    base = build_qubo_with_genus_penalty(seq, encoding='stem', mu=-0.25)
    rb = base_solve(base, time_limit_sec=30)
    q = build_stem_qubo(seq, model='baseline', mode='maximal', min_len=2,
                        pk_beta=-0.25)
    r = cpsat_solve(q, time_limit=30)
    assert rb.is_optimal and r.optimal
    assert q.n == base.n
    assert abs(r.energy - rb.energy) < 1e-4   # baseline keeps float32 noise
    # Optima can be tied (pk_htype_001 has two), so instead of requiring the
    # same structure, map M0's solution into the baseline QUBO and check it
    # is optimal there too.
    index = {tuple(c.pairs): k for k, c in enumerate(base.candidates)}
    xb = np.zeros(base.n)
    for k, b in enumerate(r.x):
        if b:
            xb[index[tuple(q.stems[k].pairs)]] = 1
    assert abs(base.evaluate(xb) - rb.energy) < 1e-4


def test_turner2_exact_for_two_stems():
    seq = 'GGGGAAACCCCAAGGGAAAACCCAAAAAGGGGAAACCCCAAA'
    q = build_stem_qubo(seq, model='turner2', mode='truncations', pk_beta=1.0)
    fc = RNA.fold_compound(seq)
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(2000):
        a, b = sorted(rng.choice(q.n, 2, replace=False))
        if relation(q.stems[a], q.stems[b]) in ('conflict', 'cross'):
            continue
        x = np.zeros(q.n, int)
        x[[a, b]] = 1
        e_true = fc.eval_structure(dotbracket(q.pairs(x), len(seq)))
        assert abs(q.energy(x) - e_true) < 0.011
        checked += 1
    assert checked > 50


def test_dense_Q_matches_energy_on_feasible():
    seq = 'GCAAUUGCUGCAUAAUGGCACAU'
    q = build_stem_qubo(seq, model='turner2', mode='truncations')
    Q = q.dense_Q()
    r = cpsat_solve(q)
    x = np.array(r.x, float)
    assert abs(x @ Q @ x - r.energy) < 1e-9
    # dense-Q brute force agrees with CP-SAT when small
    if q.n <= 18:
        best = min(np.array([(s >> k) & 1 for k in range(q.n)], float) @ Q @
                   np.array([(s >> k) & 1 for k in range(q.n)], float)
                   for s in range(2 ** q.n))
        assert abs(best - r.energy) < 1e-6


def test_topk_distinct_and_sorted():
    q = build_stem_qubo('GGGGAAACCCCAAGGGAAAACCC', model='turner2')
    sols = cpsat_topk(q, k=4, time_limit=10)
    xs = [tuple(s.x) for s in sols]
    assert len(set(xs)) == len(xs)
    es = [s.energy for s in sols]
    # ties can differ in the last bit through summation order
    assert all(a <= b + 1e-9 for a, b in zip(es, es[1:]))


def test_turner_direct_exact_on_chains_without_far_pairs():
    # Any feasible nested selection whose nested pairs are all "direct"
    # must score its exact Turner energy (multiloop-free structures).
    from mrnasqd.qubo_v2 import _far_nested_pairs
    seq = 'GGGACUUCGGUCCCAAAGGCAGUAAUGCCAAAA'
    q = build_stem_qubo(seq, model='turner_direct', mode='truncations')
    fc = RNA.fold_compound(seq)
    r = cpsat_solve(q)
    e_true = fc.eval_structure(dotbracket(q.pairs(r.x), len(seq)))
    sel = [i for i, b in enumerate(r.x) if b]
    rel = {(a, b): relation(q.stems[a], q.stems[b]) for a in range(q.n)
           for b in range(a + 1, q.n)}
    far = _far_nested_pairs(q.stems, rel)
    if not any((a, b) in far for a in sel for b in sel if a < b):
        assert abs(r.energy - e_true) < 0.011


def test_pk_loop_lengths_htype():
    from mrnasqd.qubo_v2 import pk_loop_lengths
    from mrnasqd.encoding import _make_stem
    # s: (2,17),(3,16),(4,15); t: (8,25),(9,24),(10,23)
    s = _make_stem([(2, 17), (3, 16), (4, 15)])
    t = _make_stem([(8, 25), (9, 24), (10, 23)])
    # L1 = 8-4-1 = 3, L2 = 15-10-1 = 4, L3 = 23-17-1 = 5
    assert pk_loop_lengths(s, t) == (3, 4, 5)
    assert pk_loop_lengths(t, s) == (3, 4, 5)


def test_pk_loop_term_value():
    seq = 'GGCGCGGCACCGUCCGCGGAACAAACGG'   # BWYV pseudoknot (1L2X)
    q = build_stem_qubo(seq, model='turner_direct', pk_beta=0.0,
                        pk_mode='loop', pk_init=9.0, pk_unp=0.1)
    from mrnasqd.qubo_v2 import _direct_crossing_pairs, pk_loop_lengths
    rel = {(a, b): relation(q.stems[a], q.stems[b]) for a in range(q.n)
           for b in range(a + 1, q.n)}
    dc = _direct_crossing_pairs(q.stems, rel)
    assert dc and dc <= {k for k, r in rel.items() if r == 'cross'}
    fc = RNA.fold_compound(seq)
    a, b = sorted(dc)[0]
    hp = [fc.eval_hp_loop(q.stems[i].inner[0] + 1, q.stems[i].inner[1] + 1) / 100
          for i in (a, b)]
    want = 9.0 + 0.1 * sum(pk_loop_lengths(q.stems[a], q.stems[b])) - sum(hp)
    assert abs(q.J.get((a, b), 0.0) - round(want, 2)) < 1e-9


def test_rerank_components_nested_is_exact():
    from mrnasqd.qubo_v2 import rerank_components
    seq = 'GGGACUUCGGUCCCAAAGGCAGUAAUGCCAAAA'
    q = build_stem_qubo(seq, model='turner_direct', pk_beta=0.0)
    r = cpsat_solve(q)
    e0, srest, nrest = rerank_components(q, r.x)
    fc = RNA.fold_compound(seq)
    assert nrest == 0 and srest == 0
    assert abs(e0 - fc.eval_structure(dotbracket(q.pairs(r.x), len(seq)))) < 0.011
