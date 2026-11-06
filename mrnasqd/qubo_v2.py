"""Stem-level QUBO models for RNA secondary structure (v2).

A model is ``h`` (linear) + ``J`` (pairwise energy couplings) over stem
candidates, plus hard *conflict* pairs (stems sharing a nucleotide).  Keeping
conflicts separate lets CP-SAT treat them as constraints; ``dense_Q`` folds
them back in as a penalty for the quantum solvers.

Models (``model=``):

``baseline``
    The WISER-Moderna QUBO semantics on the chosen candidate set: one-body
    = ViennaRNA ``eval_structure`` of the stem alone; a two-body term only for
    directly stacked neighbours and only when stabilising; a flat ``pk_beta``
    per crossing pair (the baseline's ``mu* = -0.25``).

``turner2``
    One-body as above.  Every compatible (non-overlapping, non-crossing)
    pair gets the exact two-stem correction
        J(s,t) = E(s u t) - E(s) - E(t).
    For t inside s this is I(s,t) - H(s) - X(t): the interior loop replaces
    s's hairpin.  The energy of any structure with at most two stems is then
    exactly its Turner energy; with more stems the pairwise sum misses the
    multiloop and over-subtracts s's hairpin once per extra child.
    ``ml_sibling`` adds a constant to every side-by-side stem pair to absorb
    that error on average (calibrated).  Crossing pairs get ``pk_beta``.

``turner_direct``
    As ``turner2``, but the two-stem correction is kept only for nested
    pairs where no third candidate can sit in the loop between them, so t
    is necessarily s's direct child and the term is exact.  Other nested
    pairs get the constant ``c_far``.  Without this, chains s > u > t of
    truncated stems collect a spurious "interior loop replaces hairpin"
    bonus for every ancestor-descendant pair.

``fox22``
    Fox et al. (2022) length-based Hamiltonian (c_L = 10, c_B = 1,
    pseudoknot factor 0.5, mu = longest stem), for comparison.

Energies are kcal/mol (ViennaRNA rounds to 0.01, so CP-SAT scales by 100
exactly).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from mrnasqd import baseline_path
from mrnasqd.encoding import generate_substem_candidates

baseline_path()
import RNA  # noqa: E402
from stems import Stem  # noqa: E402


# ── geometry ────────────────────────────────────────────────────────────────

def _positions(s: Stem) -> set:
    return {b for p in s.pairs for b in p}


def relation(a: Stem, b: Stem) -> str:
    """'conflict' | 'cross' | 'inside_a' (b inside a) | 'inside_b' | 'side'."""
    if _positions(a) & _positions(b):
        return 'conflict'
    (ai, aj), (bi, bj) = a.outer, b.outer
    if ai < bi < aj < bj or bi < ai < bj < aj:
        return 'cross'
    if ai < bi and bj < aj:
        return 'inside_a'
    if bi < ai and aj < bj:
        return 'inside_b'
    return 'side'


def dotbracket(pairs: Sequence[Tuple[int, int]], n: int) -> str:
    db = ['.'] * n
    for i, j in pairs:
        db[i], db[j] = '(', ')'
    return ''.join(db)


def stacking_energy(fc, s: Stem) -> float:
    """Sum of the stacking energies inside a stem (kcal/mol)."""
    e = 0
    for (i, j), (k, l) in zip(s.pairs[:-1], s.pairs[1:]):
        e += fc.eval_int_loop(i + 1, j + 1, k + 1, l + 1)
    return e / 100.0


# ── model container ─────────────────────────────────────────────────────────

@dataclass
class StemQUBO:
    sequence: str
    stems: List[Stem]
    h: np.ndarray
    J: Dict[Tuple[int, int], float]
    conflicts: List[Tuple[int, int]]
    crossing: List[Tuple[int, int]]
    model: str
    params: dict = field(default_factory=dict)
    build_seconds: float = 0.0

    @property
    def n(self) -> int:
        return len(self.stems)

    def energy(self, x: Sequence[int]) -> float:
        """Model energy; +inf if a conflict pair is co-selected."""
        sel = [i for i, b in enumerate(x) if b]
        ss = set(sel)
        for a, b in self.conflicts:
            if a in ss and b in ss:
                return float('inf')
        e = float(sum(self.h[i] for i in sel))
        for a_idx in range(len(sel)):
            for b_idx in range(a_idx + 1, len(sel)):
                e += self.J.get((sel[a_idx], sel[b_idx]), 0.0)
        return e

    def pairs(self, x: Sequence[int]) -> List[Tuple[int, int]]:
        out = set()
        for i, b in enumerate(x):
            if b:
                out.update(self.stems[i].pairs)
        return sorted(out)

    def penalty_weight(self, multiplier: float = 10.0) -> float:
        mags = [abs(v) for v in self.h] + [abs(v) for v in self.J.values()]
        return multiplier * max(max(mags, default=1.0), 1e-6)

    def dense_Q(self, penalty: Optional[float] = None) -> np.ndarray:
        """Symmetric QUBO matrix (x^T Q x) with conflicts as a penalty."""
        P = self.penalty_weight() if penalty is None else penalty
        Q = np.diag(self.h.astype(float))
        for (a, b), v in self.J.items():
            Q[a, b] += v / 2
            Q[b, a] += v / 2
        for a, b in self.conflicts:
            Q[a, b] += P / 2
            Q[b, a] += P / 2
        return Q


# ── builders ────────────────────────────────────────────────────────────────

def _far_nested_pairs(stems: List[Stem], rel: Dict[Tuple[int, int], str]
                      ) -> set:
    """Nested pairs (s, t), t inside s, for which some third candidate u,
    compatible with both, could sit in the loop that s closes alongside or
    around t (u inside s and not inside t).  For such pairs t is not
    guaranteed to be s's direct child, so the two-stem correction
    E(s u t) - E(s) - E(t) is not valid in general.
    """
    n = len(stems)
    if n < 3:
        return set()
    C = np.zeros((n, n), bool)       # conflict
    X = np.zeros((n, n), bool)       # cross
    In = np.zeros((n, n), bool)      # In[a, b]: b inside a
    for (a, b), r in rel.items():
        if r == 'conflict':
            C[a, b] = C[b, a] = True
        elif r == 'cross':
            X[a, b] = X[b, a] = True
        elif r == 'inside_a':
            In[a, b] = True
        elif r == 'inside_b':
            In[b, a] = True
    far = set()
    for s_idx in range(n):
        U = In[s_idx] & ~C[s_idx]
        ts = np.flatnonzero(U)
        if len(ts) < 2:
            continue
        # M[k, u]: u is a compatible third stem for the pair (s, ts[k])
        M = U[None, :] & ~C[ts] & ~X[ts] & ~In[ts]
        M[np.arange(len(ts)), ts] = False
        for t_idx in ts[M.any(axis=1)]:
            far.add((min(s_idx, t_idx), max(s_idx, t_idx)))
    return far


def _direct_crossing_pairs(stems: List[Stem], rel: Dict[Tuple[int, int], str]
                           ) -> set:
    """Crossing pairs (s, t) for which no third candidate u, compatible with
    both and crossing neither, can sit nested inside s or inside t.  For such
    pairs neither stem can close a real hairpin or carry a nested child once
    both are selected, so replacing their two hairpin terms by a pseudoknot
    loop term cannot double-count against the nested corrections."""
    n = len(stems)
    C = np.zeros((n, n), bool)
    X = np.zeros((n, n), bool)
    In = np.zeros((n, n), bool)
    for (a, b), r in rel.items():
        if r == 'conflict':
            C[a, b] = C[b, a] = True
        elif r == 'cross':
            X[a, b] = X[b, a] = True
        elif r == 'inside_a':
            In[a, b] = True
        elif r == 'inside_b':
            In[b, a] = True
    direct = set()
    for a in range(n):
        B = np.flatnonzero(X[a])
        B = B[B > a]
        if not len(B):
            continue
        M = ((~C[a] & ~X[a])[None, :] & ~C[B] & ~X[B]
             & (In[a][None, :] | In[B]))
        for b in B[~M.any(axis=1)]:
            direct.add((a, int(b)))
    return direct


def pk_loop_lengths(s: Stem, t: Stem) -> Tuple[int, int, int]:
    """Unpaired lengths (L1, L2, L3) of an H-type pseudoknot formed by two
    crossing, non-conflicting stems: s 5' strand, L1, t 5' strand, L2,
    s 3' strand, L3, t 3' strand."""
    if s.outer[0] > t.outer[0]:
        s, t = t, s
    (i, j), (ii, jj) = s.outer, s.inner
    (k, l), (kk, ll) = t.outer, t.inner
    return k - ii - 1, jj - kk - 1, ll - j - 1


def build_stem_qubo(sequence: str, model: str = 'turner2',
                    min_len: int = 2, mode: str = 'truncations',
                    pk_beta: float = 1.0, ml_sibling: float = 0.0,
                    c_far: float = 0.0,
                    pk_mode: str = 'flat', pk_init: float = 8.0,
                    pk_unp: float = 0.1,
                    stack_max: Optional[float] = None,
                    max_stems: Optional[int] = None,
                    j_tol: float = 1e-9) -> StemQUBO:
    """Build a stem QUBO.

    ``pk_mode='loop'`` (turner models): for crossing pairs from
    ``_direct_crossing_pairs`` the coupling is
        pk_init + pk_unp * (L1 + L2 + L3) - H(s) - H(t),
    i.e. both stems' hairpin terms (which no longer exist in an H-type
    pseudoknot) are replaced by a Dirks-Pierce / HotKnots-style loop cost.
    Other crossing pairs keep the flat ``pk_beta``.

    ``stack_max``: drop stems whose internal stacking energy is above this
    (kcal/mol), e.g. -1.0.  ``max_stems``: then keep only the most stable
    stems by stacking energy.  Both prune the qubit count.
    """
    t0 = time.time()
    seq = sequence.upper().replace('T', 'U')
    n_nt = len(seq)
    if model == 'fox22':
        min_len, mode = max(min_len, 3), 'all'
    stems = generate_substem_candidates(seq, min_len=min_len, mode=mode)
    fc = RNA.fold_compound(seq)

    if stack_max is not None or max_stems is not None:
        st_e = [stacking_energy(fc, s) for s in stems]
        order = sorted(range(len(stems)), key=lambda i: st_e[i])
        keep = [i for i in order if stack_max is None or st_e[i] <= stack_max]
        if max_stems is not None:
            keep = keep[:max_stems]
        stems = [stems[i] for i in sorted(keep)]

    n = len(stems)
    rel: Dict[Tuple[int, int], str] = {}
    for a in range(n):
        for b in range(a + 1, n):
            rel[(a, b)] = relation(stems[a], stems[b])
    conflicts = [k for k, r in rel.items() if r == 'conflict']
    crossing = [k for k, r in rel.items() if r == 'cross']

    h = np.zeros(n)
    J: Dict[Tuple[int, int], float] = {}

    if model == 'fox22':
        L = [len(s.pairs) for s in stems]
        mu = max(L) if L else 0
        for i, k in enumerate(L):
            h[i] = 10 * (k - mu) ** 2 - k * k
        for (a, b), r in rel.items():
            if r == 'conflict':
                continue
            base = -2.0 * L[a] * L[b]
            J[(a, b)] = base + 0.5 * abs(base) if r == 'cross' else base
        params = {'c_L': 10, 'c_B': 1, 'pseudo_factor': 0.5, 'mu': mu}
    else:
        dbs = [dotbracket(s.pairs, n_nt) for s in stems]
        for i in range(n):
            h[i] = round(fc.eval_structure(dbs[i]), 2)
        far = (_far_nested_pairs(stems, rel) if model == 'turner_direct'
               else set())
        dcross = (_direct_crossing_pairs(stems, rel) if pk_mode == 'loop'
                  else set())
        hp = ([fc.eval_hp_loop(st.inner[0] + 1, st.inner[1] + 1) / 100.0
               for st in stems] if pk_mode == 'loop' else [])
        for (a, b), r in rel.items():
            if r == 'conflict':
                continue
            if r == 'cross':
                if (a, b) in dcross:
                    L1, L2, L3 = pk_loop_lengths(stems[a], stems[b])
                    v = pk_init + pk_unp * (L1 + L2 + L3) - hp[a] - hp[b]
                    if abs(v) > j_tol:
                        J[(a, b)] = round(v, 2)
                elif abs(pk_beta) > j_tol:
                    J[(a, b)] = pk_beta
                continue
            if (a, b) in far:
                if abs(c_far) > j_tol:
                    J[(a, b)] = c_far
                continue
            if model == 'baseline':
                sa, sb = stems[a], stems[b]
                adjacent = (
                    (sb.outer[0] == sa.inner[0] + 1 and sb.outer[1] == sa.inner[1] - 1)
                    or (sa.outer[0] == sb.inner[0] + 1 and sa.outer[1] == sb.inner[1] - 1))
                if not adjacent:
                    continue
            e2 = fc.eval_structure(dotbracket(stems[a].pairs + stems[b].pairs,
                                              n_nt)) - h[a] - h[b]
            if model == 'baseline' and e2 >= 0:
                continue
            if model in ('turner2', 'turner_direct') and r == 'side':
                e2 += ml_sibling
            if abs(e2) > j_tol:
                J[(a, b)] = round(e2, 2)
        params = {'pk_beta': pk_beta, 'ml_sibling': ml_sibling,
                  'c_far': c_far, 'n_far': len(far), 'pk_mode': pk_mode,
                  'pk_init': pk_init, 'pk_unp': pk_unp,
                  'n_direct_cross': len(dcross)}
    params.update({'min_len': min_len, 'mode': mode, 'stack_max': stack_max,
                   'max_stems': max_stems})
    return StemQUBO(sequence=seq, stems=stems, h=h, J=J, conflicts=conflicts,
                    crossing=crossing, model=model, params=params,
                    build_seconds=time.time() - t0)


# ── exact solver ────────────────────────────────────────────────────────────

@dataclass
class CPSATResult:
    x: List[int]
    energy: float
    optimal: bool
    seconds: float
    status: str


def cpsat_solve(q: StemQUBO, time_limit: float = 60.0, workers: int = 8,
                scale: int = 100, exclude: Sequence[Sequence[int]] = (),
                hint: Optional[Sequence[int]] = None) -> CPSATResult:
    """Exact minimisation with conflicts as hard constraints.

    Products x_a x_b are linearised one-sidedly: for a negative coefficient
    only y <= x_a, y <= x_b is needed (minimisation pushes y up); for a
    positive one only y >= x_a + x_b - 1.  ``exclude`` adds no-good cuts
    (used for top-K enumeration).
    """
    from ortools.sat.python import cp_model

    t0 = time.time()
    m = cp_model.CpModel()
    x = [m.new_bool_var(f'x{i}') for i in range(q.n)]
    terms = []
    for i in range(q.n):
        c = int(round(q.h[i] * scale))
        if c:
            terms.append(c * x[i])
    for a, b in q.conflicts:
        m.add_bool_or([x[a].Not(), x[b].Not()])
    for (a, b), v in q.J.items():
        c = int(round(v * scale))
        if not c:
            continue
        y = m.new_bool_var(f'y{a}_{b}')
        if c < 0:
            m.add_implication(y, x[a])
            m.add_implication(y, x[b])
        else:
            m.add_bool_or([x[a].Not(), x[b].Not(), y])
        terms.append(c * y)
    for sol in exclude:
        m.add_bool_or([x[i].Not() if sol[i] else x[i] for i in range(q.n)])
    if hint is not None:
        for i, v in enumerate(hint):
            m.add_hint(x[i], int(v))
    m.minimize(sum(terms) if terms else 0)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    st = solver.solve(m)
    name = solver.status_name(st)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        sol = [int(solver.value(v)) for v in x]
        return CPSATResult(sol, q.energy(sol), st == cp_model.OPTIMAL,
                           time.time() - t0, name)
    return CPSATResult([0] * q.n, 0.0, False, time.time() - t0, name)


def cpsat_topk(q: StemQUBO, k: int = 5, time_limit: float = 30.0,
               workers: int = 8, next_time_limit: Optional[float] = None
               ) -> List[CPSATResult]:
    """The k lowest-energy distinct solutions (no-good cuts, one solve each).

    The first solve gets ``time_limit``, later ones ``next_time_limit``
    (default: the same).  A later solution is the k-th best only if every
    solve up to it was proven optimal."""
    out: List[CPSATResult] = []
    for n_done in range(k):
        tl = time_limit if n_done == 0 or next_time_limit is None else next_time_limit
        r = cpsat_solve(q, tl, workers, exclude=[s.x for s in out])
        if r.status not in ('OPTIMAL', 'FEASIBLE'):
            break
        out.append(r)
    return out


# ── reference energy for re-ranking ─────────────────────────────────────────

def rerank_components(q: StemQUBO, x: Sequence[int], fc=None
                      ) -> Tuple[float, float, int]:
    """(nested Turner energy, stacking of the crossing remainder, number of
    crossing remainder stems), so that the re-rank energy for any beta is
    e_nested + stack_rest + beta * n_rest.  See ``turner_rerank_energy``."""
    sel = [i for i, b in enumerate(x) if b]
    if not sel:
        return 0.0, 0.0, 0
    fc = fc or RNA.fold_compound(q.sequence)
    stack = {i: stacking_energy(fc, q.stems[i]) for i in sel}
    sel.sort(key=lambda i: stack[i])
    page1: List[int] = []
    rest: List[int] = []
    for i in sel:
        if all(relation(q.stems[i], q.stems[j]) != 'cross' for j in page1):
            page1.append(i)
        else:
            rest.append(i)
    pairs = [p for i in page1 for p in q.stems[i].pairs]
    e = fc.eval_structure(dotbracket(pairs, len(q.sequence)))
    return round(e, 2), round(sum(stack[i] for i in rest), 2), len(rest)


def turner_rerank_energy(q: StemQUBO, x: Sequence[int],
                         pk_beta: float = 1.0) -> float:
    """Full-structure energy used to re-rank candidate solutions.

    The nested part (the largest non-crossing subset of the selected stems,
    chosen greedily by stacking energy) is evaluated exactly with ViennaRNA,
    with multiloops and dangles included.  Each remaining crossing stem
    adds its stacking energy plus ``pk_beta``.
    """
    sel = [i for i, b in enumerate(x) if b]
    if not sel:
        return 0.0
    fc = RNA.fold_compound(q.sequence)
    sel.sort(key=lambda i: stacking_energy(fc, q.stems[i]))
    page1: List[int] = []
    rest: List[int] = []
    for i in sel:
        if all(relation(q.stems[i], q.stems[j]) != 'cross' for j in page1):
            page1.append(i)
        else:
            rest.append(i)
    pairs = [p for i in page1 for p in q.stems[i].pairs]
    e = fc.eval_structure(dotbracket(pairs, len(q.sequence)))
    for i in rest:
        e += stacking_energy(fc, q.stems[i]) + pk_beta
    return e
