"""Samplers and SQD post-processing for stem QUBOs.

Samplers (all return basis indices; same shot budget for every sampler):
  * ``skqd``    Krylov states exp(-i k dt H(G))|psi0>, k = 1..r-1 (the
                reference state psi0 itself enters the subspace for free);
  * ``lrqaoa``  fixed linear-ramp QAOA state (SQD's "sampling circuit");
  * ``random``  uniform bitstrings (control);
  * ``sa``      final states of independent simulated-annealing runs
                (classical control; one run = one sample).

Post-processing (identical for every sampler):
  * ``best``    lowest H_C among the unique samples.  For a diagonal H_C this
                *is* SQD: the projected matrix is diagonal.
  * ``sqd``     project H(G_d) = H_C/scale - G_d sum X onto the unique
                samples with ``qiskit_addon_sqd.qubit.solve_qubit``, take the
                lowest eigenvector; read out its largest-weight bitstring.
  * ``rep_best`` lowest H_C after repair only (drop the least stable stem of
                each violated conflict).
  * ``recovery`` configuration recovery for bitstrings: repair infeasible
                samples (drop the least stable stem of each conflict), then
                ``rounds`` of single-flip expansion around the ``keep`` best
                configurations, then ``sqd`` on the enlarged subspace.  This
                is classical local search; it is reported separately so the
                quantum contribution can be isolated.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

# qiskit-addon-sqd uses JAX; keep it on the CPU (the WSL GPU is too small
# for its default preallocation and only produces OOM noise).
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import numpy as np

from mrnasqd import quantum as Qm


# ── samplers ────────────────────────────────────────────────────────────────

def sample_skqd(E: np.ndarray, n: int, gamma: float, dt: float, r: int,
                shots: int, rngs, psi0_index: int = 0,
                trotter_steps: int = 1, x_opt: Optional[int] = None,
                nb: Optional[np.ndarray] = None, noise=None
                ) -> Tuple[list, Dict]:
    """E is the scaled cost vector.  The Krylov states are computed once and
    each generator in ``rngs`` (one per seed) draws ``shots // (r-1)`` shots
    from every state k = 1..r-1.  Returns (one index array per rng, info);
    each array is [psi0_index] followed by the (r-1) x per block, row-major.
    info holds the exact probability of the optimum in each Krylov state.

    ``nb``: conflict masks -> constrained (feasibility-preserving) drive.
    ``noise``: optional list of (label, fidelity_fn(k), eps); for each entry
    an extra noisy sample set per rng is drawn from the same states with
    ``noisy_sample`` and returned in info['noisy'][label]."""
    if isinstance(rngs, np.random.Generator):
        rngs = [rngs]
    per = max(1, shots // max(1, r - 1))
    outs = [[np.array([psi0_index], dtype=np.int64)] for _ in rngs]
    noisy = {lab: [[np.array([psi0_index], dtype=np.int64)] for _ in rngs]
             for lab, _, _ in (noise or [])}
    p_opt = []
    gen = Qm.krylov_states(E, n, gamma, dt, r, Qm.basis_state(n, psi0_index),
                           trotter_steps, nb=nb)
    next(gen)                                   # k = 0: reference state
    for k, psi in enumerate(gen, start=1):
        if x_opt is not None:
            p_opt.append(float(abs(psi[x_opt]) ** 2))
        for o, g in zip(outs, rngs):
            o.append(Qm.sample(psi, per, g))
        for lab, fid, eps in (noise or []):
            for o, g in zip(noisy[lab], rngs):
                o.append(noisy_sample(psi, per, g, n, fid(k), eps))
        del psi
    info = {'p_opt_k': p_opt}
    if noise:
        info['noisy'] = {lab: [np.concatenate(o) for o in v]
                         for lab, v in noisy.items()}
    return [np.concatenate(o) for o in outs], info


def sample_skqd_traj(E: np.ndarray, n: int, gamma: float, dt: float, r: int,
                     shots: int, rng: np.random.Generator, psi0_index: int,
                     err: np.ndarray, trajectories: int, eps: float = 0.0,
                     nb: Optional[np.ndarray] = None) -> np.ndarray:
    """Noisy SKQD samples from Pauli trajectories (quantum.krylov_states_noisy):
    ``trajectories`` independent noisy runs, each contributing
    shots // (r-1) // trajectories shots per Krylov state, then readout
    bit flips with probability ``eps``.  Layout as ``sample_skqd``:
    [psi0_index] then an (r-1) x per block, row-major."""
    per_t = max(1, shots // max(1, r - 1) // trajectories)
    blocks = [[] for _ in range(r - 1)]
    for _ in range(trajectories):
        gen = Qm.krylov_states_noisy(E, n, gamma, dt, r,
                                     Qm.basis_state(n, psi0_index), err, rng, nb)
        next(gen)
        for k, psi in enumerate(gen):
            x = Qm.sample(psi, per_t, rng)
            if eps > 0:
                flips = np.zeros(per_t, dtype=np.int64)
                for q in range(n):
                    flips |= (rng.random(per_t) < eps).astype(np.int64) << q
                x ^= flips
            blocks[k].append(x)
            del psi
    body = np.stack([np.concatenate(b) for b in blocks])      # (r-1, per)
    return np.concatenate([np.array([psi0_index], dtype=np.int64), body.ravel()])


def noisy_sample(psi: np.ndarray, shots: int, rng: np.random.Generator,
                 n: int, fidelity: float, eps: float) -> np.ndarray:
    """Global white-noise model of a noisy circuit plus readout error: each
    shot comes from |psi|^2 with probability ``fidelity`` and is uniformly
    random otherwise; then every bit flips independently with probability
    ``eps``.  fidelity = (1 - p2)^N2q for N2q two-qubit gates (validated
    against Aer density-matrix runs in scripts/validate_noise.py)."""
    x = Qm.sample(psi, shots, rng)
    junk = rng.random(shots) >= fidelity
    if junk.any():
        x[junk] = rng.integers(0, 1 << n, int(junk.sum()), dtype=np.int64)
    if eps > 0:
        flips = np.zeros(shots, dtype=np.int64)
        for q in range(n):
            flips |= (rng.random(shots) < eps).astype(np.int64) << q
        x ^= flips
    return x


def warm_start_index(q, sequence: str) -> int:
    """ViennaRNA MFE structure mapped onto the candidate stems: candidates
    whose pairs all lie in the MFE structure, longest first, skipping any
    that conflict with one already chosen.  Always feasible."""
    from mrnasqd import baseline_path
    baseline_path()
    import RNA
    from genus import parse_dotbracket
    mfe = set(parse_dotbracket(RNA.fold(sequence)[0]))
    chosen, used = [], set()
    for i in sorted(range(q.n), key=lambda i: -len(q.stems[i].pairs)):
        st = q.stems[i]
        pos = {b for p in st.pairs for b in p}
        if set(st.pairs) <= mfe and not (pos & used):
            chosen.append(i)
            used |= pos
    return int(sum(1 << i for i in chosen))


def run_vqe(E: np.ndarray, n: int, reps: int, rng: np.random.Generator,
            maxiter: int = 300, rhobeg: float = 0.5):
    """Baseline VQE (WISER-Moderna ``run_vqe``): two-local RY/CZ ansatz,
    exact expectation of the scaled H_C, COBYLA (maxiter 300, rhobeg 0.5),
    x0 ~ U(-pi, pi).  Returns (final state, info)."""
    from scipy.optimize import minimize
    signs = Qm.cz_chain_signs(n)
    n_par = n * (reps + 1)
    evals = [0]

    def f(theta):
        evals[0] += 1
        psi = Qm.two_local_state(n, theta, reps, signs)
        return float(np.dot(np.abs(psi) ** 2, E))

    x0 = rng.uniform(-np.pi, np.pi, n_par)
    res = minimize(f, x0, method='COBYLA',
                   options={'maxiter': maxiter, 'rhobeg': rhobeg})
    psi = Qm.two_local_state(n, res.x, reps, signs)
    return psi, {'vqe_energy': float(res.fun), 'n_evals': evals[0],
                 'argmax': int(np.argmax(np.abs(psi)))}


def sample_lrqaoa(E: np.ndarray, n: int, p: int, d_gamma: float, d_beta: float,
                  shots: int, rng: np.random.Generator,
                  x_opt: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
    psi = Qm.lr_qaoa_state(E, n, p, d_gamma, d_beta)
    info = {'p_opt': float(abs(psi[x_opt]) ** 2) if x_opt is not None else None}
    return Qm.sample(psi, shots, rng), info


def sample_random(n: int, shots: int, rng: np.random.Generator):
    return rng.integers(0, 1 << n, shots, dtype=np.int64), {}


def sample_sa(E: np.ndarray, n: int, shots: int, rng: np.random.Generator,
              sweeps: int = 50, t_hi: float = 5.0, t_lo: float = 0.05,
              init: Optional[int] = None):
    """Vectorised single-flip Metropolis on the (scaled) energy vector;
    ``shots`` independent runs from random starts (or all from ``init``),
    geometric schedule."""
    if init is None:
        x = rng.integers(0, 1 << n, shots, dtype=np.int64)
    else:
        x = np.full(shots, init, dtype=np.int64)
    e = E[x]
    # A single sweep runs at t_lo (a greedy descent).  geomspace(t_hi, t_lo, 1)
    # would give [t_hi], i.e. one hot random-walk sweep (the schedule used for
    # the rows tagged ``sweeps1_hot`` / ``warm_sweeps1_hot``).
    temps = np.geomspace(t_hi, t_lo, sweeps) if sweeps > 1 else np.array([t_lo])
    for T in temps:
        for q in rng.permutation(n):
            y = x ^ (1 << int(q))
            ey = E[y]
            acc = (ey <= e) | (rng.random(shots) < np.exp(-(ey - e) / T))
            x = np.where(acc, y, x)
            e = np.where(acc, ey, e)
    return x, {'sweeps': sweeps}


# ── post-processing ─────────────────────────────────────────────────────────

def project_drive(idx: np.ndarray, E: np.ndarray, n: int, gamma: float,
                  nb: Optional[np.ndarray] = None):
    """Sparse H(G) = diag(E) - G sum X projected onto basis states ``idx``
    (sorted, unique).  With ``nb`` the drive is the constrained mixer
    -G sum X_q P_q: a flip of q couples only when q's conflict partners are
    all 0."""
    from scipy.sparse import coo_matrix
    pos = {int(v): k for k, v in enumerate(idx)}
    rows, cols = [], []
    for k, v in enumerate(idx):
        for q in range(n):
            if nb is not None and (int(v) & int(nb[q])):
                continue
            w = pos.get(int(v) ^ (1 << q))
            if w is not None:
                rows.append(k)
                cols.append(w)
    m = len(idx)
    off = coo_matrix((np.full(len(rows), -gamma), (rows, cols)), shape=(m, m))
    diag = coo_matrix((E[idx], (np.arange(m), np.arange(m))), shape=(m, m))
    return (off + diag).tocsr()


def sqd_diagonalise(idx: np.ndarray, E: np.ndarray, n: int, gamma: float,
                    use_addon: bool = True, H=None,
                    nb: Optional[np.ndarray] = None) -> Tuple[int, float, float]:
    """Lowest eigenpair of H(G) in span(idx).  Returns (readout index,
    eigenvalue, weight of the readout).  With ``use_addon`` the projection and
    diagonalisation are done by qiskit-addon-sqd's ``solve_qubit`` on the
    Pauli form of H(G); otherwise by ``project_drive`` + eigsh.  The
    constrained drive (``nb``) has an exponentially long Pauli form, so it
    always uses ``project_drive``."""
    idx = np.unique(idx)
    if len(idx) == 1:
        return int(idx[0]), float(E[idx[0]]), 1.0
    if use_addon and H is not None and nb is None:
        from qiskit_addon_sqd.qubit import solve_qubit
        op = Qm.pauli_cost(H, H.scale) + Qm.pauli_drive(n, gamma)
        # solve_qubit wants bitstrings left = highest qubit
        bits = Qm.bits_of(idx, n)[:, ::-1].astype(bool)
        vals, vecs = solve_qubit(bits, op, k=1, which='SA')
        # solve_qubit sorts its rows; recover the order it used
        order_bits = np.unique(bits, axis=0)
        sorted_idx = Qm.index_of(order_bits[:, ::-1])
        v = np.abs(vecs[:, 0]) ** 2
        k = int(np.argmax(v))
        return int(sorted_idx[k]), float(vals[0]), float(v[k])
    from scipy.sparse.linalg import eigsh
    M = project_drive(idx, E, n, gamma, nb)
    if len(idx) <= 3:
        w, V = np.linalg.eigh(M.toarray())
        vals, vecs = w[:1], V[:, :1]
    else:
        vals, vecs = eigsh(M, k=1, which='SA')
    v = np.abs(vecs[:, 0]) ** 2
    k = int(np.argmax(v))
    return int(idx[k]), float(vals[0]), float(v[k])


def repair(idx: np.ndarray, H: Qm.CostHamiltonian, conflicts) -> np.ndarray:
    """Make each sampled configuration feasible by dropping, for every
    violated conflict, the stem with the higher one-body energy."""
    out = idx.copy()
    for a, b in conflicts:
        both = ((out >> a) & 1).astype(bool) & ((out >> b) & 1).astype(bool)
        if not both.any():
            continue
        drop = a if H.h[a] >= H.h[b] else b
        out[both] &= ~np.int64(1 << drop)
    return out


def expand(idx: np.ndarray, E: np.ndarray, n: int, keep: int, rounds: int
           ) -> np.ndarray:
    """Add all single-flip neighbours of the ``keep`` lowest-energy
    configurations, ``rounds`` times."""
    cur = np.unique(idx)
    for _ in range(rounds):
        best = cur[np.argsort(E[cur])[:keep]]
        nb = (best[:, None] ^ (np.int64(1) << np.arange(n, dtype=np.int64))[None, :]).ravel()
        cur = np.unique(np.concatenate([cur, nb]))
    return cur


def postprocess(idx: np.ndarray, E: np.ndarray, H: Qm.CostHamiltonian,
                conflicts, gamma_d: float = 0.1, keep: int = 20,
                rounds: int = 3, use_addon: bool = True,
                nb: Optional[np.ndarray] = None) -> Dict:
    """``nb``: use the constrained drive in the SQD projection (for samples
    from the constrained sampler)."""
    uniq = np.unique(idx)
    best = int(uniq[np.argmin(E[uniq])])
    sqd_x, sqd_val, sqd_w = sqd_diagonalise(uniq, E, H.n, gamma_d, use_addon, H, nb)
    rep = np.unique(repair(uniq, H, conflicts))
    rep_best = int(rep[np.argmin(E[rep])])
    sub = expand(rep, E, H.n, keep, rounds)
    rec_best = int(sub[np.argmin(E[sub])])
    rec_x, rec_val, rec_w = sqd_diagonalise(sub, E, H.n, gamma_d, use_addon, H, nb)
    feas = np.ones(len(uniq), bool)
    for a, b in conflicts:
        feas &= ~(((uniq >> a) & 1).astype(bool) & ((uniq >> b) & 1).astype(bool))
    return {'n_unique': int(len(uniq)), 'feasible_frac': float(feas.mean()),
            'best': best, 'sqd': sqd_x, 'sqd_eig': sqd_val, 'sqd_w': sqd_w,
            'rep_best': rep_best,
            'rec_best': rec_best, 'rec': rec_x, 'rec_eig': rec_val,
            'rec_dim': int(len(sub))}
