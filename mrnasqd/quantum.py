"""Exact statevector machinery for SQD / SKQD on stem QUBOs (<= ~26 qubits).

Conventions
-----------
Qubit i <-> stem candidate i; basis index = sum_i x_i 2^i (Qiskit's
little-endian order), x_i = 1 means stem i is selected.

Cost Hamiltonian (diagonal):  H_C(x) = sum_i h_i x_i + sum_{i<j} J_ij x_i x_j
                                       + P * sum_{conflicts} x_a x_b,
with P = max_i (|h_i| + sum_j |J_ij|) + 1.  With this P every infeasible x has
a strictly lower-energy neighbour (drop one stem of a conflicting pair), so
the minimum of H_C is the constrained optimum.

Driven Hamiltonian:  H(G) = H_C - G * sum_i X_i.   H_C is diagonal, so on its
own its projection onto any set of sampled bitstrings is diagonal and SQD
reduces to keeping the best sample; the drive makes SQD/SKQD non-trivial.

The circuits simulated here are exactly the Trotter / QAOA circuits a device
would run (diagonal phase layer exp(-i t H_C) followed by single-qubit X
rotations); ``qiskit_*`` builders produce the same circuits in Qiskit for
cross-checks and resource counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from mrnasqd.qubo_v2 import StemQUBO


# ── cost Hamiltonian ────────────────────────────────────────────────────────

@dataclass
class CostHamiltonian:
    n: int
    h: np.ndarray                         # linear terms (kcal/mol)
    J: Dict[Tuple[int, int], float]       # pairwise terms incl. penalties
    penalty: float
    scale: float                          # max |h_i|, used to make G, dt dimensionless

    def energy(self, x: Sequence[int]) -> float:
        x = np.asarray(x)
        e = float(self.h @ x)
        for (a, b), v in self.J.items():
            if x[a] and x[b]:
                e += v
        return e


def cost_hamiltonian(q: StemQUBO, penalty: Optional[float] = None) -> CostHamiltonian:
    n = q.n
    row = np.abs(q.h).astype(float).copy()
    for (a, b), v in q.J.items():
        row[a] += abs(v)
        row[b] += abs(v)
    P = float(row.max() + 1.0) if penalty is None else float(penalty)
    J = dict(q.J)
    for a, b in q.conflicts:
        J[(a, b)] = J.get((a, b), 0.0) + P
    scale = float(max(np.abs(q.h).max(initial=0.0), 1e-6))
    return CostHamiltonian(n=n, h=np.asarray(q.h, float), J=J, penalty=P,
                           scale=scale)


def conflict_masks(q: StemQUBO) -> np.ndarray:
    """nb[i] = bitmask of the stems that conflict with stem i."""
    nb = np.zeros(q.n, dtype=np.int64)
    for a, b in q.conflicts:
        nb[a] |= np.int64(1) << b
        nb[b] |= np.int64(1) << a
    return nb


def unpenalised(H: CostHamiltonian, q: StemQUBO) -> CostHamiltonian:
    """H_C without the conflict penalties.  Inside the feasible subspace,
    where the constrained drive keeps the state, the penalties are
    identically zero, so a device does not need to implement them."""
    return CostHamiltonian(n=H.n, h=H.h.copy(), J=dict(q.J), penalty=0.0,
                           scale=H.scale)


def energy_vector(H: CostHamiltonian) -> np.ndarray:
    """H_C(x) for all 2^n basis states, by in-place doubling (O(n 2^n) time,
    one 2^n float64 array plus O(2^(n-1)) work space)."""
    n = H.n
    nbr: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    for (a, b), v in H.J.items():
        lo, hi = min(a, b), max(a, b)
        nbr[hi].append((lo, v))
    E = np.empty(1 << n, dtype=np.float64)
    E[0] = 0.0
    for k in range(n):
        m = 1 << k
        add = np.full(m, H.h[k], dtype=np.float64)
        if nbr[k]:
            idx = np.arange(m, dtype=np.int64)
            for j, v in nbr[k]:
                add += v * ((idx >> j) & 1)
            del idx
        np.add(E[:m], add, out=E[m:2 * m])
        del add
    return E


def bits_of(index: np.ndarray, n: int) -> np.ndarray:
    """(m, n) uint8 matrix of x_i for each basis index."""
    index = np.asarray(index, dtype=np.int64)
    return ((index[:, None] >> np.arange(n)[None, :]) & 1).astype(np.uint8)


def index_of(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.int64)
    return bits @ (1 << np.arange(bits.shape[1], dtype=np.int64))


# ── statevector primitives ──────────────────────────────────────────────────

def _np_phase(psi, E, t):
    psi *= np.exp(-1j * t * E)


def _np_rx_all(psi, n, theta):
    c, s = np.cos(theta), -1j * np.sin(theta)
    for q in range(n):
        v = psi.reshape(1 << (n - q - 1), 2, 1 << q)
        a = v[:, 0, :].copy()
        b = v[:, 1, :]
        v[:, 0, :] = c * a + s * b
        v[:, 1, :] = s * a + c * b


try:                                    # numba: parallel, in place, no temporaries
    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def _nb_phase(psi, E, t):
        for i in prange(psi.shape[0]):
            a = -t * E[i]
            psi[i] *= complex(np.cos(a), np.sin(a))

    @njit(parallel=True, cache=True)
    def _nb_rx_all(psi, n, c, s_im):
        half = psi.shape[0] >> 1
        for q in range(n):
            stride = 1 << q
            mask = stride - 1
            for i in prange(half):
                lo = ((i >> q) << (q + 1)) | (i & mask)
                hi = lo | stride
                a = psi[lo]
                b = psi[hi]
                psi[lo] = c * a + 1j * s_im * b
                psi[hi] = 1j * s_im * a + c * b

    @njit(parallel=True, cache=True)
    def _nb_crx_all(psi, n, c, s_im, nb):
        half = psi.shape[0] >> 1
        for q in range(n):
            stride = 1 << q
            mask = stride - 1
            m = nb[q]
            for i in prange(half):
                lo = ((i >> q) << (q + 1)) | (i & mask)
                if lo & m:
                    continue
                hi = lo | stride
                a = psi[lo]
                b = psi[hi]
                psi[lo] = c * a + 1j * s_im * b
                psi[hi] = 1j * s_im * a + c * b

    @njit(parallel=True, cache=True)
    def _nb_ry_layer(psi, n, cs, ss):
        half = psi.shape[0] >> 1
        for q in range(n):
            stride = 1 << q
            mask = stride - 1
            c = cs[q]
            s = ss[q]
            for i in prange(half):
                lo = ((i >> q) << (q + 1)) | (i & mask)
                hi = lo | stride
                a = psi[lo]
                b = psi[hi]
                psi[lo] = c * a - s * b
                psi[hi] = s * a + c * b

    @njit(parallel=True, cache=True)
    def _nb_pauli_frame(psi, out, xmask, zmask):
        for i in prange(psi.shape[0]):
            j = i ^ xmask
            v = psi[j]
            c = j & zmask
            par = 0
            while c:
                c &= c - 1
                par ^= 1
            out[i] = -v if par else v

    HAVE_NUMBA = True
except ImportError:                      # pragma: no cover
    HAVE_NUMBA = False


def apply_pauli_frame(psi: np.ndarray, xmask: int, zmask: int) -> np.ndarray:
    """Return P psi for the Pauli P = X^xmask Z^zmask (Y = XZ up to a global
    phase).  Z acts first: (X Z psi)[i] = (-1)^{popcount(j & z)} psi[j], j = i ^ x."""
    if xmask == 0 and zmask == 0:
        return psi
    if HAVE_NUMBA:
        out = np.empty_like(psi)
        _nb_pauli_frame(psi, out, np.int64(xmask), np.int64(zmask))
        return out
    idx = np.arange(psi.shape[0], dtype=np.int64)
    j = idx ^ xmask
    par = np.zeros(psi.shape[0], dtype=np.int64)
    c = j & zmask
    while c.any():
        par ^= c & 1
        c >>= 1
    return np.where(par == 1, -psi[j], psi[j])


def step_error_probs(n: int, cx_per_qubit: np.ndarray, p2: float) -> np.ndarray:
    """Per-qubit probability of an X, of a Y and of a Z error per Trotter step
    when every CX touching the qubit carries a two-qubit depolarizing error
    p2: each of X, Y, Z on a given qubit has probability 4 p2 / 15 per CX
    (correlations between the two qubits ignored)."""
    a = 4.0 * p2 / 15.0
    pe = 1.0 - (1.0 - a) ** np.asarray(cx_per_qubit, float)
    return np.repeat(pe[:, None], 3, axis=1)


def krylov_states_noisy(E: np.ndarray, n: int, gamma: float, dt: float, r: int,
                        psi0: np.ndarray, err: np.ndarray,
                        rng: np.random.Generator,
                        nb: Optional[np.ndarray] = None):
    """One Pauli trajectory of the noisy Krylov circuits: after every
    Trotter step each qubit independently gets X / Y / Z with probabilities
    err[q] (from ``step_error_probs``).  Yields psi_k for k = 0..r-1 of this
    trajectory; average over trajectories for the noisy distribution."""
    psi = psi0.copy()
    yield psi.copy()
    for _ in range(1, r):
        apply_phase(psi, E, dt)
        if nb is None:
            apply_rx_all(psi, n, -gamma * dt)
        else:
            apply_crx_all(psi, n, -gamma * dt, nb)
        u = rng.random((n, 3))
        xm = zm = 0
        for q in range(n):
            if u[q, 0] < err[q, 0]:
                xm ^= 1 << q
            if u[q, 1] < err[q, 1]:
                xm ^= 1 << q
                zm ^= 1 << q
            if u[q, 2] < err[q, 2]:
                zm ^= 1 << q
        psi = apply_pauli_frame(psi, xm, zm)
        yield psi.copy()


def cx_per_qubit(H: CostHamiltonian) -> np.ndarray:
    """CX gates touching each qubit in one all-to-all Trotter step of the
    penalised cost layer (2 CX per ZZ term, on both of its qubits)."""
    c = np.zeros(H.n)
    for (a, b), v in H.J.items():
        if abs(v) > 1e-12:
            c[a] += 2
            c[b] += 2
    return c


def _np_crx_all(psi, n, theta, nb):
    c, s = np.cos(theta), -1j * np.sin(theta)
    idx = np.arange(psi.shape[0], dtype=np.int64)
    for q in range(n):
        lo = idx[(((idx >> q) & 1) == 0) & ((idx & nb[q]) == 0)]
        hi = lo | (1 << q)
        a, b = psi[lo].copy(), psi[hi].copy()
        psi[lo] = c * a + s * b
        psi[hi] = s * a + c * b


def _np_ry_layer(psi, n, thetas):
    for q in range(n):
        c, s = np.cos(thetas[q] / 2), np.sin(thetas[q] / 2)
        v = psi.reshape(1 << (n - q - 1), 2, 1 << q)
        a = v[:, 0, :].copy()
        b = v[:, 1, :]
        v[:, 0, :] = c * a - s * b
        v[:, 1, :] = s * a + c * b


def apply_phase(psi: np.ndarray, E: np.ndarray, t: float) -> None:
    """psi <- exp(-i t E) psi (in place)."""
    if HAVE_NUMBA:
        _nb_phase(psi, E, float(t))
    else:
        _np_phase(psi, E, t)


def apply_rx_all(psi: np.ndarray, n: int, theta: float) -> None:
    """psi <- prod_q exp(-i theta X_q) psi (in place)."""
    if HAVE_NUMBA:
        _nb_rx_all(psi, n, float(np.cos(theta)), float(-np.sin(theta)))
    else:
        _np_rx_all(psi, n, theta)


def apply_crx_all(psi: np.ndarray, n: int, theta: float, nb: np.ndarray) -> None:
    """psi <- prod_q exp(-i theta X_q P_q) psi, P_q = projector onto all of
    q's conflicting stems being 0 (the independent-set mixer of Hadfield et
    al. 2019), applied for q = 0..n-1 in order.  Preserves feasibility."""
    if HAVE_NUMBA:
        _nb_crx_all(psi, n, float(np.cos(theta)), float(-np.sin(theta)),
                    np.asarray(nb, dtype=np.int64))
    else:
        _np_crx_all(psi, n, theta, nb)


def apply_ry_layer(psi: np.ndarray, n: int, thetas: np.ndarray) -> None:
    """psi <- prod_q RY(theta_q) psi."""
    thetas = np.asarray(thetas, dtype=np.float64)
    if HAVE_NUMBA:
        _nb_ry_layer(psi, n, np.cos(thetas / 2), np.sin(thetas / 2))
    else:
        _np_ry_layer(psi, n, thetas)


def cz_chain_signs(n: int) -> np.ndarray:
    """Diagonal of the linear CZ chain CZ(0,1) CZ(1,2) ... CZ(n-2,n-1)."""
    idx = np.arange(1 << n, dtype=np.int64)
    par = np.zeros(1 << n, dtype=np.int8)
    for q in range(n - 1):
        par ^= (((idx >> q) & (idx >> (q + 1))) & 1).astype(np.int8)
    return 1.0 - 2.0 * par.astype(np.float64)


def two_local_state(n: int, params: np.ndarray, reps: int,
                    cz_signs: np.ndarray) -> np.ndarray:
    """Baseline VQE ansatz (WISER-Moderna ``build_two_local_ansatz``):
    reps x [RY layer, linear CZ chain], then a final RY layer."""
    psi = basis_state(n, 0)
    k = 0
    for _ in range(reps):
        apply_ry_layer(psi, n, params[k:k + n])
        k += n
        psi *= cz_signs
    apply_ry_layer(psi, n, params[k:k + n])
    return psi


def plus_state(n: int) -> np.ndarray:
    return np.full(1 << n, (1 << n) ** -0.5, dtype=np.complex128)


def basis_state(n: int, index: int = 0) -> np.ndarray:
    psi = np.zeros(1 << n, dtype=np.complex128)
    psi[index] = 1.0
    return psi


# ── SKQD: Krylov states of H(G) ─────────────────────────────────────────────

def krylov_states(E: np.ndarray, n: int, gamma: float, dt: float, r: int,
                  psi0: np.ndarray, trotter_steps: int = 1,
                  nb: Optional[np.ndarray] = None):
    """Yield psi_k = U^k psi0 for k = 0..r-1, U = (e^{-i dt/m H_C}
    e^{+i G dt/m sum X})^m, m = trotter_steps (first-order Lie-Trotter, Z
    terms first, matching qiskit LieTrotter with the Z terms listed first).
    Times are in units of 1/scale (callers pass E already scaled).
    With ``nb`` (conflict masks) the drive is the constrained mixer
    -G sum_q X_q P_q instead of -G sum_q X_q."""
    psi = psi0.copy()
    yield psi.copy()
    tau = dt / trotter_steps
    for _ in range(1, r):
        for _ in range(trotter_steps):
            apply_phase(psi, E, tau)
            if nb is None:
                apply_rx_all(psi, n, -gamma * tau)
            else:
                apply_crx_all(psi, n, -gamma * tau, nb)
        yield psi.copy()


# ── linear-ramp QAOA (fixed schedule, no variational loop) ─────────────────

def lr_qaoa_state(E: np.ndarray, n: int, p: int, d_gamma: float,
                  d_beta: float) -> np.ndarray:
    """|+>^n, then for k = 1..p: exp(-i g_k H_C) then exp(-i b_k sum X),
    with the linear ramp g_k = (k / p) d_gamma, b_k = (1 - (k - 1) / p) d_beta
    (fixed schedule in the style of Montanez-Barrera et al. 2024, no
    variational optimisation).  E is the scaled cost vector."""
    psi = plus_state(n)
    for k in range(1, p + 1):
        g = (k / p) * d_gamma
        b = (1.0 - (k - 1) / p) * d_beta
        apply_phase(psi, E, g)
        apply_rx_all(psi, n, b)
    return psi


# ── sampling ────────────────────────────────────────────────────────────────

def sample(psi: np.ndarray, shots: int, rng: np.random.Generator) -> np.ndarray:
    """Basis indices drawn from |psi|^2."""
    p = np.abs(psi) ** 2
    c = np.cumsum(p)
    c /= c[-1]
    return np.searchsorted(c, rng.random(shots), side='right').astype(np.int64)


# ── Qiskit builders (cross-checks and resource counts) ─────────────────────

def pauli_cost(H: CostHamiltonian, scale: float = 1.0):
    """H_C / scale as a SparsePauliOp (x_i = (1 - Z_i) / 2); Z terms only."""
    from qiskit.quantum_info import SparsePauliOp
    n = H.n
    terms: Dict[str, float] = {}

    def add(label_pos: Tuple[int, ...], c: float):
        lab = ['I'] * n
        for qb in label_pos:
            lab[n - 1 - qb] = 'Z'
        key = ''.join(lab)
        terms[key] = terms.get(key, 0.0) + c

    for i in range(n):
        add((), H.h[i] / 2 / scale)
        add((i,), -H.h[i] / 2 / scale)
    for (a, b), v in H.J.items():
        w = v / 4 / scale
        add((), w)
        add((a,), -w)
        add((b,), -w)
        add((a, b), w)
    items = [(k, v) for k, v in terms.items() if abs(v) > 1e-12]
    return SparsePauliOp.from_list(items)


def pauli_drive(n: int, gamma: float):
    from qiskit.quantum_info import SparsePauliOp
    return SparsePauliOp.from_list(
        [(''.join('X' if n - 1 - k == q else 'I' for k in range(n)), -gamma)
         for q in range(n)])


def qiskit_krylov_circuit(H: CostHamiltonian, gamma: float, dt: float, k: int,
                          trotter_steps: int = 1, init_index: int = 0):
    """Reference basis state, then k applications of the Trotterised U."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.synthesis import LieTrotter
    n = H.n
    op = pauli_cost(H, H.scale) + pauli_drive(n, gamma)
    qc = QuantumCircuit(n)
    for q in range(n):
        if (init_index >> q) & 1:
            qc.x(q)
    if k:
        gate = PauliEvolutionGate(op, time=dt / trotter_steps,
                                  synthesis=LieTrotter(reps=1))
        for _ in range(k * trotter_steps):
            qc.append(gate, range(n))
    return qc


def qiskit_constrained_krylov_circuit(H0: CostHamiltonian, nb: np.ndarray,
                                      gamma: float, dt: float, k: int,
                                      init_index: int = 0):
    """Reference basis state, then k Trotter steps of the constrained drive:
    exp(-i dt H0/scale) (H0 without penalties), then for q = 0..n-1 an
    RX(-2 G dt) on q controlled on all of q's conflicting stems being 0."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate, RXGate
    n = H0.n
    qc = QuantumCircuit(n)
    for q in range(n):
        if (init_index >> q) & 1:
            qc.x(q)
    cost = pauli_cost(H0, H0.scale)
    for _ in range(k):
        qc.append(PauliEvolutionGate(cost, time=dt), range(n))
        for q in range(n):
            ctrls = [j for j in range(n) if (int(nb[q]) >> j) & 1]
            if ctrls:
                gate = RXGate(-2 * gamma * dt).control(
                    len(ctrls), ctrl_state='0' * len(ctrls))
                qc.append(gate, ctrls + [q])
            else:
                qc.rx(-2 * gamma * dt, q)
    return qc


def qiskit_two_local_circuit(n: int, params: np.ndarray, reps: int):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    k = 0
    for _ in range(reps):
        for q in range(n):
            qc.ry(float(params[k]), q)
            k += 1
        for q in range(n - 1):
            qc.cz(q, q + 1)
    for q in range(n):
        qc.ry(float(params[k]), q)
        k += 1
    return qc


def qiskit_lr_qaoa_circuit(H: CostHamiltonian, p: int, d_gamma: float,
                           d_beta: float):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    n = H.n
    cost = pauli_cost(H, H.scale)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for k in range(1, p + 1):
        g = (k / p) * d_gamma
        b = (1.0 - (k - 1) / p) * d_beta
        qc.append(PauliEvolutionGate(cost, time=g), range(n))
        for q in range(n):
            qc.rx(2 * b, q)
    return qc
