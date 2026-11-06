import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrnasqd.qubo_v2 import build_stem_qubo, cpsat_solve  # noqa: E402
from mrnasqd import quantum as Qm  # noqa: E402

SEQS = ['GCAAUUGCUGCAUAAUGGCACAU', 'GGCGCGGCACCGUCCGCGGAACAAACGG',
        'GGGACUUCGGUCCCAAAGGCAGUAAUGCCAAAA']


def small(seq, k=10):
    return build_stem_qubo(seq, model='turner_direct', pk_beta=0.0,
                           max_stems=k)


@pytest.mark.parametrize('seq', SEQS)
def test_energy_vector_matches_direct(seq):
    q = small(seq, 10)
    H = Qm.cost_hamiltonian(q)
    E = Qm.energy_vector(H)
    rng = np.random.default_rng(1)
    for idx in rng.integers(0, 1 << q.n, 200):
        x = Qm.bits_of(np.array([idx]), q.n)[0]
        assert abs(E[idx] - H.energy(x)) < 1e-9


@pytest.mark.parametrize('seq', SEQS)
def test_penalty_makes_global_min_the_constrained_optimum(seq):
    q = small(seq, 14)
    H = Qm.cost_hamiltonian(q)
    E = Qm.energy_vector(H)
    r = cpsat_solve(q)
    assert r.optimal
    assert abs(E.min() - r.energy) < 1e-6
    xmin = Qm.bits_of(np.array([int(E.argmin())]), q.n)[0]
    assert np.isfinite(q.energy(xmin))          # feasible


def _qiskit_probs(qc):
    # Transpile first: Statevector would otherwise evaluate a
    # PauliEvolutionGate as an exact matrix exponential instead of the
    # Trotter circuit a device runs.
    from qiskit import transpile
    from qiskit.quantum_info import Statevector
    qc = transpile(qc, basis_gates=['cx', 'rz', 'rx', 'h', 'x', 'sx'],
                   optimization_level=0)
    return np.abs(Statevector(qc).data) ** 2


def test_krylov_matches_qiskit_trotter_circuit():
    q = small(SEQS[0], 8)
    H = Qm.cost_hamiltonian(q)
    E = Qm.energy_vector(H) / H.scale
    gamma, dt = 0.7, 0.3
    states = list(Qm.krylov_states(E, q.n, gamma, dt, r=4,
                                   psi0=Qm.basis_state(q.n, 0), trotter_steps=2))
    for k in (0, 1, 3):
        qc = Qm.qiskit_krylov_circuit(H, gamma, dt, k, trotter_steps=2)
        assert np.allclose(np.abs(states[k]) ** 2, _qiskit_probs(qc), atol=1e-10)


def test_lr_qaoa_matches_qiskit_circuit():
    q = small(SEQS[1], 8)
    H = Qm.cost_hamiltonian(q)
    E = Qm.energy_vector(H) / H.scale
    psi = Qm.lr_qaoa_state(E, q.n, p=3, d_gamma=0.4, d_beta=0.6)
    qc = Qm.qiskit_lr_qaoa_circuit(H, p=3, d_gamma=0.4, d_beta=0.6)
    assert np.allclose(np.abs(psi) ** 2, _qiskit_probs(qc), atol=1e-10)


def test_sampling_distribution():
    rng = np.random.default_rng(0)
    psi = np.sqrt(np.array([0.1, 0.2, 0.3, 0.4], complex))
    s = Qm.sample(psi, 200000, rng)
    freq = np.bincount(s, minlength=4) / len(s)
    assert np.allclose(freq, [0.1, 0.2, 0.3, 0.4], atol=0.01)


def test_solve_qubit_matches_own_projection():
    from mrnasqd import sqd as S
    q = small(SEQS[2], 12)
    H = Qm.cost_hamiltonian(q)
    E = Qm.energy_vector(H) / H.scale
    rng = np.random.default_rng(3)
    idx = np.unique(rng.integers(0, 1 << q.n, 400))
    for g in (0.05, 0.5):
        a = S.sqd_diagonalise(idx, E, q.n, g, use_addon=True, H=H)
        b = S.sqd_diagonalise(idx, E, q.n, g, use_addon=False)
        assert abs(a[1] - b[1]) < 1e-8
        assert a[0] == b[0]


def test_repair_makes_feasible_and_sa_finds_optimum():
    from mrnasqd import sqd as S
    q = small(SEQS[0], 12)
    H = Qm.cost_hamiltonian(q)
    E = Qm.energy_vector(H) / H.scale
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 1 << q.n, 500)
    rep = S.repair(idx, H, q.conflicts)
    for v in rep:
        x = Qm.bits_of(np.array([v]), q.n)[0]
        assert np.isfinite(q.energy(x))
    x, _ = S.sample_sa(E, q.n, 64, rng, sweeps=60)
    assert E[x].min() <= E.min() + 1e-9


def test_numba_kernels_match_numpy():
    if not Qm.HAVE_NUMBA:
        pytest.skip('numba not installed')
    rng = np.random.default_rng(5)
    n = 11
    psi = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    E = rng.normal(size=1 << n)
    a, b = psi.copy(), psi.copy()
    Qm._np_phase(a, E, 0.37); Qm.apply_phase(b, E, 0.37)
    assert np.allclose(a, b)
    Qm._np_rx_all(a, n, 0.61); Qm.apply_rx_all(b, n, 0.61)
    assert np.allclose(a, b)


def _feasible_mask(q):
    nb = Qm.conflict_masks(q)
    idx = np.arange(1 << q.n, dtype=np.int64)
    ok = np.ones(1 << q.n, bool)
    for i in range(q.n):
        ok &= ~((((idx >> i) & 1) == 1) & ((idx & nb[i]) != 0))
    return ok


def test_constrained_drive_preserves_feasibility_and_matches_qiskit():
    q = small(SEQS[0], 7)
    H = Qm.cost_hamiltonian(q)
    H0 = Qm.unpenalised(H, q)
    E0 = Qm.energy_vector(H0) / H0.scale
    nb = Qm.conflict_masks(q)
    states = list(Qm.krylov_states(E0, q.n, 0.8, 0.4, r=3,
                                   psi0=Qm.basis_state(q.n, 0), nb=nb))
    ok = _feasible_mask(q)
    for psi in states:
        assert np.sum(np.abs(psi[~ok]) ** 2) < 1e-12
    for k in (1, 2):
        qc = Qm.qiskit_constrained_krylov_circuit(H0, nb, 0.8, 0.4, k)
        assert np.allclose(np.abs(states[k]) ** 2, _qiskit_probs(qc), atol=1e-9)


def test_two_local_matches_qiskit():
    rng = np.random.default_rng(2)
    n, reps = 6, 2
    th = rng.uniform(-np.pi, np.pi, n * (reps + 1))
    psi = Qm.two_local_state(n, th, reps, Qm.cz_chain_signs(n))
    qc = Qm.qiskit_two_local_circuit(n, th, reps)
    assert np.allclose(np.abs(psi) ** 2, _qiskit_probs(qc), atol=1e-10)


def test_crx_and_ry_numba_match_numpy():
    if not Qm.HAVE_NUMBA:
        pytest.skip('numba not installed')
    rng = np.random.default_rng(6)
    n = 9
    nb = rng.integers(0, 1 << n, n).astype(np.int64)
    for q in range(n):
        nb[q] &= ~np.int64(1 << q)
    psi = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    a, b = psi.copy(), psi.copy()
    Qm._np_crx_all(a, n, 0.3, nb); Qm.apply_crx_all(b, n, 0.3, nb)
    assert np.allclose(a, b)
    th = rng.uniform(-3, 3, n)
    Qm._np_ry_layer(a, n, th); Qm.apply_ry_layer(b, n, th)
    assert np.allclose(a, b)


def test_noisy_sample_limits():
    from mrnasqd import sqd as S
    rng = np.random.default_rng(1)
    n = 4
    psi = Qm.basis_state(n, 5)
    assert (S.noisy_sample(psi, 2000, rng, n, 1.0, 0.0) == 5).all()
    u = S.noisy_sample(psi, 40000, rng, n, 0.0, 0.0)
    assert np.allclose(np.bincount(u, minlength=16) / 40000, 1 / 16, atol=0.01)
    f = S.noisy_sample(psi, 40000, rng, n, 1.0, 0.1)
    assert abs((f == 5).mean() - 0.9 ** 4) < 0.01


def test_warm_start_is_feasible():
    from mrnasqd import sqd as S
    for seq in SEQS:
        q = small(seq, 14)
        w = S.warm_start_index(q, seq)
        x = Qm.bits_of(np.array([w]), q.n)[0]
        assert np.isfinite(q.energy(x))


def test_constrained_projection_couples_feasible_only():
    from mrnasqd import sqd as S
    q = small(SEQS[1], 8)
    H = Qm.cost_hamiltonian(q)
    E = Qm.energy_vector(H) / H.scale
    nb = Qm.conflict_masks(q)
    ok = _feasible_mask(q)
    idx = np.arange(1 << q.n, dtype=np.int64)[ok]
    M = S.project_drive(idx, E, q.n, 0.5, nb).tocoo()
    for r_, c_ in zip(M.row, M.col):
        if r_ != c_:
            assert ok[idx[r_]] and ok[idx[c_]]


def test_pauli_frame_matches_dense_operator():
    from functools import reduce
    rng = np.random.default_rng(4)
    n = 4
    psi = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    I2 = np.eye(2); X = np.array([[0, 1], [1, 0]]); Z = np.diag([1, -1])
    for xm, zm in [(0b0101, 0), (0, 0b0011), (0b1001, 0b1100)]:
        ops = []
        for q in reversed(range(n)):          # kron order: qubit n-1 first
            m = I2
            if (zm >> q) & 1:
                m = Z @ m
            if (xm >> q) & 1:
                m = X @ m
            ops.append(m)
        P = reduce(np.kron, ops)
        assert np.allclose(Qm.apply_pauli_frame(psi, xm, zm), P @ psi)


def test_noisy_trajectories_without_errors_are_ideal():
    q = small(SEQS[0], 8)
    H = Qm.cost_hamiltonian(q)
    E = Qm.energy_vector(H) / H.scale
    err = np.zeros((q.n, 3))
    rng = np.random.default_rng(0)
    a = list(Qm.krylov_states(E, q.n, 0.5, 0.3, 4, Qm.basis_state(q.n, 0)))
    b = list(Qm.krylov_states_noisy(E, q.n, 0.5, 0.3, 4, Qm.basis_state(q.n, 0), err, rng))
    for x, y in zip(a, b):
        assert np.allclose(x, y)
