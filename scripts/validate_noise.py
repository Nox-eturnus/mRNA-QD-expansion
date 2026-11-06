"""Validate the global white-noise model used for the noisy runs.

For small instances (8 qubits, pruned M3 on short Q-dev chains) the SKQD
Krylov circuits (k = 1..3 Trotter steps, transverse drive) are transpiled to
{cx, rz, sx, x} and simulated with Aer's density-matrix method under a
depolarizing error p2 on every CX (and p2/10 on every single-qubit gate).
The exact noisy output distribution is compared with two models:
  * white noise: P_model = F P_ideal + (1 - F) U,  F = (1 - p2)^N_cx;
  * Pauli trajectories (quantum.krylov_states_noisy): X/Y/Z errors after
    each Trotter step with per-qubit rates from the CX count;
by total variation distance and by the probability of the optimum.

    python scripts/validate_noise.py --out results/noise_validation.jsonl
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mrnasqd import DATA_DIR  # noqa: E402


def main():
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    from mrnasqd.qubo_v2 import build_stem_qubo
    from mrnasqd import quantum as Qm

    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/noise_validation.jsonl')
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--chains', type=int, default=4)
    ap.add_argument('--trajectories', type=int, default=400)
    args = ap.parse_args()
    qs = json.load(open(os.path.join(DATA_DIR, 'qset.json')))
    seqs = {json.loads(l)['id']: json.loads(l)['sequence']
            for l in open(os.path.join(DATA_DIR, 'pdb_fr3d_chains.jsonl'))}
    items = qs['sets']['qdev'][:args.chains]
    rows = []
    for it in items:
        q = build_stem_qubo(seqs[it['id']], model='turner_direct', pk_beta=0.0,
                            c_far=0.0, max_stems=args.n)
        H = Qm.cost_hamiltonian(q)
        E = Qm.energy_vector(H) / H.scale
        x_opt = int(E.argmin())
        U = np.full(1 << q.n, 1.0 / (1 << q.n))
        states = list(Qm.krylov_states(E, q.n, 0.5, 0.3, 4, Qm.basis_state(q.n, 0)))
        base_cx = Qm.cx_per_qubit(H)
        step1 = transpile(Qm.qiskit_krylov_circuit(H, 0.5, 0.3, 1),
                          basis_gates=['cx', 'rz', 'sx', 'x'],
                          optimization_level=1, seed_transpiler=0)
        n_cx1 = int(step1.count_ops().get('cx', 0))
        cxq = base_cx * (2 * n_cx1 / max(base_cx.sum(), 1))   # match the real CX count
        traj_cache = {}
        for k in (1, 2, 3):
            qc = Qm.qiskit_krylov_circuit(H, 0.5, 0.3, k)
            tq = transpile(qc, basis_gates=['cx', 'rz', 'sx', 'x'],
                           optimization_level=1, seed_transpiler=0)
            n_cx = int(tq.count_ops().get('cx', 0))
            p_ideal = np.abs(states[k]) ** 2
            for p2 in (1e-3, 3e-3, 1e-2):
                nm = NoiseModel()
                nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ['cx'])
                nm.add_all_qubit_quantum_error(depolarizing_error(p2 / 10, 1),
                                               ['sx', 'x'])
                sim = AerSimulator(method='density_matrix', noise_model=nm)
                c2 = tq.copy()
                c2.save_density_matrix()
                rho = sim.run(c2).result().data(0)['density_matrix']
                p_noisy = np.real(np.diag(np.asarray(rho)))
                F = (1 - p2) ** n_cx
                p_model = F * p_ideal + (1 - F) * U
                if p2 not in traj_cache:
                    err = Qm.step_error_probs(q.n, cxq, p2)
                    rng = np.random.default_rng(7)
                    acc = np.zeros((4, 1 << q.n))
                    M = args.trajectories
                    for _ in range(M):
                        for kk, psi in enumerate(Qm.krylov_states_noisy(
                                E, q.n, 0.5, 0.3, 4, Qm.basis_state(q.n, 0), err, rng)):
                            acc[kk] += np.abs(psi) ** 2
                    traj_cache[p2] = acc / M
                p_traj = traj_cache[p2][k]
                rows.append({
                    'id': it['id'], 'n_qubits': q.n, 'k': k, 'n_cx': n_cx, 'p2': p2,
                    'F_model': F,
                    'tvd_model': 0.5 * float(np.abs(p_noisy - p_model).sum()),
                    'tvd_ideal': 0.5 * float(np.abs(p_noisy - p_ideal).sum()),
                    'tvd_traj': 0.5 * float(np.abs(p_noisy - p_traj).sum()),
                    'p_opt_traj': float(p_traj[x_opt]),
                    'p_opt_ideal': float(p_ideal[x_opt]),
                    'p_opt_noisy': float(p_noisy[x_opt]),
                    'p_opt_model': float(p_model[x_opt])})
                print(json.dumps({k2: (round(v, 4) if isinstance(v, float) else v)
                                  for k2, v in rows[-1].items()}), flush=True)
    with open(args.out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


if __name__ == '__main__':
    main()
