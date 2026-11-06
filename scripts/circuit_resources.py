"""Circuit resources for the quantum circuits on every Q-set / Q-dev chain.

Per chain: one SKQD Trotter step U = exp(-i dt H_C) exp(+i G dt sum X), one
constrained SKQD step (penalty-free H_C phase + conflict-controlled RX gates),
one LR-QAOA layer and the baseline VQE ansatz (reps 1), transpiled (optimization_level=1) for
  * an all-to-all device, basis {cx, rz, sx, x};
  * IBM heavy-hex: FakeSherbrooke (127 qubits, ECR), with routing.
Gate counts per step do not depend on G, dt or the ramp values, only on the
Hamiltonian's terms.  A Krylov state k needs k steps; an LR-QAOA state p layers.

    python scripts/circuit_resources.py --out results/resources.jsonl
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mrnasqd import DATA_DIR  # noqa: E402


def work(item, seq, budget):
    from qiskit import transpile
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
    from mrnasqd.qubo_v2 import build_stem_qubo
    from mrnasqd import quantum as Qm
    q = build_stem_qubo(seq, model='turner_direct', mode='truncations',
                        min_len=2, max_stems=budget, pk_beta=0.0, c_far=0.0)
    H = Qm.cost_hamiltonian(q)
    out = {'id': item['id'], 'pk_kind': item['pk_kind'], 'n_qubits': q.n,
           'n_zz_terms': sum(1 for k in Qm.pauli_cost(H).paulis.to_labels()
                             if k.count('Z') == 2)}
    import numpy as np
    nb = Qm.conflict_masks(q)
    circs = {'skqd_step': Qm.qiskit_krylov_circuit(H, 0.5, 0.3, 1),
             'qaoa_layer': Qm.qiskit_lr_qaoa_circuit(H, 1, 1.0, 0.6),
             'cskqd_step': Qm.qiskit_constrained_krylov_circuit(
                 Qm.unpenalised(H, q), nb, 0.5, 0.3, 1),
             'vqe_reps1': Qm.qiskit_two_local_circuit(q.n, np.full(2 * q.n, 0.3), 1)}
    out['conflict_degree_max'] = max(bin(int(m)).count('1') for m in nb)
    backend = FakeSherbrooke()
    for name, qc in circs.items():
        a2a = transpile(qc, basis_gates=['cx', 'rz', 'sx', 'x'],
                        optimization_level=1, seed_transpiler=0)
        ops = a2a.count_ops()
        out[f'{name}_a2a_2q'] = int(ops.get('cx', 0))
        out[f'{name}_a2a_depth'] = a2a.depth()
        hh = transpile(qc, backend=backend, optimization_level=1,
                       seed_transpiler=0)
        ops = hh.count_ops()
        out[f'{name}_hh_2q'] = int(ops.get('ecr', 0))
        out[f'{name}_hh_depth'] = hh.depth()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/resources.jsonl')
    ap.add_argument('--procs', type=int, default=4)
    args = ap.parse_args()
    qs = json.load(open(os.path.join(DATA_DIR, 'qset.json')))
    items = qs['sets']['qset'] + qs['sets']['qdev']
    seqs = {}
    for line in open(os.path.join(DATA_DIR, 'pdb_fr3d_chains.jsonl')):
        d = json.loads(line)
        seqs[d['id']] = d['sequence']
    with ProcessPoolExecutor(args.procs) as pool, open(args.out, 'w') as f:
        for k, row in enumerate(pool.map(work, items, [seqs[i['id']] for i in items],
                                         [qs['budget']] * len(items)), 1):
            f.write(json.dumps(row) + '\n')
            if k % 20 == 0:
                print(f'  {k}/{len(items)}', flush=True)


if __name__ == '__main__':
    main()
