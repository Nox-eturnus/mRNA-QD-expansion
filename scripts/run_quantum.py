"""SQD / SKQD and controls on the Q-set (or Q-dev) chains.

Each chain uses the M3 model pruned to ``--budget`` stems (default: the
Q-set budget in data/qset.json; parameters from results/m3_params.json).
The exact optimum comes from enumerating all 2^n energies.  For every
(chain, method, seed) the largest shot budget is drawn once; smaller budgets
use its prefix (iid draws, so a prefix is a valid smaller sample).  Every
budget goes through the same post-processing (mrnasqd.sqd.postprocess) and
five readouts are scored:
  best      lowest H_C among samples (= SQD on the diagonal H_C)
  sqd       largest-weight bitstring of the H(G_d) ground state in the sample span
  rep_best  lowest H_C after repair only (classical)
  rec_best  lowest H_C after repair + single-flip expansion (classical)
  rec       sqd readout on the expanded subspace

Methods: skqd (``--drive x`` transverse field on the penalised H_C, or
``--drive cx`` the feasibility-preserving conflict-controlled mixer on the
penalty-free H_C), lrqaoa, vqe (baseline two-local ansatz + COBYLA), random,
sa.  ``--warm`` starts SKQD / SA from the ViennaRNA MFE structure mapped onto
the candidate stems.

``--noise "lab:dev:p2:eps[:model];..."`` (quantum samplers, 24-qubit budget
only) adds, for each entry, a noisy sample set.  model ``white`` (default):
samples from the same states with sqd.noisy_sample, circuit fidelity
(1 - p2)^N2q, N2q from results/resources.jsonl for device ``dev``
(a2a | hh) times the number of steps / layers.  model ``traj`` (SKQD only):
``--trajectories`` Pauli trajectories per seed (sqd.sample_skqd_traj), per-
qubit error rates from the same CX counts.  Readout flips with prob eps.
Rows get tag ``<tag>|<lab>``.

    python scripts/run_quantum.py --set qdev --method skqd --gamma 0.5 --dt 0.3 \
        --r 10 --tag g0.5_dt0.3 --out results/qdev.jsonl
Run in WSL; set NUMBA_NUM_THREADS so procs x threads <= cores.
"""

import argparse
import json
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mrnasqd import DATA_DIR  # noqa: E402

ROOT = os.path.dirname(DATA_DIR)
CIRCUIT_KEY = {('skqd', 'x'): 'skqd_step', ('skqd', 'cx'): 'cskqd_step',
               ('lrqaoa', 'x'): 'qaoa_layer', ('vqe', 'x'): 'vqe_reps1'}


def model_params():
    p = json.load(open(os.path.join(ROOT, 'results', 'm3_params.json')))
    return {k: p[k] for k in ('c_far', 'pk_beta', 'ml_sibling')}


def parse_noise(spec):
    """'lab:dev:p2:eps[:white|traj]' entries separated by ';'."""
    out = []
    for part in [p for p in (spec or '').split(';') if p.strip()]:
        f = part.split(':')
        model = f[4] if len(f) > 4 else 'white'
        out.append((f[0], f[1], float(f[2]), float(f[3]), model))
    return out


def work(item, rec, budget, args, resources):
    import numpy as np
    from mrnasqd.qubo_v2 import build_stem_qubo
    from mrnasqd import quantum as Qm, sqd as S
    from mrnasqd.metrics import structure_metrics

    t0 = time.time()
    q = build_stem_qubo(rec['sequence'], model='turner_direct',
                        mode='truncations', min_len=2, max_stems=budget,
                        **model_params())
    H = Qm.cost_hamiltonian(q)
    E = Qm.energy_vector(H) / H.scale
    x_opt = int(E.argmin())
    e_opt = float(E[x_opt])
    truth = [tuple(p) for p in rec['pairs']]
    n_nt = rec['length']
    nb = Qm.conflict_masks(q) if args.drive == 'cx' else None
    E_dyn = E
    if args.drive == 'cx' and args.method in ('skqd', 'lrqaoa'):
        E_dyn = Qm.energy_vector(Qm.unpenalised(H, q)) / H.scale
    psi0 = S.warm_start_index(q, rec['sequence']) if args.warm else 0

    def score(index):
        x = Qm.bits_of(np.array([index]), q.n)[0]
        feas = bool(np.isfinite(q.energy(x)))
        m = structure_metrics(q.pairs(x), truth, n_nt) if feas else {}
        gap = (float(E[index]) - e_opt) * H.scale
        # an infeasible readout is not a structure: it scores 0 on every metric
        return {'mcc': m.get('mcc', 0.0), 'sens': m.get('sensitivity', 0.0),
                'ppv': m.get('ppv', 0.0), 'gap': round(gap, 4),
                'hit': bool(gap < 1e-6), 'feasible': feas}

    opt_score = score(x_opt)
    opt_mcc = opt_score['mcc']
    warm_score = score(psi0)
    shots_list = sorted(int(s) for s in args.shots.split(','))
    smax = shots_list[-1]
    rngs = [np.random.default_rng([args.seed0, seed, zlib.crc32(rec['id'].encode())])
            for seed in range(args.seeds)]

    # noise specs -> fidelity functions of the step / layer count
    noise, traj = [], []
    quantum = args.method in ('skqd', 'lrqaoa', 'vqe')
    if args.noise and quantum:
        res = resources.get(item['id'])
        if res is None or res['n_qubits'] != q.n:
            raise RuntimeError(f'no resources for {item["id"]} at {q.n} qubits')
        ckey = CIRCUIT_KEY[(args.method, args.drive)]
        for lab, dev, p2, eps, model in parse_noise(args.noise):
            n2q = res[f'{ckey}_{dev}_2q']
            if model == 'traj':
                if args.method != 'skqd':
                    continue
                # per-qubit CX per step: penalised layer's pattern (transverse
                # drive) or uniform (constrained drive), scaled to the
                # transpiled count for this device
                base = Qm.cx_per_qubit(H) if args.drive == 'x' else np.ones(q.n)
                cxq = base * (2.0 * n2q / max(base.sum(), 1e-9))
                traj.append((lab, Qm.step_error_probs(q.n, cxq, p2), eps))
                continue
            if args.method == 'skqd':
                fid = (lambda k, n2q=n2q, p2=p2: (1 - p2) ** (k * n2q))
            elif args.method == 'lrqaoa':
                fid = (lambda k, n2q=n2q, p2=p2: (1 - p2) ** (args.p * n2q))
            else:
                fid = (lambda k, n2q=n2q, p2=p2: (1 - p2) ** (args.reps * n2q))
            noise.append((lab, fid, eps))

    # sets[label][seed] = {shots: indices}
    sets = {'': [None] * len(rngs)}
    for lab, _, _ in noise + traj:
        sets[lab] = [None] * len(rngs)
    state_info = {}
    extra_rows = [{} for _ in rngs]
    if args.method == 'skqd':
        ideal, info = S.sample_skqd(E_dyn, q.n, args.gamma, args.dt, args.r, smax,
                                    rngs, psi0_index=psi0,
                                    trotter_steps=args.trotter_steps,
                                    x_opt=x_opt, nb=nb, noise=noise)
        def prefixes(idx):
            per = (len(idx) - 1) // (args.r - 1)
            ref, rest = idx[:1], idx[1:].reshape(args.r - 1, per)
            return {s: np.concatenate([ref, rest[:, :max(1, s * per // smax)].ravel()])
                    for s in shots_list}
        for seed in range(len(rngs)):
            sets[''][seed] = prefixes(ideal[seed])
            for lab, _, _ in noise:
                sets[lab][seed] = prefixes(info['noisy'][lab][seed])
        for lab, err, eps in traj:
            for seed, rng in enumerate(rngs):
                tidx = S.sample_skqd_traj(E_dyn, q.n, args.gamma, args.dt, args.r,
                                          smax, rng, psi0, err, args.trajectories,
                                          eps, nb)
                sets[lab][seed] = prefixes(tidx)
        state_info = {'p_opt_mean': float(np.mean(info['p_opt_k'])),
                      'p_opt_max': float(np.max(info['p_opt_k']))}
    elif args.method in ('lrqaoa', 'vqe'):
        psi = None
        for seed, rng in enumerate(rngs):
            if args.method == 'lrqaoa':
                if psi is None:
                    psi = Qm.lr_qaoa_state(E_dyn, q.n, args.p, args.d_gamma, args.d_beta)
                    state_info = {'p_opt': float(abs(psi[x_opt]) ** 2)}
            else:
                psi, vinfo = S.run_vqe(E, q.n, args.reps, rng, maxiter=args.maxiter)
                extra_rows[seed] = {'p_opt': float(abs(psi[x_opt]) ** 2),
                                    'vqe_energy': vinfo['vqe_energy'] * H.scale,
                                    'vqe_evals': vinfo['n_evals'],
                                    **{f'vqe_argmax_{k}': v
                                       for k, v in score(vinfo['argmax']).items()}}
            idx = Qm.sample(psi, smax, rng)
            sets[''][seed] = {s: idx[:s] for s in shots_list}
            for lab, fid, eps in noise:
                nidx = S.noisy_sample(psi, smax, rng, q.n, fid(0), eps)
                sets[lab][seed] = {s: nidx[:s] for s in shots_list}
            if args.method == 'vqe':
                psi = None
    else:
        for seed, rng in enumerate(rngs):
            if args.method == 'random':
                idx, _ = S.sample_random(q.n, smax, rng)
            elif args.method == 'sa':
                idx, _ = S.sample_sa(E, q.n, smax, rng, sweeps=args.sweeps,
                                     init=psi0 if args.warm else None)
            else:
                raise ValueError(args.method)
            sets[''][seed] = {s: idx[:s] for s in shots_list}

    rows = []
    for lab, per_seed in sets.items():
        for seed, pref in enumerate(per_seed):
            for s in shots_list:
                pp = S.postprocess(pref[s], E, H, q.conflicts, gamma_d=args.gamma_d,
                                   keep=args.keep, rounds=args.rounds,
                                   use_addon=args.use_addon, nb=nb)
                row = {'id': item['id'], 'pk_kind': item['pk_kind'], 'n_qubits': q.n,
                       'method': args.method, 'tag': args.tag + (f'|{lab}' if lab else ''),
                       'drive': args.drive, 'warm': bool(args.warm), 'noise': lab,
                       'shots': s, 'seed': seed,
                       'ref_cpsat_mcc': item['ref_cpsat_mcc'] if budget == args.qset_budget else None,
                       'opt_mcc': opt_mcc, 'opt_sens': opt_score['sens'],
                       'opt_ppv': opt_score['ppv'], 'warm_mcc': warm_score['mcc'],
                       'warm_gap': warm_score['gap'],
                       'n_unique': pp['n_unique'], 'feasible_frac': pp['feasible_frac'],
                       'rec_dim': pp['rec_dim'], 'sqd_w': pp['sqd_w'], **state_info,
                       **extra_rows[seed]}
                for key in ('best', 'sqd', 'rep_best', 'rec_best', 'rec'):
                    for k2, v2 in score(pp[key]).items():
                        row[f'{key}_{k2}'] = v2
                row['seconds'] = round(time.time() - t0, 1)
                rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='qdev', choices=['qdev', 'qset'])
    ap.add_argument('--method', required=True,
                    choices=['skqd', 'lrqaoa', 'vqe', 'random', 'sa'])
    ap.add_argument('--tag', default='')
    ap.add_argument('--drive', default='x', choices=['x', 'cx'])
    ap.add_argument('--warm', action='store_true')
    ap.add_argument('--budget', type=int, default=0,
                    help='stem budget (qubits); default: the Q-set budget')
    ap.add_argument('--min-stems', type=int, default=0,
                    help='only chains with at least this many candidate stems')
    ap.add_argument('--gamma', type=float, default=0.5)
    ap.add_argument('--dt', type=float, default=0.3)
    ap.add_argument('--r', type=int, default=10)
    ap.add_argument('--trotter-steps', type=int, default=1)
    ap.add_argument('--p', type=int, default=8)
    ap.add_argument('--d-gamma', type=float, default=1.0)
    ap.add_argument('--d-beta', type=float, default=0.6)
    ap.add_argument('--reps', type=int, default=1)
    ap.add_argument('--maxiter', type=int, default=300)
    ap.add_argument('--sweeps', type=int, default=50)
    ap.add_argument('--shots', default='1000,10000')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--seed0', type=int, default=20261105)
    ap.add_argument('--gamma-d', type=float, default=0.1)
    ap.add_argument('--keep', type=int, default=20)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--noise', default='')
    ap.add_argument('--trajectories', type=int, default=11)
    ap.add_argument('--use-addon', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--every', type=int, default=1,
                    help='take every k-th chain of the set (subsets for costly methods)')
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    qs = json.load(open(os.path.join(DATA_DIR, 'qset.json')))
    args.qset_budget = qs['budget']
    budget = args.budget or qs['budget']
    items = qs['sets'][args.set][::args.every]
    recs = {}
    want = {it['id'] for it in items}
    for line in open(os.path.join(DATA_DIR, 'pdb_fr3d_chains.jsonl')):
        d = json.loads(line)
        if d['id'] in want:
            recs[d['id']] = d
    if args.min_stems:
        from mrnasqd.qubo_v2 import build_stem_qubo
        keep = []
        for it in items:
            n_all = len(build_stem_qubo(recs[it['id']]['sequence'], model='turner_direct',
                                        **model_params()).stems)
            if n_all >= args.min_stems:
                keep.append(it)
        items = keep
    if args.limit:
        items = items[:args.limit]
    resources = {}
    rp = os.path.join(ROOT, 'results', 'resources.jsonl')
    if args.noise and os.path.exists(rp):
        resources = {json.loads(l)['id']: json.loads(l) for l in open(rp)}
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            if line.strip():
                d = json.loads(line)
                done.add((d['id'], d['method'], d['tag']))
    todo = [it for it in items if (it['id'], args.method, args.tag) not in done]
    print(f'{args.method} {args.tag}: {len(todo)} chains at budget {budget}', flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.procs) as pool, open(args.out, 'a') as f:
        futs = [pool.submit(work, it, recs[it['id']], budget, args, resources)
                for it in todo]
        for k, fu in enumerate(futs, 1):
            for row in fu.result():
                f.write(json.dumps(row) + '\n')
            f.flush()
            if k % 10 == 0 or k == len(futs):
                print(f'  {k}/{len(futs)} {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
