"""E9: is the quadratization penalty dynamical?  Add consistency-preserving moves.

On the quadratized model (M = A+1) a single-bit SA move flips one bit, original or
auxiliary.  A *coordinated* move flips an original bit x_i together with every
auxiliary y = x_i x_b that depends on it, so that the auxiliaries stay consistent.
Both move types are Metropolis moves on the SAME quadratized energy, so the
equilibrium distribution is unchanged; only the dynamics differ.  We mix them with
probability q of a coordinated move (q = 0 is the plain single-bit SA of E3; q = 1
on a consistent start is single-bit SA of the native model in disguise) and record
the success rate P(E0) and, at the end of the anneal, the fraction of replicas
with at least one inconsistent auxiliary bit.

Same instances and schedule rule as E3 (native beta_cold used for every variant).
"""
import json, os, sys, time
sys.path.insert(0, "..")
import numpy as np, torch
import instances as I
from sa import SA

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
FAM = {"3spin_m2n": lambda n, s: I.random_kspin(n, 2 * n, 3, s),
       "3SAT_4.26": lambda n, s: I.random_ksat(n, int(round(4.26 * n)), 3, s)}
Q = [0.0, 0.1, 0.25, 0.5, 1.0]
REPS, SWEEPS = 1024, 1000


def energy_torch(hq, X):
    E = torch.zeros(X.shape[0], dtype=X.dtype)
    for idx, c in hq.terms.items():
        p = X[:, idx[0]]
        for i in idx[1:]:
            p = p * X[:, i]
        E = E + c * p
    return E + hq.offset


def anneal_mixed(h, hq, aux, q, n_reps, n_sweeps, beta_hot, beta_cold, seed):
    """Metropolis SA on hq with single-bit moves (prob 1-q) and coordinated moves (prob q).
    Energies recomputed in full for coordinated moves (small n); single-bit moves use the
    colour-parallel kernel of sa.SA for one sweep."""
    g = torch.Generator().manual_seed(seed)
    n, nq = h.n, hq.n
    dep = {}  # original bit -> list of (y, other parent)
    for y, a, b, M in aux:
        dep.setdefault(a, []).append((y, b)); dep.setdefault(b, []).append((y, a))
    # consistent random start: original bits random, auxiliaries derived
    X = torch.zeros(n_reps, nq + 1, dtype=torch.float64); X[:, -1] = 1.0
    X[:, :n] = torch.randint(0, 2, (n_reps, n), generator=g).double()
    for y, a, b, M in aux:
        X[:, y] = X[:, a] * X[:, b]
    sa = SA(hq)
    E = energy_torch(hq, X[:, :nq])
    best = E.clone()
    betas = np.geomspace(beta_hot, beta_cold, n_sweeps)
    for beta in betas:
        if q < 1.0 and torch.rand(1, generator=g).item() >= q:
            # one single-bit sweep (all bits, colour-parallel), reusing SA's groups
            for vars_c, owner, others, coef in sa.groups:
                contrib = X[:, others].prod(dim=2) * coef if others.shape[1] else coef.expand(n_reps, -1)
                field = torch.zeros(n_reps, len(vars_c), dtype=torch.float64)
                field.index_add_(1, owner, contrib)
                xc = X[:, vars_c]
                dE = (1 - 2 * xc) * field
                u = torch.rand(n_reps, len(vars_c), generator=g, dtype=torch.float64)
                acc = (dE <= 0) | (u < torch.exp(-beta * dE.clamp(min=0)))
                X[:, vars_c] = torch.where(acc, 1 - xc, xc)
                E = E + (dE * acc).sum(dim=1)
        else:
            # one coordinated sweep: for each original bit in random order, propose flipping it with its auxiliaries
            for i in torch.randperm(n, generator=g).tolist():
                Xp = X.clone()
                Xp[:, i] = 1 - Xp[:, i]
                for y, b in dep.get(i, []):
                    Xp[:, y] = Xp[:, i] * Xp[:, b]
                Ep = energy_torch(hq, Xp[:, :nq])
                dE = Ep - E
                u = torch.rand(n_reps, generator=g, dtype=torch.float64)
                acc = (dE <= 0) | (u < torch.exp(-beta * dE.clamp(min=0)))
                X = torch.where(acc[:, None], Xp, X)
                E = torch.where(acc, Ep, E)
        best = torch.minimum(best, E)
    incons = torch.zeros(n_reps, dtype=torch.bool)
    for y, a, b, M in aux:
        incons |= X[:, y] != X[:, a] * X[:, b]
    return best.numpy(), float(incons.double().mean())


out = []
jobs = [("3spin_m2n", n, s) for n in (24, 32, 40) for s in (0, 1, 2)] + [("3SAT_4.26", n, s) for n in (24, 32) for s in (0, 1, 2)]
for fam, n, s in jobs:
    h = FAM[fam](n, s)
    E0 = h.ground_state()[0] if h.n <= 30 else None
    if E0 is None:
        from tn import TNHOBO
        E0 = TNHOBO(h, method="greedy", max_repeats=16, target_size=2 ** 22, basis="natural", parallel=False).ground_energy()
    hq, aux = h.quadratize(1.0, 1.0)
    sa_nat = SA(h)
    rec = {"family": fam, "n": n, "seed": s, "E0": E0, "n_aux": len(aux), "beta": [sa_nat.beta_hot, sa_nat.beta_cold]}
    r = sa_nat.run(REPS, SWEEPS, seed=s)
    rec["native_P0"] = float(np.mean(np.abs(r["best_E"] - E0) < 1e-6))
    for q in Q:
        t0 = time.time()
        best, inc = anneal_mixed(h, hq, aux, q, REPS, SWEEPS, sa_nat.beta_hot, sa_nat.beta_cold, seed=100 + s)
        rec[f"q{q}"] = {"P0": float(np.mean(np.abs(best - E0) < 1e-6)), "inconsistent_end": inc, "sec": time.time() - t0}
    out.append(rec)
    print(f"{fam:10s} n={n} s{s} E0={E0:+.0f} aux={len(aux):3d} | native {rec['native_P0']:.3f} | " +
          " ".join(f"q={q}: {rec[f'q{q}']['P0']:.3f} (inc {rec[f'q{q}']['inconsistent_end']:.2f})" for q in Q) +
          f" | {sum(rec[f'q{q}']['sec'] for q in Q):.0f}s", flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "..", "results", "e9_moves.json"), "w"), indent=1)
