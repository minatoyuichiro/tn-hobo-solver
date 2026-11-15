"""Batched simulated annealing on a HOBO, native or quadratized, same code path.

Single-bit-flip Metropolis on B independent replicas at once (torch; CPU or GPU).
The flip cost of bit i only needs the terms that contain i,

    dE_i = (1 - 2 x_i) * sum_{S ni i} c_S prod_{j in S, j != i} x_j ,

so the variables are greedily coloured on the interaction graph and all bits of
one colour (which share no term) are updated in parallel - a chromatic sweep,
which keeps detailed balance.  A quadratized instance is just another HOBO
(order 2, more bits), so native vs quadratized is a fair comparison: same
kernel, same schedule rule, same number of sweeps.

Schedule rule (the same one dwave-neal uses): geometric in beta from
beta_hot = ln 2 / max_i A_i to beta_cold = ln 100 / min_S |c_S|, where A_i is
the sum of |c_S| over the terms containing i.
"""
from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
import torch

from hobo import HOBO


class SA:
    def __init__(self, h: HOBO, device: str = "cpu", dtype=torch.float64):
        self.h = h
        self.n = h.n
        self.device = device
        self.dtype = dtype
        n = h.n
        terms = list(h.terms.items())
        k = max(h.order, 1)
        # colour the interaction graph (greedy, largest degree first)
        adj = [set() for _ in range(n)]
        for idx, _ in terms:
            for a in idx:
                adj[a].update(j for j in idx if j != a)
        colour = [-1] * n
        for v in sorted(range(n), key=lambda u: -len(adj[u])):
            used = {colour[u] for u in adj[v]}
            c = 0
            while c in used:
                c += 1
            colour[v] = c
        self.n_colours = max(colour) + 1 if n else 0
        # per colour: variables, and every (owner, others, coeff) term incidence
        self.groups = []
        for c in range(self.n_colours):
            vars_c = [v for v in range(n) if colour[v] == c]
            pos = {v: p for p, v in enumerate(vars_c)}
            owner, others, coef = [], [], []
            for idx, cs in terms:
                for i in idx:
                    if i in pos:
                        rest = [j for j in idx if j != i]
                        owner.append(pos[i])
                        others.append(rest + [n] * (k - 1 - len(rest)))  # n = constant-1 column
                        coef.append(cs)
            t = lambda a, dt=torch.long: torch.tensor(a, dtype=dt, device=device)
            self.groups.append((
                t(vars_c),
                t(owner),
                t(others).reshape(len(owner), max(k - 1, 0)) if owner else t([]).reshape(0, max(k - 1, 0)),
                t(coef, dtype),
            ))
        A = np.zeros(n)
        for idx, cs in terms:
            for i in idx:
                A[i] += abs(cs)
        cmin = min((abs(c) for _, c in terms), default=1.0)
        self.beta_hot = math.log(2) / max(A.max(), 1e-300)
        self.beta_cold = math.log(100) / cmin

    def energy(self, X: torch.Tensor) -> torch.Tensor:
        return self.h.energy(X[:, : self.n])

    def run(self, n_reps: int, n_sweeps: int, seed: int = 0,
            beta_hot: Optional[float] = None, beta_cold: Optional[float] = None,
            X0: Optional[torch.Tensor] = None) -> dict:
        g = torch.Generator(device=self.device).manual_seed(seed)
        b0 = beta_hot if beta_hot is not None else self.beta_hot
        b1 = beta_cold if beta_cold is not None else self.beta_cold
        betas = np.geomspace(b0, b1, n_sweeps) if n_sweeps > 1 else np.array([b1])
        B, n = n_reps, self.n
        X = torch.ones(B, n + 1, dtype=self.dtype, device=self.device)  # last column stays 1
        if X0 is None:
            X[:, :n] = torch.randint(0, 2, (B, n), generator=g, device=self.device).to(self.dtype)
        else:
            X[:, :n] = X0
        E = self.energy(X)
        best_E, best_X = E.clone(), X[:, :n].clone()
        t0 = time.perf_counter()
        for beta in betas:
            for vars_c, owner, others, coef in self.groups:
                if others.shape[1]:
                    contrib = X[:, others].prod(dim=2) * coef          # (B, T)
                else:
                    contrib = coef.expand(B, -1)
                field = torch.zeros(B, len(vars_c), dtype=self.dtype, device=self.device)
                field.index_add_(1, owner, contrib)
                xc = X[:, vars_c]
                dE = (1 - 2 * xc) * field
                u = torch.rand(B, len(vars_c), generator=g, dtype=self.dtype, device=self.device)
                acc = (dE <= 0) | (u < torch.exp(-beta * dE.clamp(min=0)))
                X[:, vars_c] = torch.where(acc, 1 - xc, xc)
                E = E + (dE * acc).sum(dim=1)
            better = E < best_E
            best_E = torch.where(better, E, best_E)
            best_X[better] = X[better, :n]
        if self.device != "cpu":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        return {"best_E": best_E.cpu().numpy(), "best_X": best_X.cpu().numpy().astype(np.int8),
                "final_E": E.cpu().numpy(), "seconds": elapsed, "sweeps": n_sweeps,
                "beta": (b0, b1), "colours": self.n_colours}


def solve_quadratized(h: HOBO, n_reps: int, n_sweeps: int, penalty_scale: float = 1.5,
                      seed: int = 0, device: str = "cpu", penalty_add: float = 0.0) -> dict:
    """Quadratize, anneal, and report energies of the ORIGINAL cost on the first n bits
    (auxiliaries that end up inconsistent simply don't matter for the reported value)."""
    q, aux = h.quadratize(penalty_scale, penalty_add)
    r = SA(q, device=device).run(n_reps, n_sweeps, seed=seed)
    Xo = r["best_X"][:, : h.n].astype(np.float64)
    r["orig_E"] = h.energy(Xo)
    r["n_aux"] = len(aux)
    r["n_terms_q"] = len(q)
    return r


# ------------------------------------------------------------------ self test
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    def random_3spin(n, m, rng):
        J = {}
        for _ in range(m):
            J[tuple(sorted(rng.choice(n, 3, replace=False)))] = float(rng.choice([-1, 1]))
        return HOBO.from_ising(n, J)

    # 1. incremental energy bookkeeping == recomputed energy
    h = random_3spin(20, 40, rng)
    sa = SA(h)
    r = sa.run(64, 50, seed=1)
    Xb = torch.as_tensor(r["best_X"], dtype=torch.float64)
    assert np.allclose(h.energy(Xb).numpy(), r["best_E"], atol=1e-9)
    print(f"incremental dE bookkeeping == recomputed energy   ok  ({sa.n_colours} colours)")

    # 2. SA reaches the exact (tropical) ground state on small instances
    for n in (16, 24, 30):
        h = random_3spin(n, 2 * n, rng)
        Eg, _ = h.ground_state()
        r = SA(h).run(256, 300, seed=2)
        p = float(np.mean(np.abs(r["best_E"] - Eg) < 1e-9))
        rq = solve_quadratized(h, 256, 300, seed=2)
        pq = float(np.mean(np.abs(rq["orig_E"] - Eg) < 1e-9))
        print(f"3-spin n={n}: E0={Eg:+.1f}  native P(E0)={p:.2f} ({r['seconds']:.2f}s)   "
              f"quadratized (+{rq['n_aux']} aux) P(E0)={pq:.2f} ({rq['seconds']:.2f}s)")
