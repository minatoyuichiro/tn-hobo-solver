"""Instance generators.  Every generator returns a HOBO (bit form, x in {0,1});
spin models are built in spins and converted with HOBO.from_ising, so the
reported energies are the spin-model energies.

  random_kspin      random k-spin glass, m terms, J = +-1
  xorsat_planted    k-regular k-XORSAT with a planted solution (energy 0 reachable),
                    degeneracy = 2^(n - rank_GF2(A)) is known exactly
  labs              low-autocorrelation binary sequences, E = sum_k C_k^2 (4-body)
  plaquette_2d      2D plaquette (4-spin) Ising on an L x L torus, random signs
  triangle_2d       3-spin model on the up-triangles of an L x L triangular lattice
  random_ksat       random k-SAT as a HOBO: one term per clause, E = #violated clauses
"""
from __future__ import annotations

import itertools
from typing import Dict, Tuple

import numpy as np

from hobo import HOBO

Index = Tuple[int, ...]


def _spin(n, J: Dict[Index, float], offset: float = 0.0) -> HOBO:
    """Spin model -> HOBO, keeping each spin term as a natural factor table
    (J prod s over its own scope), so bounds can start from the model's own terms."""
    h = HOBO.from_ising(n, J, offset=offset)
    facs = []
    for idx, c in J.items():
        k = len(idx)
        g = np.indices((2,) * k)
        t = np.full((2,) * k, float(c))
        for p in range(k):
            t = t * (1 - 2 * g[p])
        facs.append((tuple(idx), t))
    h.factors = facs
    h.factors_offset = offset
    return h


def random_kspin(n: int, m: int, k: int = 3, seed: int = 0) -> HOBO:
    import math
    if m > math.comb(n, k):
        raise ValueError("more terms requested than distinct k-subsets")
    rng = np.random.default_rng(seed)
    J: Dict[Index, float] = {}
    while len(J) < m:
        J[tuple(sorted(rng.choice(n, k, replace=False).tolist()))] = float(rng.choice([-1, 1]))
    return _spin(n, J)


def xorsat_planted(n: int, k: int = 3, seed: int = 0, max_tries: int = 1000):
    """k-regular k-XORSAT (each variable in exactly k clauses, m = n clauses, no repeated
    clause, no repeated variable in a clause).  Clause c: prod_{i in c} s_i = b_c with
    b drawn from a planted s*; energy term -b_c prod s_i, so E_min = -m.
    Returns (hobo, A (m x n over GF2), planted x*)."""
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        stubs = np.repeat(np.arange(n), k)
        rng.shuffle(stubs)
        clauses = stubs.reshape(-1, k)
        if all(len(set(c)) == k for c in clauses) and len({tuple(sorted(c)) for c in clauses}) == len(clauses):
            break
    else:
        raise RuntimeError("could not draw a simple k-regular hypergraph")
    xs = rng.integers(0, 2, n)
    s = 1 - 2 * xs
    J = {}
    A = np.zeros((len(clauses), n), dtype=np.uint8)
    for r, c in enumerate(clauses):
        key = tuple(sorted(int(i) for i in c))
        J[key] = -float(np.prod(s[list(key)]))
        A[r, list(key)] = 1
    return _spin(n, J), A, xs.astype(np.int8)


def gf2_rank(A: np.ndarray) -> int:
    A = A.copy() % 2
    r = 0
    rows, cols = A.shape
    for c in range(cols):
        piv = next((i for i in range(r, rows) if A[i, c]), None)
        if piv is None:
            continue
        A[[r, piv]] = A[[piv, r]]
        for i in range(rows):
            if i != r and A[i, c]:
                A[i] ^= A[r]
        r += 1
        if r == rows:
            break
    return r


def _spin_poly_mul(P: Dict[frozenset, float], Q: Dict[frozenset, float]) -> Dict[frozenset, float]:
    out: Dict[frozenset, float] = {}
    for a, ca in P.items():
        for b, cb in Q.items():
            key = a ^ b  # s_i^2 = 1
            out[key] = out.get(key, 0.0) + ca * cb
    return {k: v for k, v in out.items() if v != 0.0}


def labs(N: int) -> HOBO:
    """E(s) = sum_{k=1}^{N-1} C_k^2, C_k = sum_i s_i s_{i+k}.  Merit factor F = N^2 / (2E)."""
    total: Dict[frozenset, float] = {}
    for k in range(1, N):
        C = {frozenset((i, i + k)): 1.0 for i in range(N - k)}
        for key, v in _spin_poly_mul(C, C).items():
            total[key] = total.get(key, 0.0) + v
    off = total.pop(frozenset(), 0.0)
    J = {tuple(sorted(k)): v for k, v in total.items() if v != 0.0}
    return _spin(N, J, offset=off)


def plaquette_2d(L: int, seed: int = 0) -> HOBO:
    if L < 3:
        raise ValueError("L >= 3 required (smaller tori make the corner sites coincide)")
    rng = np.random.default_rng(seed)
    idx = lambda r, c: (r % L) * L + (c % L)
    J = {}
    for r in range(L):
        for c in range(L):
            key = tuple(sorted({idx(r, c), idx(r, c + 1), idx(r + 1, c), idx(r + 1, c + 1)}))
            J[key] = float(rng.choice([-1, 1]))
    return _spin(L * L, J)


def triangle_2d(L: int, seed: int = 0) -> HOBO:
    """3-spin terms on the up-triangles (r,c),(r,c+1),(r+1,c) of a periodic L x L lattice."""
    if L < 3:
        raise ValueError("L >= 3 required (smaller tori make the corner sites coincide)")
    rng = np.random.default_rng(seed)
    idx = lambda r, c: (r % L) * L + (c % L)
    J = {}
    for r in range(L):
        for c in range(L):
            key = tuple(sorted({idx(r, c), idx(r, c + 1), idx(r + 1, c)}))
            J[key] = float(rng.choice([-1, 1]))
    return _spin(L * L, J)


def random_ksat(n: int, m: int, k: int = 3, seed: int = 0) -> HOBO:
    """Clause (l_1 or ... or l_k) is violated iff every literal is false:
    penalty prod_j (1 - l_j), l_j = x or 1 - x.  E = number of violated clauses."""
    rng = np.random.default_rng(seed)
    h = HOBO(n)
    facs = []
    for _ in range(m):
        vars_ = rng.choice(n, k, replace=False)
        neg = rng.integers(0, 2, k)  # 1 -> literal is NOT x
        order = np.argsort(vars_)
        t = np.zeros((2,) * k)
        t[tuple(int(neg[j]) for j in order)] = 1.0   # the single violating assignment
        facs.append((tuple(int(vars_[j]) for j in order), t))
        # prod over j of f_j, f_j = (1 - x) for positive literal, x for negated
        poly = {(): 1.0}
        for v, ng in zip(vars_, neg):
            new = {}
            for key, c in poly.items():
                if ng:  # factor x_v
                    new[tuple(sorted(key + (int(v),)))] = new.get(tuple(sorted(key + (int(v),))), 0.0) + c
                else:   # factor 1 - x_v
                    new[key] = new.get(key, 0.0) + c
                    kk = tuple(sorted(key + (int(v),)))
                    new[kk] = new.get(kk, 0.0) - c
            poly = new
        for key, c in poly.items():
            h.add(key, c)
    h.factors = facs
    h.factors_offset = 0.0
    return h


if __name__ == "__main__":
    # sanity: planted XORSAT has E_min = -m and degeneracy 2^(n - rank)
    h, A, xs = xorsat_planted(18, 3, seed=1)
    Eb, _ = h.brute_force()
    X = ((np.arange(1 << 18)[:, None] >> np.arange(18)[None, :]) & 1).astype(float)
    E = h.energy(X)
    deg = int(np.sum(np.abs(E - Eb) < 1e-9))
    print(f"3R3X n=18: E0={Eb:+.0f} (expect {-A.shape[0]}), degeneracy {deg} (expect 2^{18 - gf2_rank(A)} = {2 ** (18 - gf2_rank(A))}),"
          f" planted E={h.energy(xs.astype(float)):+.0f}")
    # LABS known optima: N=10 E=13, N=13 E=6 (Barker), N=16 E=24
    for N, Eopt in ((10, 13), (13, 6), (16, 24)):
        e, _ = labs(N).brute_force()
        print(f"LABS N={N}: E0={e:.0f} (known {Eopt}), terms {len(labs(N))}, order {labs(N).order}")
    h = random_ksat(14, 50, 3, seed=0)
    X = ((np.arange(1 << 14)[:, None] >> np.arange(14)[None, :]) & 1).astype(float)
    print("3-SAT n=14 m=50: min violated =", h.energy(X).min())
    print("plaquette L=4 E0 =", plaquette_2d(4).brute_force()[0], " triangle L=4 E0 =", triangle_2d(4).brute_force()[0])
