"""HOBO (higher-order binary optimisation) as one coefficient tensor.

A HOBO cost function is a multilinear polynomial in x in {0,1}^n,

    E(x) = c + sum_S  T_S  prod_{i in S} x_i ,

where S runs over index sets (x_i^2 = x_i, so repeated indices carry no new
information).  This module keeps the tensor sparse (dict: index set -> coefficient)
and provides the three views the paper needs, all from the same object:

  * cost evaluation      energy(X) for a batch of configurations (numpy / torch),
                          energy_einsum(x) in the dense v1 form (one order-k tensor,
                          lower-order terms embedded on repeated indices);
  * exact ground state    ground_state(): tropical (min,+) variable elimination,
                          exponential in the induced width of the elimination order,
                          with the state recovered by back-substitution;
  * partition function    log_partition(beta): the same elimination with the
                          (logsumexp, +) semiring - only the reduction changes.

Everything here is CPU numpy; batched GPU sampling lives in sa.py.
"""
from __future__ import annotations

import itertools
import math
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

Index = Tuple[int, ...]


class HOBO:
    def __init__(self, n: int, terms: Optional[Dict[Index, float]] = None, offset: float = 0.0):
        self.n = int(n)
        self.offset = float(offset)
        self.terms: Dict[Index, float] = {}
        if terms:
            for idx, c in terms.items():
                self.add(idx, c)

    # ---------------------------------------------------------------- building
    def add(self, idx: Iterable[int], coeff: float) -> "HOBO":
        """Add coeff * prod_{i in idx} x_i.  Repeated indices collapse (x_i^2 = x_i)."""
        key = tuple(sorted(set(int(i) for i in idx)))
        for i in key:
            if not 0 <= i < self.n:
                raise IndexError(f"variable {i} out of range for n={self.n}")
        if not key:
            self.offset += float(coeff)
            return self
        c = self.terms.get(key, 0.0) + float(coeff)
        if c == 0.0:
            self.terms.pop(key, None)
        else:
            self.terms[key] = c
        return self

    @property
    def order(self) -> int:
        return max((len(k) for k in self.terms), default=0)

    def __len__(self) -> int:
        return len(self.terms)

    def copy(self) -> "HOBO":
        return HOBO(self.n, dict(self.terms), self.offset)

    # ------------------------------------------------------------ spin <-> bit
    @classmethod
    def from_ising(cls, n: int, couplings: Dict[Index, float], offset: float = 0.0) -> "HOBO":
        """Build from a polynomial in spins s in {-1,+1}: E = offset + sum_S J_S prod s_i.

        Uses s_i = 1 - 2 x_i and expands the product, so a k-body spin term becomes
        2^k - 1 bit terms of order <= k (this is the price of the {0,1} basis; the
        sparse dict absorbs it).
        """
        h = cls(n, offset=offset)
        for idx, J in couplings.items():
            key = tuple(sorted(set(int(i) for i in idx)))
            # prod (1 - 2 x_i) = sum over subsets U of (-2)^|U| prod_{i in U} x_i
            for r in range(len(key) + 1):
                for U in itertools.combinations(key, r):
                    h.add(U, J * (-2.0) ** r)
        return h

    def to_ising(self) -> Tuple[Dict[Index, float], float]:
        """Inverse of from_ising: coefficients of the same function in spins s = 1 - 2x
        (x_i = (1 - s_i)/2).  Returns (couplings, offset)."""
        J: Dict[Index, float] = {}
        off = self.offset
        for idx, c in self.terms.items():
            k = len(idx)
            for r in range(k + 1):
                for U in itertools.combinations(idx, r):
                    val = c * (-0.5) ** r * 0.5 ** (k - r)
                    if U:
                        J[U] = J.get(U, 0.0) + val
                    else:
                        off += val
        J = {k: v for k, v in J.items() if v != 0.0}
        return J, off

    # ------------------------------------------------------------- evaluation
    def energy(self, X) -> np.ndarray:
        """Batched cost.  X: array (B, n) or (n,) of {0,1}; returns (B,) or scalar.
        Works for numpy arrays and torch tensors (same code path)."""
        single = X.ndim == 1
        if single:
            X = X[None, :]
        E = None
        for idx, c in self.terms.items():
            p = X[:, idx[0]]
            for i in idx[1:]:
                p = p * X[:, i]
            E = p * c if E is None else E + p * c
        if E is None:
            E = X[:, 0] * 0.0
        E = E + self.offset
        return E[0] if single else E

    def to_dense_tensor(self, order: Optional[int] = None) -> np.ndarray:
        """The v1 (arXiv:2407.16106) representation: a single order-k tensor H with
        the coefficient of prod_{i in S} x_i stored at H[s_1, ..., s_k] where the index
        list is S padded by repeating its first element (H_iii = c_i, H_iij = q_ij, ...).
        Dense: n^k entries - only for illustration / small n."""
        k = order or max(self.order, 1)
        if self.order > k:
            raise ValueError("order smaller than the polynomial's order")
        H = np.zeros((self.n,) * k)
        for idx, c in self.terms.items():
            pos = idx + (idx[0],) * (k - len(idx))
            H[pos] += c
        return H

    def energy_einsum(self, x, H: Optional[np.ndarray] = None) -> float:
        """cost = einsum('ijk...,i,j,k,...->', H, x, x, x, ...) + offset  (v1, eq. in sec. 4).
        Works with numpy or torch (torch.einsum has the same signature)."""
        if H is None:
            H = self.to_dense_tensor()
        k = H.ndim
        letters = "abcdefghijklmnopqrstuvwxyz"[:k]
        spec = letters + "," + ",".join(letters) + "->"
        if type(x).__module__.startswith("torch"):
            import torch
            Ht = torch.as_tensor(H, dtype=x.dtype, device=x.device)
            return torch.einsum(spec, Ht, *([x] * k)) + self.offset
        return float(np.einsum(spec, H, *([x] * k))) + self.offset

    # --------------------------------------------------------------- exact
    def brute_force(self, chunk: int = 1 << 18) -> Tuple[float, np.ndarray]:
        """Enumerate all 2^n configurations (n <= ~26).  Returns (E_min, x_min)."""
        if self.n > 28:
            raise ValueError("too many variables for brute force")
        best = math.inf
        best_x = None
        N = 1 << self.n
        bits = np.arange(self.n, dtype=np.int64)
        for start in range(0, N, chunk):
            ids = np.arange(start, min(N, start + chunk), dtype=np.int64)
            X = ((ids[:, None] >> bits[None, :]) & 1).astype(np.float64)
            E = self.energy(X)
            j = int(np.argmin(E))
            if E[j] < best:
                best, best_x = float(E[j]), X[j].astype(np.int8)
        return best, best_x

    def elimination_order(self, method: str = "min-fill") -> Tuple[list, int]:
        """Greedy elimination order on the primal (interaction) graph.
        Returns (order, induced_width) where induced width = max clique size - 1 met
        during elimination; the tropical contraction costs O(sum 2^(width+1))."""
        adj = {i: set() for i in range(self.n)}
        for idx in self.terms:
            for a in idx:
                adj[a].update(j for j in idx if j != a)
        remaining = set(range(self.n))
        order, width = [], 0
        while remaining:
            if method == "min-degree":
                v = min(remaining, key=lambda u: (len(adj[u]), u))
            else:  # min-fill
                def fill(u):
                    nb = list(adj[u])
                    return sum(1 for a, b in itertools.combinations(nb, 2) if b not in adj[a])
                v = min(remaining, key=lambda u: (fill(u), len(adj[u]), u))
            nb = adj[v]
            width = max(width, len(nb))
            for a in nb:
                adj[a].update(nb - {a})
                adj[a].discard(v)
            for a in nb:
                adj[a].discard(v)
            adj[v] = set()
            remaining.discard(v)
            order.append(v)
        return order, width

    def _factors(self):
        """Sparse terms as tables over {0,1}^|S|: value c only at all-ones corner."""
        facs = []
        for idx, c in self.terms.items():
            t = np.zeros((2,) * len(idx))
            t[(1,) * len(idx)] = c
            facs.append((tuple(idx), t))
        return facs

    @staticmethod
    def _combine(factors, scope):
        """Sum (in the exponent / tropical sense) of tables over a common scope."""
        pos = {v: i for i, v in enumerate(scope)}
        out = np.zeros((2,) * len(scope))
        for vars_, t in factors:
            perm = sorted(range(len(vars_)), key=lambda a: pos[vars_[a]])
            tt = np.transpose(t, perm)
            shape = [1] * len(scope)
            for a in perm:
                shape[pos[vars_[a]]] = 2
            out = out + tt.reshape(shape)
        return out

    def _eliminate(self, reduce, order: Optional[Sequence[int]] = None, keep_argmin: bool = False):
        """Generic bucket elimination.  reduce(table, axis) -> table with that axis removed.
        Combination is always '+' (energies add), so tropical = (min, +) and
        Boltzmann = (logsumexp, +) differ only in `reduce`."""
        if order is None:
            order, _ = self.elimination_order()
        factors = self._factors()
        trace = []  # (v, scope_without_v, combined table) for back-substitution
        for v in order:
            bucket = [f for f in factors if v in f[0]]
            factors = [f for f in factors if v not in f[0]]
            scope = sorted(set().union(*[set(f[0]) for f in bucket]) | {v})
            table = self._combine(bucket, scope) if bucket else np.zeros((2,))
            if not bucket:
                scope = [v]
            ax = scope.index(v)
            new_scope = tuple(s for s in scope if s != v)
            if keep_argmin:
                trace.append((v, new_scope, np.moveaxis(table, ax, -1)))
            reduced = reduce(table, ax)
            if new_scope:
                factors.append((new_scope, reduced))
            else:
                factors.append(((), reduced))
        total = sum(float(np.asarray(t).reshape(())) for vars_, t in factors if not vars_)
        return total, trace

    def ground_state(self, order: Optional[Sequence[int]] = None) -> Tuple[float, np.ndarray]:
        """Exact minimum by tropical (min,+) contraction, state by back-substitution.
        Cost ~ sum over eliminations of 2^(scope size); see elimination_order()."""
        E, trace = self._eliminate(lambda t, ax: t.min(axis=ax), order, keep_argmin=True)
        x = np.zeros(self.n, dtype=np.int8)
        for v, scope, table in reversed(trace):
            sub = table[tuple(int(x[s]) for s in scope)]
            x[v] = int(np.argmin(sub))
        return E + self.offset, x

    def log_partition(self, beta: float, order: Optional[Sequence[int]] = None) -> float:
        """log Z(beta) = log sum_x exp(-beta E(x)), by the same elimination with
        reduce = logsumexp (energies scaled by -beta before contraction)."""
        scaled = HOBO(self.n, {k: -beta * c for k, c in self.terms.items()})

        def lse(t, ax):
            m = t.max(axis=ax, keepdims=True)
            return (m + np.log(np.exp(t - m).sum(axis=ax, keepdims=True))).squeeze(axis=ax)

        val, _ = scaled._eliminate(lse, order)
        return val - beta * self.offset

    # ---------------------------------------------------------- quadratization
    def quadratize(self, penalty_scale: float = 1.5, penalty_add: float = 0.0) -> Tuple["HOBO", list]:
        """Reduce to order <= 2 by Rosenberg substitution (Rosenberg 1975): repeatedly
        pick the pair (a, b) shared by the most terms of order >= 3, add an auxiliary
        bit y standing for x_a x_b, replace a,b by y in those terms and add the penalty

            M (x_a x_b - 2 x_a y - 2 x_b y + 3 y)      (= 0 iff y = x_a x_b, else >= M).

        M = penalty_scale * A + penalty_add, where A = sum |c| of the terms that received y.
        Any M > A is exact (penalty_scale > 1, or scale 1 with penalty_add > 0; for integer
        coefficients M = A + 1 is the tightest exact integer choice): min over the auxiliaries equals the original cost
        for every x (the last auxiliary is forced first, then induct backwards).
        Returns (quadratic HOBO on n + n_aux bits, list of (y, a, b, M));
        the first n bits are the original variables."""
        if penalty_scale < 1.0 or (penalty_scale == 1.0 and penalty_add <= 0.0):
            raise ValueError("need M > A: penalty_scale > 1, or penalty_scale == 1 with penalty_add > 0")
        terms = dict(self.terms)
        n = self.n
        aux = []
        while True:
            count: Dict[Tuple[int, int], int] = {}
            for idx in terms:
                if len(idx) >= 3:
                    for p in itertools.combinations(idx, 2):
                        count[p] = count.get(p, 0) + 1
            if not count:
                break
            a, b = max(count, key=lambda p: (count[p], -p[0], -p[1]))
            y = n
            n += 1
            new_terms: Dict[Index, float] = {}
            A = 0.0
            for idx, c in terms.items():
                if len(idx) >= 3 and a in idx and b in idx:
                    key = tuple(sorted([i for i in idx if i not in (a, b)] + [y]))
                    A += abs(c)
                else:
                    key = idx
                new_terms[key] = new_terms.get(key, 0.0) + c
            M = penalty_scale * A + penalty_add
            for key, c in (((a, b), M), ((a, y), -2 * M), ((b, y), -2 * M), ((y,), 3 * M)):
                new_terms[key] = new_terms.get(key, 0.0) + c
            terms = {k: v for k, v in new_terms.items() if v != 0.0}
            aux.append((y, a, b, M))
        return HOBO(n, terms, self.offset), aux


# ------------------------------------------------------------------ self test
if __name__ == "__main__":
    import time

    rng = np.random.default_rng(0)

    def random_hobo(n, m, k, rng):
        h = HOBO(n)
        for _ in range(m):
            idx = rng.choice(n, size=rng.integers(1, k + 1), replace=False)
            h.add(idx, rng.normal())
        return h

    # 1. energy() == energy_einsum() (v1 dense form) on random points
    h = random_hobo(7, 15, 3, rng)
    for _ in range(20):
        x = rng.integers(0, 2, size=7).astype(float)
        assert abs(h.energy(x) - h.energy_einsum(x)) < 1e-9
    print("energy == einsum(v1 dense tensor)      ok")

    # 2. spin <-> bit round trip
    J = {(0, 1, 2): 1.3, (1, 3): -0.7, (2,): 0.4, (0, 1, 2, 3): 0.9}
    hs = HOBO.from_ising(4, J, offset=0.1)
    J2, off2 = hs.to_ising()
    assert all(abs(J[k] - J2.get(k, 0.0)) < 1e-12 for k in J) and abs(off2 - 0.1) < 1e-12
    print("from_ising / to_ising round trip       ok")

    # 3. tropical ground state == brute force, incl. degenerate / all-positive cases
    for n, m, k in [(8, 12, 3), (12, 30, 3), (16, 40, 4), (20, 60, 3), (22, 44, 3)]:
        h = random_hobo(n, m, k, rng)
        t0 = time.time(); Eb, xb = h.brute_force(); tb = time.time() - t0
        order, w = h.elimination_order()
        t0 = time.time(); Et, xt = h.ground_state(order); tt = time.time() - t0
        assert abs(Eb - Et) < 1e-9, (Eb, Et)
        assert abs(h.energy(xt.astype(float)) - Et) < 1e-9
        print(f"n={n:2d} m={m:3d} k={k}: brute {Eb:+.6f} ({tb:5.2f}s)  tropical {Et:+.6f} "
              f"({tt:5.3f}s, width {w})   ok")

    # 4. log Z by elimination == log Z by enumeration
    h = random_hobo(12, 30, 3, rng)
    N = 1 << 12
    X = ((np.arange(N)[:, None] >> np.arange(12)[None, :]) & 1).astype(float)
    E = h.energy(X)
    for beta in (0.3, 1.0, 3.0):
        lz_enum = float(np.log(np.exp(-beta * E).sum()))
        lz = h.log_partition(beta)
        assert abs(lz - lz_enum) < 1e-8, (lz, lz_enum)
    print("log_partition == enumeration            ok")

    # 5. a 3-regular-ish 3-spin glass on a ring of n=25 (sparse -> small width)
    n = 25
    Jr = {tuple(sorted(((i) % n, (i + 1) % n, (i + 2) % n))): float(rng.choice([-1, 1])) for i in range(n)}
    h = HOBO.from_ising(n, Jr)
    order, w = h.elimination_order()
    t0 = time.time(); Et, xt = h.ground_state(order); tt = time.time() - t0
    t0 = time.time(); Eb, xb = h.brute_force(); tb = time.time() - t0
    assert abs(Eb - Et) < 1e-9
    print(f"3-spin ring n=25: tropical {Et:+.1f} ({tt:.3f}s, width {w})  brute {Eb:+.1f} ({tb:.1f}s)   ok")
