"""Certified lower bounds on min_x E(x) by width-bounded tropical contraction.

Mini-bucket elimination (Dechter & Rish 2003) in tensor-network language: when
eliminating variable v would create a table over more than ibound+1 variables,
the tables that share v are split into groups ("mini-buckets") and v is
minimised in each group separately - i.e. the hyperedge v is cut and every
group gets its own copy of v.  Since min_v (f + g) >= min_v f + min_v g, the
contraction value is a lower bound, and it is exact once ibound >= width.

moment_matching=True reparameterises the mini-buckets before the split so that
their min-marginals over v agree (the sum of the tables is unchanged, so the result
is still a valid lower bound).  It usually tightens the bound but is not guaranteed
to: downstream combination can undo the gain (counterexample in notes/codex_verification.md),
so best_bound() takes the better of the two.

Upper bounds come from any configuration (SA); together they give a certified
optimality gap at sizes where exact contraction is out of reach.
"""
from __future__ import annotations

import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

from hobo import HOBO
from tn import factor_tables

Factor = Tuple[Tuple[int, ...], np.ndarray]


def _combine(factors: List[Factor], scope: Sequence[int]) -> np.ndarray:
    pos = {v: i for i, v in enumerate(scope)}
    out = np.zeros((2,) * len(scope))
    for vars_, t in factors:
        if not vars_:
            out = out + float(t)
            continue
        perm = sorted(range(len(vars_)), key=lambda a: pos[vars_[a]])
        tt = np.transpose(t, perm)
        shape = [1] * len(scope)
        for a in perm:
            shape[pos[vars_[a]]] = 2
        out = out + tt.reshape(shape)
    return out


def minfill_order(tables: List[Factor], n: int) -> List[int]:
    adj = {i: set() for i in range(n)}
    for vars_, _ in tables:
        for a in vars_:
            adj[a].update(j for j in vars_ if j != a)
    remaining = set(range(n))
    order = []
    while remaining:
        def fill(u):
            nb = list(adj[u])
            return sum(1 for i, a in enumerate(nb) for b in nb[i + 1:] if b not in adj[a])
        v = min(remaining, key=lambda u: (fill(u), len(adj[u]), u))
        nb = adj[v]
        for a in nb:
            adj[a].update(nb - {a})
            adj[a].discard(v)
        adj[v] = set()
        remaining.discard(v)
        order.append(v)
    return order


def mini_bucket(h: HOBO, ibound: int, order: Optional[Sequence[int]] = None,
                moment_matching: bool = True, tables: Optional[List[Factor]] = None, basis: str = "bit") -> dict:
    """Lower bound on min E.  Returns {'lb', 'exact' (no split happened), 'max_scope', 'seconds'}.
    `tables`, if given, must sum to E - h.offset (e.g. the output of dual_ascent or factor_tables
    in any basis); `basis` is only used when `tables` is None.  `order` must cover every variable
    that appears in the tables (variables fixed by conditioning may be absent)."""
    t0 = time.perf_counter()
    factors: List[Factor] = [(tuple(v), np.array(t, dtype=float)) for v, t in (tables if tables is not None else factor_tables(h, basis=basis))]
    if order is None:
        order = minfill_order(factors, h.n)
    const = 0.0
    splits = 0
    max_scope = 0
    for v in order:
        bucket = [f for f in factors if v in f[0]]
        factors = [f for f in factors if v not in f[0]]
        if not bucket:
            continue
        # first-fit partition, largest scopes first
        bucket.sort(key=lambda f: -len(f[0]))
        groups: List[Tuple[set, List[Factor]]] = []
        for f in bucket:
            for sc, fs in groups:
                if len(sc | set(f[0])) <= ibound + 1:
                    sc |= set(f[0])
                    fs.append(f)
                    break
            else:
                groups.append((set(f[0]), [f]))
        if len(groups) > 1:
            splits += 1
        tabs = []
        for sc, fs in groups:
            scope = sorted(sc)
            max_scope = max(max_scope, len(scope))
            tabs.append((scope, _combine(fs, scope)))
        if moment_matching and len(tabs) > 1:
            # min-marginal of each group over v, shift to the average
            mm = []
            for scope, T in tabs:
                ax = scope.index(v)
                other = tuple(a for a in range(T.ndim) if a != ax)
                mm.append(T.min(axis=other) if other else T)
            avg = np.mean(mm, axis=0)
            new = []
            for (scope, T), m in zip(tabs, mm):
                shape = [1] * T.ndim
                shape[scope.index(v)] = 2
                new.append((scope, T + (avg - m).reshape(shape)))
            tabs = new
        for scope, T in tabs:
            R = T.min(axis=scope.index(v))
            rest = tuple(s for s in scope if s != v)
            if rest:
                factors.append((rest, R))
            else:
                const += float(R)
    if any(vars_ for vars_, _ in factors):
        raise ValueError("elimination order does not cover every variable that appears in a table")
    const += sum(float(t) for vars_, t in factors if not vars_)
    return {"lb": const + h.offset, "exact": splits == 0, "splits": splits, "max_scope": max_scope,
            "seconds": time.perf_counter() - t0}


def dual_ascent(h: HOBO, iters: int = 200, tol: float = 1e-9, tables: Optional[List[Factor]] = None,
                time_limit: Optional[float] = None, basis: str = "bit") -> dict:
    """Min-sum diffusion / MPLP star updates on the factor network.

    Reparameterise theta_f(x_S) += delta_{f,v}(x_v), theta_v(x_v) -= sum_f delta_{f,v}(x_v):
    the total energy is unchanged, so sum_f min theta_f + sum_v min theta_v is a lower bound
    (the dual of the local-polytope LP relaxation); each star update at v equalises the
    min-marginals over x_v of v's tables and never decreases the bound.
    Returns the reparameterised tables (factors + unary tables) and the bound history."""
    t0 = time.perf_counter()
    facs = [(tuple(v), np.array(t, dtype=float)) for v, t in (tables if tables is not None else factor_tables(h, basis=basis))]
    n = h.n
    unary = np.zeros((n, 2))
    inc = [[] for _ in range(n)]
    for k, (vars_, _) in enumerate(facs):
        for v in vars_:
            inc[v].append(k)

    def bound():
        return sum(float(t.min()) for _, t in facs) + float(unary.min(axis=1).sum()) + h.offset

    hist = [bound()]
    for it in range(iters):
        for v in range(n):
            if not inc[v]:
                continue
            mm = []
            for k in inc[v]:
                vars_, T = facs[k]
                ax = vars_.index(v)
                other = tuple(a for a in range(T.ndim) if a != ax)
                mm.append(T.min(axis=other) if other else T)
            total = unary[v] + np.sum(mm, axis=0)
            share = total / (len(inc[v]) + 1)
            for k, m in zip(inc[v], mm):
                vars_, T = facs[k]
                shape = [1] * T.ndim
                shape[vars_.index(v)] = 2
                facs[k] = (vars_, T + (share - m).reshape(shape))
            unary[v] = share
        hist.append(bound())
        if hist[-1] - hist[-2] <= tol * (1 + abs(hist[-1])):
            break
        if time_limit and time.perf_counter() - t0 > time_limit:
            break
    tabs = facs + [((v,), unary[v].copy()) for v in range(n) if np.any(unary[v] != 0)]
    return {"lb": hist[-1], "history": hist, "tables": tabs, "iters": len(hist) - 1,
            "seconds": time.perf_counter() - t0}


def best_bound(h: HOBO, ibound: int, iters: int = 200, basis: str = "bit") -> dict:
    """Dual ascent, then mini-bucket (with and without moment matching) on the
    reparameterised network; every number here is a valid lower bound, return the max."""
    t0 = time.perf_counter()
    da = dual_ascent(h, iters, basis=basis)
    order = minfill_order(da["tables"], h.n)
    a = mini_bucket(h, ibound, order, moment_matching=True, tables=da["tables"])
    b = mini_bucket(h, ibound, order, moment_matching=False, tables=da["tables"])
    lb = max(da["lb"], a["lb"], b["lb"])
    return {"lb": lb, "dual": da["lb"], "mbe_mm": a["lb"], "mbe": b["lb"], "exact": a["exact"],
            "iters": da["iters"], "seconds": time.perf_counter() - t0}


def trivial_bound(h: HOBO, basis: str = "bit") -> float:
    """Sum of per-table minima: the ibound = 0 limit, for reference."""
    return sum(float(np.min(t)) for _, t in factor_tables(h, basis=basis)) + h.offset


if __name__ == "__main__":
    import instances as I
    rng = np.random.default_rng(0)
    for name, h in [("3spin n=22", I.random_kspin(22, 44, 3, 3)), ("3R3X n=22", I.xorsat_planted(22, 3, 1)[0]),
                    ("LABS 16", I.labs(16)), ("plaq L=4", I.plaquette_2d(4, 1)), ("3SAT n=20", I.random_ksat(20, 90, 3, 2))]:
        E0 = h.brute_force()[0] if h.n <= 22 else h.ground_state()[0]
        row = [f"{name:11s} E0={E0:+7.1f} triv={trivial_bound(h):+8.1f}"]
        for ib in (2, 4, 6, 8, 12, 16):
            a = mini_bucket(h, ib, moment_matching=False)["lb"]
            r = mini_bucket(h, ib)
            assert a <= E0 + 1e-9 and r["lb"] <= E0 + 1e-9, (a, r["lb"], E0)
            row.append(f"i{ib}:{a:+.1f}/{r['lb']:+.1f}{'*' if r['exact'] else ''}")
        print("  ".join(row))
        row = [f"{'':11s} dual ascent -> {dual_ascent(h)['lb']:+.1f}  then MBE:"]
        for ib in (2, 4, 6, 8, 12, 16):
            r = best_bound(h, ib)
            assert r["lb"] <= E0 + 1e-6, (r, E0)
            row.append(f"i{ib}:{r['lb']:+.1f}")
        print("  ".join(row))
