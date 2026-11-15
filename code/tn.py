"""HOBO as a tensor network, contracted with a pluggable semiring along a
cotengra-optimised (optionally sliced) contraction tree.

Formulation.  The v1 object is one order-k coefficient tensor.  For contraction
we use its factor-graph form: every term (after merging terms whose scope is
contained in another term's scope) becomes a small table over {0,1}^|S| holding
that term's energy, and every variable is a hyperedge (a COPY / delta tensor)
shared by the tables that mention it.  The cost of any configuration is the sum
of the tables, so

    min_x E(x)            = contraction in the tropical (min, +) semiring
    (E_min, #argmin)      = counting-tropical semiring
    log Z(beta)           = (logsumexp, +) semiring on -beta E
    argmin / samples      = top-down pass through the same tree

and only the semiring changes.  Contraction order and slicing come from
cotengra (the same trees cuTensorNet would use), so the exact methods scale
with the contraction width of the network, not with n.
"""
from __future__ import annotations

import itertools
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from hobo import HOBO


# --------------------------------------------------------------------- network
def _tables_from_terms(n, terms, merge, poly):
    """Tables from monomials: poly(scope_positions, host_rank) -> table of one monomial."""
    scopes = sorted(terms, key=len, reverse=True)
    if merge:
        maximal: List[Tuple[int, ...]] = []
        for s in scopes:
            if not any(set(s) <= set(m) for m in maximal):
                maximal.append(s)
    else:
        maximal = scopes
    tables = {m: np.zeros((2,) * len(m)) for m in maximal}
    for s, c in terms.items():
        host = next(m for m in maximal if set(s) <= set(m)) if merge else s
        tables[host] = tables[host] + c * poly([host.index(i) for i in s], len(host))
    seen = set(i for m in maximal for i in m)
    out = [(m, t) for m, t in tables.items()]
    for v in range(n):
        if v not in seen:
            out.append(((v,), np.zeros(2)))
    return out


def _bit_monomial(pos, k):
    g = np.indices((2,) * k)
    m = np.ones((2,) * k)
    for p in pos:
        m = m * g[p]
    return m


def _spin_monomial(pos, k):
    g = np.indices((2,) * k)
    m = np.ones((2,) * k)
    for p in pos:
        m = m * (1 - 2 * g[p])
    return m


def factor_tables(h: HOBO, merge: bool = True, basis: str = "bit") -> List[Tuple[Tuple[int, ...], np.ndarray]]:
    """Energy tables of the factor graph (the sum of all tables = E - offset).

    basis='bit'     tables built from the {0,1} monomial coefficients (the v1 coefficient
                    tensor, QUBO-matrix style); lower-order monomials are absorbed into
                    some maximal scope.
    basis='spin'    tables built from the spin (+-1) monomials of the same function.
    basis='natural' the generator's own factors (h.factors: clause / term tables),
                    if it attached them; else falls back to 'spin'.
    All three give identical exact contractions; they differ as starting points for
    relaxations (bounds, BP, approximate contraction).  The spin/natural offsets are
    folded into one scalar table so that the offset convention stays h.offset."""
    if basis == "natural" and getattr(h, "factors", None) is not None:
        tabs = [(tuple(v), np.asarray(t, dtype=float)) for v, t in h.factors]
        const = float(getattr(h, "factors_offset", 0.0)) - h.offset
        seen = set(i for v, _ in tabs for i in v)
        tabs += [((v,), np.zeros(2)) for v in range(h.n) if v not in seen]
        if const != 0.0:
            tabs.append(((), np.array(const)))
        return tabs
    if basis in ("spin", "natural"):
        J, off = h.to_ising()
        tabs = _tables_from_terms(h.n, J, merge, _spin_monomial)
        const = off - h.offset
        if const != 0.0:
            tabs.append(((), np.array(const)))
        return tabs
    return _tables_from_terms(h.n, h.terms, merge, _bit_monomial)


def _factor_tables_old(h: HOBO, merge: bool = True) -> List[Tuple[Tuple[int, ...], np.ndarray]]:
    scopes = sorted(h.terms, key=len, reverse=True)
    if merge:
        maximal: List[Tuple[int, ...]] = []
        for s in scopes:
            if not any(set(s) <= set(m) for m in maximal):
                maximal.append(s)
    else:
        maximal = scopes
    tables = {m: np.zeros((2,) * len(m)) for m in maximal}
    for s, c in h.terms.items():
        host = next(m for m in maximal if set(s) <= set(m)) if merge else s
        pos = [host.index(i) for i in s]
        # add c * prod_{i in s} x_i to the host table
        grids = np.indices((2,) * len(host))
        mask = np.ones((2,) * len(host), dtype=bool)
        for p in pos:
            mask &= grids[p] == 1
        tables[host] = tables[host] + c * mask
    # variables that appear in no term still need a leg (free bit)
    seen = set(i for m in maximal for i in m)
    out = [(m, t) for m, t in tables.items()]
    for v in range(h.n):
        if v not in seen:
            out.append(((v,), np.zeros(2)))
    return out


def optimise_tree(tables, method: str = "hyper", max_repeats: int = 64, target_size: Optional[int] = None,
                  max_time: Optional[float] = None, seed: int = 0, parallel="auto"):
    """cotengra contraction tree for the factor network (all indices of size 2,
    variables as hyperedges, scalar output)."""
    import cotengra as ctg
    inputs = [tuple(str(i) for i in vars_) for vars_, _ in tables]
    size = {str(i): 2 for vars_, _ in tables for i in vars_}
    kw = dict(minimize="flops", progbar=False, parallel=parallel)
    if target_size is not None:
        kw["slicing_reconf_opts"] = {"target_size": target_size}
    if method == "greedy":
        opt = ctg.HyperOptimizer(methods=["greedy"], max_repeats=max_repeats, **kw)
    elif method == "hyper":
        opt = ctg.HyperOptimizer(methods=["greedy", "kahypar"], max_repeats=max_repeats,
                                 reconf_opts={}, max_time=max_time, **kw)
    else:
        raise ValueError(method)
    return opt.search(inputs, (), size)


def elimination_stats(h: HOBO, order=None):
    """log2 of the summed table sizes of plain min-fill variable elimination
    (hobo.py), comparable with tree_stats() below."""
    if order is None:
        order, w = h.elimination_order()
    adj = {i: set() for i in range(h.n)}
    for idx in h.terms:
        for a in idx:
            adj[a].update(j for j in idx if j != a)
    cost, width = 0.0, 0
    for v in order:
        nb = adj[v]
        cost += 2.0 ** (len(nb) + 1)
        width = max(width, len(nb))
        for a in nb:
            adj[a].update(nb - {a})
            adj[a].discard(v)
        adj[v] = set()
    return {"log2_cost": math.log2(cost), "width": width}


def tree_stats(tree):
    return {"log2_cost": math.log2(max(float(tree.contraction_cost()), 1.0)),
            "width": float(tree.contraction_width()),
            "n_sliced": len(getattr(tree, "sliced_inds", {}) or {})}


# -------------------------------------------------------------------- semirings
class Tropical:
    """(min, +) on energies."""
    name = "tropical"

    @staticmethod
    def leaf(E):
        return E

    @staticmethod
    def mul(a, b):
        return a + b

    @staticmethod
    def sum(a, axes):
        return a.min(axis=axes) if axes else a

    @staticmethod
    def add_many(vals):
        return min(vals)


class CountingTropical:
    """(E_min, number of minimisers).  Counts in float64 (exact below 2^53)."""
    name = "counting"
    tol = 1e-9

    @staticmethod
    def leaf(E):
        return (E, np.ones_like(E))

    @staticmethod
    def mul(a, b):
        return (a[0] + b[0], a[1] * b[1])

    @classmethod
    def sum(cls, a, axes):
        if not axes:
            return a
        E, C = a
        m = E.min(axis=axes, keepdims=True)
        C = np.where(np.isclose(E, m, rtol=1e-12, atol=cls.tol), C, 0.0).sum(axis=axes)
        return (m.squeeze(axis=axes), C)

    @classmethod
    def add_many(cls, vals):
        m = min(float(v[0]) for v in vals)
        return (m, sum(float(np.sum(v[1])) for v in vals if np.isclose(float(v[0]), m, rtol=1e-12, atol=cls.tol)))


class LogSumExp:
    """(logsumexp, +) on -beta E: contraction gives log Z."""
    name = "logsumexp"

    @staticmethod
    def leaf(L):
        return L

    @staticmethod
    def mul(a, b):
        return a + b

    @staticmethod
    def sum(a, axes):
        if not axes:
            return a
        m = a.max(axis=axes, keepdims=True)
        m = np.where(np.isfinite(m), m, 0.0)
        return (m + np.log(np.exp(a - m).sum(axis=axes, keepdims=True))).squeeze(axis=axes)

    @staticmethod
    def add_many(vals):
        v = np.array([float(x) for x in vals])
        m = v.max()
        return float(m + np.log(np.exp(v - m).sum()))


class TruncatedPoly:
    """Low-energy spectrum: element = (E_min, counts of E_min, E_min+u, ..., E_min+(K-1)u).
    Energies must be integer multiples of the unit u (set LEVELS/UNIT before use).
    Counts in float64.  This is GenericTensorNetworks' TruncatedPoly / CountingMin(K)."""
    name = "truncpoly"
    K = 8
    UNIT = 1.0

    @classmethod
    def leaf(cls, E):
        C = np.zeros(E.shape + (cls.K,))
        C[..., 0] = 1.0
        return (E, C)

    @classmethod
    def mul(cls, a, b):
        Ea, Ca = a
        Eb, Cb = b
        E = Ea + Eb
        shp = np.broadcast_shapes(Ca.shape, Cb.shape)
        C = np.zeros(shp)
        for k in range(cls.K):
            C[..., k:] += Ca[..., k:k + 1] * Cb[..., : cls.K - k]
        return (E, C)

    @classmethod
    def sum(cls, a, axes):
        if not axes:
            return a
        E, C = a
        E = np.broadcast_to(E, C.shape[:-1])
        m = E.min(axis=axes, keepdims=True)
        shift = np.rint((E - m) / cls.UNIT).astype(np.int64)
        out = np.zeros(np.squeeze(m, axis=axes).shape + (cls.K,))
        for s in range(cls.K):
            mask = (shift == s)[..., None]
            part = np.where(mask, C, 0.0).sum(axis=axes)
            out[..., s:] += part[..., : cls.K - s]
        return (np.squeeze(m, axis=axes), out)

    @classmethod
    def add_many(cls, vals):
        m = min(float(v[0]) for v in vals)
        out = np.zeros(cls.K)
        for e, c in vals:
            s = int(round((float(e) - m) / cls.UNIT))
            if s < cls.K:
                out[s:] += np.asarray(c).reshape(-1, cls.K).sum(axis=0)[: cls.K - s]  # sum any leading dims
        return (m, out)


# ------------------------------------------------------------------ contraction
class TNHOBO:
    def __init__(self, h: HOBO, tree=None, merge: bool = True, basis: str = "bit", **tree_kw):
        self.h = h
        tabs = factor_tables(h, merge, basis)
        self.const = sum(float(t) for v, t in tabs if len(v) == 0)
        self.tables = [(v, t) for v, t in tabs if len(v) > 0]
        t0 = time.perf_counter()
        self.tree = tree if tree is not None else optimise_tree(self.tables, **tree_kw)
        self.t_path = time.perf_counter() - t0
        self.sliced = [int(i) for i in (getattr(self.tree, "sliced_inds", None) or {})]
        self.path = self.tree.get_path()

    # one contraction of a (possibly sliced) network; keeps the steps for top-down passes
    def _contract(self, sr, leaves, keep=False):
        tensors = [(tuple(v), sr.leaf(t)) for v, t in leaves]
        refs = {}
        for v, _ in tensors:
            for i in v:
                refs[i] = refs.get(i, 0) + 1
        steps = []
        for (p, q) in self.path:
            p, q = sorted((p, q))
            (vb, B) = tensors.pop(q)
            (va, A) = tensors.pop(p)
            union = list(dict.fromkeys(va + vb))
            for i in va:
                refs[i] -= 1
            for i in vb:
                refs[i] -= 1
            out = tuple(i for i in union if refs[i] > 0)
            Ab = self._bcast(sr, A, va, union)
            Bb = self._bcast(sr, B, vb, union)
            S = sr.mul(Ab, Bb)
            axes = tuple(k for k, i in enumerate(union) if i not in out)
            # move output indices to the front in `out` order before reducing
            R = sr.sum(S, axes)
            for i in out:
                refs[i] += 1
            if keep:
                steps.append((va, vb, tuple(union), out, S))
            tensors.append((out, R))
        assert len(tensors) == 1 and tensors[0][0] == ()
        val = tensors[0][1]
        return val, steps

    @staticmethod
    def _bcast(sr, A, va, union):
        def one(a):
            a = np.asarray(a, dtype=float)
            perm = sorted(range(len(va)), key=lambda k: union.index(va[k]))
            a = np.transpose(a, perm) if len(va) > 1 else a
            shape = [1] * len(union)
            for k in perm:
                shape[union.index(va[k])] = 2
            return a.reshape(shape) if len(union) else a
        if isinstance(A, tuple):
            if A[1].ndim == np.ndim(A[0]) + 1:  # TruncatedPoly: trailing level axis
                E = one(A[0])
                Cs = np.stack([one(A[1][..., k]) for k in range(A[1].shape[-1])], axis=-1)
                return (E, Cs)
            return tuple(one(x) for x in A)
        return one(A)

    def _leaves(self, fix: Dict[int, int], transform=lambda t: t):
        out = []
        for vars_, t in self.tables:
            t = transform(t)
            if fix and any(v in fix for v in vars_):
                sl = tuple(fix[v] if v in fix else slice(None) for v in vars_)
                t = t[sl]
                vars_ = tuple(v for v in vars_ if v not in fix)
            out.append((vars_, t))
        return out

    def _slices(self):
        for vals in itertools.product((0, 1), repeat=len(self.sliced)):
            yield dict(zip(self.sliced, vals))

    def _run(self, sr, transform=lambda t: t, keep=False):
        vals, steps = [], []
        for fix in self._slices():
            v, st = self._contract(sr, self._leaves(fix, transform), keep)
            vals.append(v)
            steps.append((fix, st))
        return vals, steps

    # ---------------------------------------------------------------- queries
    def ground_energy(self):
        vals, _ = self._run(Tropical)
        return float(Tropical.add_many([float(v) for v in vals])) + self.h.offset + self.const

    def ground_count(self):
        vals, _ = self._run(CountingTropical)
        E, C = CountingTropical.add_many([(float(a), float(b)) for a, b in vals])
        return E + self.h.offset + self.const, C

    def log_partition(self, beta):
        vals, _ = self._run(LogSumExp, transform=lambda t: -beta * t)
        return LogSumExp.add_many([float(v) for v in vals]) - beta * (self.h.offset + self.const)

    def spectrum(self, K: int = 8, unit: float = 1.0):
        """(E_min, [#states at E_min + k*unit for k < K])."""
        TruncatedPoly.K, TruncatedPoly.UNIT = K, unit
        vals, _ = self._run(TruncatedPoly)
        E, C = TruncatedPoly.add_many([(float(a), np.asarray(b)) for a, b in vals])
        return E + self.h.offset + self.const, C

    def ground_state(self):
        vals, steps = self._run(Tropical, keep=True)
        k = int(np.argmin([float(v) for v in vals]))
        fix, st = steps[k]
        x = dict(fix)
        for va, vb, union, out, S in reversed(st):
            idx = tuple(x[i] if i in out else slice(None) for i in union)
            sub = S[idx]
            free = [i for i in union if i not in out]
            if free:
                flat = int(np.argmin(sub))
                for i, b in zip(free, np.unravel_index(flat, sub.shape)):
                    x[i] = int(b)
        xs = np.array([x.get(i, 0) for i in range(self.h.n)], dtype=np.int8)
        return float(self.h.energy(xs.astype(float))), xs

    def sample(self, beta, n_samples: int, seed: int = 0):
        """Exact Boltzmann samples: contract once in (logsumexp,+), then draw the
        contracted indices of each node top-down conditioned on its outputs.
        The stored S are on the -beta*E scale, so exp(S) is the unnormalised weight."""
        rng = np.random.default_rng(seed)
        vals, steps = self._run(LogSumExp, transform=lambda t: -beta * t, keep=True)
        w = np.array([float(v) for v in vals])
        p = np.exp(w - w.max())
        p /= p.sum()
        which = rng.choice(len(vals), size=n_samples, p=p)
        X = np.zeros((n_samples, self.h.n), dtype=np.int8)
        for k in np.unique(which):
            rows = np.nonzero(which == k)[0]
            fix, st = steps[k]
            vals_ = {i: np.full(len(rows), b, dtype=np.int64) for i, b in fix.items()}
            for va, vb, union, out, S in reversed(st):
                free = [i for i in union if i not in out]
                if not free:
                    continue
                # gather S[out values per sample, :free]
                Sf = np.moveaxis(S, [union.index(i) for i in out] + [union.index(i) for i in free],
                                 list(range(len(union))))
                Sf = np.broadcast_to(Sf, (2,) * len(union)).reshape((2,) * len(out) + (-1,))
                logits = Sf[tuple(vals_[i] for i in out)] if out else np.broadcast_to(Sf, (len(rows), Sf.shape[-1]))
                logits = logits - logits.max(axis=1, keepdims=True)
                pr = np.exp(logits)
                pr /= pr.sum(axis=1, keepdims=True)
                u = rng.random((len(rows), 1))
                choice = (pr.cumsum(axis=1) < u).sum(axis=1)
                choice = np.minimum(choice, pr.shape[1] - 1)
                bits = np.unravel_index(choice, (2,) * len(free))
                for i, b in zip(free, bits):
                    vals_[i] = b.astype(np.int64)
            for i in range(self.h.n):
                if i in vals_:
                    X[rows, i] = vals_[i]
                else:  # variable in no factor: uniform
                    X[rows, i] = rng.integers(0, 2, len(rows))
        return X


# ------------------------------------------------------------------ self test
if __name__ == "__main__":
    import instances as I

    rng = np.random.default_rng(0)
    cases = [("3spin n=20", I.random_kspin(20, 40, 3, 1)), ("3R3X n=20", I.xorsat_planted(20, 3, 2)[0]),
             ("LABS N=14", I.labs(14)), ("plaq L=4", I.plaquette_2d(4, 3)), ("tri L=4", I.triangle_2d(4, 2)),
             ("3SAT n=16", I.random_ksat(16, 70, 3, 4))]
    for name, h in cases:
        X = ((np.arange(1 << h.n)[:, None] >> np.arange(h.n)[None, :]) & 1).astype(float)
        E = h.energy(X)
        E0, C0 = E.min(), int(np.sum(np.abs(E - E.min()) < 1e-9))
        for tgt in (None, 2 ** 4):
            tn = TNHOBO(h, method="greedy", max_repeats=8, target_size=tgt)
            e = tn.ground_energy()
            e2, c = tn.ground_count()
            es, xs = tn.ground_state()
            beta = 0.7
            lz = tn.log_partition(beta)
            lz_ref = float(np.log(np.exp(-beta * (E - E0)).sum()) - beta * E0)
            assert abs(e - E0) < 1e-8 and abs(e2 - E0) < 1e-8 and c == C0 and abs(es - E0) < 1e-8, (e, e2, c, C0, es)
            assert abs(lz - lz_ref) < 1e-7, (lz, lz_ref)
            print(f"{name:11s} slices={2 ** len(tn.sliced):3d}  E0={e:+8.2f}  #ground={int(c):4d}  logZ ok  argmin ok")
        # exact sampling vs enumeration: compare <E> and a 2-point marginal
        tn = TNHOBO(h, method="greedy", max_repeats=8, target_size=2 ** 4)
        S = tn.sample(0.5, 40000, seed=1)
        Es = h.energy(S.astype(float))
        w = np.exp(-0.5 * (E - E0)); w /= w.sum()
        print(f"{'':11s} sampling beta=0.5: <E> exact {np.dot(w, E):+.3f}  sampled {Es.mean():+.3f} ± {Es.std() / 200:.3f}")
