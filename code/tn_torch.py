"""Tropical / counting-tropical contraction on GPU (torch), along the same
cotengra tree (and slices) as tn.TNHOBO.

cuTensorNet / cuTENSOR only contract in (+, x), so the path and the slicing are
planned by cotengra (hyper-optimised, the same kind of tree cuTensorNet builds)
and each pairwise contraction is executed here as a batched min-plus matrix
product, chunked over the summed index so that memory stays at the size of the
output tensor:

    R[b, i, j] = min_k  A[b, i, k] + B[b, k, j]        (b: shared kept indices)

For the counting semiring the multiplicities are carried along (float64).
"""
from __future__ import annotations

import itertools
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from tn import TNHOBO


def _perm_reshape(T, labels, groups):
    """Permute tensor T (axes labelled `labels`) into the concatenation of `groups`
    and reshape to one axis per group."""
    order = [labels.index(l) for g in groups for l in g]
    T = T.permute(*order) if order else T
    return T.reshape([1 << len(g) for g in groups])


def _reduce_only(E, C, labels, drop):
    """min (and counting) over indices that occur in this tensor only."""
    if not drop:
        return E, C, labels
    axes = [labels.index(l) for l in drop]
    keep = [l for l in labels if l not in drop]
    m = E.amin(dim=axes, keepdim=True)
    if C is not None:
        C = torch.where(E == m, C, torch.zeros((), dtype=C.dtype, device=C.device)).sum(dim=axes)
    return m.reshape([E.shape[i] for i in range(E.ndim) if i not in axes]), C, keep


def contract_torch(tn: TNHOBO, device: str = "cuda", dtype=torch.float32, counting: bool = False,
                   chunk_elems: int = 1 << 27, fix: Optional[Dict[int, int]] = None) -> Tuple[float, Optional[float]]:
    """Contract one slice (fix = values of the sliced indices).  Returns (E_min, count)."""
    leaves = tn._leaves(fix or {})
    tensors = []
    for v, t in leaves:
        E = torch.as_tensor(np.asarray(t, dtype=np.float64), dtype=dtype, device=device)
        C = torch.ones_like(E, dtype=torch.float64) if counting else None
        tensors.append((list(v), E, C))
    refs: Dict[int, int] = {}
    for v, _, _ in tensors:
        for i in v:
            refs[i] = refs.get(i, 0) + 1
    for (p, q) in tn.path:
        p, q = sorted((p, q))
        vb, Eb, Cb = tensors.pop(q)
        va, Ea, Ca = tensors.pop(p)
        for i in va:
            refs[i] -= 1
        for i in vb:
            refs[i] -= 1
        # indices that disappear and live in one operand only: reduce them first
        only_a = [i for i in va if i not in vb and refs[i] == 0]
        only_b = [i for i in vb if i not in va and refs[i] == 0]
        Ea, Ca, va = _reduce_only(Ea, Ca, va, only_a)
        Eb, Cb, vb = _reduce_only(Eb, Cb, vb, only_b)
        shared = [i for i in va if i in vb]
        batch = [i for i in shared if refs[i] > 0]
        summed = [i for i in shared if refs[i] == 0]
        a_out = [i for i in va if i not in vb]
        b_out = [i for i in vb if i not in va]
        A = _perm_reshape(Ea, va, [batch, a_out, summed])
        B = _perm_reshape(Eb, vb, [batch, summed, b_out])
        if counting:
            CA = _perm_reshape(Ca, va, [batch, a_out, summed])
            CB = _perm_reshape(Cb, vb, [batch, summed, b_out])
        nb, na, nk = A.shape
        nbo = B.shape[2]
        step = max(1, min(nk, chunk_elems // max(1, nb * na * nbo)))
        R = None
        RC = None
        for k0 in range(0, nk, step):
            k1 = min(nk, k0 + step)
            T = A[:, :, k0:k1, None] + B[:, None, k0:k1, :]           # (nb, na, ch, nbo)
            m = T.amin(dim=2)
            if counting:
                TC = CA[:, :, k0:k1, None] * CB[:, None, k0:k1, :]
                c = torch.where(T == m[:, :, None, :], TC, torch.zeros((), dtype=TC.dtype, device=TC.device)).sum(dim=2)
            if R is None:
                R = m
                RC = c if counting else None
            else:
                newR = torch.minimum(R, m)
                if counting:
                    RC = torch.where(R == newR, RC, 0.0) + torch.where(m == newR, c, 0.0)
                R = newR
        out_labels = batch + a_out + b_out
        R = R.reshape([2] * len(out_labels)) if out_labels else R.reshape(())
        if counting:
            RC = RC.reshape([2] * len(out_labels)) if out_labels else RC.reshape(())
        for i in out_labels:
            refs[i] += 1
        tensors.append((out_labels, R, RC))
    assert len(tensors) == 1
    _, E, C = tensors[0]
    # scalar tables that never met a contraction (isolated tensors)
    return float(E.reshape(-1).min()), (float(C.reshape(-1).sum()) if counting else None)


def ground_torch(tn: TNHOBO, device="cuda", dtype=torch.float32, counting=False, progress=False, **kw):
    t0 = time.perf_counter()
    if counting and dtype != torch.float64:
        # ties are detected with ==, which is exact only if every partial sum is exactly
        # representable: integer-valued tables below 2^24 in float32.  Otherwise use float64.
        if not all(np.all(np.asarray(t) == np.round(t)) and np.abs(t).sum() < 2 ** 20 for _, t in tn.tables):
            dtype = torch.float64
    vals = []
    n_sl = 2 ** len(tn.sliced)
    for k, fix in enumerate(tn._slices()):
        vals.append(contract_torch(tn, device, dtype, counting, fix=fix, **kw))
        if progress and (k + 1) % max(1, n_sl // 32) == 0:
            el = time.perf_counter() - t0
            print(f"  slice {k + 1}/{n_sl}  {el:.0f}s  eta {el / (k + 1) * (n_sl - k - 1):.0f}s  best {min(v[0] for v in vals) + tn.h.offset + tn.const:+.1f}",
                  flush=True)
    E = min(v[0] for v in vals)
    tol = 1e-3 if dtype == torch.float32 else 1e-9
    C = sum(v[1] for v in vals if abs(v[0] - E) <= tol * (1 + abs(E))) if counting else None
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return E + tn.h.offset + tn.const, C, time.perf_counter() - t0


if __name__ == "__main__":
    import instances as I
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for name, h in [("3spin n=20", I.random_kspin(20, 40, 3, 1)), ("LABS 14", I.labs(14)),
                    ("plaq L=6", I.plaquette_2d(6, 1)), ("3SAT n=30", I.random_ksat(30, 128, 3, 1))]:
        for tgt in (None, 2 ** 10):
            tn = TNHOBO(h, method="greedy", max_repeats=8, target_size=tgt)
            if len(tn.sliced) > 8:
                continue
            e_ref, c_ref = tn.ground_count()
            e, c, t = ground_torch(tn, dev, torch.float64, counting=True, chunk_elems=1 << 10)
            assert abs(e - e_ref) < 1e-6 and abs(c - c_ref) < 0.5, (e, e_ref, c, c_ref)
            print(f"{name:11s} slices={2 ** len(tn.sliced):3d} {dev}: E0={e:+.1f} #={c:.0f}  == numpy  ({t:.2f}s)")
