"""Exact solving beyond one contraction: slices + bounds (branch and bound over
the slice tree).

cotengra picks the sliced indices S (the cut that bounds the intermediate
size).  Every assignment of S is an independent subproblem.  Before contracting
a slice we bound it from below with the width-bounded relaxation of bounds.py
(dual ascent is done once on the full network; each slice just conditions the
reparameterised tables and runs mini-bucket).  A slice whose lower bound
cannot beat the incumbent (from SA, then improved by contracted slices) is
skipped.  The result is exact: every skipped slice is certified not to contain
a better solution.
"""
from __future__ import annotations

import itertools
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from bounds import dual_ascent, mini_bucket, minfill_order
from hobo import HOBO
from tn import TNHOBO
from tn_torch import contract_torch


def condition(tables, fix: Dict[int, int]):
    out = []
    const = 0.0
    for v, t in tables:
        t = np.asarray(t, dtype=float)
        if len(v) == 0:
            const += float(t)
            continue
        if any(i in fix for i in v):
            t = t[tuple(fix[i] if i in fix else slice(None) for i in v)]
            v = tuple(i for i in v if i not in fix)
            if len(v) == 0:
                const += float(t)
                continue
        out.append((tuple(v), t))
    out.append(((), np.array(const)))
    return out


def solve_bnb(h: HOBO, ub: float, unit: float, target_size: int = 2 ** 20, ibound: int = 16,
              basis: str = "natural", device: str = "cpu", tree_time: float = 60, log=print) -> dict:
    t0 = time.perf_counter()
    tn = TNHOBO(h, method="hyper", max_repeats=512, max_time=tree_time, target_size=target_size, basis=basis)
    t_tree = time.perf_counter() - t0
    da = dual_ascent(h, 200, basis=basis)
    rep = da["tables"]
    sliced = tn.sliced
    order_all = minfill_order(rep, h.n)
    slices = [dict(zip(sliced, vals)) for vals in itertools.product((0, 1), repeat=len(sliced))]
    t1 = time.perf_counter()
    lbs = []
    for fix in slices:
        tabs = condition(rep, fix)
        order = [v for v in order_all if v not in fix]
        lbs.append(mini_bucket(h, ibound, order, True, tabs)["lb"])
    t_bound = time.perf_counter() - t1
    best = ub
    contracted = 0
    t2 = time.perf_counter()
    for k in np.argsort(lbs):
        if lbs[k] > best - unit + 1e-6:  # cannot contain anything strictly better than the incumbent
            continue
        e, _ = contract_torch(tn, device, torch.float64, counting=False, fix=slices[k])
        e += h.offset + tn.const
        contracted += 1
        best = min(best, e)
    t_contract = time.perf_counter() - t2
    return {"E0": best, "n_slices": len(slices), "contracted": contracted,
            "width": float(tn.tree.contraction_width()), "log2_cost_total": float(np.log2(float(tn.tree.contraction_cost()))),
            "t_tree": t_tree, "t_dual": da["seconds"], "t_bound": t_bound, "t_contract": t_contract,
            "lb_min": float(min(lbs)), "lbs": [float(x) for x in lbs]}


if __name__ == "__main__":
    import sys
    import instances as I
    from sa import SA
    torch.set_num_threads(8)
    for fam, gen, L, s, E0_known in (("plaq2d", I.plaquette_2d, 12, 0, -136), ("tri2d", I.triangle_2d, 12, 0, -140),
                                     ("plaq2d", I.plaquette_2d, 12, 1, -130)):
        h = gen(L, s)
        ub = float(SA(h).run(1024, 2000, seed=s)["best_E"].min())
        for tgt in (2 ** 22, 2 ** 18):
            r = solve_bnb(h, ub, 2.0, target_size=tgt)
            print(f"{fam} L={L} s{s}: UB(SA) {ub:+.0f} -> E0 {r['E0']:+.0f} (known {E0_known}) | slices {r['n_slices']} contracted "
                  f"{r['contracted']} ({100 * r['contracted'] / r['n_slices']:.1f}%) | width {r['width']:.0f} | "
                  f"t tree {r['t_tree']:.0f}s dual {r['t_dual']:.0f}s bound {r['t_bound']:.0f}s contract {r['t_contract']:.0f}s", flush=True)
