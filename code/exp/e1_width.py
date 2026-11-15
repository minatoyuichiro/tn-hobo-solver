"""E1: how far does exact contraction reach?  Contraction cost (log2 flops-like)
and width of plain min-fill elimination vs cotengra greedy vs cotengra hyper
(kahypar + reconfiguration), per instance family and size.  Where the cost is
small enough the tropical contraction is actually run and timed."""
import json, sys, time, math
sys.path.insert(0, "..")
sys.path.insert(0, ".")
import numpy as np
import instances as I
from tn import TNHOBO, factor_tables, optimise_tree, elimination_stats, tree_stats

FAM = {
    "3R3X": (lambda n, s: I.xorsat_planted(n, 3, s)[0], [20, 40, 60, 80, 100, 150, 200]),
    "3spin_m2n": (lambda n, s: I.random_kspin(n, 2 * n, 3, s), [20, 30, 40, 50, 60, 80, 100]),
    "3SAT_4.26": (lambda n, s: I.random_ksat(n, int(round(4.26 * n)), 3, s), [20, 30, 40, 50, 60, 80, 100]),
    "LABS": (lambda n, s: I.labs(n), [10, 15, 20, 25, 30, 35, 40]),
    "plaq2d": (lambda L, s: I.plaquette_2d(L, s), [4, 6, 8, 10, 12, 16]),
    "tri2d": (lambda L, s: I.triangle_2d(L, s), [4, 6, 8, 10, 12, 16]),
}
RUN_MAX_LOG2 = 30
out = []
fams = sys.argv[1:] or list(FAM)
for fam in fams:
    gen, sizes = FAM[fam]
    for n in sizes:
        seeds = [0] if fam == "LABS" else [0, 1, 2]
        for s in seeds:
            h = gen(n, s)
            rec = {"family": fam, "size": n, "seed": s, "n": h.n, "terms": len(h), "order": h.order}
            t0 = time.time(); rec["minfill"] = elimination_stats(h); rec["minfill"]["t"] = time.time() - t0
            tabs = factor_tables(h)
            rec["n_tensors"] = len(tabs)
            t0 = time.time(); tg = optimise_tree(tabs, "greedy", max_repeats=32)
            rec["greedy"] = tree_stats(tg); rec["greedy"]["t"] = time.time() - t0
            t0 = time.time(); th = optimise_tree(tabs, "hyper", max_repeats=256, max_time=30)
            rec["hyper"] = tree_stats(th); rec["hyper"]["t"] = time.time() - t0
            if rec["hyper"]["log2_cost"] <= RUN_MAX_LOG2:
                tn = TNHOBO(h, tree=th)
                t0 = time.time(); E, C = tn.ground_count(); rec["run"] = {"E0": E, "count": C, "t": time.time() - t0}
            out.append(rec)
            r = rec
            print(f"{fam:10s} {n:4d} s{s} n={r['n']:4d} | minfill w={r['minfill']['width']:3d} c={r['minfill']['log2_cost']:6.1f}"
                  f" | greedy w={r['greedy']['width']:5.1f} c={r['greedy']['log2_cost']:6.1f}"
                  f" | hyper w={r['hyper']['width']:5.1f} c={r['hyper']['log2_cost']:6.1f} ({r['hyper']['t']:.0f}s)"
                  + (f" | E0={r['run']['E0']:+.0f} #={r['run']['count']:.0f} {r['run']['t']:.1f}s" if "run" in r else ""),
                  flush=True)
            json.dump(out, open(f"../../results/e1_width_{'_'.join(fams)}.json", "w"), indent=1)
