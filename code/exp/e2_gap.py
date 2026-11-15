"""E2: certified optimality gaps beyond exact reach.
UB = best SA energy (1024 replicas x 2000 sweeps), LB = dual ascent + mini-bucket
(width-bounded tropical contraction) at several i-bounds.  Energies are integer
multiples of `unit`, so the instance is certified optimal when ceil(LB) >= UB."""
import json, math, sys, time
sys.path.insert(0, "..")
import os
import numpy as np
import instances as I
from sa import SA
from bounds import dual_ascent, mini_bucket, minfill_order

FAM = {
    "plaq2d": (lambda L, s: I.plaquette_2d(L, s), [8, 12, 16, 20, 24], 2),
    "tri2d": (lambda L, s: I.triangle_2d(L, s), [8, 12, 16, 20, 24], 2),
    "3spin_m2n": (lambda n, s: I.random_kspin(n, 2 * n, 3, s), [40, 60, 100, 150], 2),
    "3SAT_4.26": (lambda n, s: I.random_ksat(n, int(round(4.26 * n)), 3, s), [40, 60, 100, 150], 1),
    "3R3X": (lambda n, s: I.xorsat_planted(n, 3, s)[0], [40, 60, 100, 200], 2),
}
IB = [8, 12, 16, 20]
BASIS = os.environ.get("BASIS", "natural")
fams = sys.argv[1:] or list(FAM)
out = []
for fam in fams:
    gen, sizes, unit = FAM[fam]
    for size in sizes:
        for s in (0, 1):
            h = gen(size, s)
            rec = {"family": fam, "size": size, "seed": s, "n": h.n, "unit": unit, "basis": BASIS}
            t0 = time.time(); r = SA(h).run(1024, 2000, seed=s)
            rec["ub"] = float(r["best_E"].min()); rec["ub_sec"] = time.time() - t0
            rec["ub_hits"] = int(np.sum(np.abs(r["best_E"] - rec["ub"]) < 1e-6))
            da = dual_ascent(h, 300, time_limit=600, basis=BASIS)
            rec["lb_dual"] = da["lb"]; rec["dual_sec"] = da["seconds"]; rec["dual_iters"] = da["iters"]
            order = minfill_order(da["tables"], h.n)
            rec["mbe"] = {}
            for ib in IB:
                a = mini_bucket(h, ib, order, True, da["tables"])
                b = mini_bucket(h, ib, order, False, da["tables"])
                rec["mbe"][ib] = {"lb": max(a["lb"], b["lb"]), "exact": a["exact"], "sec": a["seconds"] + b["seconds"]}
            lb = max([rec["lb_dual"]] + [v["lb"] for v in rec["mbe"].values()])
            rec["lb"] = lb
            lbr = math.ceil(lb / unit - 1e-6) * unit
            rec["certified"] = lbr >= rec["ub"] - 1e-6
            rec["gap"] = rec["ub"] - lbr
            out.append(rec)
            print(f"{fam:10s} {size:4d} s{s} n={h.n:4d} | UB {rec['ub']:+8.1f} ({rec['ub_hits']:4d}/1024) | dual {da['lb']:+9.1f}"
                  + "".join(f" | i{ib} {v['lb']:+9.1f}{'*' if v['exact'] else ''} ({v['sec']:.0f}s)" for ib, v in rec["mbe"].items())
                  + f" | gap {rec['gap']:.0f}{'  CERTIFIED' if rec['certified'] else ''}", flush=True)
            json.dump(out, open(f"../../results/e2{BASIS}_gap_{'_'.join(fams)}.json", "w"), indent=1)
