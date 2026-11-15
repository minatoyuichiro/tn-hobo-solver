"""E6: exact ground energy + degeneracy of 2D plaquette / triangle models at L=12
(n=144) with the torch min-plus executor on CPU, vs the SA upper bound of E2."""
import json, sys, time
sys.path.insert(0, "..")
import numpy as np, torch
import instances as I
from tn import TNHOBO
from tn_torch import ground_torch
from sa import SA
torch.set_num_threads(8)
out = []
for fam, gen in (("plaq2d", I.plaquette_2d), ("tri2d", I.triangle_2d)):
    for L in [int(a) for a in sys.argv[1:]] or [12]:
        for s in (0, 1):
            h = gen(L, s)
            t0 = time.time()
            tn = TNHOBO(h, method="hyper", max_repeats=1024, max_time=180, target_size=2 ** 27, basis="natural")
            tp = time.time() - t0
            E, C, sec = ground_torch(tn, "cpu", torch.float32, counting=True, chunk_elems=1 << 26)
            r = SA(h).run(1024, 2000, seed=s)
            rec = {"family": fam, "L": L, "seed": s, "width": float(tn.tree.contraction_width()),
                   "log2_cost": float(np.log2(float(tn.tree.contraction_cost()))), "slices": 2 ** len(tn.sliced),
                   "path_sec": tp, "E0": E, "count": C, "tn_sec": sec, "sa_best": float(r["best_E"].min()),
                   "sa_p": float(np.mean(np.abs(r["best_E"] - E) < 1e-6)), "sa_sec": r["seconds"]}
            out.append(rec)
            print(f"{fam} L={L} s{s}: width {rec['width']:.0f} log2cost {rec['log2_cost']:.1f} slices {rec['slices']} | "
                  f"E0 {E:+.0f} #={C:.4g} ({sec:.0f}s) | SA best {rec['sa_best']:+.0f} P={rec['sa_p']:.3f} ({rec['sa_sec']:.0f}s)", flush=True)
            json.dump(out, open("../../results/e6_lattice_exact.json", "w"), indent=1)
