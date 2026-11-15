"""E8 (one GPU): exact tropical + counting contraction beyond CPU reach.
3R3X n=150 (planted, degeneracy checked against GF(2)) and 2D plaquette / triangle
L=16 (n=256).  SA baseline run on the same GPU (1024 replicas, 1e3 / 1e4 sweeps).
VRAM is capped at 30% of the card."""
import json, sys, time, os
sys.path.insert(0, "..")
import numpy as np, torch
torch.cuda.set_per_process_memory_fraction(0.30, 0)
import instances as I
from tn import TNHOBO
from tn_torch import ground_torch
from sa import SA

jobs = []
which = sys.argv[1]
seeds = [int(a) for a in sys.argv[2:]] or [0]
if which == "xorsat":
    for s in seeds:
        h, A, xs = I.xorsat_planted(150, 3, s)
        jobs.append(("3R3X", 150, s, h, {"E0_planted": -float(A.shape[0]), "deg_gf2": 2.0 ** (150 - I.gf2_rank(A))}))
else:
    for fam, gen in (("plaq2d", I.plaquette_2d), ("tri2d", I.triangle_2d)):
        for s in seeds:
            jobs.append((fam, 16, s, gen(16, s), {}))
out = []
for fam, size, s, h, extra in jobs:
    rec = {"family": fam, "size": size, "seed": s, **extra}
    t0 = time.time()
    tn = TNHOBO(h, method="hyper", max_repeats=2048, max_time=300, target_size=2 ** 26, basis="natural", parallel=False)
    rec["path_sec"] = time.time() - t0
    rec["width"] = float(tn.tree.contraction_width())
    rec["log2_cost"] = float(np.log2(float(tn.tree.contraction_cost())))
    rec["slices"] = 2 ** len(tn.sliced)
    print(f"{fam} {size} s{s}: tree width {rec['width']:.0f} log2cost {rec['log2_cost']:.1f} slices {rec['slices']} ({rec['path_sec']:.0f}s)", flush=True)
    E, C, sec = ground_torch(tn, "cuda", torch.float32, counting=True, chunk_elems=1 << 24, progress=True)
    rec.update({"E0": E, "count": C, "tn_sec": sec, "peak_mem_GB": torch.cuda.max_memory_allocated(0) / 2 ** 30})
    for sw in (1000, 10000):
        r = SA(h, device="cuda").run(1024, sw, seed=s)
        rec[f"sa{sw}"] = {"best": float(r["best_E"].min()), "p": float(np.mean(np.abs(r["best_E"] - E) < 1e-6)), "sec": r["seconds"]}
    out.append(rec)
    print(f"{fam} {size} s{s}: E0 {E:+.0f} #={C:.4g} ({sec:.0f}s, peak {rec['peak_mem_GB']:.1f}GB)"
          + (f" | planted {extra['E0_planted']:+.0f} GF2 {extra['deg_gf2']:.0f}" if extra else "")
          + f" | SA1e3 {rec['sa1000']['best']:+.0f} P={rec['sa1000']['p']:.3f} | SA1e4 {rec['sa10000']['best']:+.0f} P={rec['sa10000']['p']:.3f} ({rec['sa10000']['sec']:.0f}s)", flush=True)
    json.dump(out, open(f"e8_gpu_{which}_{'_'.join(map(str, seeds))}.json", "w"), indent=1)
