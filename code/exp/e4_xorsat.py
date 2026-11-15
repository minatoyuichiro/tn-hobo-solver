"""E4: planted 3-regular 3-XORSAT, the classic case where SA stalls.
Exact tropical/counting contraction (cotengra hyper tree, torch executor with
slicing) vs SA at increasing effort.  Ground energy is -n by construction and the
ground-state degeneracy is 2^(n - rank_GF2(A)), so both TN outputs are checked
against independent algebra."""
import json, sys, time
sys.path.insert(0, "..")
import numpy as np, torch
import instances as I
from tn import TNHOBO
from tn_torch import ground_torch
from sa import SA

torch.set_num_threads(8)
out = []
for n in [int(a) for a in sys.argv[1:]] or [60, 80, 100]:
    for s in (0, 1, 2):
        h, A, xs = I.xorsat_planted(n, 3, s)
        rec = {"n": n, "seed": s, "E0_planted": -float(A.shape[0]), "deg_gf2": 2.0 ** (n - I.gf2_rank(A))}
        t0 = time.time()
        tn = TNHOBO(h, method="hyper", max_repeats=512, max_time=120, target_size=2 ** 26, basis="natural")
        rec["path_sec"] = time.time() - t0
        rec["width"] = float(tn.tree.contraction_width()); rec["log2_cost"] = float(np.log2(float(tn.tree.contraction_cost())))
        rec["slices"] = 2 ** len(tn.sliced)
        E, C, sec = ground_torch(tn, "cpu", torch.float32, counting=True, chunk_elems=1 << 26)
        rec.update({"tn_E0": E, "tn_count": C, "tn_sec": sec})
        for sw in (1000, 10000):
            t0 = time.time(); r = SA(h).run(1024, sw, seed=s)
            rec[f"sa{sw}"] = {"best": float(r["best_E"].min()), "p_ground": float(np.mean(np.abs(r["best_E"] - rec["E0_planted"]) < 1e-6)),
                              "sec": time.time() - t0}
        out.append(rec)
        print(f"3R3X n={n} s{s}: width {rec['width']:.0f} log2cost {rec['log2_cost']:.1f} slices {rec['slices']} | "
              f"TN E0 {E:+.0f} (planted {rec['E0_planted']:+.0f}) #={C:.0f} (GF2 {rec['deg_gf2']:.0f}) {sec:.1f}s | "
              f"SA1e3 best {rec['sa1000']['best']:+.0f} P={rec['sa1000']['p_ground']:.3f} | SA1e4 best {rec['sa10000']['best']:+.0f} "
              f"P={rec['sa10000']['p_ground']:.3f} ({rec['sa10000']['sec']:.0f}s)", flush=True)
        json.dump(out, open(f"../../results/e4_xorsat_{'_'.join(sys.argv[1:]) or 'default'}.json", "w"), indent=1)
