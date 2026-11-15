"""E10: SVD-compressed contraction (quimb contract_compressed) vs exact log Z on the 2D
plaquette model, L=6 (Sec. 7.8).  Records value, relative error, sign of the error, time."""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import quimb.tensor as qtn
import instances as I
from tn import TNHOBO, factor_tables

out = []
L = 6
h = I.plaquette_2d(L, 0)
ex = TNHOBO(h, method="greedy", max_repeats=8, basis="natural", parallel=False)
for beta in (0.3, 1.0, 3.0):
    t0 = time.time(); lz = ex.log_partition(beta); t_exact = time.time() - t0
    ts = [qtn.Tensor(np.exp(-beta * np.asarray(t)), inds=[f"x{i}" for i in v]) for v, t in factor_tables(h, basis="natural") if len(v)]
    tn = qtn.TensorNetwork(ts).hyperinds_resolve("tree")
    for chi in (4, 16, 64):
        t0 = time.time()
        m, e = tn.contract_compressed(optimize="greedy", max_bond=chi, strip_exponent=True, progbar=False)
        val = float(np.log(abs(m)) + e * np.log(10))
        rec = {"L": L, "beta": beta, "chi": chi, "logZ_exact": lz, "logZ_compressed": val, "rel_err": abs(val - lz) / abs(lz),
               "signed_err": val - lz, "sec": time.time() - t0, "sec_exact": t_exact}
        out.append(rec)
        print(f"L={L} beta={beta} chi={chi}: {val:.6f} vs exact {lz:.6f}  rel {rec['rel_err']:.1e}  signed {rec['signed_err']:+.2e}  {rec['sec']:.0f}s (exact {t_exact * 1e3:.0f} ms)", flush=True)
        json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "e10_compressed.json"), "w"), indent=1)
