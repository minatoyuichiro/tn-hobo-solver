"""E3 fix: equilibrium ground-state probability P0(beta) = N0 exp(-beta E0)/Z(beta)
for native and both quadratizations at ONE common beta per instance (the native
schedule's beta_cold), and also at 2x that beta.  E3 had used each variant's own
beta_cold, which differs when quadratization changes the smallest coefficient."""
import json, sys, time
sys.path.insert(0, "..")
import numpy as np
import instances as I
from tn import TNHOBO

FAM = {"3spin_m2n": lambda n, s: I.random_kspin(n, 2 * n, 3, s),
       "3SAT_4.26": lambda n, s: I.random_ksat(n, int(round(4.26 * n)), 3, s),
       "LABS": lambda n, s: I.labs(n)}
recs = json.load(open("../../results/e3_quad_3spin_3sat.json")) + json.load(open("../../results/e3_quad_LABS.json"))
out = []
for r in recs:
    h = FAM[r["family"]](r["size"], r["seed"])
    beta = r["native"]["beta_cold"]
    variants = {"native": h, "quad_A+1": h.quadratize(1.0, 1.0)[0], "quad_1.5A": h.quadratize(1.5, 0.0)[0]}
    rec = {"family": r["family"], "size": r["size"], "seed": r["seed"], "beta": beta}
    for tag, hv in variants.items():
        assert hv.n == r[tag]["n_bits"]
        tn = TNHOBO(hv, method="greedy", max_repeats=16, target_size=2 ** 22, basis="natural" if tag == "native" else "bit")
        E0, N0 = r[tag]["E0"], r[tag]["spectrum"][0]
        p = {}
        for mult in (1.0, 2.0):
            lz = tn.log_partition(beta * mult)
            p[mult] = float(np.exp(np.log(N0) - beta * mult * E0 - lz))
        rec[tag] = {"P0_beta": p[1.0], "P0_2beta": p[2.0], "own_beta_cold": r[tag]["beta_cold"], "P0_own": r[tag]["P0_eq_beta_cold"]}
    out.append(rec)
    print(f"{r['family']:10s} n={r['size']:3d} s{r['seed']} beta={beta:.3f} | P0(beta) " +
          "/".join(f"{rec[t]['P0_beta']:.3f}" for t in variants) + " | P0(2beta) " +
          "/".join(f"{rec[t]['P0_2beta']:.3f}" for t in variants) + " | SA " +
          "/".join(f"{r[t]['sa1000_P0']:.3f}" for t in variants), flush=True)
    json.dump(out, open("../../results/e3_common_beta.json", "w"), indent=1)
